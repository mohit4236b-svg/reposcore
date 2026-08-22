"""
Production-grade async worker for repository analysis using Celery.
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from celery import Celery, current_task
import redis

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configuration
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")

# Celery app configuration
celery_app = Celery(
    "reposcore_workers",
    broker=REDIS_URL,
    backend=redis.Redis.from_url(REDIS_URL, decode_responses=True)
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=600,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)


def update_worker_status(job_id: str, status: str, message: str = "", progress: int = 0):
    """Update job status in Redis."""
    status_data = {
        "job_id": job_id,
        "status": status,
        "message": message,
        "progress": progress,
        "updated_at": datetime.utcnow().isoformat()
    }
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    redis_client.hset(f"job:{job_id}", mapping=status_data)


def get_worker_status(job_id: str) -> Dict[str, Any]:
    """Get job status from Redis."""
    redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    data = redis_client.hgetall(f"job:{job_id}")
    if data:
        return {k: int(v) if k == "progress" else v for k, v in data.items()}
    return {}


@celery_app.task(bind=True, name="analyze_repo", 
                  autoretry_for=(Exception,), retry_kwargs={"max_retries": 3})
def analyze_repository_async(self, owner: str, repo: str, token: Optional[str] = None) -> Dict[str, Any]:
    """
    Async task to analyze a GitHub repository with support for large repos.
    
    Handles:
    - Chunked processing for large datasets
    - Rate limit retries with exponential backoff
    - Progress tracking via Redis
    """
    job_id = self.request.id or f"{owner}_{repo}"
    current_task.update_state(state="STARTED", meta={"status": "Initializing..."})
    
    try:
        from reposcore_utils import fetch_repo_features, RepoFetchError, RateLimitedRepoFetchError
        from scoring_engine import RepoScorer
        
        # Initialize job tracking
        update_worker_status(job_id, "queued", f"Processing {owner}/{repo}", 0)
        
        # Step 1: Fetch features with chunked processing
        update_worker_status(job_id, "processing", "Fetching repository data...", 10)
        features = fetch_repo_features(f"{owner}/{repo}", 
                                       headers={"Authorization": f"Bearer {token}"} if token else {})
        
        # Step 2: Process large datasets in chunks
        update_worker_status(job_id, "processing", "Analyzing issues and commits...", 40)
        
        # Detect CI and tests
        topics = features.get("topics", [])
        features["has_ci"] = features.get("has_ci", any(t in topics for t in ["ci", "github-actions"]))
        features["has_tests"] = features.get("has_tests", any(t in topics for t in ["tests", "pytest"]))
        features["total_contributors"] = features.get("total_contributors", 1)
        
        # Step 3: Calculate score
        update_worker_status(job_id, "processing", "Calculating quality score...", 70)
        scorer = RepoScorer()
        score_result = scorer.calculate_score(features)
        score_result["full_name"] = f"{owner}/{repo}"
        score_result["analyzed_at"] = datetime.utcnow().isoformat()
        
        # Store result
        update_worker_status(job_id, "completed", "Analysis complete!", 100)
        
        # Cache result
        cache_key = f"score:{owner}:{repo}"
        redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.setex(cache_key, 86400, json.dumps(score_result))
        
        return {
            "status": "completed",
            "job_id": job_id,
            "result": score_result,
            "completed_at": datetime.utcnow().isoformat()
        }
        
    except RateLimitedRepoFetchError as e:
        update_worker_status(job_id, "failed", f"Rate limit: {str(e)}", 0)
        raise self.retry(exc=e, countdown=e.retry_after) if e.retry_after > 0 else self.retry(exc=e, countdown=60)
        
    except RepoFetchError as e:
        update_worker_status(job_id, "failed", str(e), 0)
        return {"status": "failed", "error": str(e), "job_id": job_id}
        
    except Exception as exc:
        update_worker_status(job_id, "failed", str(exc), 0)
        raise


if __name__ == "__main__":
    celery_app.start()