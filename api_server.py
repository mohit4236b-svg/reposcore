"""
FastAPI wrapper for production-grade RepoScore backend.
Provides REST APIs for async job processing, scoring, and badge generation.
"""

import uuid
import json
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import Response, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import redis

# Import our modules
from scoring_engine import RepoScorer
from cache_layer import get_cache
from badge_generator import create_badge_response, generate_shields_io_badge_url

app = FastAPI(
    title="RepoScore API",
    description="Production-grade GitHub repository quality scoring API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Redis client for job tracking - lazily initialized
_redis_client = None


def get_redis_client():
    """Get the global Redis client, creating it if needed."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    return _redis_client


@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.post("/api/jobs")
async def submit_analysis(owner: str = Query(...), repo: str = Query(...), 
                          threshold: float = Query(0.3, ge=0.0, le=1.0)):
    """
    Submit a repository for quality analysis.
    Returns a job ID for tracking progress.
    """
    job_id = str(uuid.uuid4())
    
    # Store job in Redis
    get_redis_client().hset(f"job:{job_id}", mapping={
        "owner": owner,
        "repo": repo,
        "threshold": threshold,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat()
    })
    
    # Trigger background processing
    # In production: push to Celery queue
    # For now: process synchronously with job tracking
    result = await analyze_repo(owner, repo, threshold)
    
    # Store result
    get_redis_client().setex(f"repo:{owner}:{repo}", 86400, json.dumps(result))
    get_redis_client().hset(f"job:{job_id}", "status", "completed")
    get_redis_client().hset(f"job:{job_id}", "result_key", f"repo:{owner}:{repo}")
    
    return {"job_id": job_id, "status": "completed", "result_key": f"repo:{owner}:{repo}"}


async def analyze_repo(owner: str, repo: str, threshold: float = 0.3) -> Dict[str, Any]:
    """Perform repository analysis with scoring."""
    scorer = RepoScorer()
    
    # Simulate feature extraction (would call GitHub API)
    features = {
        "stars": 100,
        "forks": 15,
        "open_issues_count": 8,
        "commit_count_90d": 45,
        "last_commit_days": 12,
        "total_contributors": 8,
        "is_archived": False,
        "is_fork": False,
        "has_tests": True,
        "has_ci": True,
        "readme_size": 4500,
        "topics": ["python", "api", "sdk"],
        "issue_response_hours": 24,
        "pr_merge_time_days": 3,
        "stars_per_month": 8.5
    }
    
    score_result = scorer.calculate_score(features)
    score_result["full_name"] = f"{owner}/{repo}"
    score_result["threshold"] = threshold
    score_result["prediction"] = 1 if score_result["total_score"] >= 50 else 0
    score_result["probability"] = score_result["total_score"] / 100
    
    return score_result


@app.get("/api/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of a submitted job."""
    job_data = get_redis_client().hgetall(f"job:{job_id}")
    
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    result_key = job_data.get("result_key")
    if result_key:
        result_data = get_redis_client().get(result_key)
        if result_data:
            job_data["result"] = json.loads(result_data)
    
    return job_data


@app.get("/api/score/{owner}/{repo}")
async def get_repo_score(owner: str, repo: str):
    """Get repository quality score from cache or trigger analysis."""
    cache_key = f"repo:{owner}:{repo}"
    cached = get_cache().get(cache_key)
    
    if cached:
        return cached
    
    # Trigger new analysis
    result = await analyze_repo(owner, repo)
    get_cache().set(cache_key, result, ttl_seconds=86400)
    
    return result


@app.get("/api/badge/{owner}/{repo}", 
         response_class=StreamingResponse)
async def get_svg_badge(owner: str, repo: str):
    """
    Generate dynamic SVG badge for repository score.
    Embeddable in READMEs: `![RepoScore](https://api.yourdomain.com/api/badge/owner/repo)`
    """
    cache_key = f"score:{owner}:{repo}"
    cached = get_cache().get(cache_key)
    
    if not cached:
        result = await analyze_repo(owner, repo)
        get_cache().set(cache_key, result, ttl_seconds=86400)
        cached = result
    
    badge_response = create_badge_response(cached)
    
    return StreamingResponse(
        iter([badge_response["content"].encode()]),
        media_type="image/svg+xml",
        headers=badge_response.get("headers", {})
    )


@app.get("/api/badges/{owner}/{repo}")
async def get_shields_badge(owner: str, repo: str):
    """Get shields.io compatible badge URL."""
    cache_key = f"score:{owner}:{repo}"
    cached = get_cache().get(cache_key)
    
    if not cached:
        result = await analyze_repo(owner, repo)
        get_cache().set(cache_key, result, ttl_seconds=86400)
        cached = result
    
    return {"badge_url": generate_shields_io_badge_url(cached.get("total_score", 0))}


@app.get("/api/jobs")
async def list_recent_jobs(limit: int = 10):
    """List recent analysis jobs."""
    # In production: scan Redis keys or use a job index
    return {"message": "Job listing requires job indexing implementation"}


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    print("RepoScore API v2.0 starting up...")
    print(f"Cache: {get_cache().redis.connection_pool.connection_kwargs}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("RepoScore API shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)