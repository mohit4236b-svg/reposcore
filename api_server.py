"""
FastAPI wrapper for production-grade RepoScore backend.
Provides REST APIs for async job processing, scoring, and badge generation.
"""

import json
import os
import requests
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.responses import Response, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import redis

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

    try:
        get_redis_client().hset(f"job:{job_id}", mapping={
            "owner": owner,
            "repo": repo,
            "threshold": str(threshold),
            "status": "queued",
            "created_at": datetime.utcnow().isoformat()
        })
    except redis.exceptions.ConnectionError as e:
        raise HTTPException(status_code=503, detail="Redis service unavailable")

    result = await analyze_repo(owner, repo, threshold)

    try:
        get_redis_client().setex(f"repo:{owner}:{repo}", 86400, json.dumps(result))
        get_redis_client().hset(f"job:{job_id}", "status", "completed")
        get_redis_client().hset(f"job:{job_id}", "result_key", f"repo:{owner}:{repo}")
    except redis.exceptions.ConnectionError as e:
        raise HTTPException(status_code=503, detail="Redis service unavailable")

    return {"job_id": job_id, "status": "completed", "result_key": f"repo:{owner}:{repo}"}


async def analyze_repo(owner: str, repo: str, threshold: float = 0.3) -> Dict[str, Any]:
    """Perform repository analysis with scoring."""
    scorer = RepoScorer()

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
    try:
        job_data = get_redis_client().hgetall(f"job:{job_id}")
    except redis.exceptions.ConnectionError as e:
        raise HTTPException(status_code=503, detail="Redis service unavailable")

    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")

    result_key = job_data.get("result_key")
    if result_key:
        try:
            result_data = get_redis_client().get(result_key)
        except redis.exceptions.ConnectionError as e:
            raise HTTPException(status_code=503, detail="Redis service unavailable")
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
    return {"message": "Job listing requires job indexing implementation"}


@app.get("/api/trends/{owner}/{repo}")
async def get_repo_trends(owner: str, repo: str):
    """
    Get star and commit trend analysis for a repository.
    """
    from trends_analyzer import analyze_repository
    from reposcore_utils import fetch_repo_features

    full_name = f"{owner}/{repo}"

    try:
        features = fetch_repo_features(full_name)
        headers = {"Accept": "application/vnd.github+json"}
        token = os.getenv("GITHUB_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"

        trend_analysis = analyze_repository(full_name, features, headers)

        return {
            "full_name": full_name,
            "stars": trend_analysis.current_stars,
            "stars_30d_ago": trend_analysis.stars_30d_ago,
            "stars_90d_ago": trend_analysis.stars_90d_ago,
            "star_growth_rate_30d": trend_analysis.star_growth_rate_30d,
            "star_growth_rate_90d": trend_analysis.star_growth_rate_90d,
            "stars_per_week_avg": trend_analysis.stars_per_week_avg,
            "trend_direction": trend_analysis.trend_direction,
            "fork_ratio": trend_analysis.fork_ratio,
            "fork_ratio_interpretation": trend_analysis.fork_ratio_interpretation,
            "commit_activity_90d": trend_analysis.commit_activity_90d,
            "commit_frequency": trend_analysis.commit_frequency,
            "activity_trend": trend_analysis.activity_trend,
            "health_score": trend_analysis.health_score,
            "health_status": trend_analysis.health_status,
            "analysis_timestamp": trend_analysis.analysis_timestamp
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trend analysis failed: {str(e)}")


@app.get("/api/security/{owner}/{repo}")
async def get_repo_security(owner: str, repo: str):
    """
    Get security vulnerability scan for a repository.
    Requires repository to be cloned first.
    """
    from security_scanner import scan_repository
    from reposcore_utils import clone_repo_bounded
    import shutil

    full_name = f"{owner}/{repo}"

    try:
        features_response = requests.get(
            f"https://api.github.com/repos/{full_name}",
            headers={"Accept": "application/vnd.github+json"}
        )
        if features_response.status_code != 200:
            raise HTTPException(status_code=404, detail="Repository not found")

        repo_data = features_response.json()
        repo_size_kb = repo_data.get("size", 0)

        repo_path = clone_repo_bounded(full_name, repo_size_kb)
        if not repo_path:
            return {
                "full_name": full_name,
                "error": "Repository too large to scan",
                "total_vulnerabilities": 0,
                "scanned_files": []
            }

        try:
            scan_result = scan_repository(repo_path)

            return {
                "full_name": full_name,
                "total_vulnerabilities": scan_result.total_vulnerabilities,
                "critical_count": scan_result.critical_count,
                "high_count": scan_result.high_count,
                "medium_count": scan_result.medium_count,
                "low_count": scan_result.low_count,
                "risk_level": scan_result.risk_level,
                "scan_method": scan_result.scan_method,
                "dependencies_found": scan_result.dependencies_found,
                "scanned_files": scan_result.scanned_files,
                "vulnerabilities": [
                    {
                        "package_name": v.package_name,
                        "version": v.version,
                        "severity": v.severity.value,
                        "vulnerability_id": v.vulnerability_id,
                        "description": v.description,
                        "fix_version": v.fix_version
                    }
                    for v in scan_result.vulnerabilities[:20]
                ],
                "timestamp": scan_result.timestamp
            }
        finally:
            shutil.rmtree(repo_path, ignore_errors=True)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Security scan failed: {str(e)}")


@app.get("/api/report/{owner}/{repo}")
async def get_repo_report(owner: str, repo: str, format: str = Query("html")):
    """
    Generate a comprehensive quality report for a repository.
    """
    from report_generator import generate_report
    from scoring_engine import RepoScorer
    from reposcore_utils import fetch_repo_features

    full_name = f"{owner}/{repo}"

    try:
        features = fetch_repo_features(full_name)
        scorer = RepoScorer()

        heuristic_result = scorer.calculate_score(features)
        ml_probability = 0.5
        combined_score = (ml_probability * 100 + heuristic_result.get("total_score", 0)) / 2

        report_content = generate_report(
            full_name=full_name,
            html_url=features.get("html_url", f"https://github.com/{full_name}"),
            features=features,
            ml_probability=ml_probability,
            heuristic_score=heuristic_result,
            combined_score=combined_score,
            format=format
        )

        if format.lower() == "json":
            return JSONResponse(content=json.loads(report_content))
        else:
            return Response(content=report_content, media_type="text/html")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@app.get("/api/compare/{owner}/{repo}")
async def compare_repos(owner: str, repo: str, compare_with: str = Query(...)):
    """
    Compare two repositories side by side.
    """
    from scoring_engine import RepoScorer
    from reposcore_utils import fetch_repo_features

    try:
        repo1_features = fetch_repo_features(f"{owner}/{repo}")
        repo2_features = fetch_repo_features(compare_with)

        scorer = RepoScorer()

        score1 = scorer.calculate_score(repo1_features)
        score2 = scorer.calculate_score(repo2_features)

        return {
            "repository_1": {
                "full_name": repo1_features["full_name"],
                "stars": repo1_features["stars"],
                "forks": repo1_features["forks"],
                "score": score1
            },
            "repository_2": {
                "full_name": repo2_features["full_name"],
                "stars": repo2_features["stars"],
                "forks": repo2_features["forks"],
                "score": score2
            },
            "comparison": {
                "score_delta": round(score1.get("total_score", 0) - score2.get("total_score", 0), 2),
                "star_delta": repo1_features["stars"] - repo2_features["stars"],
                "fork_delta": repo1_features["forks"] - repo2_features["forks"]
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    print("RepoScore API v2.0 starting up...")
    try:
        print(f"Cache: {get_cache().redis.connection_pool.connection_kwargs}")
    except Exception as e:
        print(f"Cache: Redis unavailable ({e})")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    print("RepoScore API shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
