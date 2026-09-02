"""
FastAPI wrapper for production-grade RepoScore backend.
Provides REST APIs for async job processing, scoring, and badge generation.
"""

import json
import os
import re
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Header, Request
from fastapi.responses import Response, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import redis

from scoring_engine import RepoScorer
from cache_layer import get_cache
from badge_generator import create_badge_response, generate_shields_io_badge_url

try:
    from reposcore_utils import fetch_repo_features, featurize, RepoFetchError, RateLimitedRepoFetchError
    REPOSCORE_UTILS_AVAILABLE = True
except ImportError:
    REPOSCORE_UTILS_AVAILABLE = False

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="RepoScore API",
    description="Production-grade GitHub repository quality scoring API",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_redis_client = None

REPO_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_.-]+$')


def validate_repo_path(owner: str, repo: str) -> tuple[bool, str]:
    if not owner or not repo:
        return False, "Owner and repo must not be empty"
    if len(owner) > 100 or len(repo) > 100:
        return False, "Owner or repo name too long"
    if not REPO_NAME_PATTERN.match(owner):
        return False, "Invalid owner name"
    if not REPO_NAME_PATTERN.match(repo):
        return False, "Invalid repo name"
    return True, ""


def get_redis_client():
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
    return _redis_client


def get_github_headers() -> Dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


@app.post("/api/jobs")
@limiter.limit("30/minute")
async def submit_analysis(request: Request, owner: str = Query(...), repo: str = Query(...),
                          threshold: float = Query(0.3, ge=0.0, le=1.0)):
    is_valid, error_msg = validate_repo_path(owner, repo)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    job_id = str(uuid.uuid4())
    full_name = f"{owner}/{repo}"

    try:
        get_redis_client().hset(f"job:{job_id}", mapping={
            "owner": owner, "repo": repo, "threshold": str(threshold),
            "status": "queued", "created_at": datetime.utcnow().isoformat()
        })
    except redis.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Redis service unavailable")

    result = await analyze_repo(owner, repo, threshold)

    try:
        get_redis_client().setex(f"repo:{full_name}", 86400, json.dumps(result))
        get_redis_client().hset(f"job:{job_id}", "status", "completed")
        get_redis_client().hset(f"job:{job_id}", "result_key", f"repo:{full_name}")
    except redis.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Redis service unavailable")

    return {"job_id": job_id, "status": "completed", "result_key": f"repo:{full_name}"}


async def analyze_repo(owner: str, repo: str, threshold: float = 0.3) -> Dict[str, Any]:
    is_valid, error_msg = validate_repo_path(owner, repo)
    if not is_valid:
        raise ValueError(error_msg)
    
    full_name = f"{owner}/{repo}"
    
    if not REPOSCORE_UTILS_AVAILABLE:
        raise RuntimeError("reposcore_utils module not available")
    
    scorer = RepoScorer()
    headers = get_github_headers()
    
    try:
        features = fetch_repo_features(full_name, headers=headers)
    except RateLimitedRepoFetchError as e:
        raise HTTPException(status_code=429, detail=f"GitHub API rate limited. Retry after {e.retry_after} seconds.")
    except RepoFetchError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch repository: {str(e)}")
    
    score_result = scorer.calculate_score(features)
    score_result["full_name"] = full_name
    score_result["threshold"] = threshold
    score_result["prediction"] = 1 if score_result["total_score"] >= 50 else 0
    score_result["probability"] = score_result["total_score"] / 100
    return score_result


@app.get("/api/jobs/{job_id}")
@limiter.limit("60/minute")
async def get_job_status(request: Request, job_id: str):
    if len(job_id) > 100:
        raise HTTPException(status_code=400, detail="Invalid job ID")
    
    try:
        job_data = get_redis_client().hgetall(f"job:{job_id}")
    except redis.exceptions.ConnectionError:
        raise HTTPException(status_code=503, detail="Redis service unavailable")

    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")

    result_key = job_data.get("result_key")
    if result_key:
        try:
            result_data = get_redis_client().get(result_key)
        except redis.exceptions.ConnectionError:
            raise HTTPException(status_code=503, detail="Redis service unavailable")
        if result_data:
            job_data["result"] = json.loads(result_data)

    return job_data


@app.get("/api/score/{owner}/{repo}")
@limiter.limit("30/minute")
async def get_repo_score(request: Request, owner: str, repo: str):
    is_valid, error_msg = validate_repo_path(owner, repo)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    full_name = f"{owner}/{repo}"
    cache_key = f"repo:{full_name}"
    cached = get_cache().get(cache_key)

    if cached:
        cached["from_cache"] = True
        return cached

    result = await analyze_repo(owner, repo)
    result["from_cache"] = False
    get_cache().set(cache_key, result, ttl_seconds=86400)
    return result


@app.get("/api/badge/{owner}/{repo}", response_class=StreamingResponse)
@limiter.limit("60/minute")
async def get_svg_badge(request: Request, owner: str, repo: str):
    is_valid, error_msg = validate_repo_path(owner, repo)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    full_name = f"{owner}/{repo}"
    cache_key = f"score:{full_name}"
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
@limiter.limit("60/minute")
async def get_shields_badge(request: Request, owner: str, repo: str):
    is_valid, error_msg = validate_repo_path(owner, repo)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    full_name = f"{owner}/{repo}"
    cache_key = f"score:{full_name}"
    cached = get_cache().get(cache_key)

    if not cached:
        result = await analyze_repo(owner, repo)
        get_cache().set(cache_key, result, ttl_seconds=86400)
        cached = result

    return {"badge_url": generate_shields_io_badge_url(cached.get("total_score", 0))}


@app.get("/api/trends/{owner}/{repo}")
@limiter.limit("30/minute")
async def get_repo_trends(request: Request, owner: str, repo: str):
    is_valid, error_msg = validate_repo_path(owner, repo)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    full_name = f"{owner}/{repo}"

    try:
        from trends_analyzer import analyze_repository
        headers = get_github_headers()
        features = fetch_repo_features(full_name, headers=headers)
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
    except RateLimitedRepoFetchError as e:
        raise HTTPException(status_code=429, detail=f"GitHub API rate limited. Retry after {e.retry_after} seconds.")
    except RepoFetchError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trend analysis failed: {str(e)}")


@app.get("/api/security/{owner}/{repo}")
@limiter.limit("10/minute")
async def get_repo_security(request: Request, owner: str, repo: str):
    is_valid, error_msg = validate_repo_path(owner, repo)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    import requests
    full_name = f"{owner}/{repo}"

    try:
        features_response = requests.get(
            f"https://api.github.com/repos/{full_name}",
            headers=get_github_headers(), timeout=10
        )
        if features_response.status_code == 403:
            raise HTTPException(status_code=429, detail="GitHub API rate limited")
        if features_response.status_code == 404:
            raise HTTPException(status_code=404, detail="Repository not found")
        if features_response.status_code != 200:
            raise HTTPException(status_code=features_response.status_code, detail="GitHub API error")

        repo_data = features_response.json()
        repo_size_kb = repo_data.get("size", 0)

        from security_scanner import scan_repository
        from reposcore_utils import clone_repo_bounded
        import shutil

        repo_path = clone_repo_bounded(full_name, repo_size_kb)
        if not repo_path:
            return {"full_name": full_name, "error": "Repository too large to scan (max 50MB)", "total_vulnerabilities": 0, "scanned_files": []}

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
                    {"package_name": v.package_name, "version": v.version, "severity": v.severity.value,
                     "vulnerability_id": v.vulnerability_id, "description": v.description, "fix_version": v.fix_version}
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
@limiter.limit("20/minute")
async def get_repo_report(request: Request, owner: str, repo: str, format: str = Query("html")):
    is_valid, error_msg = validate_repo_path(owner, repo)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    full_name = f"{owner}/{repo}"

    try:
        from report_generator import generate_report
        headers = get_github_headers()
        features = fetch_repo_features(full_name, headers=headers)
        scorer = RepoScorer()
        heuristic_result = scorer.calculate_score(features)
        ml_probability = 0.5
        combined_score = (ml_probability * 100 + heuristic_result.get("total_score", 0)) / 2

        report_content = generate_report(
            full_name=full_name,
            html_url=features.get("html_url", f"https://github.com/{full_name}"),
            features=features, ml_probability=ml_probability,
            heuristic_score=heuristic_result, combined_score=combined_score, format=format
        )

        if format.lower() == "json":
            return JSONResponse(content=json.loads(report_content))
        else:
            return Response(content=report_content, media_type="text/html")

    except RateLimitedRepoFetchError as e:
        raise HTTPException(status_code=429, detail=f"GitHub API rate limited. Retry after {e.retry_after} seconds.")
    except RepoFetchError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@app.get("/api/compare/{owner}/{repo}")
@limiter.limit("20/minute")
async def compare_repos(request: Request, owner: str, repo: str, compare_with: str = Query(...)):
    is_valid, error_msg = validate_repo_path(owner, repo)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    compare_owner, compare_repo = compare_with.split('/')
    is_valid, error_msg = validate_repo_path(compare_owner, compare_repo)
    if not is_valid:
        raise HTTPException(status_code=400, detail=f"Invalid compare_with: {error_msg}")

    try:
        headers = get_github_headers()
        repo1_features = fetch_repo_features(f"{owner}/{repo}", headers=headers)
        repo2_features = fetch_repo_features(compare_with, headers=headers)
        scorer = RepoScorer()
        score1 = scorer.calculate_score(repo1_features)
        score2 = scorer.calculate_score(repo2_features)

        return {
            "repository_1": {"full_name": repo1_features["full_name"], "stars": repo1_features["stars"],
                           "forks": repo1_features["forks"], "score": score1},
            "repository_2": {"full_name": repo2_features["full_name"], "stars": repo2_features["stars"],
                           "forks": repo2_features["forks"], "score": score2},
            "comparison": {"score_delta": round(score1.get("total_score", 0) - score2.get("total_score", 0), 2),
                          "star_delta": repo1_features["stars"] - repo2_features["stars"],
                          "fork_delta": repo1_features["forks"] - repo2_features["forks"]}
        }
    except RateLimitedRepoFetchError as e:
        raise HTTPException(status_code=429, detail=f"GitHub API rate limited. Retry after {e.retry_after} seconds.")
    except RepoFetchError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")


@app.on_event("startup")
async def startup_event():
    print("RepoScore API v2.0 starting up...")
    try:
        cache = get_cache()
        print(f"Cache: {cache.redis.connection_pool.connection_kwargs}")
    except Exception as e:
        print(f"Cache: Redis unavailable ({e})")
    if not REPOSCORE_UTILS_AVAILABLE:
        print("WARNING: reposcore_utils not available")


@app.on_event("shutdown")
async def shutdown_event():
    print("RepoScore API shutting down...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
