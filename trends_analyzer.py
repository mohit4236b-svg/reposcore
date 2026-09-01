"""
Trend analyzer for repository star growth and commit activity.
Calculates star growth rate, commit frequency trends, and fork ratio indicators.
"""

import math
import requests
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta


@dataclass
class StarHistory:
    date: str
    stars: int
    daily_change: int


@dataclass
class CommitActivity:
    week: str
    commits: int
    is_active: bool


@dataclass
class TrendAnalysis:
    current_stars: int
    stars_30d_ago: Optional[int]
    stars_90d_ago: Optional[int]
    star_growth_rate_30d: float
    star_growth_rate_90d: float
    stars_per_week_avg: float
    trend_direction: str
    fork_ratio: float
    fork_ratio_interpretation: str
    commit_activity_90d: int
    commit_frequency: str
    activity_trend: str
    health_score: int
    health_status: str
    analysis_timestamp: str


def fetch_star_history(full_name: str, headers: Dict[str, str]) -> List[StarHistory]:
    """
    Fetch star history using GitHub API (requires authentication for detailed history).
    Falls back to estimates if API is unavailable.
    """
    try:
        response = requests.get(
            f"https://api.github.com/repos/{full_name}/stats/stars",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            star_data = response.json()
            history = []
            for entry in star_data:
                history.append(StarHistory(
                    date=entry.get("date", "")[:10],
                    stars=entry.get("total_count", 0),
                    daily_change=0
                ))
            
            for i in range(1, len(history)):
                history[i].daily_change = history[i].stars - history[i-1].stars
            
            return history
    except Exception:
        pass
    
    return []


def fetch_commit_activity(full_name: str, headers: Dict[str, str]) -> List[CommitActivity]:
    """
    Fetch commit activity by week for the last year.
    """
    try:
        response = requests.get(
            f"https://api.github.com/repos/{full_name}/stats/commit_activity",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            activity_data = response.json()
            activities = []
            for week_data in activity_data:
                timestamp = datetime.fromtimestamp(week_data.get("week", 0))
                total_commits = sum(week_data.get("days", []))
                activities.append(CommitActivity(
                    week=timestamp.strftime("%Y-%W"),
                    commits=total_commits,
                    is_active=total_commits > 0
                ))
            return activities
    except Exception:
        pass
    
    return []


def calculate_star_trend(star_history: List[StarHistory], current_stars: int) -> Dict[str, Any]:
    """Calculate star growth trends."""
    if not star_history:
        return {
            "stars_30d_ago": None,
            "stars_90d_ago": None,
            "growth_rate_30d": 0.0,
            "growth_rate_90d": 0.0,
            "stars_per_week_avg": 0.0,
            "trend_direction": "unknown"
        }
    
    now = datetime.now()
    thirty_days_ago = (now - timedelta(days=30)).strftime("%Y-%m-%d")
    ninety_days_ago = (now - timedelta(days=90)).strftime("%Y-%m-%d")
    
    stars_30d = None
    stars_90d = None
    
    for entry in star_history:
        if entry.date <= thirty_days_ago and stars_30d is None:
            stars_30d = entry.stars
        if entry.date <= ninety_days_ago and stars_90d is None:
            stars_90d = entry.stars
    
    if stars_30d is None and len(star_history) > 30:
        stars_30d = star_history[-30].stars
    if stars_90d is None and len(star_history) > 90:
        stars_90d = star_history[-90].stars
    
    growth_30d = 0.0
    if stars_30d is not None and stars_30d > 0:
        growth_30d = ((current_stars - stars_30d) / stars_30d) * 100
    
    growth_90d = 0.0
    if stars_90d is not None and stars_90d > 0:
        growth_90d = ((current_stars - stars_90d) / stars_90d) * 100
    
    weeks_tracked = min(len(star_history), 12)
    stars_per_week = 0.0
    if weeks_tracked > 0 and star_history:
        stars_change = current_stars - star_history[-weeks_tracked].stars
        stars_per_week = stars_change / weeks_tracked
    
    trend = "stable"
    if growth_30d > 10:
        trend = "growing"
    elif growth_30d < -10:
        trend = "declining"
    
    return {
        "stars_30d_ago": stars_30d,
        "stars_90d_ago": stars_90d,
        "growth_rate_30d": growth_30d,
        "growth_rate_90d": growth_90d,
        "stars_per_week_avg": stars_per_week,
        "trend_direction": trend
    }


def calculate_commit_trend(commit_activity: List[CommitActivity]) -> Dict[str, Any]:
    """Calculate commit frequency trends."""
    if not commit_activity:
        return {
            "total_commits_90d": 0,
            "active_weeks": 0,
            "frequency": "unknown",
            "activity_trend": "unknown"
        }
    
    now = datetime.now()
    ninety_days_ago = now - timedelta(days=90)
    
    recent_commits = 0
    active_weeks = 0
    
    for activity in commit_activity:
        try:
            week_date = datetime.strptime(activity.week + "-1", "%Y-%W-%w")
            if week_date >= ninety_days_ago:
                recent_commits += activity.commits
                if activity.is_active:
                    active_weeks += 1
        except ValueError:
            continue
    
    frequency = "inactive"
    if recent_commits >= 100:
        frequency = "very_active"
    elif recent_commits >= 50:
        frequency = "active"
    elif recent_commits >= 10:
        frequency = "moderately_active"
    elif recent_commits > 0:
        frequency = "low_activity"
    
    activity_trend = "stable"
    if len(commit_activity) >= 8:
        first_half = sum(a.commits for a in commit_activity[:len(commit_activity)//2])
        second_half = sum(a.commits for a in commit_activity[len(commit_activity)//2:])
        
        if second_half > first_half * 1.2:
            activity_trend = "improving"
        elif second_half < first_half * 0.8:
            activity_trend = "declining"
    
    return {
        "total_commits_90d": recent_commits,
        "active_weeks": active_weeks,
        "frequency": frequency,
        "activity_trend": activity_trend
    }


def calculate_fork_ratio(stars: int, forks: int) -> Dict[str, Any]:
    """Calculate fork ratio as community indicator."""
    ratio = forks / max(stars, 1)
    
    interpretation = "normal"
    if stars > 100 and ratio < 0.01:
        interpretation = "low_fork_ratio"
    elif stars > 1000 and ratio > 0.5:
        interpretation = "high_fork_ratio"
    
    return {
        "fork_ratio": round(ratio, 3),
        "interpretation": interpretation
    }


def calculate_health_score(trend_data: Dict[str, Any], commit_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate overall repository health score."""
    score = 50
    
    if trend_data.get("trend_direction") == "growing":
        score += 15
    elif trend_data.get("trend_direction") == "declining":
        score -= 20
    
    if commit_data.get("activity_trend") == "improving":
        score += 15
    elif commit_data.get("activity_trend") == "declining":
        score -= 20
    
    commits = commit_data.get("total_commits_90d", 0)
    if commits >= 50:
        score += 10
    elif commits >= 10:
        score += 5
    elif commits == 0:
        score -= 15
    
    active_weeks = commit_data.get("active_weeks", 0)
    if active_weeks >= 8:
        score += 10
    elif active_weeks >= 4:
        score += 5
    elif active_weeks < 2:
        score -= 10
    
    status = "unknown"
    if score >= 80:
        status = "excellent"
    elif score >= 60:
        status = "healthy"
    elif score >= 40:
        status = "moderate"
    elif score >= 20:
        status = "concerning"
    else:
        status = "critical"
    
    return {
        "health_score": max(0, min(100, score)),
        "health_status": status
    }


def analyze_repository(full_name: str, features: Dict[str, Any], headers: Dict[str, str]) -> TrendAnalysis:
    """
    Perform comprehensive trend analysis on a repository.
    
    Args:
        full_name: Repository full name (owner/repo)
        features: Repository features from fetch_repo_features()
        headers: GitHub API headers
        
    Returns:
        TrendAnalysis with all trend metrics
    """
    from datetime import datetime
    
    stars = features.get("stars", 0)
    forks = features.get("forks", 0)
    
    star_history = fetch_star_history(full_name, headers)
    commit_activity = fetch_commit_activity(full_name, headers)
    
    star_trend = calculate_star_trend(star_history, stars)
    commit_trend = calculate_commit_trend(commit_activity)
    fork_ratio_data = calculate_fork_ratio(stars, forks)
    health = calculate_health_score(star_trend, commit_trend)
    
    return TrendAnalysis(
        current_stars=stars,
        stars_30d_ago=star_trend.get("stars_30d_ago"),
        stars_90d_ago=star_trend.get("stars_90d_ago"),
        star_growth_rate_30d=star_trend.get("growth_rate_30d", 0.0),
        star_growth_rate_90d=star_trend.get("growth_rate_90d", 0.0),
        stars_per_week_avg=star_trend.get("stars_per_week_avg", 0.0),
        trend_direction=star_trend.get("trend_direction", "unknown"),
        fork_ratio=fork_ratio_data.get("fork_ratio", 0.0),
        fork_ratio_interpretation=fork_ratio_data.get("interpretation", "unknown"),
        commit_activity_90d=commit_trend.get("total_commits_90d", 0),
        commit_frequency=commit_trend.get("frequency", "unknown"),
        activity_trend=commit_trend.get("activity_trend", "unknown"),
        health_score=health.get("health_score", 0),
        health_status=health.get("health_status", "unknown"),
        analysis_timestamp=datetime.utcnow().isoformat()
    )


def get_trend_summary(analysis: TrendAnalysis) -> str:
    """Generate human-readable trend summary."""
    parts = []
    
    if analysis.trend_direction == "growing":
        parts.append(f"Stars growing at {analysis.star_growth_rate_30d:.1f}%/month")
    elif analysis.trend_direction == "declining":
        parts.append(f"Stars declining at {abs(analysis.star_growth_rate_30d):.1f}%/month")
    else:
        parts.append("Star count stable")
    
    if analysis.activity_trend == "improving":
        parts.append("Commit activity increasing")
    elif analysis.activity_trend == "declining":
        parts.append("Commit activity declining")
    
    parts.append(f"Health: {analysis.health_status}")
    
    return " | ".join(parts)
