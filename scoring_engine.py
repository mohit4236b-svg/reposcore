"""
Production-grade scoring engine with time decay, dynamic features, and edge case handling.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import math
import numpy as np


@dataclass
class RatingTier:
    """Score tier definition."""
    min_score: float
    max_score: float
    label: str
    emoji: str
    color: str


# Define rating tiers A+ to F
RATING_TIERS: List[RatingTier] = [
    RatingTier(90, 100, "A+", "🏆", "#4CAF50"),
    RatingTier(80, 89, "A", "⭐", "#8BC34A"),
    RatingTier(70, 79, "B", "👍", "#CDDC39"),
    RatingTier(60, 69, "C", "👌", "#FF9800"),
    RatingTier(50, 59, "D", "⚠️", "#FF5722"),
    RatingTier(0, 49, "F", "❌", "#F44336"),
]


class RepoScorer:
    """
    Production-grade repository quality scorer.
    
    Scores 0-100 across 4 dimensions:
    1. Maintenance Activity (30%)
    2. Community Health (25%)
    3. Documentation Quality (25%)
    4. Contributor Distribution (20%)
    """
    
    DIMENSION_WEIGHTS = {
        "maintenance": 0.30,
        "community": 0.25,
        "docs": 0.25,
        "contributors": 0.20
    }
    
    def __init__(self):
        self.tiers = RATING_TIERS
    
    def calculate_score(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate comprehensive repository quality score.
        
        Args:
            features: Dictionary of repository features from fetch_repo_features()
            
        Returns:
            Score breakdown with component scores, tier, and explanations
        """
        # Calculate individual component scores
        maintenance_score = self._score_maintenance(features)
        community_score = self._score_community(features)
        docs_score = self._score_documentation(features)
        contributor_score = self._score_contributors(features)
        
        # Apply time decay for inactive/archived repos
        decay_factor = self._calculate_decay_factor(features)
        
        final_scores = {
            "maintenance": maintenance_score * decay_factor,
            "community": community_score * decay_factor,
            "docs": docs_score * decay_factor,
            "contributors": contributor_score * decay_factor
        }
        
        # Calculate weighted total
        total = sum(
            final_scores[key] * self.DIMENSION_WEIGHTS[key]
            for key in self.DIMENSION_WEIGHTS
        )
        
        # Determine tier
        tier = self._get_tier(total)
        
        # Generate explanations
        explanations = self._generate_explanations(features, final_scores, decay_factor)
        
        return {
            "total_score": round(total, 2),
            "total_score_percent": round(total, 2),
            "components": {
                "maintenance": round(final_scores["maintenance"], 2),
                "community": round(final_scores["community"], 2),
                "documentation": round(final_scores["docs"], 2),
                "contributors": round(final_scores["contributors"], 2)
            },
            "tier": tier.label,
            "tier_emoji": tier.emoji,
            "tier_color": tier.color,
            "decay_factor": round(decay_factor, 3),
            "explanations": explanations,
            "calculated_at": datetime.utcnow().isoformat(),
            "version": "2.0"
        }
    
    def _score_maintenance(self, features: Dict[str, Any]) -> float:
        """Score maintenance activity (0-100)."""
        score = 100.0
        
        # Commit frequency (last 90 days)
        commits_90d = features.get("commit_count_90d", 0)
        if commits_90d == 0:
            score -= 40  # No activity
        elif commits_90d < 10:
            score -= 20
        elif commits_90d < 30:
            score -= 5
        elif commits_90d >= 50:
            score += 10  # Bonus for very active
        
        # Days since last commit
        last_commit = features.get("last_commit_days", 999999)
        if last_commit > 365:
            score -= min(30, last_commit / 20)
        elif last_commit > 180:
            score -= 15
        elif last_commit < 30:
            score += 5
        
        # Issue response time
        issue_resp = features.get("issue_response_hours", 24)
        if issue_resp > 168:  # 1 week
            score -= 10
        elif issue_resp > 72:  # 3 days
            score -= 5
        elif issue_resp < 24:
            score += 5
        
        # PR merge time
        pr_merge = features.get("pr_merge_time_days", 7)
        if pr_merge > 30:
            score -= 10
        elif pr_merge > 14:
            score -= 5
        
        # Archived penalty
        if features.get("is_archived", False):
            score -= 30
        
        return max(0, min(100, score))
    
    def _score_community(self, features: Dict[str, Any]) -> float:
        """Score community health (0-100)."""
        score = 100.0
        
        # Stars/Forks
        stars = features.get("stars", 0)
        forks = features.get("forks", 0)
        
        if stars > 10000:
            score += 15
        elif stars > 1000:
            score += 10
        elif stars > 100:
            score += 5
        
        if forks > 500:
            score += 10
        elif forks > 50:
            score += 5
        
        # Open issues
        open_issues = features.get("open_issues_count", 0)
        if open_issues > 100:
            score -= min(20, open_issues / 5)
        elif open_issues > 50:
            score -= 10
        
        # Issue/PR activity
        if features.get("has_ci", False):
            score += 10
        
        # Topic diversity
        topics = features.get("topics", [])
        if len(topics) > 5:
            score += 5
        elif len(topics) < 2:
            score -= 5
        
        # Negative score penalty
        if stars > 5000 and forks < 10:
            score -= 20  # Vanity stars
        
        return max(0, min(100, score))
    
    def _score_documentation(self, features: Dict[str, Any]) -> float:
        """Score documentation quality (0-100)."""
        score = 100.0
        
        # README presence and size
        readme_size = features.get("readme_size", 0)
        if readme_size < 100:
            score -= 30  # No README
        elif readme_size < 500:
            score -= 15
        elif readme_size > 5000:
            score += 10  # Good documentation
        
        # Tests
        if features.get("has_tests", False):
            score += 20
        else:
            score -= 15
        
        # CI
        if features.get("has_ci", False):
            score += 15
        else:
            score -= 10
        
        # Topics as documentation indicator
        topics = features.get("topics", [])
        if len(topics) >= 3:
            score += 5
        elif len(topics) == 0:
            score -= 5
        
        return max(0, min(100, score))
    
    def _score_contributors(self, features: Dict[str, Any]) -> float:
        """Score contributor distribution (0-100)."""
        score = 100.0
        
        # Forks are not included
        if features.get("is_fork", False):
            return 0
        
        # Total contributors (bus factor)
        total_contributors = features.get("total_contributors", 1)
        
        if total_contributors < 2:
            score -= 50  # Single contributor
        elif total_contributors < 3:
            score -= 30
        elif total_contributors < 5:
            score -= 10
        elif total_contributors > 20:
            score += 10
        
        # Contributor commits (activity distribution)
        contrib_commits = features.get("contributor_commits_last_90d", 0)
        if contrib_commits > 20:
            score += 5
        
        # Vanity stars penalty
        stars = features.get("stars", 0)
        if stars > 5000 and total_contributors < 5:
            score -= 40  # High stars, low contribution
        
        return max(0, min(100, score))
    
    def _calculate_decay_factor(self, features: Dict[str, Any]) -> float:
        """
        Calculate time decay factor for inactive/archived repos.
        
        Returns:
            Float between 0.5 and 1.0 (1.0 = no decay)
        """
        decay = 1.0
        
        # Archiving penalty
        if features.get("is_archived", False):
            decay *= 0.7
        
        # Inactivity decay (exponential)
        last_commit_days = features.get("last_commit_days", 999999)
        if last_commit_days > 365:
            # Exponential decay: e^(-t/365) clipped to 0.5
            decay *= max(0.5, math.exp(-last_commit_days / 365))
        
        # Vanity star penalty
        stars = features.get("stars", 0)
        if stars > 5000 and last_commit_days > 180:
            decay *= 0.85
        
        return max(0.5, decay)
    
    def _get_tier(self, total_score: float) -> RatingTier:
        """Get rating tier based on total score."""
        for tier in self.tiers:
            if tier.min_score <= total_score <= tier.max_score:
                return tier
        return self.tiers[-1]
    
    def _generate_explanations(self, features: Dict[str, Any], 
                               scores: Dict[str, float], 
                               decay: float) -> List[str]:
        """Generate human-readable explanations for the score."""
        explanations = []
        
        # Maintenance
        if features.get("is_archived"):
            explanations.append("Archived repositories receive a 30% maintenance penalty")
        
        if features.get("last_commit_days", 0) > 365:
            days = features.get("last_commit_days", 0)
            explanations.append(f"No commits in {days} days triggers time decay")
        
        if not features.get("has_tests", False):
            explanations.append("No test suite detected (-15 points)")
        
        if not features.get("has_ci", False):
            explanations.append("No CI configuration (-10 points)")
        
        # Community
        contributors = features.get("total_contributors", 0)
        if contributors < 3:
            explanations.append(f"Low bus factor: only {contributors} contributors")
        
        stars = features.get("stars", 0)
        forks = features.get("forks", 0)
        if stars > 0 and forks / max(stars, 1) < 0.01:
            explanations.append("Low fork ratio suggests limited community adoption")
        
        # Positive indicators
        if features.get("has_ci") and features.get("has_tests"):
            explanations.append("Has CI pipeline and test suite")
        
        if decay < 1.0:
            explanations.append(f"Time decay applied: {decay:.0%} retained score")
        
        # Default message
        if not explanations:
            explanations.append("Repository shows healthy activity and community engagement")
        
        return explanations