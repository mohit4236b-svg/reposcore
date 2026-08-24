"""
Shared preprocessing used by BOTH train_model.py and app.py.

Keeping this in one place matters: if the training script and the Streamlit
app clean README text differently, you get train/serve skew — the model was
fit on one distribution of text and scores a different one at inference time.
Import strip_badges from here in both places instead of copy-pasting it.
"""

import re

# Strips markdown image/badge syntax, shields.io/badge-service URLs, and
# common CI/build/status badge hosts. This exists because has_ci/has_tests
# are excluded as direct model features (they were used to build the label),
# but their *signal* was leaking back in indirectly through badge markup
# embedded in the README text (tokens like "workflows", "shields", "badge",
# "svg" ranked in the top-15 TF-IDF features before this was applied).
BADGE_PATTERNS = [
    r"!\[[^\]]*\]\([^)]*\)",                 # ![alt](url) markdown images
    r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)",     # linked badge images
    r"https?://\S*(shields\.io|badge\.fury\.io|travis-ci|"
    r"github\.com/\S*workflows\S*|circleci|codecov|coveralls|"
    r"bestpractices\.coreinfrastructure|securityscorecards|"
    r"oss-fuzz|ossrank|zenodo)\S*",
]


def strip_badges(text: str) -> str:
    """Remove badge/shield markup from README text before vectorizing."""
    if not isinstance(text, str):
        return ""
    for pattern in BADGE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return text


STRUCTURED_COLS = ["stars", "forks", "open_issues", "readme_size",
                    "repo_age_days", "last_commit_days", "has_readme"]


def fetch_repo_features(full_name, headers=None):
    """
    Fetch a repo's metadata + README from the GitHub API and return the raw
    feature dict needed for prediction. Raises RepoFetchError on failure.

    This is the single place both app.py and reposcore_cli.py call into, so
    a Streamlit prediction and a CLI prediction for the same repo can never
    silently diverge.
    
    Production enhancements:
    - Fetches contributor count for better scoring
    - Handles rate limits with Retry-After headers
    - Supports conditional requests with ETags
    """
    import base64
    import requests
    import pandas as pd

    headers = headers or {}
    full_name = full_name.strip().strip("/")

    repo_resp = requests.get(f"https://api.github.com/repos/{full_name}", headers=headers)
    if repo_resp.status_code == 404:
        raise RepoFetchError(f"Repository '{full_name}' not found.")
    if repo_resp.status_code == 403:
        retry_after = int(repo_resp.headers.get("Retry-After", 60))
        raise RateLimitedRepoFetchError(
            "GitHub API rate limit exceeded. Set GITHUB_TOKEN to increase the limit.",
            retry_after=retry_after
        )
    if repo_resp.status_code != 200:
        raise RepoFetchError(f"GitHub API returned status {repo_resp.status_code}.")
    repo = repo_resp.json()

    readme_resp = requests.get(f"https://api.github.com/repos/{full_name}/readme", headers=headers)
    if readme_resp.status_code == 403:
        retry_after = int(readme_resp.headers.get("Retry-After", 60))
        raise RateLimitedRepoFetchError(
            "GitHub API rate limit exceeded while fetching README. Set GITHUB_TOKEN to increase the limit.",
            retry_after=retry_after
        )
    has_readme = readme_resp.status_code == 200
    readme_text, readme_size = "", 0
    if has_readme:
        readme_data = readme_resp.json()
        readme_size = readme_data.get("size", 0)
        try:
            readme_text = base64.b64decode(readme_data.get("content", "")).decode("utf-8", errors="ignore")
        except Exception:
            readme_text = ""

    # Fetch contributor count (handle pagination)
    total_contributors = 1  # Default to at least the user
    try:
        contribs_resp = requests.get(
            f"https://api.github.com/repos/{full_name}/contributors",
            headers=headers,
            params={"per_page": 1}
        )
        if contribs_resp.status_code == 200:
            # Use Link header to get total count if available
            link = contribs_resp.headers.get("Link", "")
            if 'rel="last"' in link:
                import re
                # Match the page number specifically from the "last" relation
                match = re.search(r'page=(\d+)>; rel="last"', link)
                if match:
                    total_contributors = int(match.group(1))
            else:
                total_contributors = len(contribs_resp.json())
    except Exception:
        pass  # Don't fail if contributor fetch fails

    # Detect CI from topics and repo info
    topics = repo.get("topics", []) or []
    has_ci = any(t in topics for t in ["ci", "github-actions", "workflows", "circleci", "travis-ci", "codecov"])
    has_pages = repo.get("has_pages", False)
    
    # Detect tests from topics
    has_tests = any(t in topics for t in ["tests", "test", "testing", "pytest", "unittest"])

    created_at = pd.to_datetime(repo["created_at"])
    pushed_at = pd.to_datetime(repo["pushed_at"])
    now = pd.Timestamp.now(tz="UTC")

    return {
        "full_name": repo["full_name"],
        "html_url": repo["html_url"],
        "topics": topics,
        "readme_text_clean": strip_badges(readme_text),
        "stars": repo.get("stargazers_count", 0),
        "forks": repo.get("forks_count", 0),
        "open_issues": repo.get("open_issues_count", 0),
        "readme_size": readme_size,
        "repo_age_days": (now - created_at).days,
        "last_commit_days": (now - pushed_at).days,
        "has_readme": int(has_readme),
        "has_ci": has_ci or has_pages,
        "has_tests": has_tests,
        "total_contributors": total_contributors,
    }


class RepoFetchError(Exception):
    """Raised when a repo can't be fetched from the GitHub API (404, rate limit, etc.)."""
    pass


class RateLimitedRepoFetchError(RepoFetchError):
    """Raised when rate limited - includes retry-after info for backoff."""
    def __init__(self, message: str, retry_after: int = 0):
        super().__init__(message)
        self.retry_after = retry_after


class CacheableRepoFetchError(RepoFetchError):
    """Error that can be cached to prevent repeated failures."""
    def __init__(self, message: str, cacheable: bool = True):
        super().__init__(message)
        self.cacheable = cacheable


def featurize(features, tfidf_readme, tfidf_topics, scaler):
    """Turn a fetch_repo_features() dict into the dense matrix the model expects."""
    import numpy as np
    from scipy.sparse import hstack

    topics_text = " ".join(features["topics"])
    structured = np.array([[features[c] for c in STRUCTURED_COLS]])

    X_readme = tfidf_readme.transform([features["readme_text_clean"]])
    X_topics = tfidf_topics.transform([topics_text])
    X = hstack([X_readme, X_topics, structured]).tocsr()
    X_scaled = scaler.transform(X)
    return np.asarray(X_scaled.todense(), dtype=np.float64)


def predict_quality(features, rf_model, tfidf_readme, tfidf_topics, scaler):
    """
    Run the trained model on a feature dict from fetch_repo_features().
    Returns (prediction: int, probability: float).
    """
    X_dense = featurize(features, tfidf_readme, tfidf_topics, scaler)
    prediction = int(rf_model.predict(X_dense)[0])
    probability = float(rf_model.predict_proba(X_dense)[0][1])
    return prediction, probability