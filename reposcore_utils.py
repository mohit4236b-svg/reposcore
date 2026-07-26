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
                    "repo_age_days", "days_since_last_commit", "has_readme"]


def fetch_repo_features(full_name, headers=None):
    """
    Fetch a repo's metadata + README from the GitHub API and return the raw
    feature dict needed for prediction. Raises RepoFetchError on failure.

    This is the single place both app.py and reposcore_cli.py call into, so
    a Streamlit prediction and a CLI prediction for the same repo can never
    silently diverge.
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
        raise RepoFetchError("GitHub API rate limit exceeded. Set GITHUB_TOKEN to increase the limit.")
    if repo_resp.status_code != 200:
        raise RepoFetchError(f"GitHub API returned status {repo_resp.status_code}.")
    repo = repo_resp.json()

    readme_resp = requests.get(f"https://api.github.com/repos/{full_name}/readme", headers=headers)
    has_readme = readme_resp.status_code == 200
    readme_text, readme_size = "", 0
    if has_readme:
        readme_data = readme_resp.json()
        readme_size = readme_data.get("size", 0)
        try:
            readme_text = base64.b64decode(readme_data.get("content", "")).decode("utf-8", errors="ignore")
        except Exception:
            readme_text = ""

    created_at = pd.to_datetime(repo["created_at"])
    pushed_at = pd.to_datetime(repo["pushed_at"])
    now = pd.Timestamp.now(tz="UTC")

    return {
        "full_name": repo["full_name"],
        "html_url": repo["html_url"],
        "topics": repo.get("topics", []),
        "readme_text_clean": strip_badges(readme_text),
        "stars": repo.get("stargazers_count", 0),
        "forks": repo.get("forks_count", 0),
        "open_issues": repo.get("open_issues_count", 0),
        "readme_size": readme_size,
        "repo_age_days": (now - created_at).days,
        "days_since_last_commit": (now - pushed_at).days,
        "has_readme": int(has_readme),
    }


class RepoFetchError(Exception):
    """Raised when a repo can't be fetched from the GitHub API (404, rate limit, etc.)."""
    pass


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