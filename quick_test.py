import os
import json
from dotenv import load_dotenv
load_dotenv()
token = os.getenv("GITHUB_TOKEN")
headers = {"Accept": "application/vnd.github+json"}
if token:
    headers["Authorization"] = f"Bearer {token}"

from reposcore_utils import fetch_repo_features, featurize
from app import load_ml_assets

rf_model, tfidf_readme, tfidf_topics, scaler = load_ml_assets()

repos = [
    "microsoft/vscode",
    "facebook/react",
    "nodejs/node",
    "python/cpython",
    "golang/go",
    "octocat/Spoon-Knife",
    "torvalds/linux",
    "scikit-learn/scikit-learn"
]

for repo in repos:
    try:
        features = fetch_repo_features(repo, headers=headers)
    except Exception as e:
        print(f"{repo}: Error - {e}")
        continue
    X_dense = featurize(features, tfidf_readme, tfidf_topics, scaler)
    probability = float(rf_model.predict_proba(X_dense)[0][1])
    low_confidence = 0.4 <= probability <= 0.6
    print(f"{repo}: probability={probability:.4f}, low_confidence={low_confidence}")
