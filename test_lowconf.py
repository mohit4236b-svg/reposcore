import os
import json
from dotenv import load_dotenv
load_dotenv()
token = os.getenv("GITHUB_TOKEN")
headers = {"Accept": "application/vnd.github+json"}
if token:
    headers["Authorization"] = f"Bearer {token}"

from reposcore_utils import fetch_repo_features, featurize
from app import load_ml_assets, log_audit_trail, check_exceptions

rf_model, tfidf_readme, tfidf_topics, scaler = load_ml_assets()

repo = "python/cpython"
print(f"Fetching features for {repo}...")
features = fetch_repo_features(repo, headers=headers)
print(f"Stars: {features['stars']}, Forks: {features['forks']}, Open issues: {features['open_issues']}")

X_dense = featurize(features, tfidf_readme, tfidf_topics, scaler)
probability = float(rf_model.predict_proba(X_dense)[0][1])
threshold = 0.3
prediction = 1 if probability >= threshold else 0
print(f"Probability: {probability:.4f}, Prediction: {prediction} (threshold={threshold})")

exceptions = check_exceptions(features)
low_confidence = 0.4 <= probability <= 0.6
warning_messages = exceptions.copy()
if low_confidence:
    warning_messages.append("Low confidence prediction (probability near 0.5).")
print(f"Exceptions: {exceptions}")
print(f"Low confidence flag: {low_confidence}")
print(f"Warning messages (caveats): {warning_messages}")

print("Logging to audit trail...")
log_audit_trail(features, probability, prediction, threshold, caveats=warning_messages)
print("Done.")
