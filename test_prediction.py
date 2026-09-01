import os
import json
from dotenv import load_dotenv
load_dotenv()
token = os.getenv("GITHUB_TOKEN")
headers = {"Accept": "application/vnd.github+json"}
if token:
    headers["Authorization"] = f"Bearer {token}"

from reposcore_utils import fetch_repo_features, featurize
import joblib
import pickle
import numpy as np
from datetime import datetime

def safe_load(file_path):
    try:
        return joblib.load(file_path)
    except Exception:
        with open(file_path, "rb") as f:
            return pickle.load(f)

def load_ml_assets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "models")
    files = {
        "rf_model": "rf_model.pkl",
        "tfidf_readme": "tfidf_readme.pkl",
        "tfidf_topics": "tfidf_topics.pkl",
        "scaler": "scaler.pkl"
    }
    loaded = {}
    for key, filename in files.items():
        file_path = os.path.join(model_dir, filename)
        loaded[key] = safe_load(file_path)
    return loaded["rf_model"], loaded["tfidf_readme"], loaded["tfidf_topics"], loaded["scaler"]

def check_exceptions(features):
    exceptions = []
    if features.get("has_readme", 1) == 0:
        exceptions.append("No README detected.")
    elif features.get("readme_size", 0) < 50:
        exceptions.append("Very small README (less than 50 characters).")
    if not features.get("topics"):
        exceptions.append("No topics specified.")
    last_commit_days = features.get("last_commit_days")
    if last_commit_days is not None and last_commit_days > 730:
        exceptions.append("No commits in over 2 years.")
    return exceptions

def log_audit_trail(features, probability, prediction, threshold, caveats=None):
    import csv
    audit_dir = "audit_trail"
    if not os.path.exists(audit_dir):
        os.makedirs(audit_dir)
    csv_file = os.path.join(audit_dir, "scoring_decisions.csv")
    jsonl_file = os.path.join(audit_dir, "predictions.jsonl")
    timestamp = datetime.now().isoformat()
    repo_id = features.get("full_name", "")
    logged_features = {
        "full_name": features.get("full_name", ""),
        "html_url": features.get("html_url", ""),
        "stars": features.get("stars", 0),
        "forks": features.get("forks", 0),
        "open_issues": features.get("open_issues", 0),
        "readme_size": features.get("readme_size", 0),
        "repo_age_days": features.get("repo_age_days", 0),
        "last_commit_days": features.get("last_commit_days", 0),
        "has_readme": features.get("has_readme", 0),
        "topics_count": len(features.get("topics", [])),
        "probability": f"{probability:.6f}",
        "prediction": prediction,
        "threshold": f"{threshold:.2f}",
        "timestamp": timestamp
    }
    fieldnames = [
        "timestamp", "repo_id", "repo_url", "stars", "forks", "open_issues",
        "readme_size", "repo_age_days", "last_commit_days", "has_readme",
        "topics_count", "probability", "prediction", "threshold"
    ]
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": timestamp,
            "repo_id": repo_id,
            "repo_url": logged_features["html_url"],
            "stars": logged_features["stars"],
            "forks": logged_features["forks"],
            "open_issues": logged_features["open_issues"],
            "readme_size": logged_features["readme_size"],
            "repo_age_days": logged_features["repo_age_days"],
            "last_commit_days": logged_features["repo_age_days"],
            "has_readme": logged_features["has_readme"],
            "topics_count": logged_features["topics_count"],
            "probability": logged_features["probability"],
            "prediction": logged_features["prediction"],
            "threshold": logged_features["threshold"]
        })
    logged_features_with_caveats = logged_features.copy()
    logged_features_with_caveats["caveats"] = caveats if caveats is not None else []
    with open(jsonl_file, "a", encoding="utf-8") as f:
        json.dump(logged_features_with_caveats, f)
        f.write("\n")

# Load models
print("Loading models...")
rf_model, tfidf_readme, tfidf_topics, scaler = load_ml_assets()
print("Models loaded.")

# Choose a repo
repo = "octocat/Spoon-Knife"  # small test repo
print(f"Fetching features for {repo}...")
try:
    features = fetch_repo_features(repo, headers=headers)
except Exception as e:
    print(f"Error fetching features: {e}")
    exit(1)

print("Features fetched.")
print(f"Stars: {features['stars']}, Forks: {features['forks']}, Open issues: {features['open_issues']}")

X_dense = featurize(features, tfidf_readme, tfidf_topics, scaler)
probability = float(rf_model.predict_proba(X_dense)[0][1])
threshold = 0.3  # default
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

# Log to audit trail
print("Logging to audit trail...")
log_audit_trail(features, probability, prediction, threshold, caveats=warning_messages)
print("Done."
