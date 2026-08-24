import requests
import os
from reposcore_utils import fetch_repo_features, featurize, predict_quality
import joblib
import pickle

def safe_load(file_path):
    try:
        return joblib.load(file_path)
    except Exception:
        with open(file_path, "rb") as f:
            return pickle.load(f)

token = os.getenv('GITHUB_TOKEN')
headers = {'Accept': 'application/vnd.github+json'}
if token:
    headers['Authorization'] = f'Bearer {token}'

# Test check_exceptions from app.py
def check_exceptions(features):
    """Check for data quality issues that might affect prediction reliability."""
    exceptions = []
    if features.get("has_readme", 1) == 0:
        exceptions.append("⚠️ No README detected.")
    elif features.get("readme_size", 0) < 50:
        exceptions.append("⚠️ Very small README (less than 50 characters).")
    if not features.get("topics"):
        exceptions.append("⚠️ No topics specified.")
    last_commit_days = features.get("last_commit_days")
    if last_commit_days is not None and last_commit_days > 730:  # over 2 years
        exceptions.append("⚠️ No commits in over 2 years.")
    return exceptions

# Load model assets
base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, "models")

rf_model = safe_load(os.path.join(model_dir, "rf_model.pkl"))
tfidf_readme = safe_load(os.path.join(model_dir, "tfidf_readme.pkl"))
tfidf_topics = safe_load(os.path.join(model_dir, "tfidf_topics.pkl"))
scaler = safe_load(os.path.join(model_dir, "scaler.pkl"))

# Test multiple repos
repos = [
    'razorpay/go-financial',
    'razorpay/razorpay-python', 
    'razorpay/razorpay-node',
    'razorpay/razorpay-php',
    'razorpay/razorpay-java',
]

for repo in repos:
    print(f"\n=== Testing {repo} ===")
    try:
        features = fetch_repo_features(repo, headers)
        print(f"  last_commit_days: {features.get('last_commit_days')}")
        print(f"  has_readme: {features.get('has_readme')}")
        print(f"  readme_size: {features.get('readme_size')}")
        print(f"  topics: {features.get('topics')}")
        print(f"  Exceptions: {check_exceptions(features)}")
        
        # Test full prediction
        prediction, probability = predict_quality(features, rf_model, tfidf_readme, tfidf_topics, scaler)
        print(f"  Prediction: {prediction}, Probability: {probability:.3f}")
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()