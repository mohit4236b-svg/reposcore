import os
import sys
import warnings
sys.path.insert(0, 'c:/Users/ASUS/OneDrive/Documents/GitHub/reposcore')

# Load .env file
from dotenv import load_dotenv
load_dotenv('c:/Users/ASUS/OneDrive/Documents/GitHub/reposcore/.env')

# Suppress streamlit warnings
warnings.filterwarnings("ignore")

from reposcore_utils import fetch_repo_features
from ai_review import generate_ai_review, format_ai_review_for_display

# Test with a mediocre repo - real README but smaller project
repo = "pallets/markupsafe"

headers = {"Accept": "application/vnd.github+json"}
token = os.getenv("GITHUB_TOKEN")
if token:
    headers["Authorization"] = f"Bearer {token}"

print(f"{'='*60}")
print(f"TESTING: {repo}")
print(f"{'='*60}")

try:
    features = fetch_repo_features(repo, headers=headers)
    print(f"Stars: {features.get('stars', 0)}")
    print(f"README length: {features.get('readme_size', 0)} chars")
    print(f"Has README: {features.get('has_readme', 0)}")
    print(f"Topics: {features.get('topics', [])}")
    print(f"Has CI: {features.get('has_ci', False)}")
    print(f"Has Tests: {features.get('has_tests', False)}")
    print(f"Contributors: {features.get('total_contributors', 'Unknown')}")
    print(f"Days since last commit: {features.get('last_commit_days', 'Unknown')}")
    
    # Use the model's own prediction
    from app import rf_model, tfidf_readme, tfidf_topics, scaler, featurize
    X_dense = featurize(features, tfidf_readme, tfidf_topics, scaler)
    probability = rf_model.predict_proba(X_dense)[0][1]
    prediction = 1 if probability >= 0.3 else 0
    print(f"Model prediction: {'High Quality' if prediction == 1 else 'Low Quality'} ({probability:.1%})")
    
    # Generate AI review using the module function
    ai_result = generate_ai_review(
        readme_content=features.get("readme_text_clean", ""),
        features=features,
        prediction=prediction,
        probability=probability
    )
    
    print(f"\nStatus: {ai_result['status']}")
    if ai_result['status'] != 'success':
        print(f"Error type: {ai_result['error_type']}")
        print(f"Error message: {ai_result['error_message']}")
    
    print(f"\n--- AI REVIEW OUTPUT ---")
    review_text = format_ai_review_for_display(ai_result)
    print(review_text)
    print(f"\n--- RAW REVIEW TEXT ---")
    print(repr(ai_result.get('review', 'NO REVIEW KEY')))
    
except Exception as e:
    print(f"ERROR fetching {repo}: {e}")
    import traceback
    traceback.print_exc()