import os
import sys
sys.path.insert(0, 'c:/Users/ASUS/OneDrive/Documents/GitHub/reposcore')

from dotenv import load_dotenv
load_dotenv('c:/Users/ASUS/OneDrive/Documents/GitHub/reposcore/.env')

import warnings
warnings.filterwarnings("ignore")

from reposcore_utils import fetch_repo_features
from ai_review import generate_ai_review, format_ai_review_for_display
from app import rf_model, tfidf_readme, tfidf_topics, scaler, featurize

headers = {"Accept": "application/vnd.github+json"}
token = os.getenv("GITHUB_TOKEN")
if token:
    headers["Authorization"] = f"Bearer {token}"

repo = "octocat/Spoon-Knife"
print(f"TESTING: {repo}")

features = fetch_repo_features(repo, headers=headers)
print(f"Stars: {features.get('stars', 0)}")
print(f"README length: {features.get('readme_size', 0)} chars")

X_dense = featurize(features, tfidf_readme, tfidf_topics, scaler)
probability = rf_model.predict_proba(X_dense)[0][1]
prediction = 1 if probability >= 0.3 else 0
print(f"Model prediction: {'High Quality' if prediction == 1 else 'Low Quality'} ({probability:.1%})")

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