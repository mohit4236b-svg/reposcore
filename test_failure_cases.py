import os
import sys
sys.path.insert(0, 'c:/Users/ASUS/OneDrive/Documents/GitHub/reposcore')

# Load .env file
from dotenv import load_dotenv
load_dotenv('c:/Users/ASUS/OneDrive/Documents/GitHub/reposcore/.env')

# Suppress streamlit warnings
import warnings
warnings.filterwarnings("ignore")

from reposcore_utils import fetch_repo_features
from ai_review import generate_ai_review, format_ai_review_for_display

headers = {"Accept": "application/vnd.github+json"}
token = os.getenv("GITHUB_TOKEN")
if token:
    headers["Authorization"] = f"Bearer {token}"

# Test 1: Invalid API key
print("=" * 60)
print("TEST 1: Invalid API Key")
print("=" * 60)

features = fetch_repo_features("scikit-learn/scikit-learn", headers=headers)
print(f"Stars: {features.get('stars', 0)}")

ai_result = generate_ai_review(
    readme_content=features.get("readme_text_clean", ""),
    features=features,
    prediction=1,
    probability=0.68,
    api_key="INVALID_KEY_12345"
)

print(f"\nStatus: {ai_result['status']}")
print(f"Error type: {ai_result['error_type']}")
print(f"Error message: {ai_result['error_message']}")
print(f"\n--- AI REVIEW OUTPUT ---")
print(format_ai_review_for_display(ai_result))

# Test 2: Empty README
print("\n" + "=" * 60)
print("TEST 2: Empty README")
print("=" * 60)

ai_result = generate_ai_review(
    readme_content="",
    features={"full_name": "test/repo"},
    prediction=0,
    probability=0.05
)

print(f"\nStatus: {ai_result['status']}")
print(f"Error type: {ai_result['error_type']}")
print(f"Error message: {ai_result['error_message']}")
print(f"\n--- AI REVIEW OUTPUT ---")
print(format_ai_review_for_display(ai_result))

# Test 3: Missing API key (simulate)
print("\n" + "=" * 60)
print("TEST 3: Missing API Key (empty string)")
print("=" * 60)

ai_result = generate_ai_review(
    readme_content=features.get("readme_text_clean", ""),
    features=features,
    prediction=1,
    probability=0.68,
    api_key=""
)

print(f"\nStatus: {ai_result['status']}")
print(f"Error type: {ai_result['error_type']}")
print(f"Error message: {ai_result['error_message']}")
print(f"\n--- AI REVIEW OUTPUT ---")
print(format_ai_review_for_display(ai_result))

# Test 4: A third real repo with different characteristics
print("\n" + "=" * 60)
print("TEST 4: Third real repo - pandas/pandas (Data science library)")
print("=" * 60)

features = fetch_repo_features("pandas-dev/pandas", headers=headers)
print(f"Stars: {features.get('stars', 0)}")
print(f"README length: {features.get('readme_size', 0)} chars")

from app import rf_model, tfidf_readme, tfidf_topics, scaler, featurize
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