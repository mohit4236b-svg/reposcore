#!/usr/bin/env python3
"""Single repo test."""

import os
import sys
sys.path.insert(0, r'C:\Users\ASUS\OneDrive\Documents\GitHub\reposcore')

from dotenv import load_dotenv
load_dotenv()

import joblib
MODEL_DIR = r'C:\Users\ASUS\OneDrive\Documents\GitHub\reposcore\models'
rf_model = joblib.load(os.path.join(MODEL_DIR, 'rf_model.pkl'))
tfidf_readme = joblib.load(os.path.join(MODEL_DIR, 'tfidf_readme.pkl'))
tfidf_topics = joblib.load(os.path.join(MODEL_DIR, 'tfidf_topics.pkl'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))

from reposcore_utils import fetch_repo_features, featurize
from scoring_engine import RepoScorer

headers = {'Accept': 'application/vnd.github+json'}
token = os.getenv('GITHUB_TOKEN')
if token:
    headers['Authorization'] = f'Bearer {token}'

features = fetch_repo_features('pre-commit/pre-commit-hooks', headers=headers)
print('Features fetched:', features['full_name'])

# ML model
X_dense = featurize(features, tfidf_readme, tfidf_topics, scaler)
probability = float(rf_model.predict_proba(X_dense)[0][1])
prediction = int(rf_model.predict(X_dense)[0])
print(f'ML: pred={prediction}, prob={probability:.3f}, score={probability*100:.1f}%')

# Heuristic
scorer = RepoScorer()
result = scorer.calculate_score(features)
print(f'Heuristic: score={result["total_score"]}, tier={result["tier"]}')