#!/usr/bin/env python3
"""
Divergence test: Compare ML model vs RepoScorer on real repos.
This validates that both scoring systems produce reasonably aligned results.
"""

import os
import sys
import json
import pandas as pd
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from reposcore_utils import fetch_repo_features
from scoring_engine import RepoScorer

# ML model imports - import directly to avoid streamlit warnings
import joblib
import os

# Use the correct models directory path
MODEL_DIR = r"C:\Users\ASUS\OneDrive\Documents\GitHub\reposcore\models"
try:
    rf_model = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
    tfidf_readme = joblib.load(os.path.join(MODEL_DIR, "tfidf_readme.pkl"))
    tfidf_topics = joblib.load(os.path.join(MODEL_DIR, "tfidf_topics.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    
    # Import featurize from reposcore_utils
    from reposcore_utils import featurize
    ML_AVAILABLE = True
    print("ML model loaded successfully")
except Exception as e:
    print(f"ML model not available: {e}")
    ML_AVAILABLE = False


# Test repos - mix of high/low quality, different sizes/languages
TEST_REPOS = [
    # High quality, popular
    "pre-commit/pre-commit-hooks",
    "psf/requests",
    "pallets/flask",
    "django/django",
    
    # Medium quality
    "pytest-dev/pytest",
    "tiangolo/fastapi",
    "encode/httpx",
    
    # Smaller/specialized
    "ansible/ansible",
    "kubernetes/kubernetes",
    "golang/go",
    
    # Lower activity/older
    "numpy/numpy",
    "scipy/scipy",
    
    # Different languages
    "facebook/react",
    "vuejs/vue",
    "rust-lang/rust",
]


def run_ml_model(features: Dict[str, Any]) -> Dict[str, Any]:
    """Run the ML model on features and return prediction + probability."""
    if not ML_AVAILABLE:
        return {"prediction": None, "probability": None, "score_pct": None}
    
    X_dense = featurize(features, tfidf_readme, tfidf_topics, scaler)
    probability = float(rf_model.predict_proba(X_dense)[0][1])
    prediction = int(rf_model.predict(X_dense)[0])
    
    return {
        "prediction": prediction,
        "probability": probability,
        "score_pct": round(probability * 100, 2)  # Convert to 0-100 scale
    }


def run_repo_scorer(features: Dict[str, Any]) -> Dict[str, Any]:
    """Run the RepoScorer on features and return score breakdown."""
    scorer = RepoScorer()
    result = scorer.calculate_score(features)
    
    return {
        "total_score": result["total_score"],
        "tier": result["tier"],
        "components": result["components"],
        "decay_factor": result["decay_factor"]
    }


def compare_scores(repo: str, headers: Dict[str, str] = None) -> Dict[str, Any]:
    """Fetch features and run both scorers, return comparison."""
    print(f"\n--- Testing {repo} ---")
    
    try:
        features = fetch_repo_features(repo, headers=headers)
        print(f"  Stars: {features.get('stars', 0)}, Forks: {features.get('forks', 0)}")
        print(f"  Age: {features.get('repo_age_days', 0)} days, Last commit: {features.get('last_commit_days', 0)} days ago")
        print(f"  Contributors: {features.get('total_contributors', 0)}, Has CI: {features.get('has_ci', False)}, Has Tests: {features.get('has_tests', False)}")
        print(f"  README size: {features.get('readme_size', 0)} chars")
    except Exception as e:
        return {
            "repo": repo,
            "error": str(e),
            "ml_score": None,
            "heuristic_score": None,
            "delta": None
        }
    
    ml_result = run_ml_model(features)
    heuristic_result = run_repo_scorer(features)
    
    ml_score = ml_result.get("score_pct")
    heuristic_score = heuristic_result.get("total_score")
    
    delta = None
    if ml_score is not None and heuristic_score is not None:
        delta = round(ml_score - heuristic_score, 2)
    
    print(f"  ML Model:     {ml_score}% (pred={ml_result.get('prediction')}, prob={ml_result.get('probability'):.3f})")
    print(f"  RepoScorer:   {heuristic_score} (tier={heuristic_result.get('tier')})")
    if delta is not None:
        print(f"  Delta (ML - Heuristic): {delta:+.1f}")
    
    return {
        "repo": repo,
        "stars": features.get("stars", 0),
        "forks": features.get("forks", 0),
        "age_days": features.get("repo_age_days", 0),
        "last_commit_days": features.get("last_commit_days", 0),
        "contributors": features.get("total_contributors", 0),
        "has_ci": features.get("has_ci", False),
        "has_tests": features.get("has_tests", False),
        "readme_size": features.get("readme_size", 0),
        "ml_score": ml_score,
        "ml_prediction": ml_result.get("prediction"),
        "ml_probability": ml_result.get("probability"),
        "heuristic_score": heuristic_score,
        "heuristic_tier": heuristic_result.get("tier"),
        "heuristic_components": heuristic_result.get("components"),
        "decay_factor": heuristic_result.get("decay_factor"),
        "delta": delta
    }


def main():
    """Run divergence test on all test repos."""
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    
    print(f"Starting divergence test on {len(TEST_REPOS)} repos...")
    print(f"GitHub token configured: {bool(token)}")
    
    results = []
    for i, repo in enumerate(TEST_REPOS):
        print(f"\n[{i+1}/{len(TEST_REPOS)}] Processing {repo}...")
        result = compare_scores(repo, headers)
        results.append(result)
    
    # Print summary table
    print("\n" + "=" * 120)
    print("DIVERGENCE SUMMARY TABLE")
    print("=" * 120)
    
    df = pd.DataFrame(results)
    # Select columns for display
    display_cols = ["repo", "stars", "ml_score", "heuristic_score", "delta", "heuristic_tier"]
    print(df[display_cols].to_string(index=False))
    
    # Calculate statistics
    valid_results = [r for r in results if r.get("delta") is not None]
    if valid_results:
        deltas = [r["delta"] for r in valid_results]
        print(f"\nStatistics (n={len(valid_results)}):")
        print(f"  Mean delta: {sum(deltas)/len(deltas):+.2f}")
        print(f"  Max positive delta (ML higher): {max(deltas):+.2f}")
        print(f"  Max negative delta (Heuristic higher): {min(deltas):+.2f}")
        print(f"  Std dev: {pd.Series(deltas).std():.2f}")
        
        # Find biggest divergences
        sorted_results = sorted(valid_results, key=lambda x: abs(x["delta"]), reverse=True)
        print(f"\nTop 5 biggest divergences:")
        for r in sorted_results[:5]:
            print(f"  {r['repo']}: ML={r['ml_score']}, Heuristic={r['heuristic_score']}, Delta={r['delta']:+.1f}")
    
    # Save as CSV
    output_file = "scoring_divergence_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Save as markdown
    md_file = "scoring_divergence_results.md"
    with open(md_file, "w") as f:
        f.write("# Scoring System Divergence Test Results\n\n")
        f.write(f"Tested {len(results)} repositories.\n\n")
        f.write("| Repo | Stars | ML Score | Heuristic Score | Delta (ML - Heuristic) | Tier |\n")
        f.write("|------|-------|----------|-----------------|------------------------|------|\n")
        for r in valid_results:
            f.write(f"| {r['repo']} | {r['stars']} | {r['ml_score']} | {r['heuristic_score']} | {r['delta']:+.1f} | {r['heuristic_tier']} |\n")
        f.write(f"\n\n## Statistics\n\n")
        f.write(f"- Repos tested: {len(valid_results)}\n")
        f.write(f"- Mean delta: {sum(deltas)/len(deltas):+.2f}\n")
        f.write(f"- Std deviation: {pd.Series(deltas).std():.2f}\n")
        f.write(f"- Max ML > Heuristic: {max(deltas):+.2f}\n")
        f.write(f"- Max Heuristic > ML: {min(deltas):+.2f}\n")
    
    print(f"Results saved to {md_file}")
    
    return results


if __name__ == "__main__":
    results = main()
    
    # Fail CI if divergence is too high
    valid_results = [r for r in results if r.get("delta") is not None]
    if valid_results:
        deltas = [r["delta"] for r in valid_results]
        mean_delta = sum(deltas) / len(deltas)
        max_divergence = max(abs(d) for d in deltas)
        
        print(f"\n=== CI GATE CHECKS ===")
        print(f"Mean absolute delta: {abs(mean_delta):.2f} (threshold: 25)")
        print(f"Max absolute delta: {max_divergence:.2f} (threshold: 50)")
        
        if abs(mean_delta) > 25:
            print(f"❌ FAIL: Mean delta {abs(mean_delta):.2f} exceeds threshold 25")
            sys.exit(1)
        if max_divergence > 50:
            print(f"❌ FAIL: Max divergence {max_divergence:.2f} exceeds threshold 50")
            sys.exit(1)
        
        print(f"✅ PASS: Divergence within acceptable thresholds")