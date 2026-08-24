# -*- coding: utf-8 -*-
import base64
import os
import pickle
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from scipy.sparse import hstack

from reposcore_utils import strip_badges, fetch_repo_features, featurize, RepoFetchError, STRUCTURED_COLS

# Import AI review module
try:
    from ai_review import generate_ai_review, format_ai_review_for_display
    AI_REVIEW_AVAILABLE = True
except ImportError:
    AI_REVIEW_AVAILABLE = False

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RepoScore",
    page_icon="⭐",
    layout="wide"
)

# Configure GitHub API Headers safely
token = os.getenv("GITHUB_TOKEN")
headers = {"Accept": "application/vnd.github+json"}
if token:
    headers["Authorization"] = f"Bearer {token}"

def safe_load(file_path):
    """Attempt loading with joblib first; fall back to built-in pickle if joblib fails."""
    try:
        return joblib.load(file_path)
    except Exception:
        with open(file_path, "rb") as f:
            return pickle.load(f)

@st.cache_resource
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

        if not os.path.exists(file_path):
            st.error(f"❌ Missing file: `{filename}` was not found in the `models/` directory.")
            st.stop()

        try:
            loaded[key] = safe_load(file_path)
        except Exception as err:
            st.error(f"❌ Failed loading `{filename}`:")
            st.exception(err)
            st.stop()

    return loaded["rf_model"], loaded["tfidf_readme"], loaded["tfidf_topics"], loaded["scaler"]

@st.cache_resource
def load_explainer(_model):
    """Build a SHAP TreeExplainer once and cache it (RF trees make this fast)."""
    import shap
    return shap.TreeExplainer(_model)

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


def log_audit_trail(features, probability, prediction, threshold):
    """
    Log scoring decision to CSV file for audit trail.
    Records: repo identifier, input features used, score, timestamp.
    """
    import csv
    import os
    from datetime import datetime
    
    # Create audit trail directory if it doesn't exist
    audit_dir = "audit_trail"
    if not os.path.exists(audit_dir):
        os.makedirs(audit_dir)
    
    # CSV file path
    csv_file = os.path.join(audit_dir, "scoring_decisions.csv")
    
    # Prepare data to log
    timestamp = datetime.now().isoformat()
    repo_id = features.get("full_name", "unknown")
    
    # Extract features for logging (we'll log the key features used in scoring)
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
        "prediction": prediction,  # 1 for high quality, 0 for low quality
        "threshold": f"{threshold:.2f}",
        "timestamp": timestamp
    }
    
    # Define CSV headers
    fieldnames = [
        "timestamp", "repo_id", "repo_url", "stars", "forks", "open_issues",
        "readme_size", "repo_age_days", "last_commit_days", "has_readme",
        "topics_count", "probability", "prediction", "threshold"
    ]
    
    # Write to CSV (create file with headers if it doesn't exist)
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Write header if file is new
        if not file_exists:
            writer.writeheader()
        
        # Write the data row
        writer.writerow({
            "timestamp": timestamp,
            "repo_id": repo_id,
            "repo_url": logged_features["html_url"],
            "stars": logged_features["stars"],
            "forks": logged_features["forks"],
            "open_issues": logged_features["open_issues"],
            "readme_size": logged_features["readme_size"],
            "repo_age_days": logged_features["repo_age_days"],
            "last_commit_days": logged_features["last_commit_days"],
            "has_readme": logged_features["has_readme"],
            "topics_count": logged_features["topics_count"],
            "probability": logged_features["probability"],
            "prediction": logged_features["prediction"],
            "threshold": logged_features["threshold"]
        })


rf_model, tfidf_readme, tfidf_topics, scaler = load_ml_assets()

# Application Interface
st.title("⭐ RepoScore: GitHub Repository Quality Predictor")
st.caption("Analyze a public GitHub repository to predict its overall quality score.")

repo_input = st.text_input("Enter Repository (owner/name):", placeholder="scikit-learn/scikit-learn")
threshold = st.slider(
    "Quality threshold", min_value=0.1, max_value=0.9, value=0.3, step=0.05,
    help="Lower = higher recall, more false positives (catches more true 'quality' repos, "
         "but calls more low-quality ones 'high' too). Higher = higher precision, more false "
         "negatives. 5-fold CV: F1 peaks around 0.3 (precision 0.67/recall 0.89) vs the default "
         "0.5 (precision 0.89/recall 0.47). See README for the full table."
)

if st.button("Predict Quality", type="primary") and repo_input:
    with st.spinner("Fetching repo data from GitHub API..."):
        try:
            features = fetch_repo_features(repo_input, headers=headers)
        except RepoFetchError as e:
            st.error(str(e))
        else:
            topics = features["topics"]
            repo_age_days = features["repo_age_days"]

            X_dense = featurize(features, tfidf_readme, tfidf_topics, scaler)

            # Generate Predictions -- use the user-chosen threshold, not the
            # model's built-in 0.5 cutoff (see slider help text / README for why)
            probability = rf_model.predict_proba(X_dense)[0][1]
            prediction = 1 if probability >= threshold else 0


            # Log to audit trail
            log_audit_trail(features, probability, prediction, threshold)
            # Check for exceptions and low confidence
            exceptions = check_exceptions(features)
            low_confidence = 0.4 <= probability <= 0.6  # Model uncertainty band
            warning_messages = exceptions.copy()
            if low_confidence:
                warning_messages.append("⚠️ Low confidence prediction (probability near 0.5).")
            
            # Confidence report - explicit match rate and exceptions count
            n_exceptions = len(exceptions)
            n_low_confidence_reasons = 1 if low_confidence else 0
            total_issues = n_exceptions + n_low_confidence_reasons
            
            # Display Results UI
            st.subheader(f"Results for [{features['full_name']}]({features['html_url']})")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("⭐ Stars", features["stars"])
            col2.metric("🍴 Forks", features["forks"])
            col3.metric("🐛 Open Issues", features["open_issues"])
            col4.metric("📅 Age (Days)", repo_age_days)

            if topics:
                st.write("**Topics:** " + ", ".join([f"`{t}`" for t in topics]))

            st.divider()

            # Display confidence report
            st.caption(f"Confidence report: **{probability:.1%}** match rate | {total_issues} exception{'s' if total_issues != 1 else ''} flagged")
            if warning_messages:
                st.warning("\n\n".join(warning_messages))

            st.divider()

            res_col1, res_col2 = st.columns([2, 1])
            with res_col1:
                if prediction == 1:
                    st.success("### ✅ Predicted: High Quality Repository")
                else:
                    st.warning("### ⚠️ Predicted: Low Quality / Unmaintained Repository")

            with res_col2:
                st.metric("Model Confidence", f"{probability:.1%}")

            # --- Explainability: why did the model say this? ---
            st.divider()
            st.subheader("Why this prediction?")
            try:
                explainer = load_explainer(rf_model)
                shap_values = explainer.shap_values(X_dense, check_additivity=False)

                # Newer shap versions return one ndarray shaped
                # (n_samples, n_features, n_classes); older versions return a
                # list of one array per class. Handle both, taking class 1.
                if isinstance(shap_values, list):
                    sv = shap_values[1][0]
                elif np.ndim(shap_values) == 3:
                    sv = shap_values[0, :, 1]
                else:
                    sv = shap_values[0]

                feature_names = (
                    list(tfidf_readme.get_feature_names_out()) +
                    list(tfidf_topics.get_feature_names_out()) +
                    STRUCTURED_COLS
                )
                contrib = pd.DataFrame({"feature": feature_names, "shap_value": sv})
                contrib = contrib[contrib["shap_value"] != 0]
                top_pos = contrib.sort_values("shap_value", ascending=False).head(6)
                top_neg = contrib.sort_values("shap_value", ascending=True).head(6)

                # Get the expected value (base rate) for bounded explainability
                expected_value = explainer.expected_value
                if isinstance(expected_value, (list, np.ndarray)):
                    expected_value = np.ravel(expected_value)[-1]
                expected_value = float(expected_value)
                
                # Calculate how features move from base to prediction
                base_probability = 1 / (1 + np.exp(-expected_value))  # Convert log-odds to probability
                final_probability = probability
                  
                exp_col1, exp_col2, exp_col3 = st.columns([1, 1, 1])
                with exp_col1:
                    st.write("**Base expectation:**")
                    st.write(f"Average model output: {base_probability:.1%}")
                      
                with exp_col2:
                    st.write("**Feature contributions:**")
                    st.caption("Top features pushing prediction:")
                      
                with exp_col3:
                    st.write("**Final prediction:**")
                    st.write(f"{final_probability:.1%} probability")
                      
                st.write("")  # Spacer
                  
                # Show top positive and negative features
                feat_col1, feat_col2 = st.columns(2)
                with feat_col1:
                    st.write("**Pushed toward 'high quality':**")
                    for _, row in top_pos.iterrows():
                        st.markdown(f":green[✅ {row['feature']} (+{row['shap_value']:.3f})]")
                with feat_col2:
                    st.write("**Pushed toward 'low quality':**")
                    for _, row in top_neg.iterrows():
                        st.markdown(f":red[❌ {row['feature']} ({row['shap_value']:.3f})]")
            except Exception as err:
                st.caption(f"Explanation unavailable: {err}")

            # --- Heuristic Score (RepoScorer) Breakdown ---
            with st.expander("Heuristic Score (RepoScorer)"):
                try:
                    from scoring_engine import RepoScorer
                    scorer = RepoScorer()
                    heuristic_result = scorer.calculate_score(features)
                    
                    # Display key metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Heuristic Score", f"{heuristic_result['total_score']}/100")
                        st.write(f"**Tier:** {heuristic_result['tier_emoji']} {heuristic_result['tier']}")
                    with col2:
                        st.write("**Component Scores:**")
                        comps = heuristic_result['components']
                        st.write(f"• Maintenance: {comps['maintenance']}/100")
                        st.write(f"• Community: {comps['community']}/100")
                        st.write(f"• Documentation: {comps['documentation']}/100")
                        st.write(f"• Contributors: {comps['contributors']}/100")
                    
                    # Show delta vs ML model
                    ml_score_pct = probability * 100
                    delta = ml_score_pct - heuristic_result['total_score']
                    st.write(f"**Delta vs ML Model:** {delta:+.1f} points")
                    if abs(delta) > 15:
                        st.caption("⚠️ Large divergence suggests the ML model and heuristic scorer disagree significantly. "
                                 "This often happens with repos that have strong signals in one system but not the other "
                                 "(e.g., great code but poor documentation, or vice versa). See README for details.")
                    else:
                        st.caption("✅ Scores are well-aligned between ML model and heuristic scorer.")
                    
                    # Show explanations
                    if heuristic_result.get('explanations'):
                        st.write("**Explanations:**")
                        for exp in heuristic_result['explanations'][:3]:  # Show top 3
                            st.write(f"• {exp}")
                            
                except Exception as err:
                    st.caption(f"Heuristic score unavailable: {err}")

            # --- AI Review Section ---
            st.divider()
            st.subheader("AI Review")
            if AI_REVIEW_AVAILABLE:
                try:
                    ai_result = generate_ai_review(
                        readme_content=features.get("readme_text_clean", ""),
                        features=features,
                        prediction=prediction,
                        probability=probability
                    )
                    st.write(format_ai_review_for_display(ai_result))
                except Exception as err:
                    st.caption(f"AI review unavailable: {err}")
            else:
                st.caption("AI review unavailable: ai_review module not found.")
