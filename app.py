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

# Wire Streamlit secrets (e.g., GEMINI_API_KEY, NVIDIA_API_KEY) into os.environ so downstream modules can use os.getenv()
if "GEMINI_API_KEY" in st.secrets:
    os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
if "NVIDIA_API_KEY" in st.secrets:
    os.environ["NVIDIA_API_KEY"] = st.secrets["NVIDIA_API_KEY"]

# Page configuration
st.set_page_config(
    page_title="RepoScore",
    page_icon="⭐",
    layout="wide"
)

# Custom CSS for consistent accent color and styling
st.markdown("""
<style>
    /* Accent color for section headers */
    .section-header {
        color: #00b4d8 !important;
        font-weight: 600;
    }
    /* Combined score styling */
    .combined-score-container {
        padding: 1rem;
        border-radius: 0.5rem;
        background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%);
        border: 1px solid #3a3a4e;
    }
    /* Ensure tabs have consistent styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

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

            # Calculate heuristic and combined scores early for top-level display
            try:
                from scoring_engine import RepoScorer
                scorer = RepoScorer()
                heuristic_result = scorer.calculate_score(features)
                ml_score_pct = probability * 100
                heuristic_score = heuristic_result['total_score']
                combined_score = (ml_score_pct + heuristic_score) / 2
                divergence = abs(ml_score_pct - heuristic_score)
            except Exception:
                heuristic_result = None
                ml_score_pct = probability * 100
                heuristic_score = None
                combined_score = None
                divergence = None

            # Display Results UI
            st.subheader(f"Results for [{features['full_name']}]({features['html_url']})")

            # --- Repo Stats (always visible above tabs) ---
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("⭐ Stars", features["stars"])
            col2.metric("🍴 Forks", features["forks"])
            col3.metric("🐛 Open Issues", features["open_issues"])
            col4.metric("📅 Age (Days)", repo_age_days)

            if topics:
                st.write("**Topics:** " + ", ".join([f"`{t}`" for t in topics]))

            st.divider()

            # --- Calculate scores early ---
            try:
                from scoring_engine import RepoScorer
                scorer = RepoScorer()
                heuristic_result = scorer.calculate_score(features)
                ml_score_pct = probability * 100
                heuristic_score = heuristic_result['total_score']
                combined_score = (ml_score_pct + heuristic_score) / 2
                divergence = abs(ml_score_pct - heuristic_score)
            except Exception:
                heuristic_result = None
                ml_score_pct = probability * 100
                heuristic_score = None
                combined_score = None
                divergence = None

            # --- HEADLINE COMBINED SCORE (prominent, with progress bar) ---
            if combined_score is not None:
                # Determine color based on score range
                if combined_score >= 70:
                    score_color = "green"
                    score_emoji = "🟢"
                elif combined_score >= 40:
                    score_color = "orange"
                    score_emoji = "🟡"
                else:
                    score_color = "red"
                    score_emoji = "🔴"

                # Main score display
                st.markdown(f'<h2 class="section-header">📊 Combined Score: {combined_score:.1f}/100 {score_emoji}</h2>', unsafe_allow_html=True)
                st.progress(int(combined_score), text=f"{combined_score:.1f}% — 50% ML Model + 50% Heuristic")

                # Secondary scores in columns
                score_col1, score_col2, score_col3 = st.columns(3)
                with score_col1:
                    st.metric("ML Model", f"{ml_score_pct:.1f}%")
                with score_col2:
                    st.metric("Heuristic", f"{heuristic_score:.1f}/100")
                with score_col3:
                    delta = ml_score_pct - heuristic_score
                    st.metric("Divergence", f"{delta:+.1f}", delta_color="inverse" if abs(delta) > 15 else "normal")

                # Divergence warning
                if divergence > 15:
                    st.warning(f"⚠️ ML and Heuristic scores diverge by {divergence:.1f} points — treat this combined score with caution; review both scores individually in the **Why This Score?** tab.")

            # Confidence report & warnings
            st.caption(f"Confidence report: **{probability:.1%}** match rate | {total_issues} exception{'s' if total_issues != 1 else ''} flagged")
            if warning_messages:
                st.warning("\n\n".join(warning_messages))

            st.divider()

            # Prediction badge
            res_col1, res_col2 = st.columns([2, 1])
            with res_col1:
                if prediction == 1:
                    st.success("### ✅ Predicted: High Quality Repository")
                else:
                    st.warning("### ⚠️ Predicted: Low Quality / Unmaintained Repository")

            with res_col2:
                st.metric("Model Confidence", f"{probability:.1%}")

            # --- TABS FOR DETAILED SECTIONS ---
            tab_overview, tab_why, tab_ai = st.tabs(["📊 Overview", "🔍 Why This Score?", "🤖 AI Review"])
            # ==================== TAB 1: OVERVIEW ====================
            with tab_overview:
                st.markdown("<h3 class=\"section-header\">Repository Overview</h3>", unsafe_allow_html=True)
                ov_col1, ov_col2 = st.columns(2)
                with ov_col1:
                    st.markdown("**Prediction Details**")
                    st.write(f"• **Threshold used:** {threshold:.2f}")
                    st.write(f"• **Model probability:** {probability:.1%}")
                    st.write(f"• **Prediction:** {'High Quality' if prediction == 1 else 'Low Quality / Unmaintained'}")
                    if combined_score is not None:
                        st.write(f"• **Combined Score:** {combined_score:.1f}/100")
                        st.write(f"• **ML Score:** {ml_score_pct:.1f}%")
                        st.write(f"• **Heuristic Score:** {heuristic_score:.1f}/100")
                        st.write(f"• **Divergence:** {divergence:.1f} points")
                with ov_col2:
                    if heuristic_result:
                        st.markdown("**Heuristic Breakdown (RepoScorer)**")
                        st.write(f"• **Tier:** {heuristic_result['tier_emoji']} {heuristic_result['tier']}")
                        st.write(f"• **Total Score:** {heuristic_score:.1f}/100")
                        comps = heuristic_result['components']
                        st.write(f"  - Maintenance: {comps['maintenance']:.1f}")
                        st.write(f"  - Community: {comps['community']:.1f}")
                        st.write(f"  - Documentation: {comps['documentation']:.1f}")
                        st.write(f"  - Contributors: {comps['contributors']:.1f}")
                        if heuristic_result.get('explanations'):
                            st.write("**Key Factors:**")
                            for exp in heuristic_result['explanations'][:4]:
                                st.write(f"• {exp}")
                if exceptions or low_confidence:
                    st.markdown("**Data Quality Notes**")
                    for exc in exceptions:
                        st.caption(f"⚠️ {exc}")
                    if low_confidence:
                        st.caption("⚠️ Low confidence prediction (probability near 0.5).")

# ==================== TAB 2: WHY THIS SCORE? ====================
            with tab_why:
                st.markdown("<h3 class=\"section-header\">Feature Contribution Analysis (SHAP)</h3>", unsafe_allow_html=True)
                try:
                    explainer = load_explainer(rf_model)
                    shap_values = explainer.shap_values(X_dense, check_additivity=False)
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
                    expected_value = explainer.expected_value
                    if isinstance(expected_value, (list, np.ndarray)):
                        expected_value = np.ravel(expected_value)[-1]
                    expected_value = float(expected_value)
                    base_probability = 1 / (1 + np.exp(-expected_value))
                    ctx_col1, ctx_col2, ctx_col3 = st.columns(3)
                    with ctx_col1:
                        st.metric("Base Expectation", f"{base_probability:.1%}")
                    with ctx_col2:
                        st.metric("Final Prediction", f"{probability:.1%}")
                    with ctx_col3:
                        st.metric("Shift", f"{probability - base_probability:+.1%}")
                    st.caption("These reflect word patterns found in the README text and topics, not direct quality judgments.")
                    feat_col1, feat_col2 = st.columns(2)
                    with feat_col1:
                        st.markdown("**Pushed toward High Quality**")
                        for _, row in top_pos.iterrows():
                            st.markdown(f":green[✅ {row['feature']} (+{row['shap_value']:.3f})]")
                    with feat_col2:
                        st.markdown("**Pushed toward Low Quality**")
                        for _, row in top_neg.iterrows():
                            st.markdown(f":red[❌ {row['feature']} ({row['shap_value']:.3f})]")
                except Exception as err:
                    st.caption(f"Explanation unavailable: {err}")

            # --- Heuristic Score (RepoScorer) Breakdown ---
            with st.expander("Heuristic Score (RepoScorer)", expanded=True):
                try:
                    # Use pre-calculated heuristic_result from above if available
                    if 'heuristic_result' not in locals() or heuristic_result is None:
                        from scoring_engine import RepoScorer
                        scorer = RepoScorer()
                        heuristic_result = scorer.calculate_score(features)

                    ml_score_pct = probability * 100
                    heuristic_score = heuristic_result['total_score']
                    combined_score = (ml_score_pct + heuristic_score) / 2
                    divergence = abs(ml_score_pct - heuristic_score)

                    st.subheader("Heuristic Score Details")
                    st.write(f"**Tier:** {heuristic_result['tier_emoji']} {heuristic_result['tier']}")
                    st.write(f"**Heuristic Score:** {heuristic_score:.1f}/100")
                    st.write(f"**Component Scores:** Maintenance: {heuristic_result['components']['maintenance']:.1f}, Community: {heuristic_result['components']['community']:.1f}, Documentation: {heuristic_result['components']['documentation']:.1f}, Contributors: {heuristic_result['components']['contributors']:.1f}")
                    delta = ml_score_pct - heuristic_score
                    st.write(f"**Delta vs ML Model:** {delta:+.1f} points")
                    if abs(delta) > 15:
                        st.caption("⚠️ Large divergence between ML and heuristic scores suggests the repo has contrasting strengths (e.g., great code but poor documentation, or vice versa). See README for details.")
                    else:
                        st.caption("✅ Scores are well-aligned between ML model and heuristic scorer.")
                    if heuristic_result.get('explanations'):
                        st.write("**Explanations:**")
                        for exp in heuristic_result['explanations'][:3]:  # Show top 3
                            st.write(f"• {exp}")
                except Exception as err:
                    st.caption(f"Heuristic score unavailable: {err}")

            # ==================== TAB 3: AI REVIEW ====================
            with tab_ai:
                st.markdown("<h3 class=\"section-header\">AI Review</h3>", unsafe_allow_html=True)
                if AI_REVIEW_AVAILABLE:
                    try:
                        # Cached AI review to avoid repeated API calls for same repo
                        @st.cache_data(ttl=86400, show_spinner=False)
                        def _cached_ai_review(readme_text_clean: str, features_hashable: tuple, prediction: int, probability: float):
                            # Rebuild features dict from hashable tuple
                            features_dict = dict(features_hashable)
                            return generate_ai_review(
                                readme_content=readme_text_clean,
                                features=features_dict,
                                prediction=prediction,
                                probability=probability
                            )

                        readme_text = features.get("readme_text_clean", "")
                        features_hashable = tuple(sorted(features.items()))
                        ai_result = _cached_ai_review(readme_text, features_hashable, prediction, probability)
                        st.write(format_ai_review_for_display(ai_result))
                    except Exception as err:
                        st.caption(f"AI review unavailable: {err}")
                else:
                    st.caption("AI review unavailable: ai_review module not found.")