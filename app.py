# -*- coding: utf-8 -*-
import base64
import os
import pickle
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import textwrap
from dotenv import load_dotenv
from scipy.sparse import hstack

from reposcore_utils import strip_badges, fetch_repo_features, featurize, RepoFetchError, STRUCTURED_COLS

try:
    from ai_review import generate_ai_review, format_ai_review_for_display
    AI_REVIEW_AVAILABLE = True
except ImportError:
    AI_REVIEW_AVAILABLE = False

load_dotenv()

if "NVIDIA_API_KEY" in st.secrets:
    os.environ["NVIDIA_API_KEY"] = st.secrets["NVIDIA_API_KEY"]

st.set_page_config(
    page_title="RepoScore",
    page_icon="⭐",
    layout="wide"
)

st.markdown("""
<style>
    .section-header { color: #00b4d8 !important; font-weight: 600; }
    .combined-score-container { padding: 1rem; border-radius: 0.5rem; background: linear-gradient(135deg, #1e1e2e 0%, #2a2a3e 100%); border: 1px solid #3a3a4e; }
    .stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
    .stTabs [data-baseweb="tab"] { padding: 0.5rem 1rem; border-radius: 0.5rem; }
    .rs-card { background-color: #1a1f2e; border: 1px solid #2d3548; border-radius: 10px; padding: 20px; margin-bottom: 16px; }
    .rs-verdict-low { background-color: #3a1f1f; border: 1px solid #c0392b; border-radius: 10px; padding: 16px 20px; margin-bottom: 12px; }
    .rs-verdict-high { background-color: #1a3a26; border: 1px solid #27ae60; border-radius: 10px; padding: 16px 20px; margin-bottom: 12px; }
    .rs-note { background-color: #1e2230; border-left: 3px solid #4a5568; border-radius: 4px; padding: 10px 14px; margin-bottom: 8px; color: #9aa5b8; font-size: 0.9em; }
    .rs-caution { background-color: #2e2a1a; border-left: 3px solid #d4a017; border-radius: 4px; padding: 10px 14px; margin-bottom: 12px; color: #e0c26e; }
    .rs-component-label { display: flex; justify-content: space-between; font-size: 0.9em; color: #cbd5e0; margin-bottom: 2px; }
    .stContainer[data-testid="stVerticalBlockBorderWrapper"] .stMarkdown { font-size: 0.9rem; line-height: 1.5; }
    .stContainer[data-testid="stVerticalBlockBorderWrapper"] .stMarkdown h3 { font-size: 1rem; margin-top: 1rem; margin-bottom: 0.5rem; }
    .stContainer[data-testid="stVerticalBlockBorderWrapper"] .stMarkdown ul { margin-top: 0.25rem; margin-bottom: 0.5rem; padding-left: 1.25rem; }
    .stContainer[data-testid="stVerticalBlockBorderWrapper"] .stMarkdown li { margin-bottom: 0.25rem; font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

def render_component_bar(label, value, max_value=100):
    pct = value / max_value
    color = "#27ae60" if pct >= 0.7 else "#d4a017" if pct >= 0.4 else "#c0392b"
    st.markdown(f'<div class="rs-component-label"><span>{label}</span><span>{value:.1f}/{max_value}</span></div><div style="background-color:#2d3548;border-radius:6px;height:8px;margin-bottom:12px;"><div style="background-color:{color};width:{pct*100}%;height:8px;border-radius:6px;"></div></div>', unsafe_allow_html=True)

def render_verdict_banner(prediction, probability):
    css_class = "rs-verdict-high" if prediction == 1 else "rs-verdict-low"
    icon = "✅" if prediction == 1 else "⚠️"
    label = "High Quality Repository" if prediction == 1 else "Low Quality / Unmaintained Repository"
    st.markdown(f'<div class="{css_class}"><span style="font-size:1.2em;font-weight:600;">{icon} Predicted: {label}</span><span style="float:right;font-size:1.1em;">Model Confidence: {probability:.1%}</span></div>', unsafe_allow_html=True)

def render_note(text):
    st.markdown(f'<div class="rs-note">ℹ️ {text}</div>', unsafe_allow_html=True)

def render_caution(text):
    st.markdown(f'<div class="rs-caution">⚠️ {text}</div>', unsafe_allow_html=True)

def render_card(content_html):
    st.markdown(f'<div class="rs-card">{content_html}</div>', unsafe_allow_html=True)

token = os.getenv("GITHUB_TOKEN")
headers = {"Accept": "application/vnd.github+json"}
if token:
    headers["Authorization"] = f"Bearer {token}"

def safe_load(file_path):
    try:
        return joblib.load(file_path)
    except Exception:
        with open(file_path, "rb") as f:
            return pickle.load(f)

@st.cache_resource
def load_ml_assets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "models")
    files = {"rf_model": "rf_model.pkl", "tfidf_readme": "tfidf_readme.pkl", "tfidf_topics": "tfidf_topics.pkl", "scaler": "scaler.pkl"}
    loaded = {}
    for key, filename in files.items():
        file_path = os.path.join(model_dir, filename)
        if not os.path.exists(file_path):
            st.error(f"⭐ Missing file: `{filename}` was not found in the `models/` directory.")
            st.stop()
        try:
            loaded[key] = safe_load(file_path)
        except Exception as err:
            st.error(f"⭐ Failed loading `{filename}`:")
            st.exception(err)
            st.stop()
    return loaded["rf_model"], loaded["tfidf_readme"], loaded["tfidf_topics"], loaded["scaler"]

@st.cache_resource
def load_explainer(_model):
    import shap
    return shap.TreeExplainer(_model)

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
    import json
    from datetime import datetime
    audit_dir = "audit_trail"
    if not os.path.exists(audit_dir):
        os.makedirs(audit_dir)
    csv_file = os.path.join(audit_dir, "scoring_decisions.csv")
    timestamp = datetime.now().isoformat()
    repo_id = features.get("full_name", "")
    logged_features = {"full_name": features.get("full_name", ""), "html_url": features.get("html_url", ""), "stars": features.get("stars", 0), "forks": features.get("forks", 0), "open_issues": features.get("open_issues", 0), "readme_size": features.get("readme_size", 0), "repo_age_days": features.get("repo_age_days", 0), "last_commit_days": features.get("last_commit_days", 0), "has_readme": features.get("has_readme", 0), "topics_count": len(features.get("topics", [])), "probability": f"{probability:.6f}", "prediction": prediction, "threshold": f"{threshold:.2f}", "timestamp": timestamp}
    fieldnames = ["timestamp", "repo_id", "repo_url", "stars", "forks", "open_issues", "readme_size", "repo_age_days", "last_commit_days", "has_readme", "topics_count", "probability", "prediction", "threshold"]
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({"timestamp": timestamp, "repo_id": repo_id, "repo_url": logged_features["html_url"], "stars": logged_features["stars"], "forks": logged_features["forks"], "open_issues": logged_features["open_issues"], "readme_size": logged_features["readme_size"], "repo_age_days": logged_features["repo_age_days"], "last_commit_days": logged_features["last_commit_days"], "has_readme": logged_features["has_readme"], "topics_count": logged_features["topics_count"], "probability": logged_features["probability"], "prediction": logged_features["prediction"], "threshold": logged_features["threshold"]})
    jsonl_file = os.path.join(audit_dir, "predictions.jsonl")
    logged_features_with_caveats = logged_features.copy()
    logged_features_with_caveats["caveats"] = caveats if caveats is not None else []
    with open(jsonl_file, "a", encoding="utf-8") as f:
        json.dump(logged_features_with_caveats, f)
        f.write("\n")

rf_model, tfidf_readme, tfidf_topics, scaler = load_ml_assets()

st.title("⭐ RepoScore: GitHub Repository Quality Predictor")
st.caption("Analyze a public GitHub repository to predict its overall quality score.")

# Initialize session state
if "repo_input_stored" not in st.session_state:
    st.session_state.repo_input_stored = ""
if "prediction_data" not in st.session_state:
    st.session_state.prediction_data = None
if "report_content" not in st.session_state:
    st.session_state.report_content = None

# Use stored repo input as default, but allow editing
repo_input = st.text_input("Enter Repository (owner/name):", placeholder="scikit-learn/scikit-learn", value=st.session_state.repo_input_stored)
threshold = st.slider("Quality threshold", min_value=0.1, max_value=0.9, value=0.3, step=0.05, help="Lower = higher recall, more false positives. Higher = higher precision, more false negatives. 5-fold CV: F1 peaks around 0.3.")

# Prediction button - only process if no existing prediction data
if st.button("Predict Quality", type="primary", key="predict_btn") and repo_input and st.session_state.prediction_data is None:
    with st.spinner("Fetching repo data from GitHub API..."):
        try:
            features = fetch_repo_features(repo_input, headers=headers)
        except RepoFetchError as e:
            st.error(str(e))
        else:
            # Store all prediction data in session state
            topics = features["topics"]
            repo_age_days = features["repo_age_days"]
            X_dense = featurize(features, tfidf_readme, tfidf_topics, scaler)
            probability = rf_model.predict_proba(X_dense)[0][1]
            prediction = 1 if probability >= threshold else 0
            exceptions = check_exceptions(features)
            low_confidence = 0.4 <= probability <= 0.6
            warning_messages = exceptions.copy()
            if low_confidence:
                warning_messages.append("Low confidence prediction (probability near 0.5).")
            n_exceptions = len(exceptions)
            n_low_confidence_reasons = 1 if low_confidence else 0
            total_issues = n_exceptions + n_low_confidence_reasons
            
            try:
                from scoring_engine import RepoScorer
                scorer = RepoScorer()
                heuristic_result = scorer.calculate_score(features)
                ml_score_pct = probability * 100
                heuristic_score = heuristic_result["total_score"]
                combined_score = (ml_score_pct + heuristic_score) / 2
                divergence = abs(ml_score_pct - heuristic_score)
            except Exception:
                heuristic_result = None
                ml_score_pct = probability * 100
                heuristic_score = None
                combined_score = None
                divergence = None
            
            # Store everything needed in session state
            st.session_state.repo_input_stored = repo_input
            st.session_state.prediction_data = {
                "features": features,
                "threshold": threshold,
                "topics": topics,
                "repo_age_days": repo_age_days,
                "probability": probability,
                "prediction": prediction,
                "exceptions": exceptions,
                "low_confidence": low_confidence,
                "warning_messages": warning_messages,
                "n_exceptions": n_exceptions,
                "n_low_confidence_reasons": n_low_confidence_reasons,
                "total_issues": total_issues,
                "heuristic_result": heuristic_result,
                "ml_score_pct": ml_score_pct,
                "heuristic_score": heuristic_score,
                "combined_score": combined_score,
                "divergence": divergence
            }

# Check if we have prediction data (from session state after rerun)
if st.session_state.prediction_data is not None:
    data = st.session_state.prediction_data
    features = data["features"]
    threshold = data["threshold"]
    topics = data["topics"]
    repo_age_days = data["repo_age_days"]
    probability = data["probability"]
    prediction = data["prediction"]
    exceptions = data["exceptions"]
    low_confidence = data["low_confidence"]
    warning_messages = data["warning_messages"]
    total_issues = data["total_issues"]
    heuristic_result = data["heuristic_result"]
    combined_score = data["combined_score"]
    divergence = data["divergence"]

    st.subheader(f"Results for [{features['full_name']}]({features['html_url']})")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("⭐ Stars", features["stars"])
    col2.metric("🍴 Forks", features["forks"])
    col3.metric("🐛 Open Issues", features["open_issues"])
    col4.metric("📅 Age (Days)", repo_age_days)

    if topics:
        st.write("**Topics:** " + ", ".join([f"`{t}`" for t in topics]))

    st.divider()

    if combined_score is not None:
        if combined_score >= 70:
            score_color = "green"
            score_emoji = "🟢"
        elif combined_score >= 40:
            score_color = "orange"
            score_emoji = "🟡"
        else:
            score_color = "red"
            score_emoji = "🔴"
        score_card_html = f'<h2 class="section-header">⭐ Combined Score: {combined_score:.1f}/100 {score_emoji}</h2><div style="margin-bottom: 1rem;"><div style="background-color: #2d3548; border-radius: 6px; height: 20px; overflow: hidden;"><div style="background-color: {score_color}; width: {combined_score}%; height: 100%; border-radius: 6px; transition: width 0.3s ease;"></div></div><div style="font-size: 0.9em; color: #cbd5e0; margin-top: 0.5rem;">{combined_score:.1f}% — 50% ML Model + 50% Heuristic</div></div><div style="display: flex; gap: 1rem; margin-top: 1rem;"><div style="flex: 1; text-align: center; padding: 0.75rem; background: #2d3548; border-radius: 6px;"><div style="font-size: 0.85em; color: #94a3b8;">ML Model</div><div style="font-size: 1.5em; font-weight: 600; color: #e2e8f0;">{data["ml_score_pct"]:.1f}%</div></div><div style="flex: 1; text-align: center; padding: 0.75rem; background: #2d3548; border-radius: 6px;"><div style="font-size: 0.85em; color: #94a3b8;">Heuristic</div><div style="font-size: 1.5em; font-weight: 600; color: #e2e8f0;">{data["heuristic_score"]:.1f}/100</div></div><div style="flex: 1; text-align: center; padding: 0.75rem; background: #2d3548; border-radius: 6px;"><div style="font-size: 0.85em; color: #94a3b8;">Divergence</div><div style="font-size: 1.5em; font-weight: 600; color: {"#f87171" if abs(divergence) > 15 else "#e2e8f0"};">{divergence:+.1f}</div></div></div>'
        render_card(score_card_html)
        if divergence > 15:
            render_caution(f"ML and Heuristic scores diverge by {divergence:.1f} points — treat this combined score with caution; review both scores individually in the **Why This Score?** tab.")

    confidence_html = f'<div style="font-size: 0.95em; color: #cbd5e0;">Confidence report: <strong>{probability:.1%}</strong> match rate | {total_issues} exception{"s" if total_issues != 1 else ""} flagged</div>'
    render_card(confidence_html)
    if warning_messages:
        for msg in warning_messages:
            render_caution(msg)

    st.divider()
    render_verdict_banner(prediction, probability)

    tab_overview, tab_why, tab_ai, tab_security, tab_trends, tab_report = st.tabs(["📊 Overview", "🔍 Why This Score?", "🤖 AI Review", "🛡️ Security", "📈 Trends", "📑 Report"])

    with tab_overview:
        topics_html = ""
        if topics:
            topics_html = f'<div style="margin-top: 0.5rem; color: #27ae60;"><strong>Topics:</strong> {" , ".join([f"{t}" for t in topics])}</div>'
        components = heuristic_result["components"]
        component_bars_html = ""
        for label, value in [("Maintenance", components["maintenance"]), ("Community", components["community"]), ("Documentation", components["documentation"]), ("Contributors", components["contributors"])]:
            pct = value / 100
            color = "#27ae60" if pct >= 0.7 else "#d4a017" if pct >= 0.4 else "#c0392b"
            component_bars_html += f'<div style="margin-bottom: 12px;"><div class="rs-component-label"><span>{label}</span><span>{value:.1f}/100</span></div><div style="background-color:#2d3548;border-radius:6px;height:8px;"><div style="background-color:{color};width:{pct*100}%;height:8px;border-radius:6px;"></div></div></div>'
        data_quality_html = ""
        if exceptions or low_confidence:
            notes = []
            for exc in exceptions:
                notes.append(f'<div class="rs-note">ℹ️ {exc}</div>')
            if low_confidence:
                notes.append('<div class="rs-note">ℹ️ Low confidence prediction (probability near 0.5).</div>')
            data_quality_html = f'<div style="margin-top: 1rem;"><strong>Data Quality Notes</strong>{"".join(notes)}</div>'
        overview_card_html = f'<h3 class="section-header">📊 Overview</h3><div style="display: flex; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap;"><div style="flex: 1; min-width: 120px; text-align: center; padding: 0.75rem; background: #2d3548; border-radius: 6px;"><div style="font-size: 0.85em; color: #94a3b8;">⭐ Stars</div><div style="font-size: 1.5em; font-weight: 600; color: #e2e8f0;">{features["stars"]}</div></div><div style="flex: 1; min-width: 120px; text-align: center; padding: 0.75rem; background: #2d3548; border-radius: 6px;"><div style="font-size: 0.85em; color: #94a3b8;">🍴 Forks</div><div style="font-size: 1.5em; font-weight: 600; color: #e2e8f0;">{features["forks"]}</div></div><div style="flex: 1; min-width: 120px; text-align: center; padding: 0.75rem; background: #2d3548; border-radius: 6px;"><div style="font-size: 0.85em; color: #94a3b8;">🐛 Open Issues</div><div style="font-size: 1.5em; font-weight: 600; color: #e2e8f0;">{features["open_issues"]}</div></div><div style="flex: 1; min-width: 120px; text-align: center; padding: 0.75rem; background: #2d3548; border-radius: 6px;"><div style="font-size: 0.85em; color: #94a3b8;">📅 Age (Days)</div><div style="font-size: 1.5em; font-weight: 600; color: #e2e8f0;">{repo_age_days}</div></div></div>{topics_html}<div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #2d3548;"><div style="font-size: 0.95em; color: #cbd5e0; margin-bottom: 0.5rem;"><strong>Threshold used:</strong> 0.5</div><div style="font-size: 0.95em; color: #cbd5e0; margin-bottom: 0.5rem;"><strong>Model probability:</strong> {probability:.1%}</div><div style="font-size: 0.95em; color: #cbd5e0; margin-bottom: 1rem;"><strong>Prediction:</strong> {"High Quality" if prediction == 1 else "Low Quality / Unmaintained"}</div></div><div style="margin-top: 1rem;"><strong style="color: #e2e8f0;">Component Scores</strong>{component_bars_html}</div>{data_quality_html}'
        render_card(overview_card_html)

    with tab_why:
        st.markdown("<h3 class=\"section-header\">Feature Contribution Analysis (SHAP)</h3>", unsafe_allow_html=True)
        try:
            explainer = load_explainer(rf_model)
            X_dense_local = featurize(features, tfidf_readme, tfidf_topics, scaler)
            shap_values = explainer.shap_values(X_dense_local, check_additivity=False)
            if isinstance(shap_values, list):
                sv = shap_values[1][0]
            elif np.ndim(shap_values) == 3:
                sv = shap_values[0, :, 1]
            else:
                sv = shap_values[0]
            feature_names = list(tfidf_readme.get_feature_names_out()) + list(tfidf_topics.get_feature_names_out()) + STRUCTURED_COLS
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
                    st.markdown(f":red[⚠️ {row['feature']} ({row['shap_value']:.3f})]")
        except Exception as err:
            st.caption(f"Explanation unavailable: {err}")

        with st.expander("Heuristic Score (RepoScorer)", expanded=True):
            try:
                if data["heuristic_result"] is None:
                    from scoring_engine import RepoScorer
                    scorer = RepoScorer()
                    heuristic_result_local = scorer.calculate_score(features)
                else:
                    heuristic_result_local = data["heuristic_result"]
                ml_score_pct_local = probability * 100
                heuristic_score_local = heuristic_result_local["total_score"]
                combined_score_local = (ml_score_pct_local + heuristic_score_local) / 2
                divergence_local = abs(ml_score_pct_local - heuristic_score_local)
                st.markdown('<div class="rs-card">', unsafe_allow_html=True)
                st.subheader("Heuristic Score Details")
                st.write(f"**Tier:** {heuristic_result_local['tier_emoji']} {heuristic_result_local['tier']}")
                st.write(f"**Heuristic Score:** {heuristic_score_local:.1f}/100")
                st.markdown("**Component Scores**")
                render_component_bar("Maintenance", heuristic_result_local["components"]["maintenance"])
                render_component_bar("Community", heuristic_result_local["components"]["community"])
                render_component_bar("Documentation", heuristic_result_local["components"]["documentation"])
                render_component_bar("Contributors", heuristic_result_local["components"]["contributors"])
                delta = ml_score_pct_local - heuristic_score_local
                st.write(f"**Delta vs ML Model:** {delta:+.1f} points")
                if abs(delta) > 15:
                    render_caution(f"ML and Heuristic scores diverge by {abs(delta):.1f} points — treat this combined score with caution.")
                else:
                    st.caption("✅ Scores are well-aligned between ML model and heuristic scorer.")
                if heuristic_result_local.get("explanations"):
                    st.write("**Explanations:**")
                    for exp in heuristic_result_local["explanations"][:3]:
                        st.write(f"• {exp}")
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as err:
                st.caption(f"Heuristic score unavailable: {err}")

    with tab_ai:
        st.markdown("<h3 class=\"section-header\">🤖 AI Review</h3>", unsafe_allow_html=True)
        if AI_REVIEW_AVAILABLE:
            try:
                @st.cache_data(ttl=86400, show_spinner=False)
                def _cached_ai_review(readme_text_clean: str, features_hashable: tuple, prediction: int, probability: float):
                    features_dict = dict(features_hashable)
                    return generate_ai_review(readme_content=readme_text_clean, features=features_dict, prediction=prediction, probability=probability)
                readme_text = features.get("readme_text_clean", "")
                features_hashable = tuple(sorted(features.items()))
                with st.spinner("Generating AI review..."):
                    ai_result = _cached_ai_review(readme_text, features_hashable, prediction, probability)
                if ai_result.get("status") == "success":
                    with st.container(border=True):
                        st.markdown(format_ai_review_for_display(ai_result))
                else:
                    st.error(format_ai_review_for_display(ai_result))
            except Exception as err:
                st.error(f"AI review unavailable: {str(err)}")
        else:
            st.caption("AI review unavailable: ai_review module not found.")

    with tab_security:
        st.markdown("<h3 class=\"section-header\">🛡️ Security Analysis</h3>", unsafe_allow_html=True)
        with st.spinner("Scanning for vulnerabilities..."):
            try:
                from security_scanner import scan_repository
                from reposcore_utils import clone_repo_bounded
                import shutil
                repo_size_kb = features.get("size", 0) if "size" in features else 1000
                repo_path = clone_repo_bounded(repo_input, repo_size_kb)
                if repo_path and os.path.exists(repo_path):
                    try:
                        scan_result = scan_repository(repo_path)
                        vuln_col1, vuln_col2, vuln_col3, vuln_col4 = st.columns(4)
                        vuln_col1.metric("Total", scan_result.total_vulnerabilities)
                        vuln_col2.metric("Critical", scan_result.critical_count, delta="🚨" if scan_result.critical_count > 0 else None)
                        vuln_col3.metric("High", scan_result.high_count, delta="⚠️" if scan_result.high_count > 0 else None)
                        vuln_col4.metric("Medium", scan_result.medium_count)
                        risk_color = {"CRITICAL": "🚨", "HIGH": "⚠️", "MEDIUM": "⚡", "LOW": "ℹ️", "NONE": "✅"}
                        st.markdown(f"**Risk Level:** {risk_color.get(scan_result.risk_level, '?')} {scan_result.risk_level}")
                        st.markdown(f"**Scan Method:** {scan_result.scan_method}")
                        st.markdown(f"**Dependencies Found:** {scan_result.dependencies_found}")
                        if scan_result.vulnerabilities:
                            st.markdown("**Vulnerabilities Detected:**")
                            for vuln in scan_result.vulnerabilities[:10]:
                                with st.expander(f"{vuln.package_name} ({vuln.severity.value})"):
                                    st.markdown(f"**Version:** {vuln.version}")
                                    st.markdown(f"**ID:** {vuln.vulnerability_id}")
                                    st.markdown(f"**Description:** {vuln.description[:200]}...")
                                    if vuln.fix_version:
                                        st.markdown(f"**Fix Version:** {vuln.fix_version}")
                        else:
                            st.success("No vulnerabilities detected! 🎉")
                    finally:
                        shutil.rmtree(repo_path, ignore_errors=True)
                else:
                    st.warning("Repository too large to scan for vulnerabilities (max 50MB)")
            except ImportError:
                st.caption("Security scanner not available")
            except Exception as err:
                st.error(f"Security scan failed: {str(err)}")
        st.divider()
        st.markdown("<h4 class=\"section-header\">📄 License Check</h4>", unsafe_allow_html=True)
        try:
            from license_checker import check_license_from_repo
            lic_result = check_license_from_repo(None, github_license=features.get("_license_data"))
            lic_col1, lic_col2 = st.columns(2)
            lic_col1.markdown(f"**License:** {lic_result.license_info.name if lic_result.license_info else 'Unknown'}")
            lic_col1.markdown(f"**SPDX:** {lic_result.license_info.spdx_id if lic_result.license_info else 'NOASSERTION'}")
            lic_col2.markdown(f"**Compliance Score:** {lic_result.compliance_score}/100")
            lic_col2.markdown(f"**Commercial Compatible:** {'✅ Yes' if lic_result.commercial_compatible else '❌ No'}")
            if lic_result.warnings:
                for warning in lic_result.warnings:
                    render_caution(warning)
        except ImportError:
            st.caption("License checker not available")
        except Exception as err:
            st.caption(f"License check unavailable: {str(err)}")

    with tab_trends:
        st.markdown("<h3 class=\"section-header\">📈 Trend Analysis</h3>", unsafe_allow_html=True)
        with st.spinner("Analyzing trends..."):
            try:
                from trends_analyzer import analyze_repository, get_trend_summary
                trend_analysis = analyze_repository(repo_input, features, headers)
                trend_col1, trend_col2, trend_col3, trend_col4 = st.columns(4)
                trend_col1.metric("Star Growth (30d)", f"{trend_analysis.star_growth_rate_30d:+.1f}%")
                trend_col2.metric("Star Growth (90d)", f"{trend_analysis.star_growth_rate_90d:+.1f}%")
                trend_col3.metric("Commits (90d)", trend_analysis.commit_activity_90d)
                trend_col4.metric("Health Score", trend_analysis.health_score, delta=trend_analysis.health_status.title())
                st.markdown(f"**Trend Direction:** {trend_analysis.trend_direction.title()}")
                st.markdown(f"**Activity Trend:** {trend_analysis.activity_trend.title()}")
                st.markdown(f"**Commit Frequency:** {trend_analysis.commit_frequency.replace('_', ' ').title()}")
                st.markdown(f"**Fork Ratio:** {trend_analysis.fork_ratio:.3f} ({trend_analysis.fork_ratio_interpretation.replace('_', ' ').title()})")
                if trend_analysis.stars_30d_ago:
                    st.markdown(f"**Stars 30 days ago:** {trend_analysis.stars_30d_ago:,}")
                if trend_analysis.stars_90d_ago:
                    st.markdown(f"**Stars 90 days ago:** {trend_analysis.stars_90d_ago:,}")
                st.markdown(f"**Summary:** {get_trend_summary(trend_analysis)}")
            except ImportError:
                st.caption("Trend analyzer not available")
            except Exception as err:
                st.error(f"Trend analysis failed: {str(err)}")

    with tab_report:
        st.markdown("<h3 class=\"section-header\">📑 Quality Report</h3>", unsafe_allow_html=True)
        if "report_content" not in st.session_state:
            st.session_state.report_content = None
        report_format = st.selectbox("Report Format", ["html", "json"], label_visibility="collapsed")
        if st.button("Generate Report", type="primary", key="generate_report_btn"):
            with st.spinner("Generating report..."):
                try:
                    from report_generator import generate_report
                    st.session_state.report_content = generate_report(
                        full_name=features["full_name"],
                        html_url=features.get("html_url", ""),
                        features=features,
                        ml_probability=probability,
                        heuristic_score=heuristic_result,
                        combined_score=combined_score if combined_score else (probability * 100 + (heuristic_result.get("total_score", 0) if heuristic_result else 0)) / 2,
                        format=report_format
                    )
                except ImportError:
                    st.caption("Report generator not available")
                    st.session_state.report_content = None
                except Exception as err:
                    st.error(f"Report generation failed: {str(err)}")
                    st.session_state.report_content = None
        if st.session_state.report_content:
            report_content = st.session_state.report_content
            if report_format == "json":
                st.json(json.loads(report_content))
            else:
                st.markdown("### Report Preview")
                st.components.v1.html(report_content, height=600, scrolling=True)
                report_bytes = report_content.encode()
                st.download_button(
                    label="Download Report",
                    data=report_bytes,
                    file_name=f"reposcore_report_{features['full_name'].replace('/', '_')}.html",
                    mime="text/html",
                    key="download_report_btn"
                )

elif st.button("Clear Results", key="clear_btn"):
    # Allow clearing results to start fresh
    st.session_state.prediction_data = None
    st.session_state.repo_input_stored = ""
    st.session_state.report_content = None
    st.rerun()