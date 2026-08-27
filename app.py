# -*- coding: utf-8 -*-
import csv
import json
import os
import pickle
import re
import textwrap
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from reposcore_utils import fetch_repo_features, featurize, RepoFetchError, STRUCTURED_COLS

# Import AI review module
try:
    from ai_review import generate_ai_review, format_ai_review_for_display
    AI_REVIEW_AVAILABLE = True
except ImportError:
    AI_REVIEW_AVAILABLE = False

try:
    from scoring_engine import RepoScorer
    SCORING_ENGINE_AVAILABLE = True
except ImportError:
    SCORING_ENGINE_AVAILABLE = False

# ----------------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------------
load_dotenv()

if "NVIDIA_API_KEY" in st.secrets:
    os.environ["NVIDIA_API_KEY"] = st.secrets["NVIDIA_API_KEY"]

st.set_page_config(page_title="RepoScore", page_icon="⭐", layout="wide")

REPO_INPUT_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
AUDIT_DIR = "audit_trail"
AUDIT_CSV = os.path.join(AUDIT_DIR, "scoring_decisions.csv")
AUDIT_MAX_ROWS = 10_000  # rotate once the active log passes this many rows

st.markdown("""
 <style>
    :root {
        --bg-primary: #0d0d0f;
        --bg-secondary: #1a1a20;
        --bg-tertiary: #25252a;
        --fg-primary: #e8e8ec;
        --fg-secondary: #a0a0a8;
        --accent: #00d4aa;
        --accent-hover: #00f5c4;
        --danger: #ff5f57;
        --warning: #f0a500;
        --muted: #6b6b7a;
        --card-bg: #14141a;
        --border: #3a3a4a;
    }

    * {
        box-sizing: border-box;
    }

    .stApp {
        background-color: var(--bg-primary);
        color: var(--fg-primary);
        font-family: 'Segoe UI', Tahoma, Arial, sans-serif;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    h1, h2, h3, h4, h5, h6 {
        color: var(--fg-primary);
        font-weight: 600;
        line-height: 1.3;
    }

    .section-header {
        color: var(--accent) !important;
        font-weight: 600;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid var(--border);
        margin-bottom: 1.5rem;
    }

    .sub-section-header {
        color: var(--accent);
        font-weight: 500;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 1rem;
    }

    .stTextInput > div > div > input {
        background-color: var(--bg-tertiary);
        border: 1px solid var(--border);
        border-radius: 8px;
        color: var(--fg-primary);
        font-size: 1rem;
        padding: 0.75rem 1.2rem;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }

    .stTextInput > div > div > input:focus {
        border-color: var(--accent);
        box-shadow: 0 0 0 3px rgba(0, 212, 170, 0.15);
        outline: none;
    }

    .stTextInput > div > div > input::placeholder {
        color: var(--fg-secondary);
    }

    .stSlider > div > div > div > div {
        background-color: var(--border);
        border-radius: 4px;
    }

    .stSlider > div > div > div > div > div {
        background-color: var(--accent);
        border-radius: 4px;
        transition: width 0.15s ease;
    }

    .stButton > button {
        background: linear-gradient(135deg, var(--accent), var(--accent-hover));
        color: var(--bg-primary);
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: 500;
        width: 100%;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        margin-top: 1rem;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 212, 170, 0.3);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    .stMetric > label {
        color: var(--fg-secondary) !important;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .stMetric > div {
        color: var(--fg-primary);
    }

    /* Improve Streamlit column gaps */
    .stColumns > div {
        gap: 0.5rem;
    }

    /* Better table/text readability */
    .stDataFrame {
        background-color: var(--bg-secondary);
        border-radius: 8px;
    }

    /* Avatar-like repo name display */
    .repo-name-display {
        display: inline-block;
        background: linear-gradient(135deg, var(--bg-tertiary), var(--bg-secondary));
        padding: 0.2rem 0.6rem;
        border-radius: 4px;
        font-family: 'SF Mono', 'Consolas', monospace;
        font-size: 0.85rem;
        color: var(--accent);
        margin-left: 0.5rem;
    }

    /* Focus visible states */
    .stTextInput > div > div > input:focus-visible,
    .stSlider > div > div > div > div > div:focus-visible {
        outline: 2px solid var(--accent);
        outline-offset: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Small render helpers
# ----------------------------------------------------------------------------
def render_component_bar(label, value, max_value=100):
    pct = value / max_value
    if pct >= 0.7:
        color = "var(--accent)"
    elif pct >= 0.4:
        color = var(--warning)
    else:
        color = var(--danger)
    st.markdown(textwrap.dedent(f"""
    <div style="margin-bottom: 12px;">
        <div style="display: flex; justify-content: space-between; font-size: 0.9em; color: var(--fg-secondary); margin-bottom: 2px;">
            <span>{label}</span>
            <span>{value:.1f}/{max_value}</span>
        </div>
        <div style="background-color: var(--bg-tertiary); border-radius: 6px; height: 8px; margin-bottom: 8px;">
            <div style="background-color: {color}; width:{pct*100}%; height: 8px; border-radius: 6px; transition: width 0.3s cubic-bezier(0.4, 0, 0.2, 1);"></div>
        </div>
    </div>
    """), unsafe_allow_html=True)


def render_verdict_banner(prediction, probability):
    css_class = "rs-verdict-high" if prediction == 1 else "rs-verdict-low"
    icon = "✅" if prediction == 1 else "⚠️"
    label = "High Quality Repository" if prediction == 1 else "Low Quality / Unmaintained Repository"
    st.markdown(textwrap.dedent(f"""
    <div class="{css_class}">
        <span style="font-size:1.2em;font-weight:600;">{icon} Predicted: {label}</span>
        <span style="float:right;font-size:1.1em;">Model Confidence: {probability:.1%}</span>
    </div>
    """), unsafe_allow_html=True)


def render_note(text):
    st.markdown(f'<div class="rs-note">ℹ️ {text}</div>', unsafe_allow_html=True)


def render_caution(text):
    st.markdown(f'<div class="rs-caution">⚠️ {text}</div>', unsafe_allow_html=True)


def render_card(content_html):
    st.markdown(f'<div class="rs-card">{content_html}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# GitHub API headers
# ----------------------------------------------------------------------------
token = os.getenv("GITHUB_TOKEN")
headers = {"Accept": "application/vnd.github+json"}
if token:
    headers["Authorization"] = f"Bearer {token}"


# ----------------------------------------------------------------------------
# Model loading
# ----------------------------------------------------------------------------
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
        "scaler": "scaler.pkl",
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


# ----------------------------------------------------------------------------
# Feature / score helpers
# ----------------------------------------------------------------------------
def check_exceptions(features):
    """Check for data quality issues that might affect prediction reliability."""
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


def compute_scores(features, probability):
    """
    Single source of truth for ML / heuristic / combined scores.
    Computed once per prediction and reused everywhere in the UI —
    previously this block was duplicated verbatim in two places,
    which risked the two copies drifting apart.
    """
    ml_score_pct = probability * 100
    if not SCORING_ENGINE_AVAILABLE:
        return {
            "heuristic_result": None,
            "ml_score_pct": ml_score_pct,
            "heuristic_score": None,
            "combined_score": None,
            "divergence": None,
        }
    try:
        scorer = RepoScorer()
        heuristic_result = scorer.calculate_score(features)
        heuristic_score = heuristic_result["total_score"]
        combined_score = (ml_score_pct + heuristic_score) / 2
        divergence = abs(ml_score_pct - heuristic_score)
    except Exception:
        heuristic_result = None
        heuristic_score = None
        combined_score = None
        divergence = None

    return {
        "heuristic_result": heuristic_result,
        "ml_score_pct": ml_score_pct,
        "heuristic_score": heuristic_score,
        "combined_score": combined_score,
        "divergence": divergence,
    }


def features_cache_key(features):
    """
    JSON-based cache key instead of tuple(sorted(features.items())).
    The old approach broke as soon as a feature value was a list
    (e.g. `topics`), since a tuple containing a list isn't hashable
    and st.cache_data hashes its arguments.
    """
    return json.dumps(features, sort_keys=True, default=str)


# ----------------------------------------------------------------------------
# Audit trail (with rotation)
# ----------------------------------------------------------------------------
AUDIT_FIELDNAMES = [
    "timestamp", "repo_id", "repo_url", "stars", "forks", "open_issues",
    "readme_size", "repo_age_days", "last_commit_days", "has_readme",
    "topics_count", "probability", "prediction", "threshold",
]


def _rotate_audit_log_if_needed():
    """If the active log has grown past AUDIT_MAX_ROWS, archive it with a
    timestamp suffix and start a fresh file. Keeps the working CSV bounded
    instead of growing forever."""
    if not os.path.isfile(AUDIT_CSV):
        return
    try:
        with open(AUDIT_CSV, "r", encoding="utf-8") as f:
            row_count = sum(1 for _ in f) - 1  # minus header
    except Exception:
        return
    if row_count >= AUDIT_MAX_ROWS:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived_path = os.path.join(AUDIT_DIR, f"scoring_decisions_{stamp}.csv")
        try:
            os.rename(AUDIT_CSV, archived_path)
        except OSError:
            pass


def log_audit_trail(features, probability, prediction, threshold):
    """Log scoring decision to CSV file for audit trail."""
    os.makedirs(AUDIT_DIR, exist_ok=True)
    _rotate_audit_log_if_needed()

    timestamp = datetime.now().isoformat()
    row = {
        "timestamp": timestamp,
        "repo_id": features.get("full_name", "unknown"),
        "repo_url": features.get("html_url", ""),
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
    }

    file_exists = os.path.isfile(AUDIT_CSV)
    with open(AUDIT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=AUDIT_FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ----------------------------------------------------------------------------
# App
# ----------------------------------------------------------------------------
rf_model, tfidf_readme, tfidf_topics, scaler = load_ml_assets()

st.markdown("""
 <div style="text-align: center; margin-bottom: 2rem;">
     <h1 style="font-size: 2.5rem; font-weight: 700; background: linear-gradient(135deg, var(--fg-primary), var(--fg-secondary)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem;">⭐ RepoScore</h1>
     <p style="color: var(--fg-secondary); font-size: 1.1rem; margin-bottom: 0.5rem;">GitHub Repository Quality Predictor</p>
     <p style="color: var(--muted); font-size: 0.9rem;">Analyze any public repository to predict its overall quality score using machine learning and heuristic analysis.</p>
 </div>
""", unsafe_allow_html=True)

repo_input = st.text_input("Enter Repository (owner/name):", placeholder="e.g., scikit-learn/scikit-learn", help="Format: owner/repository name (e.g., scikit-learn/scikit-learn)")
threshold = st.slider(
    "Quality threshold", min_value=0.1, max_value=0.9, value=0.3, step=0.05,
    help="Lower = higher recall (catches more 'quality' repos, but flags more false positives). "
         "Higher = higher precision (fewer false positives, but may miss some quality repos). "
         "F1 optimal: ~0.3 (precision 0.67/recall 0.89) vs default 0.5 (precision 0.89/recall 0.47)."
)

if st.button("Predict Quality", type="primary") and repo_input:
    repo_input = repo_input.strip()
    if not REPO_INPUT_PATTERN.match(repo_input):
        st.error("Please enter the repository as `owner/name` (e.g. `scikit-learn/scikit-learn`).")
        st.stop()

    with st.spinner("Fetching repo data from GitHub API and analyzing quality..."):
        try:
            features = fetch_repo_features(repo_input, headers=headers)
        except RepoFetchError as e:
            st.error(str(e))
            st.stop()
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            st.stop()

    topics = features["topics"]
    repo_age_days = features["repo_age_days"]

    X_dense = featurize(features, tfidf_readme, tfidf_topics, scaler)

    # Use the user-chosen threshold, not the model's built-in 0.5 cutoff
    probability = rf_model.predict_proba(X_dense)[0][1]
    prediction = 1 if probability >= threshold else 0

    log_audit_trail(features, probability, prediction, threshold)

    exceptions = check_exceptions(features)
    low_confidence = 0.4 <= probability <= 0.6
    warning_messages = exceptions.copy()
    if low_confidence:
        warning_messages.append("Low confidence prediction (probability near 0.5).")

    total_issues = len(exceptions) + (1 if low_confidence else 0)

    # Scores computed exactly once, reused everywhere below.
    scores = compute_scores(features, probability)
    heuristic_result = scores["heuristic_result"]
    ml_score_pct = scores["ml_score_pct"]
    heuristic_score = scores["heuristic_score"]
    combined_score = scores["combined_score"]
    divergence = scores["divergence"]

    st.subheader(f"Results for [{features['full_name']}]({features['html_url']})")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("⭐ Stars", features["stars"])
    col2.metric("🍴 Forks", features["forks"])
    col3.metric("🐛 Open Issues", features["open_issues"])
    col4.metric("📅 Age (Days)", repo_age_days)

    if topics:
        st.write("**Topics:** " + ", ".join([f"`{t}`" for t in topics]))

    st.divider()

# --- Headline combined score ---
    if combined_score is not None:
        if combined_score >= 70:
            score_color, score_emoji = "var(--accent)", "🟢"
        elif combined_score >= 40:
            score_color, score_emoji = var(--warning), "🟡"
        else:
            score_color, score_emoji = var(--danger), "🔴"

        score_card_html = f'''<div style="background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;">
     <h2 class="section-header" style="margin-top: 0; margin-bottom: 1rem;">📊 Combined Score: {combined_score:.1f}/100 {score_emoji}</h2>
     <div style="background: var(--bg-tertiary); border-radius: 8px; height: 12px; overflow: hidden; margin-bottom: 1rem;">
         <div style="background: {score_color}; width: {combined_score}%; height: 100%; border-radius: 8px; transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1);"></div>
     </div>
     <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
         <span style="color: var(--fg-secondary); font-size: 0.85rem;">ML Model</span>
         <span style="color: var(--fg-primary); font-weight: 600; font-size: 1.1rem;">{ml_score_pct:.1f}%</span>
     </div>
     <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
         <span style="color: var(--fg-secondary); font-size: 0.85rem;">Heuristic</span>
         <span style="color: var(--fg-primary); font-weight: 600; font-size: 1.1rem;">{heuristic_score:.1f}/100</span>
     </div>
     <div style="display: flex; gap: 1rem;">
         <div style="flex: 1; text-align: center; padding: 0.5rem; background: var(--bg-tertiary); border-radius: 6px;">
             <div style="font-size: 0.75em; color: var(--fg-secondary);">Divergence</div>
             <div style="font-size: 1.2em; font-weight: 700; color: {'var(--accent)' if divergence is not None and abs(divergence) <= 15 else var(--danger)};">{divergence:+.1f if divergence is not None else '—'}</div>
         </div>
         <div style="flex: 1; text-align: center; padding: 0.5rem; background: var(--bg-tertiary); border-radius: 6px;">
             <div style="font-size: 0.75em; color: var(--fg-secondary);">Threshold</div>
             <div style="font-size: 1.2em; font-weight: 600; color: var(--fg-primary);">{threshold:.2f}</div>
         </div>
     </div>
 </div>'''
        render_card(score_card_html)
        if divergence is not None and divergence > 15:
            render_caution(
                f"ML and Heuristic scores diverge by {divergence:.1f} points — treat this combined "
                f"score with caution; review both scores individually in the **Why This Score?** tab."
            )

    confidence_html = (
        f'<div style="font-size: 0.95em; color: var(--fg-secondary);">'
        f'Confidence report: <strong>{probability:.1%}</strong> match rate | '
        f'{total_issues} exception{"s" if total_issues != 1 else ""} flagged</div>'
    )
    render_card(confidence_html)
    for msg in warning_messages:
        render_caution(msg)

    st.divider()
    render_verdict_banner(prediction, probability)

    tab_overview, tab_why, tab_ai = st.tabs(["📊 Overview", "🔍 Why This Score?", "🤖 AI Review"])

    # ==================== TAB 1: OVERVIEW ====================
    with tab_overview:
        topics_html = ""
        if topics:
            topics_html = (
                f'<div style="margin-top: 0.5rem; color: #cbd5e0;">'
                f'<strong>Topics:</strong> {", ".join([f"`{t}`" for t in topics])}</div>'
            )

        component_bars_html = ""
        if heuristic_result is not None:
            components = heuristic_result["components"]
            for label, value in [
                ("Maintenance", components["maintenance"]),
                ("Community", components["community"]),
                ("Documentation", components["documentation"]),
                ("Contributors", components["contributors"]),
            ]:
                render_component_bar(label, value)

        data_quality_html = ""
        if exceptions or low_confidence:
            notes = [f'<div class="rs-note">ℹ️ {exc}</div>' for exc in exceptions]
            if low_confidence:
                notes.append('<div class="rs-note">ℹ️ Low confidence prediction (probability near 0.5).</div>')
            data_quality_html = f'<div style="margin-top: 1rem;"><strong>Data Quality Notes</strong>{"".join(notes)}</div>'

        overview_card_html = f'''<h3 class="section-header">📊 Overview</h3>
 <div style="display: flex; gap: 1rem; margin-bottom: 1rem; flex-wrap: wrap;">
     <div style="flex: 1; min-width: 120px; text-align: center; padding: 0.75rem; background: #2d3548; border-radius: 6px;">
         <div style="font-size: 0.85em; color: #94a3b8;">⭐ Stars</div>
         <div style="font-size: 1.5em; font-weight: 600; color: #e2e8f0;">{features["stars"]}</div>
     </div>
     <div style="flex: 1; min-width: 120px; text-align: center; padding: 0.75rem; background: #2d3548; border-radius: 6px;">
         <div style="font-size: 0.85em; color: #94a3b8;">🍴 Forks</div>
         <div style="font-size: 1.5em; font-weight: 600; color: #e2e8f0;">{features["forks"]}</div>
     </div>
     <div style="flex: 1; min-width: 120px; text-align: center; padding: 0.75rem; background: #2d3548; border-radius: 6px;">
         <div style="font-size: 0.85em; color: #94a3b8;">🐛 Open Issues</div>
         <div style="font-size: 1.5em; font-weight: 600; color: #e2e8f0;">{features["open_issues"]}</div>
     </div>
     <div style="flex: 1; min-width: 120px; text-align: center; padding: 0.75rem; background: #2d3548; border-radius: 6px;">
         <div style="font-size: 0.85em; color: #94a3b8;">📅 Age (Days)</div>
         <div style="font-size: 1.5em; font-weight: 600; color: #e2e8f0;">{repo_age_days}</div>
     </div>
 </div>
{topics_html}
 <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid #2d3548;">
     <div style="font-size: 0.95em; color: #cbd5e0; margin-bottom: 0.5rem;"><strong>Threshold used:</strong> {threshold:.2f}</div>
     <div style="font-size: 0.95em; color: #cbd5e0; margin-bottom: 0.5rem;"><strong>Model probability:</strong> {probability:.1%}</div>
     <div style="font-size: 0.95em; color: #cbd5e0; margin-bottom: 1rem;"><strong>Prediction:</strong> {"High Quality" if prediction == 1 else "Low Quality / Unmaintained"}</div>
 </div>
 <div style="margin-top: 1rem;">
     <strong style="color: #e2e8f0;">Component Scores</strong>
     {component_bars_html}
 </div>
{data_quality_html}'''
        render_card(overview_card_html)

    # ==================== TAB 2: WHY THIS SCORE? ====================
    with tab_why:
        st.markdown('<h3 class="section-header">Feature Contribution Analysis (SHAP)</h3>', unsafe_allow_html=True)
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
                list(tfidf_readme.get_feature_names_out())
                + list(tfidf_topics.get_feature_names_out())
                + STRUCTURED_COLS
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
                    st.markdown(f":green[✅ {row['feature']} +{row['shap_value']:.3f}]")
            with feat_col2:
                st.markdown("**Pushed toward Low Quality**")
                for _, row in top_neg.iterrows():
                    st.markdown(f":red[❌ {row['feature']} {row['shap_value']:.3f}]")
        except Exception as err:
            st.caption(f"Explanation unavailable: {err}")

        # --- Heuristic score breakdown ---
        with st.expander("Heuristic Score (RepoScorer)", expanded=True):
            if heuristic_result is None:
                st.caption("Heuristic score unavailable.")
            else:
                delta = ml_score_pct - heuristic_score
                st.markdown('<div class="rs-card">', unsafe_allow_html=True)
                st.subheader("Heuristic Score Details")
                st.write(f"**Tier:** {heuristic_result['tier_emoji']} {heuristic_result['tier']}")
                st.write(f"**Heuristic Score:** {heuristic_score:.1f}/100")

                st.markdown("**Component Scores**")
                render_component_bar("Maintenance", heuristic_result["components"]["maintenance"])
                render_component_bar("Community", heuristic_result["components"]["community"])
                render_component_bar("Documentation", heuristic_result["components"]["documentation"])
                render_component_bar("Contributors", heuristic_result["components"]["contributors"])

                st.write(f"**Delta vs ML Model:** {delta:+.1f} points")
                if abs(delta) > 15:
                    render_caution(
                        f"ML and Heuristic scores diverge by {abs(delta):.1f} points — treat this "
                        f"combined score with caution; review both scores individually above."
                    )
                else:
                    st.caption("✅ Scores are well-aligned between ML model and heuristic scorer.")

                if heuristic_result.get("explanations"):
                    st.write("**Explanations:**")
                    for exp in heuristic_result["explanations"][:3]:
                        st.write(f"• {exp}")
                st.markdown('</div>', unsafe_allow_html=True)

    # ==================== TAB 3: AI REVIEW ====================
    with tab_ai:
        st.markdown('<h3 class="section-header">AI Review</h3>', unsafe_allow_html=True)
        if not AI_REVIEW_AVAILABLE:
            st.caption("AI review unavailable: ai_review module not found.")
        else:
            try:
                @st.cache_data(ttl=86400, show_spinner=False)
                def _cached_ai_review(readme_text_clean: str, features_key: str, prediction: int, probability: float):
                    features_dict = json.loads(features_key)
                    return generate_ai_review(
                        readme_content=readme_text_clean,
                        features=features_dict,
                        prediction=prediction,
                        probability=probability,
                    )

                readme_text = features.get("readme_text_clean", "")
                ai_result = _cached_ai_review(
                    readme_text, features_cache_key(features), prediction, probability
                )

                if ai_result.get("status") == "success":
                    with st.container(border=True):
                        st.markdown(format_ai_review_for_display(ai_result))
                else:
                    st.caption(format_ai_review_for_display(ai_result))
            except Exception as err:
                st.error(f"AI review unavailable: {str(err)}")