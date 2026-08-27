# -*- coding: utf-8 -*-
import os
import csv
import pickle
import joblib
import textwrap
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from reposcore_utils import (
    fetch_repo_features,
    featurize,
    RepoFetchError,
    STRUCTURED_COLS,
)

# Optional Imports
try:
    from ai_review import generate_ai_review, format_ai_review_for_display
    AI_REVIEW_AVAILABLE = True
except ImportError:
    AI_REVIEW_AVAILABLE = False

try:
    from scoring_engine import RepoScorer
    SCORER_AVAILABLE = True
except ImportError:
    SCORER_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Load environment variables & secrets
load_dotenv()
if "NVIDIA_API_KEY" in st.secrets:
    os.environ["NVIDIA_API_KEY"] = st.secrets["NVIDIA_API_KEY"]
if "GITHUB_TOKEN" in st.secrets:
    os.environ["GITHUB_TOKEN"] = st.secrets["GITHUB_TOKEN"]

# Page configuration
st.set_page_config(
    page_title="RepoScore - Repository Quality Intelligence",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling & Typography Hierarchy
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .app-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        font-size: 1rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .section-header {
        color: #38bdf8 !important;
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.75rem !important;
    }
    .rs-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .rs-stat-box {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .rs-stat-title {
        font-size: 0.85rem;
        font-weight: 500;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .rs-stat-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f1f5f9;
    }
    .rs-verdict-banner {
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .rs-verdict-high {
        background-color: rgba(22, 101, 52, 0.25);
        border: 1px solid #22c55e;
        color: #4ade80;
    }
    .rs-verdict-low {
        background-color: rgba(153, 27, 27, 0.25);
        border: 1px solid #ef4444;
        color: #f87171;
    }
    .rs-note {
        background-color: #1e293b;
        border-left: 4px solid #64748b;
        border-radius: 4px;
        padding: 12px 16px;
        margin-bottom: 10px;
        color: #cbd5e1;
        font-size: 0.92rem;
    }
    .rs-caution {
        background-color: rgba(180, 83, 9, 0.15);
        border-left: 4px solid #f59e0b;
        border-radius: 4px;
        padding: 12px 16px;
        margin-bottom: 14px;
        color: #fbbf24;
        font-size: 0.92rem;
    }
    .rs-component-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.9rem;
        font-weight: 500;
        color: #cbd5e1;
        margin-bottom: 4px;
    }
    .rs-progress-bg {
        background-color: #334155;
        border-radius: 9999px;
        height: 10px;
        margin-bottom: 14px;
        overflow: hidden;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        margin-bottom: 16px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px;
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


def render_component_bar(label: str, value: float, max_value: float = 100.0) -> str:
    """Returns HTML for a smooth progress bar."""
    pct = max(0.0, min(1.0, value / max_value))
    color = "#22c55e" if pct >= 0.7 else "#f59e0b" if pct >= 0.4 else "#ef4444"
    return f"""
    <div class="rs-component-label">
        <span>{label}</span>
        <span>{value:.1f}/{max_value:.0f}</span>
    </div>
    <div class="rs-progress-bg">
        <div style="background-color:{color};width:{pct*100}%;height:100%;border-radius:9999px;"></div>
    </div>
    """


def render_verdict_banner(prediction: int, probability: float):
    """Renders the primary quality prediction alert banner."""
    css_class = "rs-verdict-high" if prediction == 1 else "rs-verdict-low"
    icon = "✅" if prediction == 1 else "⚠️"
    label = "High Quality Repository" if prediction == 1 else "Low Quality / Unmaintained Repository"
    st.markdown(f"""
    <div class="rs-verdict-banner {css_class}">
        <span style="font-size:1.15rem;font-weight:600;">{icon} Predicted: {label}</span>
        <span style="font-size:1rem;font-weight:500;">Model Confidence: {probability:.1%}</span>
    </div>
    """, unsafe_allow_html=True)


def safe_load(file_path: str):
    """Safely unpickles models across joblib/pickle formats."""
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
            st.error(f"❌ Missing file: `{filename}` was not found in `models/`.")
            st.stop()
        try:
            loaded[key] = safe_load(file_path)
        except Exception as err:
            st.error(f"❌ Failed loading `{filename}`:")
            st.exception(err)
            st.stop()
    return loaded["rf_model"], loaded["tfidf_readme"], loaded["tfidf_topics"], loaded["scaler"]


def check_exceptions(features: dict) -> list:
    exceptions = []
    if features.get("has_readme", 1) == 0:
        exceptions.append("No README detected.")
    elif features.get("readme_size", 0) < 50:
        exceptions.append("Very small README (< 50 characters).")
    if not features.get("topics"):
        exceptions.append("No repository topics assigned.")
    last_commit_days = features.get("last_commit_days")
    if last_commit_days is not None and last_commit_days > 730:
        exceptions.append("Repository has been inactive for over 2 years.")
    return exceptions


def log_audit_trail(features: dict, probability: float, prediction: int, threshold: float):
    audit_dir = "audit_trail"
    os.makedirs(audit_dir, exist_ok=True)
    csv_file = os.path.join(audit_dir, "scoring_decisions.csv")

    fieldnames = [
        "timestamp", "repo_id", "repo_url", "stars", "forks", "open_issues",
        "readme_size", "repo_age_days", "last_commit_days", "has_readme",
        "topics_count", "probability", "prediction", "threshold"
    ]
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(),
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
        })


# Load ML models
rf_model, tfidf_readme, tfidf_topics, scaler = load_ml_assets()

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Evaluation Settings")
    threshold = st.slider(
        "Classification Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.05,
        help="Adjust decision boundary. Lower values increase recall; higher values prioritize precision."
    )
    st.markdown("---")
    st.markdown("**Powered by:**")
    st.caption("• Random Forest Classification\n• Heuristic Engine\n• NVIDIA NIM Reviewer")

# Main Header
st.markdown('<div class="app-title">⭐ RepoScore Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Multi-dimensional quality analysis and credibility scoring for GitHub repositories</div>', unsafe_allow_html=True)

repo_input = st.text_input("Enter GitHub Repository (owner/repo):", placeholder="e.g. psf/requests or scikit-learn/scikit-learn")

if st.button("Run Comprehensive Score", type="primary") and repo_input:
    clean_repo_name = repo_input.strip().replace("https://github.com/", "").strip("/")
    
    # Configure API headers
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    with st.spinner("Extracting repository metadata & computing score..."):
        try:
            features = fetch_repo_features(clean_repo_name, headers=headers)
        except RepoFetchError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Unexpected error fetching repository data: {e}")
        else:
            # ML Featurization & Inference
            X_dense = featurize(features, tfidf_readme, tfidf_topics, scaler)
            probability = float(rf_model.predict_proba(X_dense)[0][1])
            prediction = 1 if probability >= threshold else 0
            ml_score_pct = probability * 100

            # Log audit record
            log_audit_trail(features, probability, prediction, threshold)

            # Heuristic Calculation
            heuristic_data = None
            if SCORER_AVAILABLE:
                try:
                    scorer = RepoScorer()
                    heuristic_data = scorer.score_repository(features)
                except Exception as e:
                    st.warning(f"Heuristic scorer encountered an issue: {e}")

            # Top Verdict
            render_verdict_banner(prediction, probability)

            # High-level Stat Row
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(f'<div class="rs-stat-box"><div class="rs-stat-title">Stars</div><div class="rs-stat-val">{features.get("stars", 0):,}</div></div>', unsafe_allow_html=True)
            with c2:
                st.markdown(f'<div class="rs-stat-box"><div class="rs-stat-title">Forks</div><div class="rs-stat-val">{features.get("forks", 0):,}</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="rs-stat-box"><div class="rs-stat-title">Open Issues</div><div class="rs-stat-val">{features.get("open_issues", 0):,}</div></div>', unsafe_allow_html=True)
            with c4:
                tier_label = heuristic_data.get("tier", "N/A") if heuristic_data else "N/A"
                st.markdown(f'<div class="rs-stat-box"><div class="rs-stat-title">Tier</div><div class="rs-stat-val">{tier_label}</div></div>', unsafe_allow_html=True)

            st.write("")

            # Multi-Tab Layout
            tab_overview, tab_heuristics, tab_ai = st.tabs(["📊 Overview", "🔍 Why This Score?", "🤖 AI Review"])

            with tab_overview:
                st.markdown('<div class="section-header">Repository Overview</div>', unsafe_allow_html=True)
                st.write(f"**Full Name:** `{features.get('full_name', clean_repo_name)}`")
                st.write(f"**Description:** {features.get('description') or 'No description provided.'}")
                st.write(f"**Primary Language:** {features.get('language') or 'Not specified'}")
                st.write(f"**Repository Age:** {features.get('repo_age_days', 0)} days")
                st.write(f"**Days Since Last Commit:** {features.get('last_commit_days', 0)} days")

                # Exception warnings
                exceptions = check_exceptions(features)
                for exc in exceptions:
                    st.warning(f"⚠️ {exc}")

            with tab_heuristics:
                st.markdown('<div class="section-header">HeuristicThe syntax errors in your script are primarily caused by **invisible non-breaking space characters (`\xa0`)** inserted throughout the code, along with a cut-off block at the end where an incomplete `from` statement appears.

Here is the clean, working version of your complete `app.py`:

```python
# -*- coding: utf-8 -*-
import csv
from datetime import datetime
import os
import pickle
import textwrap

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from reposcore_utils import (
    RepoFetchError,
    STRUCTURED_COLS,
    featurize,
    fetch_repo_features,
    strip_badges,
)

# Optional feature imports with safe fallbacks
try:
    from ai_review import format_ai_review_for_display, generate_ai_review
    AI_REVIEW_AVAILABLE = True
except ImportError:
    AI_REVIEW_AVAILABLE = False

try:
    from scoring_engine import RepoScorer
    SCORER_AVAILABLE = True
except ImportError:
    SCORER_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Load environment variables
load_dotenv()

# Wire Streamlit secrets into os.environ
if "NVIDIA_API_KEY" in st.secrets:
    os.environ["NVIDIA_API_KEY"] = st.secrets["NVIDIA_API_KEY"]
if "GITHUB_TOKEN" in st.secrets:
    os.environ["GITHUB_TOKEN"] = st.secrets["GITHUB_TOKEN"]

# Page configuration
st.set_page_config(
    page_title="RepoScore - Repository Quality Intelligence",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    @import url('[https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap](https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap)');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .app-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #f8fafc;
        margin-bottom: 0.2rem;
    }
    .app-subtitle {
        font-size: 1rem;
        color: #94a3b8;
        margin-bottom: 1.5rem;
    }
    .section-header {
        color: #38bdf8 !important;
        font-size: 1.25rem !important;
        font-weight: 600 !important;
        margin-bottom: 0.75rem !important;
    }
    .rs-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 18px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2), 0 2px 4px -2px rgba(0, 0, 0, 0.2);
    }
    .rs-stat-box {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .rs-stat-title {
        font-size: 0.85rem;
        font-weight: 500;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .rs-stat-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f1f5f9;
    }
    .rs-verdict-banner {
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .rs-verdict-high {
        background-color: rgba(22, 101, 52, 0.25);
        border: 1px solid #22c55e;
        color: #4ade80;
    }
    .rs-verdict-low {
        background-color: rgba(153, 27, 27, 0.25);
        border: 1px solid #ef4444;
        color: #f87171;
    }
    .rs-note {
        background-color: #1e293b;
        border-left: 4px solid #64748b;
        border-radius: 4px;
        padding: 12px 16px;
        margin-bottom: 10px;
        color: #cbd5e1;
        font-size: 0.92rem;
    }
    .rs-caution {
        background-color: rgba(180, 83, 9, 0.15);
        border-left: 4px solid #f59e0b;
        border-radius: 4px;
        padding: 12px 16px;
        margin-bottom: 14px;
        color: #fbbf24;
        font-size: 0.92rem;
    }
    .rs-component-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.9rem;
        font-weight: 500;
        color: #cbd5e1;
        margin-bottom: 4px;
    }
    .rs-progress-bg {
        background-color: #334155;
        border-radius: 9999px;
        height: 10px;
        margin-bottom: 14px;
        overflow: hidden;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        margin-bottom: 16px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px;
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.95rem;
    }
</style>
""", unsafe_allow_html=True)


def render_component_bar(label: str, value: float, max_value: float = 100.0) -> str:
    pct = max(0.0, min(1.0, value / max_value))
    color = "#22c55e" if pct >= 0.7 else "#f59e0b" if pct >= 0.4 else "#ef4444"
    return f"""
    <div class="rs-component-label">
        <span>{label}</span>
        <span>{value:.1f}/{max_value:.0f}</span>
    </div>
    <div class="rs-progress-bg">
        <div style="background-color:{color};width:{pct*100}%;height:100%;border-radius:9999px;"></div>
    </div>
    """


def render_verdict_banner(prediction: int, probability: float):
    css_class = "rs-verdict-high" if prediction == 1 else "rs-verdict-low"
    icon = "✅" if prediction == 1 else "⚠️"
    label = "High Quality Repository" if prediction == 1 else "Low Quality / Unmaintained Repository"
    st.markdown(f"""
    <div class="rs-verdict-banner {css_class}">
        <span style="font-size:1.15rem;font-weight:600;">{icon} Predicted: {label}</span>
        <span style="font-size:1rem;font-weight:500;">Model Confidence: {probability:.1%}</span>
    </div>
    """, unsafe_allow_html=True)


# Configure GitHub API Headers
token = os.getenv("GITHUB_TOKEN")
headers = {"Accept": "application/vnd.github+json"}
if token:
    headers["Authorization"] = f"Bearer {token}"


def safe_load(file_path: str):
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
            st.error(f"❌ Missing file: `{filename}` was not found in `models/`.")
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
    if not SHAP_AVAILABLE:
        return None
    return shap.TreeExplainer(_model)


def check_exceptions(features: dict) -> list:
    exceptions = []
    if features.get("has_readme", 1) == 0:
        exceptions.append("No README detected.")
    elif features.get("readme_size", 0) < 50:
        exceptions.append("Very small README (< 50 characters).")
    if not features.get("topics"):
        exceptions.append("No repository topics assigned.")
    last_commit_days = features.get("last_commit_days")
    if last_commit_days is not None and last_commit_days > 730:
        exceptions.append("Repository has been inactive for over 2 years.")
    return exceptions


def log_audit_trail(features: dict, probability: float, prediction: int, threshold: float):
    audit_dir = "audit_trail"
    os.makedirs(audit_dir, exist_ok=True)
    csv_file = os.path.join(audit_dir, "scoring_decisions.csv")

    fieldnames = [
        "timestamp", "repo_id", "repo_url", "stars", "forks", "open_issues",
        "readme_size", "repo_age_days", "last_commit_days", "has_readme",
        "topics_count", "probability", "prediction", "threshold"
    ]
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(),
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
        })


rf_model, tfidf_readme, tfidf_topics, scaler = load_ml_assets()

# Sidebar Settings
with st.sidebar:
    st.header("⚙️ Evaluation Settings")
    threshold = st.slider(
        "Classification Threshold",
        min_value=0.1,
        max_value=0.9,
        value=0.3,
        step=0.05,
        help="Adjust decision boundary. Lower values increase recall; higher values prioritize precision."
    )
    st.markdown("---")
    st.markdown("**Powered by:**")
    st.caption("• Random Forest Classification\n• SHAP Explainability Engine\n• NVIDIA NIM LLM Reviewer")

# Main Header
st.markdown('<div class="app-title">⭐ RepoScore Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="app-subtitle">Multi-dimensional quality analysis and credibility scoring for GitHub repositories</div>', unsafe_allow_html=True)

repo_input = st.text_input("Enter GitHub Repository (owner/repo):", placeholder="e.g. psf/requests or scikit-learn/scikit-learn")

if st.button("Run Comprehensive Score", type="primary") and repo_input:
    clean_repo_name = repo_input.strip().replace("[https://github.com/](https://github.com/)", "").strip("/")

    with st.spinner("Extracting repository metadata & analyzing..."):
        try:
            features = fetch_repo_features(clean_repo_name, headers=headers)
        except RepoFetchError as e:
            st.error(str(e))
        else:
            # ML Featurization & Inference
            X_dense = featurize(features, tfidf_readme, tfidf_topics, scaler)
            probability = float(rf_model.predict_proba(X_dense)[0][1])
            prediction = 1 if probability >= threshold else 0
            ml_score_pct = probability * 100

            # Log audit record
            log_audit_trail(features, probability, prediction, threshold)

            # Quality and Divergence Checks
            exceptions = check_exceptions(features)
            low_confidence = 0.4 <= probability <= 0.6
            warning_messages = exceptions.copy()
            if low_confidence:
                warning_messages.append("Low confidence inference (probability near 0.5 decision boundary).")

            # Primary Verdict
            render_verdict_banner(prediction, probability)

            if warning_messages:
                for warn in warning_messages:
                    st.markdown(f'<div class="rs-caution">⚠️ {warn}</div>', unsafe_allow_html=True)

            # Overview Stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f'<div class="rs-stat-box"><div class="rs-stat-title">Stars</div><div class="rs-stat-val">{features.get("stars", 0):,}</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="rs-stat-box"><div class="rs-stat-title">Forks</div><div class="rs-stat-val">{features.get("forks", 0):,}</div></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="rs-stat-box"><div class="rs-stat-title">Open Issues</div><div class="rs-stat-val">{features.get("open_issues", 0):,}</div></div>', unsafe_allow_html=True)
            with col4:
                st.markdown(f'<div class="rs-stat-box"><div class="rs-stat-title">Quality Score</div><div class="rs-stat-val">{ml_score_pct:.1f}%</div></div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Tabbed Detailed Breakdowns
            tab1, tab2 = st.tabs(["📊 Metric Breakdown", "🤖 AI Repository Review"])

            with tab1:
                st.markdown(render_component_bar("Overall Machine Learning Quality", ml_score_pct, 100.0), unsafe_allow_html=True)
                st.write(f"**Repository Age:** {features.get('repo_age_days', 0)} days")
                st.write(f"**Last Commit:** {features.get('last_commit_days', 0)} days ago")
                st.write(f"**Topics Defined:** {len(features.get('topics', []))}")

            with tab2:
                if AI_REVIEW_AVAILABLE:
                    with st.spinner("Generating AI-assisted code review..."):
                        review_data = generate_ai_review(features)
                        formatted_review = format_ai_review_for_display(review_data)
                        st.markdown(formatted_review)
                else:
                    st.info("AI review module (`ai_review.py`) is not configured or available.")