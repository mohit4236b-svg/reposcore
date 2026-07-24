import base64
import os
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from dotenv import load_dotenv
from scipy.sparse import hstack

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="RepoScore", 
    page_icon="⭐", 
    layout="wide"
)

# Set up GitHub API Headers safely
token = os.getenv("GITHUB_TOKEN")
headers = {"Accept": "application/vnd.github+json"}
if token:
    headers["Authorization"] = f"Bearer {token}"

# Load models safely using relative paths
@st.cache_resource
def load_ml_assets():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "models")  # Store .pkl files in a /models folder!
    
    rf_model = joblib.load(os.path.join(model_dir, "rf_model.pkl"))
    tfidf_readme = joblib.load(os.path.join(model_dir, "tfidf_readme.pkl"))
    tfidf_topics = joblib.load(os.path.join(model_dir, "tfidf_topics.pkl"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    
    return rf_model, tfidf_readme, tfidf_topics, scaler

try:
    rf_model, tfidf_readme, tfidf_topics, scaler = load_ml_assets()
except Exception as e:
    st.error("Error loading trained models. Please check if model files exist in the 'models/' folder.")
    st.stop()

# Application UI
st.title("⭐ RepoScore: GitHub Repository Quality Predictor")
st.caption("Predict the overall quality class and probability of any public GitHub repository.")

repo_input = st.text_input("GitHub Repo (owner/name):", placeholder="scikit-learn/scikit-learn")

if st.button("Predict Quality", type="primary") and repo_input:
    clean_repo = repo_input.strip().strip("/")
    
    with st.spinner("Fetching data from GitHub API..."):
        repo_resp = requests.get(f"https://api.github.com/repos/{clean_repo}", headers=headers)

        if repo_resp.status_code == 404:
            st.error("Repository not found. Double check the `owner/repository` name.")
        elif repo_resp.status_code == 403:
            st.error("GitHub API rate limit exceeded. Add a `GITHUB_TOKEN` to your `.env` file.")
        elif repo_resp.status_code != 200:
            st.error(f"GitHub API Error: {repo_resp.status_code}")
        else:
            repo = repo_resp.json()

            # Fetch README
            readme_resp = requests.get(f"https://api.github.com/repos/{clean_repo}/readme", headers=headers)
            readme_text = ""
            readme_size = 0
            has_readme = readme_resp.status_code == 200
            
            if has_readme:
                readme_data = readme_resp.json()
                readme_size = readme_data.get("size", 0)
                content_b64 = readme_data.get("content", "")
                try:
                    readme_text = base64.b64decode(content_b64).decode("utf-8", errors="ignore")
                except Exception:
                    readme_text = ""

            # Extract metadata & compute features
            topics = repo.get("topics", [])
            topics_text = " ".join(topics)

            created_at = pd.to_datetime(repo["created_at"])
            pushed_at = pd.to_datetime(repo["pushed_at"])
            now = pd.Timestamp.now(tz="UTC")
            repo_age_days = (now - created_at).days
            days_since_last_commit = (now - pushed_at).days

            # Construct feature set
            structured = np.array([[
                repo.get("stargazers_count", 0),
                repo.get("forks_count", 0),
                repo.get("open_issues_count", 0),
                readme_size,
                repo_age_days,
                days_since_last_commit,
                int(has_readme)
            ]])

            # Transform features
            X_readme = tfidf_readme.transform([readme_text])
            X_topics = tfidf_topics.transform([topics_text])
            X = hstack([X_readme, X_topics, structured])
            X_scaled = scaler.transform(X)

            # ML Predictions
            prediction = rf_model.predict(X_scaled)[0]
            probability = rf_model.predict_proba(X_scaled)[0][1]

            st.subheader(f"Results for [{repo['full_name']}]({repo['html_url']})")

            # Clean UI metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("⭐ Stars", repo["stargazers_count"])
            col2.metric("🍴 Forks", repo["forks_count"])
            col3.metric("🐛 Open Issues", repo["open_issues_count"])
            col4.metric("📅 Age (Days)", repo_age_days)

            if topics:
                st.write("**Topics:** " + ", ".join([f"`{t}`" for t in topics]))

            st.divider()

            # Display prediction banner
            res_col1, res_col2 = st.columns([2, 1])
            with res_col1:
                if prediction == 1:
                    st.success("### 🟢 High Quality Repository")
                else:
                    st.warning("### 🔴 Low Quality / Unmaintained Repository")
            
            with res_col2:
                st.metric("Quality Score Confidence", f"{probability:.1%}")