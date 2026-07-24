import streamlit as st
import requests
import os
import joblib
import numpy as np
import base64
import pandas as pd
from scipy.sparse import hstack
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")
token = os.getenv("GITHUB_TOKEN")
headers = {
    "Authorization": f"token {token}",
    "Accept": "application/vnd.github+json"
}

# Load trained model, vectorizers, and scaler
rf_model = joblib.load(r"C:\reposcore_data\rf_model.pkl")
tfidf_readme = joblib.load(r"C:\reposcore_data\tfidf_readme.pkl")
tfidf_topics = joblib.load(r"C:\reposcore_data\tfidf_topics.pkl")
scaler = joblib.load(r"C:\reposcore_data\scaler.pkl")

st.set_page_config(page_title="RepoScore", page_icon="⭐")
st.title("⭐ RepoScore: GitHub Repository Quality Predictor")
st.write("Enter a GitHub repo (e.g. `pandas-dev/pandas`) to predict its quality score.")

repo_input = st.text_input("Repo (owner/name)", placeholder="scikit-learn/scikit-learn")

if st.button("Predict") and repo_input:
    with st.spinner("Fetching repo data..."):
        repo_resp = requests.get(f"https://api.github.com/repos/{repo_input}", headers=headers)

        if repo_resp.status_code != 200:
            st.error("Repo not found or API error.")
        else:
            repo = repo_resp.json()

            # Fetch README text
            readme_resp = requests.get(f"https://api.github.com/repos/{repo_input}/readme", headers=headers)
            readme_text = ""
            readme_size = 0
            has_readme = readme_resp.status_code == 200
            if has_readme:
                readme_size = readme_resp.json().get("size", 0)
                content_b64 = readme_resp.json().get("content", "")
                try:
                    readme_text = base64.b64decode(content_b64).decode("utf-8", errors="ignore")
                except Exception:
                    readme_text = ""

            # Get topics (already included in the repo response)
            topics = repo.get("topics", [])
            topics_text = " ".join(topics)

            # Compute derived features
            created_at = pd.to_datetime(repo["created_at"])
            pushed_at = pd.to_datetime(repo["pushed_at"])
            now = pd.Timestamp.now(tz="UTC")
            repo_age_days = (now - created_at).days
            days_since_last_commit = (now - pushed_at).days

            # Build feature vector matching training format
            structured = np.array([[
                repo["stargazers_count"],
                repo["forks_count"],
                repo["open_issues_count"],
                readme_size,
                repo_age_days,
                days_since_last_commit,
                int(has_readme)
            ]])

            X_readme = tfidf_readme.transform([readme_text])
            X_topics = tfidf_topics.transform([topics_text])
            X = hstack([X_readme, X_topics, structured])
            X_scaled = scaler.transform(X)

            prediction = rf_model.predict(X_scaled)[0]
            probability = rf_model.predict_proba(X_scaled)[0][1]

            st.success(f"Found: {repo['full_name']}")
            st.write(f"⭐ Stars: {repo['stargazers_count']}")
            st.write(f"🍴 Forks: {repo['forks_count']}")
            st.write(f"🐛 Open issues: {repo['open_issues_count']}")
            st.write(f"🕒 Last updated: {repo['pushed_at']}")
            if topics:
                st.write(f"🏷️ Topics: {', '.join(topics)}")

            st.divider()
            if prediction == 1:
                st.markdown("### 🟢 Predicted: **Quality Repo**")
            else:
                st.markdown("### 🔴 Predicted: **Not Quality**")
            st.write(f"Confidence: {probability:.1%}")