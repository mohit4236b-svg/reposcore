import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from scipy.sparse import hstack
import joblib

df = pd.read_csv(r"C:\reposcore_data\repos_final_with_topics.csv")

# Fill missing text fields
df["readme_text"] = df["readme_text"].fillna("")
df["topics"] = df["topics"].fillna("")

# Structured features — excluding anything used to build the label
structured_features = df[[
    "stars", "forks", "open_issues", "readme_size",
    "repo_age_days", "days_since_last_commit"
]].fillna(0)

df["has_readme"] = df["has_readme"].astype(int)
bool_features = df[["has_readme"]]

X_structured = np.hstack([structured_features.values, bool_features.values])

# Text features from README (TF-IDF)
tfidf_readme = TfidfVectorizer(max_features=500, stop_words="english")
X_readme = tfidf_readme.fit_transform(df["readme_text"])

# NEW: Text features from topics (separate, smaller TF-IDF — topics are short tag lists)
tfidf_topics = TfidfVectorizer(max_features=100)
X_topics = tfidf_topics.fit_transform(df["topics"])

# Combine everything
X = hstack([X_readme, X_topics, X_structured])
y = df["quality"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler(with_mean=False)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("=== Logistic Regression ===")
log_model = LogisticRegression(class_weight="balanced", max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
print(classification_report(y_test, y_pred_log))

print("=== Random Forest ===")
rf_model = RandomForestClassifier(class_weight="balanced", n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print(classification_report(y_test, y_pred_rf))

# Feature importance
feature_names = (
    list(tfidf_readme.get_feature_names_out()) +
    list(tfidf_topics.get_feature_names_out()) +
    ["stars", "forks", "open_issues", "readme_size", "repo_age_days", "days_since_last_commit", "has_readme"]
)

importances = rf_model.feature_importances_
importance_df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
print("=== Top 20 Most Important Features ===")
print(importance_df.head(20).to_string(index=False))
importance_df.to_csv(r"C:\reposcore_data\feature_importances_v2.csv", index=False)

# Save model, both vectorizers, and scaler
joblib.dump(rf_model, r"C:\reposcore_data\rf_model.pkl")
joblib.dump(tfidf_readme, r"C:\reposcore_data\tfidf_readme.pkl")
joblib.dump(tfidf_topics, r"C:\reposcore_data\tfidf_topics.pkl")
joblib.dump(scaler, r"C:\reposcore_data\scaler.pkl")
print("Model, vectorizers, and scaler saved.")