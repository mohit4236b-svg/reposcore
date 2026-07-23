import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from scipy.sparse import hstack

df = pd.read_csv(r"C:\reposcore_data\repos_combined_final.csv")
# Fill missing README text with empty string
df["readme_text"] = df["readme_text"].fillna("")

# Structured features — EXCLUDING anything used to build the label
# (stars_per_month, has_ci, has_tests were used to define "quality", so they're dropped here)
structured_features = df[[
    "stars", "forks", "open_issues", "readme_size",
    "repo_age_days", "days_since_last_commit"
]].fillna(0)

# Boolean features — only has_readme stays (not used in the label)
df["has_readme"] = df["has_readme"].astype(int)
bool_features = df[["has_readme"]]

# Combine structured + boolean
X_structured = np.hstack([structured_features.values, bool_features.values])

# Text features from README (TF-IDF)
tfidf = TfidfVectorizer(max_features=500, stop_words="english")
X_text = tfidf.fit_transform(df["readme_text"])

# Combine structured + text into one feature matrix
X = hstack([X_text, X_structured])
y = df["quality"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (helps Logistic Regression converge)
scaler = StandardScaler(with_mean=False)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression baseline
print("=== Logistic Regression ===")
log_model = LogisticRegression(class_weight="balanced", max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
print(classification_report(y_test, y_pred_log))

# Random Forest
print("=== Random Forest ===")
rf_model = RandomForestClassifier(class_weight="balanced", n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
print(classification_report(y_test, y_pred_rf))
# Feature importance analysis
feature_names = (
    list(tfidf.get_feature_names_out()) +
    ["stars", "forks", "open_issues", "readme_size", "repo_age_days", "days_since_last_commit", "has_readme"]
)

importances = rf_model.feature_importances_
importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importances
}).sort_values("importance", ascending=False)

print("=== Top 20 Most Important Features ===")
print(importance_df.head(20).to_string(index=False))

importance_df.to_csv(r"C:\reposcore_data\feature_importances.csv", index=False)
print("Saved full feature importance list to C:\\reposcore_data\\feature_importances.csv")