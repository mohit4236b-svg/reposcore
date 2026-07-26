"""
Train and evaluate the RepoScore models.

Changes from the original version:
- Relative paths (was hardcoded to C:\\reposcore_data, broke on non-Windows
  machines and for anyone who cloned the repo to a different path)
- README text is cleaned with strip_badges() before TF-IDF, to remove
  indirect leakage of the has_ci signal via badge markup (see reposcore_utils.py)
- 5-fold stratified cross-validation reported (mean +/- std), not just a
  single train/test split, which was overstating performance
- Confusion matrix + ROC-AUC added alongside precision/recall/F1
"""

import os
import sys
import json

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    make_scorer, f1_score, precision_score, recall_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
from scipy.sparse import hstack
import joblib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reposcore_utils import strip_badges

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "models")
DATA_PATH = os.path.join(DATA_DIR, "repos_final_with_topics.csv")

df = pd.read_csv(DATA_PATH)

# Fill missing text fields, clean badge markup out of the README text
df["readme_text"] = df["readme_text"].fillna("").apply(strip_badges)
df["topics"] = df["topics"].fillna("")

# Structured features -- excluding anything used to build the label
structured_cols = ["stars", "forks", "open_issues", "readme_size",
                    "repo_age_days", "days_since_last_commit"]
structured_features = df[structured_cols].fillna(0)

df["has_readme"] = df["has_readme"].astype(int)
bool_features = df[["has_readme"]]

X_structured = np.hstack([structured_features.values, bool_features.values])

tfidf_readme = TfidfVectorizer(max_features=500, stop_words="english")
X_readme = tfidf_readme.fit_transform(df["readme_text"])

tfidf_topics = TfidfVectorizer(max_features=100)
X_topics = tfidf_topics.fit_transform(df["topics"])

X = hstack([X_readme, X_topics, X_structured]).tocsr()
y = df["quality"].values

# --- Cross-validated evaluation (honest estimate of generalization) ---
scoring = {
    "f1": make_scorer(f1_score),
    "precision": make_scorer(precision_score),
    "recall": make_scorer(recall_score),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("=== 5-fold Cross-Validation (Random Forest) ===")
X_scaled_for_cv = StandardScaler(with_mean=False).fit_transform(X)
rf_cv_model = RandomForestClassifier(class_weight="balanced", n_estimators=200, random_state=42)
cv_scores = cross_validate(rf_cv_model, X_scaled_for_cv, y, cv=cv, scoring=scoring)
cv_summary = {}
for metric in ["f1", "precision", "recall"]:
    vals = cv_scores[f"test_{metric}"]
    cv_summary[metric] = {"mean": float(vals.mean()), "std": float(vals.std())}
    print(f"{metric:>10}: {vals.mean():.3f} +/- {vals.std():.3f}")

# --- Single held-out split, kept for the saved model + confusion matrix ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler(with_mean=False)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\n=== Logistic Regression (held-out test set) ===")
log_model = LogisticRegression(class_weight="balanced", max_iter=1000)
log_model.fit(X_train, y_train)
y_pred_log = log_model.predict(X_test)
print(classification_report(y_test, y_pred_log))

print("=== Random Forest (held-out test set) ===")
rf_model = RandomForestClassifier(class_weight="balanced", n_estimators=200, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred_rf))

cm = confusion_matrix(y_test, y_pred_rf)
auc = roc_auc_score(y_test, y_proba_rf)
print("Confusion matrix ([[TN, FP], [FN, TP]]):")
print(cm)
print(f"ROC-AUC: {auc:.3f}")

# --- Calibration check: does a 70% confidence score actually mean ~70% correct? ---
brier = brier_score_loss(y_test, y_proba_rf)
frac_pos, mean_pred = calibration_curve(y_test, y_proba_rf, n_bins=5, strategy="quantile")
calibration_bins = [
    {"mean_predicted_confidence": float(p), "observed_fraction_positive": float(o)}
    for p, o in zip(mean_pred, frac_pos)
]
print(f"\nBrier score (lower is better, 0=perfect, 0.25=random-guessing baseline): {brier:.3f}")
print("Calibration bins (mean predicted confidence vs. observed fraction positive):")
for b in calibration_bins:
    print(f"  predicted={b['mean_predicted_confidence']:.2f}  observed={b['observed_fraction_positive']:.2f}")

# Feature importance
feature_names = (
    list(tfidf_readme.get_feature_names_out()) +
    list(tfidf_topics.get_feature_names_out()) +
    structured_cols + ["has_readme"]
)
importances = rf_model.feature_importances_
importance_df = pd.DataFrame({"feature": feature_names, "importance": importances}) \
    .sort_values("importance", ascending=False)
print("=== Top 20 Most Important Features ===")
print(importance_df.head(20).to_string(index=False))
importance_df.to_csv(os.path.join(DATA_DIR, "feature_importances.csv"), index=False)

# Save a metrics report alongside the model so results are reproducible/auditable
metrics_report = {
    "cv_5fold": cv_summary,
    "holdout_confusion_matrix": cm.tolist(),
    "holdout_roc_auc": float(auc),
    "holdout_brier_score": float(brier),
    "holdout_calibration_bins": calibration_bins,
    "n_rows": len(df),
    "positive_rate": float(y.mean()),
}
with open(os.path.join(DATA_DIR, "metrics_report.json"), "w") as f:
    json.dump(metrics_report, f, indent=2)
print(f"\nSaved metrics report to {os.path.join(DATA_DIR, 'metrics_report.json')}")

# Save model, both vectorizers, and scaler
joblib.dump(rf_model, os.path.join(DATA_DIR, "rf_model.pkl"))
joblib.dump(tfidf_readme, os.path.join(DATA_DIR, "tfidf_readme.pkl"))
joblib.dump(tfidf_topics, os.path.join(DATA_DIR, "tfidf_topics.pkl"))
joblib.dump(scaler, os.path.join(DATA_DIR, "scaler.pkl"))
print("Model, vectorizers, and scaler saved.")
