"""
Command-line interface for RepoScore.

The Streamlit app is good for one-off interactive lookups, but isn't
scriptable -- you can't pipe a list of repos through it, use it in CI,
or get machine-readable output. This gives that: same model, same
reposcore_utils prediction path, JSON or CSV output, adjustable threshold.

Usage:
    python reposcore_cli.py owner/repo [owner/repo ...]
    python reposcore_cli.py --file repos.txt
    python reposcore_cli.py owner/repo --pretty
    python reposcore_cli.py owner/repo --format csv
    python reposcore_cli.py owner/repo --threshold 0.3

On --threshold: the model's default 0.5 cutoff is NOT F1-optimal on this
dataset. 5-fold CV shows F1 peaks around 0.3 (recall-favoring) --
precision 0.665 / recall 0.893 / F1 0.762 -- versus 0.5's precision 0.894 /
recall 0.474 / F1 0.619. Which one you want depends on what a false
negative costs you: use 0.3 if missing a genuinely good repo is worse than
a false positive (e.g. discovery/triage), stick with 0.5 or higher if a
false positive is worse (e.g. an automated gate). See README for the full
threshold table.

Exit code is 0 if all repos were scored successfully, 1 if any failed
(so this is usable as a CI gate, e.g. "fail the build if this repo's
predicted quality drops").
"""
import argparse
import csv
import io
import json
import os
import sys

import joblib
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reposcore_utils import fetch_repo_features, featurize, RepoFetchError

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

CSV_FIELDS = ["repo", "url", "predicted_quality", "confidence", "threshold",
              "stars", "forks", "open_issues", "repo_age_days", "error"]


def load_models():
    return (
        joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl")),
        joblib.load(os.path.join(MODELS_DIR, "tfidf_readme.pkl")),
        joblib.load(os.path.join(MODELS_DIR, "tfidf_topics.pkl")),
        joblib.load(os.path.join(MODELS_DIR, "scaler.pkl")),
    )


def score_repo(full_name, models, headers, threshold=0.5):
    rf_model, tfidf_readme, tfidf_topics, scaler = models
    try:
        features = fetch_repo_features(full_name, headers=headers)
    except RepoFetchError as e:
        return {"repo": full_name, "error": str(e)}

    X_dense = featurize(features, tfidf_readme, tfidf_topics, scaler)
    probability = float(rf_model.predict_proba(X_dense)[0][1])
    return {
        "repo": features["full_name"],
        "url": features["html_url"],
        "predicted_quality": "high" if probability >= threshold else "low",
        "confidence": round(probability, 4),
        "threshold": threshold,
        "stars": features["stars"],
        "forks": features["forks"],
        "open_issues": features["open_issues"],
        "repo_age_days": features["repo_age_days"],
    }


def to_csv(results):
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    for r in results:
        writer.writerow(r)
    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser(description="Score GitHub repos with RepoScore from the command line.")
    parser.add_argument("repos", nargs="*", help="owner/repo, space-separated")
    parser.add_argument("--file", help="path to a text file with one owner/repo per line")
    parser.add_argument("--pretty", action="store_true", help="pretty-print JSON output (ignored for --format csv)")
    parser.add_argument("--format", choices=["json", "csv"], default="json", help="output format (default: json)")
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="probability cutoff for 'high' quality (default: 0.5). "
                              "See module docstring / README for the CV precision/recall trade-off per threshold.")
    args = parser.parse_args()

    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0.0 and 1.0")

    repo_list = list(args.repos)
    if args.file:
        with open(args.file) as f:
            repo_list += [line.strip() for line in f if line.strip() and not line.startswith("#")]

    if not repo_list:
        parser.error("Provide at least one owner/repo, or --file repos.txt")

    load_dotenv(dotenv_path=".env")
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"token {token}"

    models = load_models()
    results = [score_repo(r, models, headers, threshold=args.threshold) for r in repo_list]

    if args.format == "csv":
        print(to_csv(results), end="")
    else:
        indent = 2 if args.pretty else None
        print(json.dumps(results, indent=indent))

    sys.exit(1 if any("error" in r for r in results) else 0)


if __name__ == "__main__":
    main()