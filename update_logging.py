import re

# Read the file
try:
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()
except UnicodeDecodeError:
    with open('app.py', 'r', encoding='latin-1') as f:
        content = f.read()

# Define the new log_audit_trail function
new_function = '''def log_audit_trail(features, probability, prediction, threshold, caveats=None):
    """
    Log scoring decision to CSV file for audit trail.
    Records: repo identifier, input features used, score, timestamp.
    """
    import csv
    import os
    import json
    from datetime import datetime

    # Create audit trail directory if it doesn't exist
    audit_dir = "audit_trail"
    if not os.path.exists(audit_dir):
        os.makedirs(audit_dir)

    # CSV file path
    csv_file = os.path.join(audit_dir, "scoring_decisions.csv")

    # Prepare data to log
    timestamp = datetime.now().isoformat()
    repo_id = features.get("full_name", "")

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
    # Also log to JSON Lines file
    jsonl_file = os.path.join(audit_dir, "predictions.jsonl")
    # Add caveats to the logged features for JSON Lines
    logged_features_with_caveats = logged_features.copy()
    logged_features_with_caveats["caveats"] = caveats if caveats is not None else []
    with open(jsonl_file, "a", encoding="utf-8") as f:
        json.dump(logged_features_with_caveats, f)
        f.write("\\n")
'''

# Replace the old function with the new one
# We'll use a regex to match the function definition and everything until the next function or end of indentation.
# We'll look for the pattern: "def log_audit_trail\\(features, probability, prediction, threshold\\):"
# and then replace until we see a line that starts with something that is not indented (or a blank line then a new function)
# But note: the function might be followed by a blank line and then another function or code.
# We'll do a simpler approach: we know the function ends before the line "rf_model, tfidf_readme, tfidf_topics, scaler = load_ml_assets()"
# We'll split the content by that line and then replace the function in the first part.

# Actually, let's do a regex that matches the function and then everything until we see a line that starts with whitespace? 
# Instead, we'll replace the function by matching from the def to the line before the next non-indented line that is not empty.
# We'll use a regex with the DOTALL flag to match across lines.

pattern = r'(def log_audit_trail\(features, probability, prediction, threshold\):.*?)(?=\\n\\S|\\nrf_model, tfidf_readme, tfidf_topics, scaler = load_ml_assets\\(\\))'

# We'll use the DOTALL flag so that . matches newlines.
new_content = re.sub(pattern, new_function, content, flags=re.DOTALL)

# Now we need to change the call to log_audit_trail to pass the caveats argument.
# We'll find the line that calls log_audit_trail and add the caveats=warning_messages argument.
# We'll look for the pattern: "log_audit_trail\\(features, probability, prediction, threshold\\)"
# and replace it with "log_audit_trail(features, probability, prediction, threshold, caveats=warning_messages)"

new_content = re.sub(r'log_audit_trail\\(features, probability, prediction, threshold\\)',
                     r'log_audit_trail(features, probability, prediction, threshold, caveats=warning_messages)',
                     new_content)

# Write the file back
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("Update complete.")
