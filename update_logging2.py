import re

# Read the file
with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find the start and end of the log_audit_trail function
start_idx = None
for i, line in enumerate(lines):
    if line.strip().startswith('def log_audit_trail(features, probability, prediction, threshold):'):
        start_idx = i
        break

if start_idx is None:
    print("Could not find log_audit_trail function")
    exit(1)

# Find the end of the function: look for the next line that starts with whitespace? Actually, we want to find the line where the indentation returns to the level of the function definition or less.
# We'll look for the next line that starts with something that is not a space (i.e., not indented) and is not empty, but we have to skip the function's docstring and body.
# Instead, we'll find the line that matches the pattern of the next function or a line that is not indented and is not part of the function.
# We'll do: from start_idx+1, we look for a line that has less indentation than the function definition line (assuming the function definition is at indent level 0? Actually, the function is at indent level 0 because it's at the module level).
# But note: there might be blank lines inside the function.
# We'll find the next line that starts with a non-whitespace character and is not part of the function.

# We'll get the indentation of the start line.
start_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())

# Now, we look for the next line after start_idx that has indentation <= start_indent and is not empty (or is a comment? but we'll treat empty as end of function?).
# Actually, the function ends when we see a line that is not indented more than the function definition.

end_idx = None
for i in range(start_idx + 1, len(lines)):
    line = lines[i]
    # If the line is empty or only whitespace, we continue? Actually, empty lines can be inside the function.
    # We'll check if the line has a non-whitespace character and its indentation is <= start_indent.
    stripped = line.lstrip()
    if stripped == '':
        # empty line, continue
        continue
    indent = len(line) - len(stripped)
    if indent <= start_indent:
        # This line is not indented more than the function definition, so it's the end of the function.
        end_idx = i
        break

if end_idx is None:
    # If we didn't find an end, we'll assume the function goes to the end of the file? But we know there is a line after the function.
    # We'll set end_idx to the line before the next function we know of.
    # We'll look for the next line that starts with 'def ' or 'class ' or '@' (decorator) or a known line like 'rf_model, tfidf_readme, tfidf_topics, scaler = load_ml_assets()'
    for i in range(start_idx + 1, len(lines)):
        if lines[i].strip().startswith('def ') or lines[i].strip().startswith('class ') or lines[i].strip().startswith('@') or lines[i].strip().startswith('rf_model, tfidf_readme, tfidf_topics, scaler = load_ml_assets()'):
            end_idx = i
            break
    if end_idx is None:
        end_idx = len(lines)

print(f"Function found from line {start_idx} to {end_idx-1}")

# Now we have the function lines: lines[start_idx:end_idx]
# We'll replace them with the new function.

new_function_lines = [
    'def log_audit_trail(features, probability, prediction, threshold, caveats=None):\n',
    '    """\n',
    '    Log scoring decision to CSV file for audit trail.\n',
    '    Records: repo identifier, input features used, score, timestamp.\n',
    '    """\n',
    '    import csv\n',
    '    import os\n',
    '    import json\n',
    '    from datetime import datetime\n',
    '\n',
    '    # Create audit trail directory if it doesn\'t exist\n',
    '    audit_dir = "audit_trail"\n',
    '    if not os.path.exists(audit_dir):\n',
    '        os.makedirs(audit_dir)\n',
    '\n',
    '    # CSV file path\n',
    '    csv_file = os.path.join(audit_dir, "scoring_decisions.csv")\n',
    '\n',
    '    # Prepare data to log\n',
    '    timestamp = datetime.now().isoformat()\n',
    '    repo_id = features.get("full_name", "")\n',
    '\n',
    '    # Extract features for logging (we\'ll log the key features used in scoring)\n',
    '    logged_features = {\n',
    '        "full_name": features.get("full_name", ""),\n',
    '        "html_url": features.get("html_url", ""),\n',
    '        "stars": features.get("stars", 0),\n',
    '        "forks": features.get("forks", 0),\n',
    '        "open_issues": features.get("open_issues", 0),\n',
    '        "readme_size": features.get("readme_size", 0),\n',
    '        "repo_age_days": features.get("repo_age_days", 0),\n',
    '        "last_commit_days": features.get("last_commit_days", 0),\n',
    '        "has_readme": features.get("has_readme", 0),\n',
    '        "topics_count": len(features.get("topics", [])),\n',
    '        "probability": f"{probability:.6f}",\n',
    '        "prediction": prediction,  # 1 for high quality, 0 for low quality\n',
    '        "threshold": f"{threshold:.2f}",\n',
    '        "timestamp": timestamp\n',
    '    }\n',
    '\n',
    '    # Define CSV headers\n',
    '    fieldnames = [\n',
    '        "timestamp", "repo_id", "repo_url", "stars", "forks", "open_issues",\n',
    '        "readme_size", "repo_age_days", "last_commit_days", "has_readme",\n',
    '        "topics_count", "probability", "prediction", "threshold"\n',
    '    ]\n',
    '\n',
    '    # Write to CSV (create file with headers if it doesn\'t exist)\n',
    '    file_exists = os.path.isfile(csv_file)\n',
    '    with open(csv_file, \'a\', newline=\'\', encoding=\'utf-8\') as f:\n',
    '        writer = csv.DictWriter(f, fieldnames=fieldnames)\n',
    '\n',
    '        # Write header if file is new\n',
    '        if not file_exists:\n',
    '            writer.writeheader()\n',
    '\n',
    '        # Write the data row\n',
    '        writer.writerow({\n',
    '            "timestamp": timestamp,\n',
    '            "repo_id": repo_id,\n',
    '            "repo_url": logged_features["html_url"],\n',
    '            "stars": logged_features["stars"],\n',
    '            "forks": logged_features["forks"],\n',
    '            "open_issues": logged_features["open_issues"],\n',
    '            "readme_size": logged_features["readme_size"],\n',
    '            "repo_age_days": logged_features["repo_age_days"],\n',
    '            "last_commit_days": logged_features["last_commit_days"],\n',
    '            "has_readme": logged_features["has_readme"],\n',
    '            "topics_count": logged_features["topics_count"],\n',
    '            "probability": logged_features["probability"],\n',
    '            "prediction": logged_features["prediction"],\n',
    '            "threshold": logged_features["threshold"]\n',
    '        })\n',
    '    # Also log to JSON Lines file\n',
    '    jsonl_file = os.path.join(audit_dir, "predictions.jsonl")\n',
    '    # Add caveats to the logged features for JSON Lines\n',
    '    logged_features_with_caveats = logged_features.copy()\n',
    '    logged_features_with_caveats["caveats"] = caveats if caveats is not None else []\n',
    '    with open(jsonl_file, "a", encoding="utf-8") as f:\n',
    '        json.dump(logged_features_with_caveats, f)\n',
    '        f.write("\\n")\n',
    '\n'
]

# Replace the function lines
lines[start_idx:end_idx] = new_function_lines

# Now we need to find the call to log_audit_trail and change it to pass caveats=warning_messages.
# We'll look for the line that contains "log_audit_trail(features, probability, prediction, threshold)"
# and replace it with "log_audit_trail(features, probability, prediction, threshold, caveats=warning_messages)"

for i, line in enumerate(lines):
    if 'log_audit_trail(features, probability, prediction, threshold)' in line:
        # Replace the call
        lines[i] = line.replace('log_audit_trail(features, probability, prediction, threshold)', 'log_audit_trail(features, probability, prediction, threshold, caveats=warning_messages)')
        break

# Write the file back
with open('app.py', 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Update complete.")
