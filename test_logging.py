import os
import json
import sys

# Add the current directory to the path so we can import app
sys.path.insert(0, '.')

# Import the function from app
from app import log_audit_trail

# Create a temporary audit trail directory for testing
test_audit_dir = "test_audit_trail"
if not os.path.exists(test_audit_dir):
    os.makedirs(test_audit_dir)

# We need to monkey-patch the audit_dir in the function? 
# Instead, we'll change the function to use our test directory by setting an environment variable? 
# But the function uses a hardcoded "audit_trail" directory.
# We'll instead change the current working directory to the test directory and then run the function? 
# Or we can modify the function to use a different directory by passing an argument? 
# We don't want to change the function again.

# We'll instead create a fake audit_trail directory in the current working directory and then run the function, 
# and then check the file there.
# But note: the function uses "audit_trail" relative to the current working directory.

# We'll change the current working directory to a temporary directory.
import tempfile
import shutil

temp_dir = tempfile.mkdtemp()
print(f"Using temp directory: {temp_dir}")
os.chdir(temp_dir)

# Now the function will create audit_trail in this temp directory.

# Create some fake features
features = {
    "full_name": "testowner/testrepo",
    "html_url": "https://github.com/testowner/testrepo",
    "stars": 10,
    "forks": 2,
    "open_issues": 1,
    "readme_size": 100,
    "repo_age_days": 365,
    "last_commit_days": 30,
    "has_readme": 1,
    "topics": ["test", "repo"]
}

probability = 0.75
prediction = 1
threshold = 0.3
caveats = ["Test caveat 1", "Test caveat 2"]

# Call the function
log_audit_trail(features, probability, prediction, threshold, caveats=caveats)

# Check that the predictions.jsonl file exists and contains the expected data
jsonl_file = os.path.join("audit_trail", "predictions.jsonl")
if not os.path.exists(jsonl_file):
    print("ERROR: predictions.jsonl file not found")
    sys.exit(1)

with open(jsonl_file, "r", encoding="utf-8") as f:
    lines = f.readlines()
    if len(lines) == 0:
        print("ERROR: predictions.jsonl file is empty")
        sys.exit(1)
    # The last line should be the one we just wrote
    last_line = lines[-1].strip()
    try:
        data = json.loads(last_line)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse JSON: {e}")
        sys.exit(1)

    # Check that the caveats field is present and matches
    if "caveats" not in data:
        print("ERROR: caveats field not found in JSON")
        sys.exit(1)
    if data["caveats"] != caveats:
        print(f"ERROR: caveats mismatch. Expected {caveats}, got {data['caveats']}")
        sys.exit(1)

    # Check that other fields are present
    expected_fields = ["full_name", "html_url", "stars", "forks", "open_issues", "readme_size", "repo_age_days", "last_commit_days", "has_readme", "topics_count", "probability", "prediction", "threshold", "timestamp", "caveats"]
    for field in expected_fields:
        if field not in data:
            print(f"ERROR: missing field {field}")
            sys.exit(1)

    print("SUCCESS: log_audit_trail function works correctly with caveats")

# Clean up
os.chdir("..")
shutil.rmtree(temp_dir)
