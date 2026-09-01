import sys
sys.path.insert(0, '.')

from reposcore_utils import fetch_repo_features, RepoFetchError
from app import log_audit_trail

def test_fetch_failure():
    repo_input = "this-repo-definitely-does-not-exist-12345/foo"
    threshold = 0.3  # arbitrary threshold value
    try:
        features = fetch_repo_features(repo_input, headers={})
    except RepoFetchError as e:
        print(f"Caught RepoFetchError: {e}")
        minimal_features = {
            "full_name": repo_input,
            "html_url": f"https://github.com/{repo_input}",
            "stars": 0,
            "forks": 0,
            "open_issues": 0,
            "readme_size": 0,
            "repo_age_days": 0,
            "last_commit_days": 0,
            "has_readme": 0,
            "topics": [],
        }
        print("Calling log_audit_trail with minimal_features and threshold:", threshold)
        log_audit_trail(minimal_features, 0.0, 0, threshold, caveats=[f"GitHub fetch failed: {str(e)}"])
        print("log_audit_trail call succeeded.")
    else:
        print("ERROR: Expected RepoFetchError but got features:", features)
        return False
    return True

if __name__ == "__main__":
    success = test_fetch_failure()
    if success:
        # Check the JSON Lines file for the last entry
        jsonl_file = "audit_trail/predictions.jsonl"
        try:
            with open(jsonl_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    print("Last JSON Lines entry:")
                    print(last_line)
                else:
                    print("JSON Lines file is empty.")
        except Exception as e:
            print(f"Error reading JSON Lines file: {e}")
    else:
        print("Test failed.")
