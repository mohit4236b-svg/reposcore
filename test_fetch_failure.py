import sys
sys.path.insert(0, '.')

from reposcore_utils import fetch_repo_features, RepoFetchError

def test_fetch_failure():
    repo_input = "nonexistent/nonexistent12345"
    headers = {}  # No token, but that's fine
    threshold = 0.4  # arbitrary
    
    try:
        features = fetch_repo_features(repo_input, headers=headers)
    except RepoFetchError as e:
        print(f"Caught RepoFetchError: {e}")
        # Simulate the logging block
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
        print("About to call log_audit_trail with threshold:" + str(threshold))
        # Import log_audit_trail
        from app import log_audit_trail
        log_audit_trail(minimal_features, 0.0, 0, threshold, caveats=[f"GitHub fetch failed: {str(e)}"])
        print("Logged successfully.")
    else:
        print("Unexpected success")

if __name__ == "__main__":
    test_fetch_failure()

