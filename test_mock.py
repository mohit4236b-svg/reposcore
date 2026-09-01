
import sys

class RepoFetchError(Exception):
    pass

def fetch_repo_features(repo_input, headers):
    raise RepoFetchError("Test error")

def log_audit_trail(features, probability, prediction, threshold, caveats=None):
    print(f"log_audit_trail called with threshold={threshold}, caveats={caveats}")

def test():
    repo_input = "bad/repo"
    headers = {}
    threshold = 0.5
    try:
        features = fetch_repo_features(repo_input, headers=headers)
    except RepoFetchError as e:
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
        log_audit_trail(minimal_features, 0.0, 0, threshold, caveats=[f"GitHub fetch failed: {str(e)}"])

test()

