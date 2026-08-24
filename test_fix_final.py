"""Test the fix for KeyError: 'last_commit_days'"""
import pandas as pd
from reposcore_utils import strip_badges

# Test the fixed logic
print("=== Testing fixed fetch_repo_features logic ===")

# Mock empty repo (pushed_at = None)
mock_repo_empty = {
    "full_name": "test/empty-repo",
    "html_url": "https://github.com/test/empty-repo",
    "topics": [],
    "stargazers_count": 0,
    "forks_count": 0,
    "open_issues_count": 0,
    "size": 0,
    "created_at": "2020-01-01T00:00:00Z",
    "pushed_at": None,
    "has_pages": False,
}

# Mock normal repo
mock_repo_normal = {
    "full_name": "test/normal-repo",
    "html_url": "https://github.com/test/normal-repo",
    "topics": ["python"],
    "stargazers_count": 100,
    "forks_count": 20,
    "open_issues_count": 5,
    "size": 1000,
    "created_at": "2020-01-01T00:00:00Z",
    "pushed_at": "2026-08-01T00:00:00Z",
    "has_pages": False,
}

def simulate_fetch_logic(repo, has_readme=False, readme_text="", readme_size=0):
    topics = repo.get("topics", []) or []
    has_ci = any(t in topics for t in ["ci", "github-actions", "workflows", "circleci", "travis-ci", "codecov"])
    has_pages = repo.get("has_pages", False)
    has_tests = any(t in topics for t in ["tests", "test", "testing", "pytest", "unittest"])
    total_contributors = 1
    
    created_at = pd.to_datetime(repo["created_at"])
    pushed_at_raw = repo.get("pushed_at")
    if pushed_at_raw is None:
        pushed_at = pd.to_datetime(repo["created_at"])  # fallback
    else:
        pushed_at = pd.to_datetime(pushed_at_raw)
    now = pd.Timestamp.now(tz="UTC")
    
    features = {
        "full_name": repo["full_name"],
        "html_url": repo["html_url"],
        "topics": topics,
        "readme_text_clean": strip_badges(readme_text),
        "stars": repo.get("stargazers_count", 0),
        "forks": repo.get("forks_count", 0),
        "open_issues": repo.get("open_issues_count", 0),
        "readme_size": readme_size,
        "repo_age_days": (now - created_at).days,
        "last_commit_days": int((now - pushed_at).days) if pd.notna((now - pushed_at).days) else 9999,
        "has_readme": int(has_readme),
        "has_ci": has_ci or has_pages,
        "has_tests": has_tests,
        "total_contributors": total_contributors,
    }
    return features

# Test empty repo
features_empty = simulate_fetch_logic(mock_repo_empty, has_readme=False, readme_text="", readme_size=0)
print(f"Empty repo last_commit_days: {features_empty['last_commit_days']}")

# Test normal repo
features_normal = simulate_fetch_logic(mock_repo_normal, has_readme=True, readme_text="# Test", readme_size=50)
print(f"Normal repo last_commit_days: {features_normal['last_commit_days']}")

# Test check_exceptions function (fixed version)
def check_exceptions_fixed(features):
    exceptions = []
    if features.get("has_readme", 1) == 0:
        exceptions.append("⚠️ No README detected.")
    elif features.get("readme_size", 0) < 50:
        exceptions.append("⚠️ Very small README (less than 50 characters).")
    if not features.get("topics"):
        exceptions.append("⚠️ No topics specified.")
    last_commit_days = features.get("last_commit_days")
    if last_commit_days is not None and last_commit_days > 730:  # over 2 years
        exceptions.append("⚠️ No commits in over 2 years.")
    return exceptions

print("\n=== Testing fixed check_exceptions ===")
print(f"Empty repo exceptions: {check_exceptions_fixed(features_empty)}")
print(f"Normal repo exceptions: {check_exceptions_fixed(features_normal)}")

# Test edge cases
print("\n=== Testing edge cases ===")
edge_cases = [
    {"name": "Missing last_commit_days", "features": {"has_readme": 1, "readme_size": 1000, "topics": ["test"]}},
    {"name": "None last_commit_days", "features": {"has_readme": 1, "readme_size": 1000, "topics": ["test"], "last_commit_days": None}},
    {"name": "NaN last_commit_days", "features": {"has_readme": 1, "readme_size": 1000, "topics": ["test"], "last_commit_days": float('nan')}},
    {"name": "Large last_commit_days", "features": {"has_readme": 1, "readme_size": 1000, "topics": ["test"], "last_commit_days": 9999}},
    {"name": "Zero last_commit_days", "features": {"has_readme": 1, "readme_size": 1000, "topics": ["test"], "last_commit_days": 0}},
]

for case in edge_cases:
    result = check_exceptions_fixed(case["features"])
    print(f"{case['name']}: {result}")

print("\n=== All tests completed successfully! ===")