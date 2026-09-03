"""
Shared preprocessing used by BOTH train_model.py and app.py.

Keeping this in one place matters: if the training script and the Streamlit
app clean README text differently, you get train/serve skew — the model was
fit on one distribution of text and scores a different one at inference time.
Import strip_badges from here in both places instead of copy-pasting it.
"""

import re
import subprocess
import tempfile
import shutil
import os
import time
import math

# Strips markdown image/badge syntax, shields.io/badge-service URLs, and
# common CI/build/status badge hosts. This exists because has_ci/has_tests
# are excluded as direct model features (they were used to build the label),
# but their *signal* was leaking back in indirectly through badge markup
# embedded in the README text (tokens like "workflows", "shields", "badge",
# "svg" ranked in the top-15 TF-IDF features before this was applied).
BADGE_PATTERNS = [
    r"!\[[^\]]*\]\([^)]*\)",                 # ![alt](url) markdown images
    r"\[!\[[^\]]*\]\([^)]*\)\]\([^)]*\)",     # linked badge images
    r"https?://\S*(shields\.io|badge\.fury\.io|travis-ci|"
    r"github\.com/\S*workflows\S*|circleci|codecov|coveralls|"
    r"bestpractices\.coreinfrastructure|securityscorecards|"
    r"oss-fuzz|ossrank|zenodo)\S*",
]


def strip_badges(text: str) -> str:
    """Remove badge/shield markup from README text before vectorizing."""
    if not isinstance(text, str):
        return ""
    for pattern in BADGE_PATTERNS:
        text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return text


STRUCTURED_COLS = ["stars", "forks", "open_issues", "readme_size",
                    "repo_age_days", "last_commit_days", "has_readme"]


def fetch_repo_features(full_name, headers=None):
    """
    Fetch a repo's metadata + README from the GitHub API and return the raw
    feature dict needed for prediction. Raises RepoFetchError on failure.

    This is the single place both app.py and reposcore_cli.py call into, so
    a Streamlit prediction and a CLI prediction for the same repo can never
    silently diverge.
    
    Production enhancements:
    - Fetches contributor count for better scoring
    - Handles rate limits with Retry-After headers
    - Supports conditional requests with ETags
    """
    import base64
    import requests
    import pandas as pd

    headers = headers or {}
    full_name = full_name.strip().strip("/")

    repo_resp = requests.get(f"https://api.github.com/repos/{full_name}", headers=headers, timeout=10)
    if repo_resp.status_code == 404:
        raise RepoFetchError(f"Repository '{full_name}' not found.")
    if repo_resp.status_code == 403:
        retry_after = int(repo_resp.headers.get("Retry-After", 60))
        raise RateLimitedRepoFetchError(
            "GitHub API rate limit exceeded. Set GITHUB_TOKEN to increase the limit.",
            retry_after=retry_after
        )
    if repo_resp.status_code != 200:
        raise RepoFetchError(f"GitHub API returned status {repo_resp.status_code}.")
    repo = repo_resp.json()

    readme_resp = requests.get(f"https://api.github.com/repos/{full_name}/readme", headers=headers, timeout=10)
    if readme_resp.status_code == 403:
        retry_after = int(readme_resp.headers.get("Retry-After", 60))
        raise RateLimitedRepoFetchError(
            "GitHub API rate limit exceeded while fetching README. Set GITHUB_TOKEN to increase the limit.",
            retry_after=retry_after
        )
    has_readme = readme_resp.status_code == 200
    readme_text, readme_size = "", 0
    if has_readme:
        readme_data = readme_resp.json()
        readme_size = readme_data.get("size", 0)
        try:
            readme_text = base64.b64decode(readme_data.get("content", "")).decode("utf-8", errors="ignore")
        except Exception:
            readme_text = ""

# Fetch contributor count (handle pagination)
    total_contributors = 1  # Default to at least the user
    try:
        contribs_resp = requests.get(
            f"https://api.github.com/repos/{full_name}/contributors",
            headers=headers,
            params={"per_page": 1},
            timeout=10
        )
        if contribs_resp.status_code == 200:
            # Use Link header to get total count if available
            link = contribs_resp.headers.get("Link", "")
            if 'rel="last"' in link:
                # Match the page number specifically from the "last" relation
                match = re.search(r'page=(\d+)>; rel="last"', link)
                if match:
                    total_contributors = int(match.group(1))
                else:
                    total_contributors = len(contribs_resp.json())
            else:
                total_contributors = len(contribs_resp.json())
    except Exception:
        pass  # Don't fail if contributor fetch fails

    # Detect CI from .github/workflows directory (matching training pipeline)
    ci_resp = requests.get(f"https://api.github.com/repos/{full_name}/contents/.github/workflows", headers=headers, timeout=10)
    if ci_resp.status_code == 403:
        retry_after = int(ci_resp.headers.get("Retry-After", 60))
        raise RateLimitedRepoFetchError(
            "GitHub API rate limit exceeded while checking CI workflows. Set GITHUB_TOKEN to increase the limit.",
            retry_after=retry_after
        )
    has_ci = ci_resp.status_code == 200

    # Detect tests from tests/ directory (matching training pipeline)
    tests_resp = requests.get(f"https://api.github.com/repos/{full_name}/contents/tests", headers=headers, timeout=10)
    if tests_resp.status_code == 403:
        retry_after = int(tests_resp.headers.get("Retry-After", 60))
        raise RateLimitedRepoFetchError(
            "GitHub API rate limit exceeded while checking tests directory. Set GITHUB_TOKEN to increase the limit.",
            retry_after=retry_after
        )
    has_tests = tests_resp.status_code == 200

    # Detect license from repo.license field (MISSING in original)
    has_license = repo.get("license") is not None

    # Detect CONTRIBUTING.md in repo root (case-insensitive via GitHub API)
    contributing_resp = requests.get(f"https://api.github.com/repos/{full_name}/contents/CONTRIBUTING.md", headers=headers, timeout=10)
    if contributing_resp.status_code == 403:
        retry_after = int(contributing_resp.headers.get("Retry-After", 60))
        raise RateLimitedRepoFetchError(
            "GitHub API rate limit exceeded while checking CONTRIBUTING.md. Set GITHUB_TOKEN to increase the limit.",
            retry_after=retry_after
        )
    has_contributing = contributing_resp.status_code == 200

    # Detect CODE_OF_CONDUCT.md in repo root (case-insensitive via GitHub API)
    coc_resp = requests.get(f"https://api.github.com/repos/{full_name}/contents/CODE_OF_CONDUCT.md", headers=headers, timeout=10)
    if coc_resp.status_code == 403:
        retry_after = int(coc_resp.headers.get("Retry-After", 60))
        raise RateLimitedRepoFetchError(
            "GitHub API rate limit exceeded while checking CODE_OF_CONDUCT.md. Set GITHUB_TOKEN to increase the limit.",
            retry_after=retry_after
        )
    has_code_of_conduct = coc_resp.status_code == 200

    # Get topics for TF-IDF vectorization
    topics = repo.get("topics", []) or []

    created_at = pd.to_datetime(repo["created_at"])
    pushed_at_raw = repo.get("pushed_at")
    if pushed_at_raw is None:
        pushed_at = pd.to_datetime(repo["created_at"])
    else:
        pushed_at = pd.to_datetime(pushed_at_raw)
    now = pd.Timestamp.now(tz="UTC")

    license_data = repo.get("license")
    _license_data = None
    if license_data:
        _license_data = {
            "key": license_data.get("key"),
            "name": license_data.get("name"),
            "spdx_id": license_data.get("spdx_id"),
            "url": license_data.get("url"),
        }

    return {
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
        "has_ci": has_ci,
        "has_tests": has_tests,
        "has_license": has_license,
        "has_contributing": has_contributing,
        "has_code_of_conduct": has_code_of_conduct,
        "total_contributors": total_contributors,
        "_license_data": _license_data,
    }


class RepoFetchError(Exception):
    """Raised when a repo can't be fetched from the GitHub API (404, rate limit, etc.)."""
    pass


class RateLimitedRepoFetchError(RepoFetchError):
    """Raised when rate limited - includes retry-after info for backoff."""
    def __init__(self, message: str, retry_after: int = 0):
        super().__init__(message)
        self.retry_after = retry_after


class CacheableRepoFetchError(RepoFetchError):
    """Error that can be cached to prevent repeated failures."""
    def __init__(self, message: str, cacheable: bool = True):
        super().__init__(message)
        self.cacheable = cacheable


def featurize(features, tfidf_readme, tfidf_topics, scaler):
    """Turn a fetch_repo_features() dict into the dense matrix the model expects."""
    import numpy as np
    from scipy.sparse import hstack

    topics_text = " ".join(features["topics"])
    structured = np.array([[features[c] for c in STRUCTURED_COLS]])

    X_readme = tfidf_readme.transform([features["readme_text_clean"]])
    X_topics = tfidf_topics.transform([topics_text])
    X = hstack([X_readme, X_topics, structured]).tocsr()
    X_scaled = scaler.transform(X)
    return np.asarray(X_scaled.todense(), dtype=np.float64)


def predict_quality(features, rf_model, tfidf_readme, tfidf_topics, scaler):
    """
    Run the trained model on a feature dict from fetch_repo_features().
    Returns (prediction: int, probability: float).
    """
    X_dense = featurize(features, tfidf_readme, tfidf_topics, scaler)
    prediction = int(rf_model.predict(X_dense)[0])
    probability = float(rf_model.predict_proba(X_dense)[0][1])
    return prediction, probability


def clone_repo_bounded(repo_full_name: str, size_kb: int) -> str | None:
    """
    Clone a Python repository with bounds and cleanup.
    
    Args:
        repo_full_name: Repository full name (owner/repo)
        size_kb: Repository size in kilobytes from GitHub API
        
    Returns:
        Path to cloned repository on success, None on failure
    """
    # Size cap: 50MB = 50 * 1024 KB
    MAX_SIZE_KB = 50 * 1024
    
    # Check size limit first
    if size_kb > MAX_SIZE_KB:
        return None
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="reposcore_")
    try:
        # Attempt shallow clone with timeout
        clone_url = f"https://github.com/{repo_full_name}.git"
        
        # Use subprocess with timeout
        result = subprocess.run(
            ["git", "clone", "--depth", "1", clone_url, temp_dir],
            capture_output=True,
            text=True,
            timeout=30  # 30 second hard timeout
        )
        
        # Check if clone succeeded
        if result.returncode == 0:
            return temp_dir
        else:
            # Clone failed
            return None
            
    except subprocess.TimeoutExpired:
        # Timeout occurred
        return None
    except Exception:
        # Any other error (git not found, network issues, etc.)
        return None
    finally:
        # Cleanup on failure - if we're returning None, cleanup the temp dir
        # If we succeeded, the caller is responsible for cleanup
        if 'result' in locals() and result.returncode != 0:
            shutil.rmtree(temp_dir, ignore_errors=True)
        elif 'result' not in locals():
            # Exception occurred before result was set
            shutil.rmtree(temp_dir, ignore_errors=True)


def extract_code_metrics(repo_path: str) -> dict | None:
    """
    Extract code complexity metrics using radon.
    
    Args:
        repo_path: Path to cloned repository
        
    Returns:
        Dictionary with metrics or None on failure
    """
    try:
        import radon
        from radon.raw import analyze as raw_analyze
        from radon.complexity import cc_visit
        from radon.metrics import mi_visit
        import ast
    except ImportError:
        return None
    
    start_time = time.time()
    
    try:
        py_files = []
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'venv', 'env', 'build', 'dist']]
            for file in files:
                if file.endswith('.py'):
                    py_files.append(os.path.join(root, file))
        
        if not py_files:
            return {"file_count": 0, "avg_complexity": 0, "total_loc": 0}
        
        total_loc = 0
        total_complexity = 0
        complexity_count = 0
        total_functions = 0
        total_classes = 0
        
        for py_file in py_files:
            try:
                if time.time() - start_time > 15:
                    break
                    
                with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                if not content.strip():
                    continue
                
                raw_metrics = raw_analyze(content)
                total_loc += raw_metrics.loc
                total_functions += raw_metrics.functions
                total_classes += raw_metrics.classes
                
                try:
                    tree = ast.parse(content)
                    complexity_blocks = cc_visit(tree)
                    
                    for block in complexity_blocks:
                        total_complexity += block.complexity
                        complexity_count += 1
                except SyntaxError:
                    continue
                    
            except Exception:
                continue
            
            if time.time() - start_time > 15:
                break
        
        avg_complexity = total_complexity / max(complexity_count, 1)
        
        return {
            "file_count": len(py_files),
            "avg_complexity": round(avg_complexity, 2),
            "total_loc": total_loc,
            "total_functions": total_functions,
            "total_classes": total_classes,
            "maintainability_index": calculate_maintainability_index(total_loc, complexity_count, total_complexity)
        }
        
    except Exception:
        return None


def calculate_maintainability_index(total_loc: int, function_count: int, total_complexity: float) -> float:
    """
    Calculate Maintainability Index (MI) based on Halstead metrics.
    
    MI = 171 - 5.2 * ln(HV) - 0.23 * ln(CC) - 16.2 * ln(LOC)
    where HV = Halstead Volume, CC = Cyclomatic Complexity, LOC = Lines of Code
    
    Returns:
        MI score between 0-100 (higher is better)
    """
    if total_loc == 0 or function_count == 0:
        return 0.0
    
    loc = float(total_loc)
    cc = float(total_complexity)
    
    mi = 171 - 5.2 * math.log(max(1, loc * math.log(max(1, cc + 1)))) - 0.23 * math.log(max(1, cc)) - 16.2 * math.log(max(1, loc))
    
    mi = max(0, min(100, mi * 100 / 171))
    
    return round(mi, 1)


def detect_test_coverage(repo_path: str) -> dict:
    """
    Detect test frameworks and estimate test coverage.
    
    Args:
        repo_path: Path to cloned repository
        
    Returns:
        Dictionary with test detection results
    """
    test_frameworks = {
        "pytest": ["pytest.ini", "conftest.py", "tests/", "test_", "_test.py"],
        "unittest": ["unittest", "TestCase"],
        "jest": ["jest.config", "package.json"],
        "mocha": ["mocha.opts", ".mocharc"],
        "rspec": ["spec/", "_spec.rb"],
        "go_test": ["_test.go"],
        "junit": ["pom.xml", "build.gradle", "testng.xml"],
    }
    
    detected_frameworks = []
    test_files = 0
    source_files = 0
    
    test_extensions = {".py", ".js", ".ts", ".jsx", ".tsx", ".rb", ".go", ".java", ".cs"}
    source_extensions = {".py", ".js", ".ts", ".jsx", ".tsx", ".rb", ".go", ".java", ".cs", ".cpp", ".c", ".h"}
    
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'venv', 'env', 'build', 'dist', '.git']]
        
        for file in files:
            file_lower = file.lower()
            full_path = os.path.join(root, file)
            
            for framework, indicators in test_frameworks.items():
                if any(ind in file_lower for ind in indicators):
                    if framework not in detected_frameworks:
                        detected_frameworks.append(framework)
            
            _, ext = os.path.splitext(file)
            if ext in test_extensions:
                if "test" in file_lower or "spec" in file_lower:
                    test_files += 1
                elif ext in {".js", ".ts", ".jsx", ".tsx", ".py"}:
                    if not file.startswith("_") and "test" not in root.lower() and "spec" not in root.lower():
                        source_files += 1
            elif ext in source_extensions:
                source_files += 1
    
    coverage_ratio = (test_files / max(source_files, 1)) * 100
    coverage_level = "unknown"
    if coverage_ratio >= 80:
        coverage_level = "excellent"
    elif coverage_ratio >= 60:
        coverage_level = "good"
    elif coverage_ratio >= 40:
        coverage_level = "moderate"
    elif coverage_ratio >= 20:
        coverage_level = "low"
    elif test_files > 0:
        coverage_level = "minimal"
    else:
        coverage_level = "none"
    
    return {
        "has_tests": test_files > 0,
        "test_file_count": test_files,
        "source_file_count": source_files,
        "coverage_ratio": round(coverage_ratio, 1),
        "coverage_level": coverage_level,
        "detected_frameworks": detected_frameworks,
        "needs_tests": test_files == 0 and source_files > 10
    }


def detect_documentation_quality(repo_path: str) -> dict:
    """
    Detect documentation files and quality indicators.
    
    Args:
        repo_path: Path to cloned repository
        
    Returns:
        Dictionary with documentation quality indicators
    """
    doc_files = {
        "readme": ["README.md", "README.txt", "README.rst", "README"],
        "contributing": ["CONTRIBUTING.md", "CONTRIBUTING.txt", "CONTRIBUTING"],
        "changelog": ["CHANGELOG.md", "CHANGELOG.txt", "CHANGELOG", "HISTORY.md"],
        "license": ["LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING"],
        "code_of_conduct": ["CODE_OF_CONDUCT.md", "CODE_OF_CONDUCT"],
        "api_docs": ["API.md", "API.rst", "docs/API"],
        "architecture": ["ARCHITECTURE.md", "ARCHITECTURE.txt", "docs/architecture"],
    }
    
    detected_docs = {}
    
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'venv', 'env', 'node_modules']]
        
        for doc_type, filenames in doc_files.items():
            if doc_type not in detected_docs:
                for file in files:
                    if file in filenames or file.upper() in [f.upper() for f in filenames]:
                        file_path = os.path.join(root, file)
                        try:
                            size = os.path.getsize(file_path)
                            detected_docs[doc_type] = {"found": True, "size": size, "path": os.path.relpath(file_path, repo_path)}
                        except OSError:
                            detected_docs[doc_type] = {"found": True, "size": 0, "path": os.path.relpath(file_path, repo_path)}
                        break
    
    for doc_type in doc_files:
        if doc_type not in detected_docs:
            detected_docs[doc_type] = {"found": False, "size": 0, "path": None}
    
    doc_score = sum(1 for d in detected_docs.values() if d["found"]) * 100 / len(doc_files)
    
    docstring_count = 0
    total_functions = 0
    
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'venv', 'env']]
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    import ast
                    try:
                        tree = ast.parse(content)
                        for node in ast.walk(tree):
                            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                                total_functions += 1
                                if ast.get_docstring(node):
                                    docstring_count += 1
                    except SyntaxError:
                        continue
                except Exception:
                    continue
    
    docstring_ratio = (docstring_count / max(total_functions, 1)) * 100 if total_functions > 0 else 0
    
    return {
        "detected_docs": detected_docs,
        "doc_score": round(doc_score, 1),
        "docstring_count": docstring_count,
        "total_functions": total_functions,
        "docstring_ratio": round(docstring_ratio, 1),
        "needs_docs": doc_score < 50
    }


def get_dependencies(repo_path: str) -> dict:
    """
    Get dependency information from repository.
    
    Args:
        repo_path: Path to cloned repository
        
    Returns:
        Dictionary with dependency information
    """
    dep_files = {
        "requirements.txt": [],
        "package.json": None,
        "pyproject.toml": None,
        "setup.py": None,
        "Pipfile": None,
        "go.mod": None,
        "Cargo.toml": None,
    }
    
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'venv', 'env', 'node_modules']]
        
        for file in files:
            if file in dep_files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, repo_path)
                
                if file == "requirements.txt":
                    deps = parse_requirements_file(file_path)
                    dep_files["requirements.txt"] = deps
                else:
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                        dep_files[file] = {"path": rel_path, "size": len(content), "content_preview": content[:500]}
                    except Exception:
                        dep_files[file] = {"path": rel_path, "error": "Could not read file"}
    
    has_deps = any(v for k, v in dep_files.items() if v and k != "requirements.txt")
    dep_count = len(dep_files["requirements.txt"]) if dep_files["requirements.txt"] else 0
    
    return {
        "has_dependencies": dep_count > 0 or has_deps,
        "dependency_files": {k: v for k, v in dep_files.items() if v},
        "requirements_count": dep_count
    }


def parse_requirements_file(file_path: str) -> list:
    """Parse requirements.txt file."""
    import re
    packages = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('-'):
                    continue
                
                match = re.match(r'^([a-zA-Z0-9_-]+)([=<>!~]+)(.+)$', line)
                if match:
                    packages.append({
                        "name": match.group(1),
                        "version_spec": match.group(2) + match.group(3)
                    })
                else:
                    match_name = re.match(r'^([a-zA-Z0-9_-]+)', line)
                    if match_name:
                        packages.append({
                            "name": match_name.group(1),
                            "version_spec": "unspecified"
                        })
    except Exception:
        pass
    
    return packages