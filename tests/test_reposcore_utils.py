"""
Tests for reposcore_utils.strip_badges.

These exist mainly to guard against regressing the badge-leakage fix:
if strip_badges stops removing badge markup, has_ci-like signal starts
leaking back into the README TF-IDF features silently.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reposcore_utils import strip_badges


def test_removes_markdown_image_badge():
    text = "# Project\n![Build Status](https://img.shields.io/badge/build-passing-green)\nSome real docs."
    cleaned = strip_badges(text)
    assert "shields.io" not in cleaned
    assert "Some real docs." in cleaned


def test_removes_linked_badge():
    text = "[![CI](https://github.com/user/repo/workflows/CI/badge.svg)](https://github.com/user/repo/actions)"
    cleaned = strip_badges(text)
    assert "workflows" not in cleaned
    assert "badge" not in cleaned.lower()


def test_removes_codecov_and_travis_links():
    text = "See coverage at https://codecov.io/gh/user/repo and https://travis-ci.org/user/repo"
    cleaned = strip_badges(text)
    assert "codecov" not in cleaned
    assert "travis-ci" not in cleaned


def test_preserves_normal_text():
    text = "Install with pip install reposcore. See CONTRIBUTING.md for details."
    cleaned = strip_badges(text)
    assert "Install with pip install reposcore." in cleaned
    assert "CONTRIBUTING.md" in cleaned


def test_handles_non_string_input():
    assert strip_badges(None) == ""
    assert strip_badges(float("nan") if False else None) == ""


def test_handles_empty_string():
    assert strip_badges("") == ""


def test_cli_score_repo_handles_fetch_error(monkeypatch):
    """score_repo() should surface a RepoFetchError as an {'error': ...} dict,
    not raise -- this is what lets the CLI report failures per-repo in its
    JSON array instead of crashing the whole batch."""
    import reposcore_cli
    from reposcore_utils import RepoFetchError

    def fake_fetch(full_name, headers=None):
        raise RepoFetchError("Repository 'nope/nope' not found.")

    monkeypatch.setattr(reposcore_cli, "fetch_repo_features", fake_fetch)
    result = reposcore_cli.score_repo("nope/nope", models=(None, None, None, None), headers={})
    assert result == {"repo": "nope/nope", "error": "Repository 'nope/nope' not found."}


def test_cli_score_repo_respects_threshold(monkeypatch):
    """A probability of 0.4 should flip from 'low' to 'high' depending on
    the --threshold value -- this is the whole point of exposing it."""
    import reposcore_cli
    import numpy as np

    class FakeModel:
        def predict_proba(self, X):
            return np.array([[0.6, 0.4]])

    monkeypatch.setattr(reposcore_cli, "fetch_repo_features",
                         lambda full_name, headers=None: {
                             "full_name": full_name, "html_url": "u", "topics": [],
                             "stars": 1, "forks": 0, "open_issues": 0, "repo_age_days": 1,
                         })
    monkeypatch.setattr(reposcore_cli, "featurize", lambda *a, **k: None)

    models = (FakeModel(), None, None, None)
    low = reposcore_cli.score_repo("a/b", models, headers={}, threshold=0.5)
    high = reposcore_cli.score_repo("a/b", models, headers={}, threshold=0.3)
    assert low["predicted_quality"] == "low"
    assert high["predicted_quality"] == "high"
    assert low["confidence"] == 0.4


def test_cli_csv_output_has_expected_header():
    import reposcore_cli
    results = [{"repo": "a/b", "url": "u", "predicted_quality": "high",
                "confidence": 0.7, "threshold": 0.5, "stars": 5, "forks": 1,
                "open_issues": 0, "repo_age_days": 10}]
    out = reposcore_cli.to_csv(results)
    assert out.splitlines()[0] == ",".join(reposcore_cli.CSV_FIELDS)
    assert "a/b" in out


def test_fetch_repo_features_raises_on_readme_rate_limit(monkeypatch):
    """A 403 on the README endpoint (rate limit) should raise RepoFetchError,
    not silently set has_readme=0. This prevents the model from silently
    degrading to using only ~20% of its features when rate-limited."""
    from unittest.mock import MagicMock
    from reposcore_utils import fetch_repo_features, RepoFetchError

    # Mock response for repo metadata - returns 200 with valid repo
    mock_repo_resp = MagicMock()
    mock_repo_resp.status_code = 200
    mock_repo_resp.json.return_value = {
        "full_name": "good/repo",
        "html_url": "https://github.com/good/repo",
        "topics": [],
        "created_at": "2020-01-01T00:00:00Z",
        "pushed_at": "2024-01-01T00:00:00Z",
        "stargazers_count": 100,
        "forks_count": 10,
        "open_issues_count": 5,
    }

    # Mock response for README - returns 403 (rate limit)
    mock_readme_resp = MagicMock()
    mock_readme_resp.status_code = 403

    import requests
    monkeypatch.setattr(requests, "get", lambda url, headers=None, **kwargs: mock_repo_resp if "/readme" not in url else mock_readme_resp)

    try:
        fetch_repo_features("good/repo", headers={})
        assert False, "Expected RepoFetchError to be raised"
    except RepoFetchError as e:
        assert "rate limit" in str(e).lower()
        assert "README" in str(e)
