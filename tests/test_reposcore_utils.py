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
