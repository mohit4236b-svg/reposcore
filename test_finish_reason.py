#!/usr/bin/env python3
"""Test finish_reason with actual README content."""

import os
from dotenv import load_dotenv
load_dotenv()

from ai_review import generate_ai_review

# Use cached features from the earlier successful run
features = {
    'full_name': 'pre-commit/pre-commit-hooks',
    'stars': 6667,
    'forks': 798,
    'open_issues': 6,
    'repo_age_days': 4546,
    'last_commit_days': 6,
    'total_contributors': 1,
    'topics': ['git', 'linter', 'pre-commit', 'python', 'refactoring'],
    'has_ci': False,
    'has_tests': False,
    'readme_size': 9208
}

# Read the actual README content from the earlier test
readme_content = """The pre-commit-hooks repository provides a collection of utility scripts written in Python to enforce code quality, security, and file formatting at the Git commit stage. It includes hooks for checking JSON, YAML, TOML, XML, and other file formats, as well as security-related checks like detecting AWS credentials and private keys. The hooks are designed to be used with the pre-commit framework."""

result = generate_ai_review(readme_content, features, 1, 0.35, timeout_seconds=60)
print(f"Status: {result['status']}")
print(f"Finish reason: {result.get('finish_reason', 'N/A')}")
print(f"Review length: {len(result.get('review', ''))}")
print(f"Review:\n{result['review']}")