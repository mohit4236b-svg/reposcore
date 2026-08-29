"""Quick test to verify 403 handling in fetch_repo_features."""
from unittest.mock import MagicMock
import requests
from reposcore_utils import fetch_repo_features, RateLimitedRepoFetchError

# Mock the repo metadata call to succeed
mock_repo = MagicMock()
mock_repo.status_code = 200
mock_repo.json.return_value = {
    'full_name': 'test/repo',
    'html_url': 'https://github.com/test/repo',
    'topics': [],
    'created_at': '2020-01-01T00:00:00Z',
    'pushed_at': '2024-01-01T00:00:00Z',
    'stargazers_count': 100,
    'forks_count': 10,
    'open_issues_count': 5,
    'license': {'key': 'mit'}
}

# Mock CI check to return 403 (rate limited)
mock_ci = MagicMock()
mock_ci.status_code = 403
mock_ci.headers = {'Retry-After': '60'}

call_count = [0]
def mock_get(url, headers=None):
    call_count[0] += 1
    if 'repos/test/repo$' in url:
        return mock_repo
    elif 'readme' in url:
        m = MagicMock()
        m.status_code = 200
        m.json.return_value = {'content': 'IyBSZWFkbWU=', 'size': 100, 'encoding': 'base64'}
        return m
    elif 'workflows' in url:
        return mock_ci
    m = MagicMock()
    m.status_code = 200
    return m

original_get = requests.get
requests.get = mock_get
try:
    fetch_repo_features('test/repo', headers={})
    print('ERROR: Should have raised RateLimitedRepoFetchError')
except RateLimitedRepoFetchError as e:
    print(f'SUCCESS: Raised RateLimitedRepoFetchError: {e}')
    print(f'Retry after: {e.retry_after}')
except Exception as e:
    print(f'ERROR: Wrong exception type: {type(e).__name__}: {e}')
finally:
    requests.get = original_get