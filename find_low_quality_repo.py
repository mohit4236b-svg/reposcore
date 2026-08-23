import os
import sys
sys.path.insert(0, 'c:/Users/ASUS/OneDrive/Documents/GitHub/reposcore')

from dotenv import load_dotenv
load_dotenv('c:/Users/ASUS/OneDrive/Documents/GitHub/reposcore/.env')

import warnings
warnings.filterwarnings("ignore")

from reposcore_utils import fetch_repo_features

headers = {"Accept": "application/vnd.github+json"}
token = os.getenv("GITHUB_TOKEN")
if token:
    headers["Authorization"] = f"Bearer {token}"

# Test a small/sparse repo
candidates = [
    "octocat/Spoon-Knife",           # GitHub's test repo
    "octocat/Hello-World",           # Minimal example (already tested)
    "facebook/react-native-tvos",    # Sparse/archived
]

for repo in candidates:
    try:
        features = fetch_repo_features(repo, headers=headers)
        readme_size = features.get('readme_size', 0)
        readme_text = features.get('readme_text_clean', '')[:200]
        print(f"{repo}: stars={features.get('stars', 0)}, readme={readme_size} chars")
        print(f"  README preview: {readme_text}")
        print()
    except Exception as e:
        print(f"{repo}: ERROR - {e}")
        print()