import os
import requests
import pandas as pd
import time
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv(dotenv_path=".env")

token = os.getenv("GITHUB_TOKEN")
headers = {
    "Authorization": f"token {token}",
    "Accept": "application/vnd.github+json"
}

session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

df = pd.read_csv(r"C:\reposcore_data\repos_basic_batch2.csv")

enriched = []

for i, row in df.iterrows()

    except requests.exceptions.RequestException as e:
        print(f"  Skipped {full_name}: {e}")
        repo_data["has_readme"] = None
        repo_data["readme_size"] = None
        repo_data["has_ci"] = None
        repo_data["has_tests"] = None

    enriched.append(repo_data)

    if (i + 1) % 100 == 0:
        pd.DataFrame(enriched).to_csv(r"C:\reposcore_data\repos_enriched_batch2_partial.csv", index=False)
        print(f"  Progress saved at {i+1}")

    time.sleep(0.5)

enriched_df = pd.DataFrame(enriched)
enriched_df.to_csv(r"C:\reposcore_data\repos_enriched_batch2.csv", index=False)
print(f"Done. Saved {len(enriched_df)} repos to C:\\reposcore_data\\repos_enriched_batch2.csv")