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

output_dir = r"C:\reposcore_data"
df = pd.read_csv(f"{output_dir}\\repos_combined_final.csv")

topics_list = []

for i, row in df.iterrows():
    full_name = row["full_name"]
    print(f"Fetching topics {i+1}/{len(df)}: {full_name}")

    topics = []
    try:
        resp = session.get(f"https://api.github.com/repos/{full_name}", headers=headers, timeout=10)
        if resp.status_code == 200:
            topics = resp.json().get("topics", [])
    except requests.exceptions.RequestException as e:
        print(f"  Skipped {full_name}: {e}")

    topics_list.append(" ".join(topics))  # space-joined string, ready for CountVectorizer

    if (i + 1) % 100 == 0:
        try:
            pd.Series(topics_list).to_csv(f"{output_dir}\\topics_partial.csv", index=False)
            print(f"  Progress saved at {i+1}")
        except Exception as e:
            print(f"  Warning: could not save progress: {e}")

    time.sleep(0.3)

df["topics"] = topics_list

saved = False
for attempt in range(5):
    try:
        df.to_csv(f"{output_dir}\\repos_final_with_topics.csv", index=False)
        saved = True
        break
    except Exception as e:
        print(f"Save attempt {attempt+1} failed: {e}, retrying in 5s...")
        time.sleep(5)

if saved:
    print(f"Done. Saved {len(df)} repos with topics to {output_dir}\\repos_final_with_topics.csv")
else:
    print("Failed to save after 5 attempts")