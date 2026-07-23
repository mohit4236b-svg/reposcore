import os
import requests
import pandas as pd
import base64
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

output_dir = r"C:\reposcore_data"
os.makedirs(output_dir, exist_ok=True)

session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retries))

df = pd.read_csv(os.path.join(output_dir, "repos_enriched_batch2.csv"))

readme_texts = []

for i, row in df.iterrows():
    full_name = row["full_name"]
    print(f"Fetching README {i+1}/{len(df)}: {full_name}")

    text = ""
    try:
        resp = session.get(f"https://api.github.com/repos/{full_name}/readme", headers=headers, timeout=10)
        if resp.status_code == 200:
            content_b64 = resp.json().get("content", "")
            try:
                text = base64.b64decode(content_b64).decode("utf-8", errors="ignore")
            except Exception:
                text = ""
    except requests.exceptions.RequestException as e:
        print(f"  Skipped {full_name}: {e}")

    readme_texts.append(text)

    if (i + 1) % 100 == 0:
        try:
            pd.Series(readme_texts).to_csv(os.path.join(output_dir, "readme_texts_batch2_partial.csv"), index=False)
            print(f"  Progress saved at {i+1}")
        except Exception as e:
            print(f"  Warning: could not save progress at {i+1}: {e}")

    time.sleep(0.5)

df["readme_text"] = readme_texts

saved = False
for attempt in range(5):
    try:
        df.to_csv(os.path.join(output_dir, "repos_enriched_batch2_with_readme.csv"), index=False)
        saved = True
        break
    except Exception as e:
        print(f"Save attempt {attempt+1} failed: {e}, retrying in 5s...")
        time.sleep(5)

if saved:
    print(f"Done. Saved {len(df)} repos to {output_dir}\\repos_enriched_batch2_with_readme.csv")
else:
    print("Failed to save after 5 attempts")