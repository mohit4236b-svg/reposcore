import os
import requests
import pandas as pd
import time
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

token = os.getenv("GITHUB_TOKEN")
headers = {
    "Authorization": f"token {token}",
    "Accept": "application/vnd.github+json"
}

topics = ["deep-learning", "nlp", "computer-vision", "data-science"]
all_repos = []

for topic in topics:
    query = f"topic:{topic}"
    print(f"--- Collecting topic: {topic} ---")

    for page in range(1, 6):  # 5 pages x 100 = up to 500 per topic
        url = "https://api.github.com/search/repositories"
        params = {
            "q": query,
            "sort": "stars",
            "order": "desc",
            "per_page": 100,
            "page": page
        }
        response = requests.get(url, headers=headers, params=params)

        if response.status_code != 200:
            print(f"Error on {topic} page {page}: {response.status_code}")
            break

        items = response.json().get("items", [])
        if not items:
            print(f"No more results for {topic} at page {page}")
            break

        for repo in items:
            all_repos.append({
                "full_name": repo["full_name"],
                "stars": repo["stargazers_count"],
                "forks": repo["forks_count"],
                "open_issues": repo["open_issues_count"],
                "created_at": repo["created_at"],
                "pushed_at": repo["pushed_at"],
                "language": repo["language"],
                "license": repo["license"]["name"] if repo["license"] else None,
                "description": repo["description"]
            })

        print(f"  Page {page} done, total so far: {len(all_repos)}")
        time.sleep(2)

df = pd.DataFrame(all_repos)
df = df.drop_duplicates(subset="full_name")  # remove overlap with original ml-topic repos
df.to_csv(r"C:\reposcore_data\repos_basic_batch2.csv", index=False)
print(f"Saved {len(df)} unique repos to C:\\reposcore_data\\repos_basic_batch2.csv")