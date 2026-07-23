import os
import requests
from dotenv import load_dotenv

load_dotenv(dotenv_path=".env")

token = os.getenv("GITHUB_TOKEN")
print("Token starts with:", token[:4] if token else "None")
print("Token length:", len(token) if token else 0)

headers = {
    "Authorization": f"token {token}",
    "Accept": "application/vnd.github+json"
}

response = requests.get("https://api.github.com/rate_limit", headers=headers)

print("Status code:", response.status_code)
print(response.json())