#!/usr/bin/env python3
"""Debug contributor fetching."""

import os
import requests
from dotenv import load_dotenv
load_dotenv(dotenv_path=r'C:\Users\ASUS\OneDrive\Documents\GitHub\reposcore\.env')

token = os.getenv('GITHUB_TOKEN')
headers = {'Accept': 'application/vnd.github+json', 'Authorization': f'Bearer {token}'}

# Test contributor fetch
url = 'https://api.github.com/repos/psf/requests/contributors'
resp = requests.get(url, headers=headers, params={'per_page': 1})
print('Status:', resp.status_code)
print('Link header:', resp.headers.get('Link'))
print('Data length:', len(resp.json()))