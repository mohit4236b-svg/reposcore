#!/usr/bin/env python3
"""Debug contributor fetching."""

import os
import requests
import re
from dotenv import load_dotenv
load_dotenv(dotenv_path=r'C:\Users\ASUS\OneDrive\Documents\GitHub\reposcore\.env')

token = os.getenv('GITHUB_TOKEN')
headers = {'Accept': 'application/vnd.github+json', 'Authorization': f'Bearer {token}'}

url = 'https://api.github.com/repos/psf/requests/contributors'
resp = requests.get(url, headers=headers, params={'per_page': 1})
link = resp.headers.get('Link', '')
print('Link header:', link)

# The link header has format:
# <https://api.github.com/repositories/1362490/contributors?per_page=1&page=2>; rel="next",
# <https://api.github.com/repositories/1362490/contributors?per_page=1&page=402>; rel="last"

# We need to extract the page number from the "last" relation
if 'rel="last"' in link:
    # Find the page number in the "last" link
    match = re.search(r'page=(\d+)>; rel="last"', link)
    if match:
        total = int(match.group(1))
        print('Total contributors:', total)
    else:
        print('No match found for last page')
else:
    print('No last rel in link')