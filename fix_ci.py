path = r'c:\Users\ASUS\OneDrive\Documents\GitHub\reposcore\.github\workflows\ci.yml'
content = '''name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    services:
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Install system dependencies
        run: |
          sudo apt-get update
          sudo apt-get install -y build-essential libssl-dev libffi-dev python3-dev redis-tools
        shell: bash

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          python -m pip install --no-cache-dir -r requirements-dev.txt --verbose
        shell: bash

      - name: Check Python syntax
        run: python -m py_compile app.py reposcore_utils.py reposcore_cli.py
        shell: bash

      - name: Run tests
        run: pytest tests/ -v --tb=long
        env:
          REDIS_URL: redis://localhost:6379/0
          GITHUB_ACTIONS: "true"
        shell: bash

      - name: Check for large files (>5MB outside LFS)
        run: |
          find . -type f -size +5M ! -path "./.git/*" ! -path "*/site-packages/*" ! -path "*/vendor/*" ! -path "*/venv/*" | while read f; do
            echo "WARNING: Large file detected: $f"
          done
          echo "Large file check complete"
        shell: bash
'''
open(path, 'w').write(content)
print('Written successfully')