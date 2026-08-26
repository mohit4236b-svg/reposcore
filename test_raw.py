from dotenv import load_dotenv
load_dotenv()

import os
from openai import OpenAI

k = os.getenv('NVIDIA_API_KEY')
c = OpenAI(
    base_url='https://integrate.api.nvidia.com/v1',
    api_key=k,
    default_headers={'Authorization': f'Bearer {k}'}
)

pt = """You are a critical GitHub repository reviewer. Provide a comprehensive, structured review covering all four sections below. Use clear section headers and bullet points where appropriate.

**Repository:** pallets/flask
**Stats:** ★72000 | Forks:16000 | Open Issues:3
**Age:** 5800 days | Last commit: 1 days ago
**Contributors:** 400
**Topics:** wsgi,web,framework,python
**Primary Language:** Python
**CI:** True | **Tests:** True | **License:** True
**README length:** 8000 characters

**README content:**
Flask is a lightweight WSGI web application framework written in Python. It is designed to make getting started quick and easy, with the ability to scale up to complex applications. It began as a simple wrapper around Werkzeug and Jinja and has become one of the most popular Python web frameworks. Features include: built-in development server and debugger, integrated unit testing support, RESTful request dispatching, uses Jinja2 templating, support for secure cookies, 100% WSGI 1.0 compliant, Unicode based, extensively documented.

---

Output the review in this exact format with these four sections:

## Project Purpose
Describe what the project does, its core features, technology stack, and architecture in 2-3 sentences.

## Key Strengths
List 3-4 bullet points highlighting the project's strongest aspects (README quality, metrics, community, engineering practices, etc.).

## Gaps & Missing Elements
List 3-4 bullet points identifying actionable gaps (missing CI, tests, documentation, contribution guidelines, license, etc.).

## Production Readiness Assessment
Provide a 2-3 sentence assessment of whether this project appears production-ready based on the available signals.

Do NOT mention models, predictions, confidence scores, or use placeholder text. Output ONLY the structured review."""

resp = c.chat.completions.create(
    model='nvidia/nemotron-3-ultra-550b-a55b',
    messages=[{'role': 'user', 'content': pt}],
    temperature=0.3,
    max_tokens=800
)

print("RAW RESPONSE:")
print(repr(resp.choices[0].message.content))
print("\n\nFORMATTED:")
print(resp.choices[0].message.content)