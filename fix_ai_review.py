# Write the fixed ai_review.py file - Part 1
content = """import os
import logging
import time
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning('google-genai not installed; AI review will be unavailable')

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning('openai not installed; NVIDIA NIM fallback will be unavailable')


class AIReviewError(Exception):
    pass

class MissingAPIKeyError(AIReviewError):
    pass

class EmptyReadmeError(AIReviewError):
    pass

class APITimeoutError(AIReviewError):
    pass

class APIRateLimitError(AIReviewError):
    pass


def _generate_nvidia_review(
    readme_content: str,
    features: dict,
    prediction: int,
    probability: float,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> dict:
    result = {
        "review": "",
        "status": "error",
        "error_type": None,
        "error_message": None,
        "finish_reason": None,
        "provider": "nvidia"
    }

    if not OPENAI_AVAILABLE:
        result["review"] = "AI review unavailable: openai package not installed."
        result["status"] = "skipped"
        result["error_type"] = "missing_dependency"
        result["error_message"] = "openai package not installed. Run pip install openai"
        logger.warning(result["error_message"])
        return result

    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key or api_key.strip() == "":
        result["review"] = "AI review unavailable: NVIDIA_API_KEY not configured."
        result["status"] = "skipped"
        result["error_type"] = "missing_api_key"
        result["error_message"] = "NVIDIA_API_KEY environment variable not set"
        logger.warning(result["error_message"])
        return result

    max_readme_chars = 8000
    readme_for_prompt = readme_content[:max_readme_chars]
    if len(readme_content) > max_readme_chars:
        readme_for_prompt += chr(10) + chr(10) + "[README truncated from " + str(len(readme_content)) + " to " + str(max_readme_chars) + " characters]"
"""

with open('C:/Users/ASUS/OneDrive/Documents/GitHub/reposcore/ai_review.py', 'w', encoding='utf-8') as f:
    f.write(content)
print('Part 1 done')