import os
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

    try:
        client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key
        )
        max_readme_chars = 8000
        readme_for_prompt = readme_content[:max_readme_chars]
        if len(readme_content) > max_readme_chars:
            readme_for_prompt += chr(10) + chr(10) + "[README truncated from " + str(len(readme_content)) + " to " + str(max_readme_chars) + " characters]"

        prompt = """You are an experienced software engineer reviewing a GitHub repository. Provide a technical assessment of the repository based on its README and observable metrics. Do not use promotional or marketing language. Do not mention the model, prediction, or confidence. Be direct, specific, and critical where warranted - identify concrete gaps alongside strengths.

Repository: {full_name}
Stars: {stars} | Forks: {forks} | Open Issues: {open_issues}
Age: {repo_age_days} days | Days since last commit: {last_commit_days}
Contributors: {total_contributors}
Topics: {topics}
Primary Language: {primary_language}
Has CI: {has_ci} | Has Tests: {has_tests} | Has License: {has_license}
README length: {readme_size} characters

README Content:
{readme_for_prompt}

Write 3-5 sentences covering:
1. What the README describes (purpose, key features, tech stack, architecture) - reference specific details from the README
2. How the observable metrics (stars, activity, contributors, CI/tests presence) align with or contradict the README claims - be explicit about discrepancies
3. 1-2 concrete, actionable gaps to address (e.g., Add a CONTRIBUTING.md, Include installation steps in the README, Set up CI with a badge, Add code coverage reporting, Document the API endpoints) - reference THIS repo specific missing elements
4. One specific strength and why it lowers risk for adopters - cite actual evidence from the repo

Each sentence must reference THIS repo actual content. No generic praise; if something is missing or weak, state it directly.""".format(
            full_name=features.get("full_name", "Unknown"),
            stars=features.get("stars", 0),
            forks=features.get("forks", 0),
            open_issues=features.get("open_issues", 0),
            repo_age_days=features.get("repo_age_days", 0),
            last_commit_days=features.get("last_commit_days", 0),
            total_contributors=features.get("total_contributors", "Unknown"),
            topics=", ".join(features.get("topics", [])) if features.get("topics") else "None",
            primary_language=features.get("primary_language", "Unknown"),
            has_ci=features.get("has_ci", False),
            has_tests=features.get("has_tests", False),
            has_license=features.get("has_license", False),
            readme_size=features.get("readme_size", 0),
            readme_for_prompt=readme_for_prompt
        )

        max_retries = 2
        base_delay = 1.0

        for attempt in range(max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model="meta/llama-3.1-70b-instruct",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2000,
                    temperature=0.3,
                )
                break
            except Exception as e:
                error_msg = str(e)
                is_rate_limit = (
                    "rate limit" in error_msg.lower() or
                    "429" in error_msg or
                    "quota" in error_msg.lower()
                )

                if is_rate_limit and attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.warning("NVIDIA rate limit hit (attempt {}/{}), retrying in {}s: {}".format(attempt + 1, max_retries + 1, delay, error_msg))
                    time.sleep(delay)
                    continue
                else:
                    raise

        if response and response.choices:
            finish_reason = response.choices[0].finish_reason
            logger.info("NVIDIA review finish_reason for {}: {}".format(features.get("full_name", "unknown"), finish_reason))
            result["finish_reason"] = finish_reason

        if response and response.choices and response.choices[0].message.content:
            result["review"] = response.choices[0].message.content.strip()
            result["status"] = "success"
            logger.info("NVIDIA AI review generated successfully for {}".format(features.get("full_name", "unknown")))
        else:
            result["review"] = "AI review unavailable: Empty response from NVIDIA API."
            result["status"] = "error"
            result["error_type"] = "empty_response"
            result["error_message"] = "NVIDIA API returned empty response"
            logger.warning(result["error_message"])

    except Exception as e:
        error_msg = str(e)
        logger.error("NVIDIA API error for {}: {}".format(features.get("full_name", "unknown"), error_msg))

        if "rate limit" in error_msg.lower() or "429" in error_msg or "quota" in error_msg.lower():
            result["review"] = "AI review unavailable: NVIDIA rate limit exceeded."
            result["error_type"] = "rate_limit"
        elif "timeout" in error_msg.lower() or "deadline" in error_msg.lower():
            result["review"] = "AI review unavailable: NVIDIA request timed out."
            result["error_type"] = "timeout"
        elif "api key" in error_msg.lower() or "invalid" in error_msg.lower() or "401" in error_msg or "403" in error_msg:
            result["review"] = "AI review unavailable: Invalid or missing NVIDIA API key."
            result["error_type"] = "invalid_api_key"
        else:
            result["review"] = "AI review unavailable due to a temporary NVIDIA error."
            result["error_type"] = "api_error"

        result["status"] = "error"
        result["error_message"] = error_msg

    return result

def _generate_gemini_review(
    readme_content: str,
    features: dict,
    prediction: int,
    probability: float,
    api_key: str = None,
    timeout_seconds: int = 30
) -> dict:
    result = {
        "review": "",
        "status": "error",
        "error_type": None,
        "error_message": None,
        "finish_reason": None,
        "provider": "gemini"
    }
    
    if not GENAI_AVAILABLE:
        result["review"] = "AI review unavailable: google-genai package not installed."
        result["status"] = "skipped"
        result["error_type"] = "missing_dependency"
        result["error_message"] = "google-genai package not installed. Run pip install google-genai==1.8.0"
        logger.warning(result["error_message"])
        return result
    
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key or api_key.strip() == "" or api_key == "your_gemini_api_key_here":
        result["review"] = "AI review unavailable: GEMINI_API_KEY not configured. Set it in your .env file or environment."
        result["status"] = "skipped"
        result["error_type"] = "missing_api_key"
        result["error_message"] = "GEMINI_API_KEY environment variable not set or using placeholder value"
        logger.warning(result["error_message"])
        return result
    
    if not readme_content or not readme_content.strip() or len(readme_content.strip()) < 50:
        result["review"] = "AI review unavailable: README is too short or empty to analyze meaningfully."
        result["status"] = "skipped"
        result["error_type"] = "empty_readme"
        result["error_message"] = "README content too short ({} chars)".format(len(readme_content.strip()) if readme_content else 0)
        logger.warning(result["error_message"])
        return result
    
    try:
        client = genai.Client(api_key=api_key)
        max_readme_chars = 8000
        readme_for_prompt = readme_content[:max_readme_chars]
        if len(readme_content) > max_readme_chars:
            readme_for_prompt += chr(10) + chr(10) + "[README truncated from " + str(len(readme_content)) + " to " + str(max_readme_chars) + " characters]"

        prompt = """You are an experienced software engineer reviewing a GitHub repository. Provide a technical assessment of the repository based on its README and observable metrics. Do not use promotional or marketing language. Do not mention the model, prediction, or confidence. Be direct, specific, and critical where warranted - identify concrete gaps alongside strengths.

Repository: {full_name}
Stars: {stars} | Forks: {forks} | Open Issues: {open_issues}
Age: {repo_age_days} days | Days since last commit: {last_commit_days}
Contributors: {total_contributors}
Topics: {topics}
Primary Language: {primary_language}
Has CI: {has_ci} | Has Tests: {has_tests} | Has License: {has_license}
README length: {readme_size} characters

README Content:
{readme_for_prompt}

Write 3-5 sentences covering:
1. What the README describes (purpose, key features, tech stack, architecture) - reference specific details from the README
2. How the observable metrics (stars, activity, contributors, CI/tests presence) align with or contradict the README claims - be explicit about discrepancies
3. 1-2 concrete, actionable gaps to address (e.g., Add a CONTRIBUTING.md, Include installation steps in the README, Set up CI with a badge, Add code coverage reporting, Document the API endpoints) - reference THIS repo specific missing elements
4. One specific strength and why it lowers risk for adopters - cite actual evidence from the repo

Each sentence must reference THIS repo actual content. No generic praise; if something is missing or weak, state it directly.""".format(
            full_name=features.get("full_name", "Unknown"),
            stars=features.get("stars", 0),
            forks=features.get("forks", 0),
            open_issues=features.get("open_issues", 0),
            repo_age_days=features.get("repo_age_days", 0),
            last_commit_days=features.get("last_commit_days", 0),
            total_contributors=features.get("total_contributors", "Unknown"),
            topics=", ".join(features.get("topics", [])) if features.get("topics") else "None",
            primary_language=features.get("primary_language", "Unknown"),
            has_ci=features.get("has_ci", False),
            has_tests=features.get("has_tests", False),
            has_license=features.get("has_license", False),
            readme_size=features.get("readme_size", 0),
            readme_for_prompt=readme_for_prompt
        )
        
        generation_config = types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=2000,
        )
        
        max_retries = 2
        base_delay = 1.0
        
        for attempt in range(max_retries + 1):
            try:
                response = client.models.generate_content(
                    model="gemini-3.5-flash",
                    contents=prompt,
                    config=generation_config,
                )
                break
            except Exception as e:
                error_msg = str(e)
                is_rate_limit = (
                    "quota" in error_msg.lower() or 
                    "rate limit" in error_msg.lower() or 
                    "429" in error_msg
                )
                
                if is_rate_limit and attempt < max_retries:
                    delay = base_delay * (2 ** attempt)
                    logger.warning("Rate limit hit (attempt {}/{}), retrying in {}s: {}".format(attempt + 1, max_retries + 1, delay, error_msg))
                    time.sleep(delay)
                    continue
                else:
                    raise

        if response and response.candidates:
            finish_reason = response.candidates[0].finish_reason
            logger.info("AI review finish_reason for {}: {}".format(features.get("full_name", "unknown"), finish_reason))
            result["finish_reason"] = finish_reason
        
        if response and response.text:
            result["review"] = response.text.strip()
            result["status"] = "success"
            logger.info("AI review generated successfully for {}".format(features.get("full_name", "unknown")))
        else:
            result["review"] = "AI review unavailable: Empty response from API."
            result["status"] = "error"
            result["error_type"] = "empty_response"
            result["error_message"] = "Gemini API returned empty response"
            logger.warning(result["error_message"])
            
    except Exception as e:
        error_msg = str(e)
        logger.error("Gemini API error for {}: {}".format(features.get("full_name", "unknown"), error_msg))
        
        if "quota" in error_msg.lower() or "rate limit" in error_msg.lower() or "429" in error_msg:
            result["review"] = "AI review unavailable: Rate limit exceeded. Please try again later."
            result["error_type"] = "rate_limit"
        elif "timeout" in error_msg.lower() or "deadline" in error_msg.lower():
            result["review"] = "AI review unavailable: Request timed out. Please try again."
            result["error_type"] = "timeout"
        elif "api key" in error_msg.lower() or "invalid" in error_msg.lower() or "401" in error_msg or "403" in error_msg:
            result["review"] = "AI review unavailable: Invalid or missing API key."
            result["error_type"] = "invalid_api_key"
        else:
            result["review"] = "AI review unavailable due to a temporary error. Please try again later."
            result["error_type"] = "api_error"
        
        result["status"] = "error"
        result["error_message"] = error_msg
    
    return result

def generate_ai_review(
    readme_content: str,
    features: dict,
    prediction: int,
    probability: float,
    api_key: str = None,
    timeout_seconds: int = 30
) -> dict:
    result = _generate_gemini_review(readme_content, features, prediction, probability, api_key, timeout_seconds)
    
    if result["error_type"] == "rate_limit":
        logger.warning("Gemini rate limited for {}, falling back to NVIDIA NIM".format(features.get("full_name", "unknown")))
        nvidia_result = _generate_nvidia_review(readme_content, features, prediction, probability)
        
        if nvidia_result["status"] == "success":
            nvidia_result["review"] = "[Fallback: NVIDIA NIM]" + chr(10) + chr(10) + nvidia_result["review"]
            return nvidia_result
        else:
            result["review"] += " NVIDIA fallback also failed."
            return result
    
    return result


def format_ai_review_for_display(ai_result: dict) -> str:
    if ai_result["status"] == "success":
        return "**AI Review:**" + chr(10) + chr(10) + ai_result["review"]
    elif ai_result["status"] == "skipped":
        return "**AI Review Unavailable:** " + ai_result["review"]
    else:
        return "**AI Review Error:** " + ai_result["review"]