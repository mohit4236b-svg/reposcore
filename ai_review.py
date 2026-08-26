import os
import time
import logging
from typing import Dict, Any

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from openai import OpenAI
    O=1
except:O=0
try:
    from google import genai
    from google.genai import types
    G=1
except:G=0

# Prompt for AI review - refined to be more explicit and avoid placeholders
P = """You are a critical GitHub repository reviewer. Provide a concise, specific review in 3-5 sentences. Cover:
1) README: purpose, features, technology, architecture
2) How metrics (stars, activity) align with README claims
3) 1-2 actionable gaps for improvement
4) Strengths in README/setup
Do NOT mention models, predictions, confidence, or use placeholder text like "*Sentence 1*". Output ONLY the review paragraph.

Repository: {full_name}
Stats: ★{stars} | Fork:{forks} | Issues:{open_issues}
Age: {repo_age_days} days | Last commit: {last_commit_days} days ago
Contributors: {total_contributors}
Topics: {topics}
Primary Language: {primary_language}
CI: {has_ci} | Tests: {has_tests} | License: {has_license}
README length: {readme_size} characters

README content:
{readme_for_prompt}
"""

def clean_ai_response(text: str) -> str:
    """Clean and sanitize AI response to remove placeholders and ensure proper format."""
    if not text:
        return ""
    
    # Remove common placeholder patterns
    lines = text.split('\n')
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        # Skip empty lines
        if not line:
            continue
        # Skip lines that look like placeholders or instructional text
        if line.startswith('*Sentence') or '(Aiming for' in line or 'placeholder' in line.lower():
            continue
        # Skip lines that are just markdown bullet points without content
        if line.startswith('- ') or line.startswith('* ') and len(line) <= 2:
            continue
        cleaned_lines.append(line)
    
    # Join lines and ensure we have proper sentence spacing
    cleaned_text = ' '.join(cleaned_lines)
    
    # Ensure we have reasonable length (if too short, might be invalid)
    if len(cleaned_text) < 10:
        return text  # Return original if cleaning removed too much
    
    return cleaned_text.strip()

def N(readme_content: str, features: dict, prediction: int, probability: float) -> Dict[str, Any]:
    """Generate review using NVIDIA API with retry logic."""
    if not O:
        logger.warning("NVIDIA review skipped: openai not installed")
        return {"review": "AI review unavailable: openai not installed.", "status": "skipped", "provider": "nvidia"}
    
    k = os.getenv("NVIDIA_API_KEY")
    if not k or k.strip() == "":
        logger.warning("NVIDIA review skipped: NVIDIA_API_KEY not set")
        return {"review": "AI review unavailable: NVIDIA_API_KEY not set.", "status": "skipped", "provider": "nvidia"}
    
    m = 8000
    rp = readme_content[:m]
    if len(readme_content) > m:
        rp += f"\n\n[README truncated from {len(readme_content)} to {m} chars]"
    
    pt = P.format(
        full_name=features.get("full_name", "Unknown"),
        stars=features.get("stars", 0),
        forks=features.get("forks", 0),
        open_issues=features.get("open_issues", 0),
        repo_age_days=features.get("repo_age_days", 0),
        last_commit_days=features.get("last_commit_days", 0),
        total_contributors=features.get("total_contributors", 0),
        topics=", ".join(features.get("topics", [])) if features.get("topics") else "None",
        primary_language=features.get("primary_language", "Unknown"),
        has_ci="Yes" if features.get("has_ci") else "No",
        has_tests="Yes" if features.get("has_tests") else "No",
        has_license="Yes" if features.get("has_license") else "No",
        readme_size=features.get("readme_size", 0),
        readme_for_prompt=rp
    )
    
    # Retry logic for transient errors
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            c = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=k)
            resp = c.chat.completions.create(
                model="meta/llama-3.1-70b-instruct",
                messages=[{"role": "user", "content": pt}],
                temperature=0.3,
                max_tokens=500
            )
            if resp.choices[0].message.content:
                cleaned_review = clean_ai_response(resp.choices[0].message.content.strip())
                logger.info(f"NVIDIA review successful on attempt {attempt + 1}")
                return {"review": cleaned_review, "status": "success", "provider": "nvidia"}
            else:
                logger.warning("NVIDIA review received empty response")
                if attempt == max_retries:
                    return {"review": "AI review unavailable: Empty response.", "status": "error", "provider": "nvidia"}
        except Exception as e:
            logger.error(f"NVIDIA API error on attempt {attempt + 1}: {str(e)}")
            # Check if it's a transient error worth retrying
            error_str = str(e).lower()
            is_transient = any(
                keyword in error_str for keyword in 
                ["timeout", "connection", "network", "429", "500", "502", "503", "504", "rate limit"]
            )
            if attempt < max_retries and is_transient:
                wait_time = 2 ** attempt  # Exponential backoff
                logger.info(f"Retrying NVIDIA call in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                return {"review": f"AI review unavailable: {str(e)}", "status": "error", "provider": "nvidia"}
    
    # This point should not be reached due to loop, but just in case
    return {"review": "AI review unavailable: Max retries exceeded.", "status": "error", "provider": "nvidia"}

def g_(readme_content: str, features: dict, prediction: int, probability: float) -> Dict[str, Any]:
    """Generate review using Gemini API."""
    if not G:
        logger.warning("Gemini review skipped: google-genai not installed")
        return {"review": "AI review unavailable: google-genai not installed.", "status": "skipped", "provider": "gemini"}
    
    k = os.getenv("GEMINI_API_KEY")
    if not k or k.strip() == "":
        logger.warning("Gemini review skipped: GEMINI_API_KEY not set")
        return {"review": "AI review unavailable: GEMINI_API_KEY not set.", "status": "skipped", "provider": "gemini"}
    
    m = 8000
    rp = readme_content[:m]
    if len(readme_content) > m:
        rp += f"\n\n[README truncated from {len(readme_content)} to {m} chars]"
    
    pt = P.format(
        full_name=features.get("full_name", "Unknown"),
        stars=features.get("stars", 0),
        forks=features.get("forks", 0),
        open_issues=features.get("open_issues", 0),
        repo_age_days=features.get("repo_age_days", 0),
        last_commit_days=features.get("last_commit_days", 0),
        total_contributors=features.get("total_contributors", 0),
        topics=", ".join(features.get("topics", [])) if features.get("topics") else "None",
        primary_language=features.get("primary_language", "Unknown"),
        has_ci="Yes" if features.get("has_ci") else "No",
        has_tests="Yes" if features.get("has_tests") else "No",
        has_license="Yes" if features.get("has_license") else "No",
        readme_size=features.get("readme_size", 0),
        readme_for_prompt=rp
    )
    
    try:
        c = genai.Client(api_key=k)
        resp = c.models.generate_content(
            model="gemini-3.6-flash",
            contents=pt,
            config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=500)
        )
        if resp.text:
            cleaned_review = clean_ai_response(resp.text.strip())
            logger.info("Gemini review successful")
            return {"review": cleaned_review, "status": "success", "provider": "gemini"}
        else:
            logger.warning("Gemini review received empty response")
            return {"review": "AI review unavailable: Empty response.", "status": "error", "provider": "gemini"}
    except Exception as e:
        logger.error(f"Gemini API error: {str(e)}")
        return {"review": f"AI review unavailable: {str(e)}", "status": "error", "provider": "gemini"}

def generate_ai_review(readme_content: str, features: dict, prediction: int, probability: float) -> Dict[str, Any]:
    """Generate AI review with fallback control via environment variable."""
    if not readme_content or not readme_content.strip():
        return {"review": "AI review unavailable: README empty.", "status": "error", "provider": "none"}
    
    # Try NVIDIA first
    nres = N(readme_content, features, prediction, probability)
    if nres["status"] == "success":
        return nres
    
    # Check if fallback is enabled via environment variable
    fallback_enabled = os.getenv("FALLBACK_ENABLED", "true").lower() == "true"
    if not fallback_enabled:
        logger.info("Gemini fallback disabled by FALLBACK_ENABLED=false")
        # Return NVIDIA result even if it failed, since fallback is disabled
        nres["provider"] = "nvidia (fallback disabled)"
        return nres
    
    # Try Gemini as fallback
    gres = g_(readme_content, features, prediction, probability)
    if gres["status"] == "success":
        return gres
    
    # Both failed
    error_msg = f"AI review unavailable: Both NVIDIA and Gemini failed.\nNVIDIA: {nres['review']}\nGemini: {gres['review']}"
    logger.error("Both NVIDIA and Gemini failed")
    return {"review": error_msg, "status": "error", "provider": "both_failed"}

def format_ai_review_for_display(ai_review_result: dict) -> str:
    """Format AI review result for display in frontend."""
    if ai_review_result.get("status") == "success":
        prov = ai_review_result.get("provider", "unknown").upper()
        return f"**AI Review** (via {prov}):\n\n{ai_review_result.get('review', '')}"
    elif ai_review_result.get("status") == "skipped":
        return f"*AI review skipped: {ai_review_result.get('review', '')}*"
    else:
        return f"*AI review unavailable: {ai_review_result.get('review', '')}*"