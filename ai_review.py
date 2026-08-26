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

# Prompt for AI review - request structured comprehensive summary
P = """You are a critical GitHub repository reviewer. Provide a comprehensive, structured review covering all four sections below. Use clear section headers and bullet points where appropriate.

**Repository:** {full_name}
**Stats:** ★{stars} | Forks:{forks} | Open Issues:{open_issues}
**Age:** {repo_age_days} days | Last commit: {last_commit_days} days ago
**Contributors:** {total_contributors}
**Topics:** {topics}
**Primary Language:** {primary_language}
**CI:** {has_ci} | **Tests:** {has_tests} | **License:** {has_license}
**README length:** {readme_size} characters

**README content:**
{readme_for_prompt}

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

Do NOT mention models, predictions, confidence scores, or use placeholder text. Output ONLY the structured review.
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
    """Generate review using NVIDIA API with robust error handling and retry logic."""
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
            c = OpenAI(
                base_url="https://integrate.api.nvidia.com/v1",
                api_key=k,
                default_headers={"Authorization": f"Bearer {k}"}
            )
            resp = c.chat.completions.create(
                model="nvidia/nemotron-3-ultra-550b-a55b",
                messages=[{"role": "user", "content": pt}],
                temperature=0.3,
                max_tokens=10000
            )
            if resp.choices[0].message.content:
                cleaned_review = clean_ai_response(resp.choices[0].message.content.strip())
                logger.info(f"NVIDIA review successful on attempt {attempt + 1}")
                return {"review": cleaned_review, "status": "success", "provider": "nvidia"}
            else:
                logger.warning("NVIDIA review received empty response")
                if attempt == max_retries:
                    return {"review": "AI review unavailable: Empty response from NVIDIA API.", "status": "error", "provider": "nvidia"}
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
                # Provide user-friendly error messages based on error type
                if "timeout" in error_str or "connection" in error_str or "network" in error_str:
                    user_msg = "AI review unavailable: Unable to connect to NVIDIA API. Please check network connectivity."
                elif "429" in error_str or "rate limit" in error_str:
                    user_msg = "AI review unavailable: NVIDIA API rate limit exceeded. Please try again later."
                elif "500" in error_str or "502" in error_str or "503" in error_str or "504" in error_str:
                    user_msg = "AI review unavailable: NVIDIA API server error. Please try again later."
                else:
                    user_msg = f"AI review unavailable: {str(e)}"
                return {"review": user_msg, "status": "error", "provider": "nvidia"}
    
    # This point should not be reached due to loop, but just in case
    return {"review": "AI review unavailable: Max retries exceeded for NVIDIA API.", "status": "error", "provider": "nvidia"}

def generate_ai_review(readme_content: str, features: dict, prediction: int, probability: float) -> Dict[str, Any]:
    """Generate AI review using NVIDIA API exclusively."""
    if not readme_content or not readme_content.strip():
        return {"review": "AI review unavailable: README empty.", "status": "error", "provider": "none"}
    
    # Use only NVIDIA API
    nres = N(readme_content, features, prediction, probability)
    return nres

def format_ai_review_for_display(ai_review_result: dict) -> str:
    """Format AI review result for display in frontend - returns plain markdown string."""
    if ai_review_result.get("status") == "success":
        prov = ai_review_result.get("provider", "unknown").upper()
        review = ai_review_result.get('review', '')
        # The AI now returns proper markdown with ## headers and bullet points
        # Just prepend the provider attribution
        return f"**AI Review** (via {prov})\n\n{review}"
    elif ai_review_result.get("status") == "skipped":
        return f"*AI review skipped: {ai_review_result.get('review', '')}*"
    else:
        return f"*AI review unavailable: {ai_review_result.get('review', '')}*"