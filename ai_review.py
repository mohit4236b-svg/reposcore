import os
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning('google-genai not installed; AI review will be unavailable')

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


def generate_ai_review(
    readme_content: str,
    features: dict,
    prediction: int,
    probability: float,
    api_key: str = None,
    timeout_seconds: int = 30
) -> dict:
    result = {
        'review': '',
        'status': 'error',
        'error_type': None,
        'error_message': None
    }
    
    if not GENAI_AVAILABLE:
        result['review'] = 'AI review unavailable: google-genai package not installed.'
        result['status'] = 'skipped'
        result['error_type'] = 'missing_dependency'
        result['error_message'] = 'google-genai package not installed. Run pip install google-genai==1.8.0'
        logger.warning(result['error_message'])
        return result
    
    api_key = api_key or os.getenv('GEMINI_API_KEY')
    if not api_key or api_key.strip() == '' or api_key == 'your_gemini_api_key_here':
        result['review'] = 'AI review unavailable: GEMINI_API_KEY not configured. Set it in your .env file or environment.'
        result['status'] = 'skipped'
        result['error_type'] = 'missing_api_key'
        result['error_message'] = 'GEMINI_API_KEY environment variable not set or using placeholder value'
        logger.warning(result['error_message'])
        return result
    
    if not readme_content or not readme_content.strip() or len(readme_content.strip()) < 50:
        result['review'] = 'AI review unavailable: README is too short or empty to analyze meaningfully.'
        result['status'] = 'skipped'
        result['error_type'] = 'empty_readme'
        result['error_message'] = f'README content too short ({len(readme_content.strip()) if readme_content else 0} chars)'
        logger.warning(result['error_message'])
        return result
    
    try:
        client = genai.Client(api_key=api_key)
        quality_label = 'high quality' if prediction == 1 else 'needs improvement'
        max_readme_chars = 8000
        readme_for_prompt = readme_content[:max_readme_chars]
        if len(readme_content) > max_readme_chars:
            readme_for_prompt += f'\n\n[README truncated from {len(readme_content)} to {max_readme_chars} characters]'
        
        prompt = f"""You are an experienced software engineer reviewing a GitHub repository. Provide a technical assessment of the repository based on its README and observable metrics. Do not use promotional or marketing language. Do not mention the model, prediction, or confidence.

Repository: {features.get('full_name', 'Unknown')}
Stars: {features.get('stars', 0)} | Forks: {features.get('forks', 0)} | Open Issues: {features.get('open_issues', 0)}
Age: {features.get('repo_age_days', 0)} days | Days since last commit: {features.get('days_since_last_commit', 0)}
Contributors: {features.get('total_contributors', 'Unknown')}
Topics: {', '.join(features.get('topics', [])) or 'None'}
Has CI: {features.get('has_ci', False)} | Has Tests: {features.get('has_tests', False)}
README length: {features.get('readme_size', 0)} characters

README Content:
{readme_for_prompt}

Write 3-5 sentences covering:
1. What the README describes (purpose, key features, tech stack, architecture) -- reference specific details
2. How the observable metrics (stars, activity, contributors, CI/tests presence) align with or contradict the README's claims
3. 1-2 concrete, actionable gaps to address (e.g., "Add a CONTRIBUTING.md", "Include installation steps in the README", "Set up CI with a badge", "Add code coverage reporting", "Document the API endpoints")
4. One specific strength and why it lowers risk for adopters

Each sentence must reference THIS repo's actual content. Be direct, specific, and engineering-focused."""
        
        generation_config = types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=1000,
        )
        
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=prompt,
            config=generation_config,
        )
        
        if response and response.text:
            result['review'] = response.text.strip()
            result['status'] = 'success'
            logger.info(f'AI review generated successfully for {features.get("full_name", "unknown")}')
        else:
            result['review'] = 'AI review unavailable: Empty response from API.'
            result['status'] = 'error'
            result['error_type'] = 'empty_response'
            result['error_message'] = 'Gemini API returned empty response'
            logger.warning(result['error_message'])
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Gemini API error for {features.get("full_name", "unknown")}: {error_msg}')
        
        if 'quota' in error_msg.lower() or 'rate limit' in error_msg.lower() or '429' in error_msg:
            result['review'] = 'AI review unavailable: Rate limit exceeded. Please try again later.'
            result['error_type'] = 'rate_limit'
        elif 'timeout' in error_msg.lower() or 'deadline' in error_msg.lower():
            result['review'] = 'AI review unavailable: Request timed out. Please try again.'
            result['error_type'] = 'timeout'
        elif 'api key' in error_msg.lower() or 'invalid' in error_msg.lower() or '401' in error_msg or '403' in error_msg:
            result['review'] = 'AI review unavailable: Invalid or missing API key.'
            result['error_type'] = 'invalid_api_key'
        else:
            result['review'] = 'AI review unavailable due to a temporary error. Please try again later.'
            result['error_type'] = 'api_error'
        
        result['status'] = 'error'
        result['error_message'] = error_msg
    
    return result


def format_ai_review_for_display(ai_result: dict) -> str:
    if ai_result['status'] == 'success':
        return f'**AI Review:**\n\n{ai_result["review"]}'
    elif ai_result['status'] == 'skipped':
        return f'**AI Review Unavailable:** {ai_result["review"]}'
    else:
        return f'**AI Review Error:** {ai_result["review"]}'
