#!/usr/bin/env python3
"""Verify Streamlit app renders AI Review section using Playwright."""

import asyncio
import os
from playwright.async_api import async_playwright


async def verify_ai_review_section(repo='pre-commit/pre-commit-hooks'):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        # Navigate to the Streamlit app
        await page.goto('http://localhost:8501/', wait_until='networkidle')
        
        # Wait for Streamlit to fully load
        await page.wait_for_timeout(5000)
        
        # Fill in the repository input
        # Find the text input for repository
        repo_input = page.locator('input[type="text"]').first
        await repo_input.fill(repo)
        print(f"Filled repository input: {repo}")
        
        # Click the Predict Quality button
        predict_button = page.locator('button:has-text("Predict Quality")').first
        await predict_button.click()
        print("Clicked Predict Quality button")
        
        # Wait for the analysis to complete and results to render
        await page.wait_for_timeout(30000)
        
        # Take a screenshot for visual verification
        screenshot_path = os.path.join(os.getcwd(), f'streamlit_ai_review_{repo.replace("/", "_")}.png')
        await page.screenshot(path=screenshot_path, full_page=True)
        print(f"Screenshot saved to {screenshot_path}")
        
        # Get all text content from the page
        page_text = await page.evaluate('document.body.innerText')
        
        # Save page text for inspection
        text_path = os.path.join(os.getcwd(), f'streamlit_page_text_{repo.replace("/", "_")}.txt')
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(page_text)
        print(f"Full page text saved to {text_path}")
        
        # Check for AI Review in visible text
        if 'AI Review' in page_text:
            print("SUCCESS: 'AI Review' found in visible page text")
            # Extract the AI Review section
            idx = page_text.find('AI Review')
            print(f"Context around AI Review: {page_text[max(0,idx-100):idx+2000]}")
        else:
            print("WARNING: 'AI Review' not found in visible page text")
            # Print first 5000 chars for debugging
            print(f"First 5000 chars of page: {page_text[:5000]}")
        
        # Also check for review content indicators
        review_indicators = [
            'pre-commit',
            'hook',
            'README',
            'Python',
            'CI',
            'test',
        ]
        
        for indicator in review_indicators:
            if indicator.lower() in page_text.lower():
                print(f"Found review indicator: '{indicator}'")
        
        await browser.close()
        return 'AI Review' in page_text


if __name__ == '__main__':
    # Test with a mediocre repo
    result = asyncio.run(verify_ai_review_section('pre-commit/pre-commit-hooks'))
    print(f"\nVerification result: {'PASS' if result else 'FAIL'}")