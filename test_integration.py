import sys
import os
sys.path.insert(0, '.')

from ai_review import generate_ai_review
from reposcore_utils import clone_repo_bounded, extract_code_metrics

def test_ai_review_integration():
    print("=== Testing AI Review Integration ===")
    
    # Test case 1: Normal case (should work even if NVIDIA API is not available)
    print("\n--- Test 1: Basic functionality ---")
    try:
        # Minimal test features
        test_features = {
            "full_name": "octocat/Spoon-Knife",
            "stars": 10,
            "forks": 5,
            "open_issues": 2,
            "repo_age_days": 365,
            "last_commit_days": 30,
            "total_contributors": 3,
            "topics": ["example", "demo"],
            "primary_language": "HTML",  # Not Python, so no code analysis
            "has_ci": False,
            "has_tests": False,
            "has_license": True,
            "has_contributing": False,
            "has_code_of_conduct": False,
            "readme_size": 1000,
            "size": 100  # 100 KB
        }
        
        result = generate_ai_review(
            readme_content="This is a test README for demonstration purposes.",
            features=test_features,
            prediction=1,
            probability=0.8
        )
        
        print(f"Result status: {result.get('status')}")
        print(f"Result provider: {result.get('provider')}")
        if result.get('status') == 'success':
            print("SUCCESS: AI review generated successfully")
            # Just check that we got a review - don't validate content as it depends on API
            review_text = result.get('review', '')
            print(f"Review length: {len(review_text)} characters")
        elif result.get('status') == 'skipped':
            print("EXPECTED: AI review skipped (likely due to missing API key)")
        else:
            print(f"WARNING: AI review failed with status: {result.get('status')}")
            
    except Exception as e:
        print(f"ERROR: Exception during AI review generation: {e}")
        return False
    
    # Test case 2: Python repo with code analysis (will fall back to normal if cloning fails or no Python files)
    print("\n--- Test 2: Python repo case ---")
    try:
        python_features = test_features.copy()
        python_features["primary_language"] = "Python"
        python_features["full_name"] = "octocat/Spoon-Knife"  # Still using same repo for consistency
        
        result = generate_ai_review(
            readme_content="This is a test README for a Python project.",
            features=python_features,
            prediction=1,
            probability=0.75
        )
        
        print(f"Result status: {result.get('status')}")
        print(f"Result provider: {result.get('provider')}")
        # For Python repos, we expect either success, skipped, or error - but not a crash
        if result.get('status') in ['success', 'skipped', 'error']:
            print("SUCCESS: AI review handled Python repo case without crashing")
        else:
            print(f"UNEXPECTED status: {result.get('status')}")
            return False
            
    except Exception as e:
        print(f"ERROR: Exception during Python repo AI review: {e}")
        return False
    
    print("\n=== All integration tests completed ===")
    return True

if __name__ == "__main__":
    if test_ai_review_integration():
        print("SUCCESS: Integration tests passed")
        sys.exit(0)
    else:
        print("FAILURE: Integration tests failed")
        sys.exit(1)