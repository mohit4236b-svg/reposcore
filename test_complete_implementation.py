"""
Comprehensive test for Python code analysis integration.
Tests Stage 1, 2, 3, and 4 functionality.
"""
import sys
import os
import tempfile
import shutil
sys.path.insert(0, '.')

from reposcore_utils import clone_repo_bounded, extract_code_metrics
from ai_review import generate_ai_review, CODE_ANALYSIS_AVAILABLE

def test_stage_1_clone_bounds():
    """Test Stage 1: Clone with bounds and cleanup."""
    print("=== Stage 1: Clone Bounds Test ===")
    
    # Test small repo (should succeed)
    result = clone_repo_bounded("octocat/Spoon-Knife", 100)
    if result is None:
        print("FAIL: Small repo should clone successfully")
        return False
    print(f"SUCCESS: Small repo cloned to {result}")
    # Cleanup handled by function on failure, success means caller cleans up
    
    # Test large repo (should fail)
    result = clone_repo_bounded("octocat/Spoon-Knife", 60*1024)  # 60MB > 50MB cap
    if result is not None:
        print("FAIL: Large repo should return None")
        return False
    print("SUCCESS: Large repo correctly returned None")
    
    # Test nonexistent repo (should fail)
    result = clone_repo_bounded("nonexistent/repo-12345", 100)
    if result is not None:
        print("FAIL: Nonexistent repo should return None")
        return False
    print("SUCCESS: Nonexistent repo correctly returned None")
    
    return True

def test_stage_2_extract_metrics():
    """Test Stage 2: Extract code metrics."""
    print("\n=== Stage 2: Extract Metrics Test ===")
    
    # Create a temporary test repo with Python files
    test_dir = tempfile.mkdtemp(prefix="metrics_test_")
    try:
        # Create test Python file
        py_file = os.path.join(test_dir, "sample.py")
        with open(py_file, "w") as f:
            f.write('def test_func():\n    return 42\n')
        
        metrics = extract_code_metrics(test_dir)
        
        if metrics is None:
            print("FAIL: extract_code_metrics returned None")
            return False
        
        if not isinstance(metrics, dict):
            print("FAIL: Metrics should be a dictionary")
            return False
        
        required_keys = ['file_count', 'avg_complexity', 'total_loc']
        for key in required_keys:
            if key not in metrics:
                print(f"FAIL: Missing key {key} in metrics")
                return False
        
        print(f"SUCCESS: Extracted metrics: {metrics}")
        return True
        
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)

def test_stage_3_integration():
    """Test Stage 3: Integration with AI review."""
    print("\n=== Stage 3: AI Review Integration Test ===")
    
    # Test non-Python repo (should skip code analysis)
    non_python_features = {
        "full_name": "octocat/Spoon-Knife",
        "primary_language": "HTML",
        "size": 100,
        "stars": 10,
        "forks": 5,
        "open_issues": 2,
        "repo_age_days": 365,
        "last_commit_days": 30,
        "total_contributors": 3,
        "topics": ["demo"],
        "has_ci": False,
        "has_tests": False,
        "has_license": True,
        "has_contributing": False,
        "has_code_of_conduct": False,
        "readme_size": 1000
    }
    
    result = generate_ai_review(
        readme_content="Test README",
        features=non_python_features,
        prediction=1,
        probability=0.8
    )
    
    # Should succeed (or skip if no API key) but not crash
    if 'status' not in result:
        print("FAIL: Result missing status field")
        return False
    
    print(f"SUCCESS: Non-Python repo handled correctly (status: {result.get('status')})")
    
    # Test Python repo (should attempt code analysis)
    python_features = non_python_features.copy()
    python_features["primary_language"] = "Python"
    
    result = generate_ai_review(
        readme_content="Test README for Python repo",
        features=python_features,
        prediction=1,
        probability=0.75
    )
    
    if 'status' not in result:
        print("FAIL: Result missing status field for Python repo")
        return False
    
    print(f"SUCCESS: Python repo handled correctly (status: {result.get('status')})")
    return True

def test_stage_4_fallback():
    """Test Stage 4: Fallback behavior."""
    print("\n=== Stage 4: Fallback Behavior Test ===")
    
    # Test with missing/invalid features (should fallback gracefully)
    minimal_features = {
        "full_name": "test/repo",
        "primary_language": "Python"
    }
    
    try:
        result = generate_ai_review(
            readme_content="Minimal test",
            features=minimal_features,
            prediction=0,
            probability=0.3
        )
        
        # Should handle gracefully without crashing
        if 'status' not in result:
            print("FAIL: Minimal features test failed - no status in result")
            return False
        
        print(f"SUCCESS: Minimal features handled gracefully (status: {result.get('status')})")
        return True
        
    except Exception as e:
        print(f"FAIL: Exception with minimal features: {e}")
        return False

def main():
    print("Running comprehensive Python code analysis tests...\n")
    
    tests = [
        ("Stage 1: Clone Bounds", test_stage_1_clone_bounds),
        ("Stage 2: Extract Metrics", test_stage_2_extract_metrics),
        ("Stage 3: AI Review Integration", test_stage_3_integration),
        ("Stage 4: Fallback Behavior", test_stage_4_fallback),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "="*50)
    print("TEST RESULTS SUMMARY")
    print("="*50)
    
    all_passed = True
    for name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{name}: {status}")
        if not success:
            all_passed = False
    
    print("="*50)
    
    if all_passed:
        print("ALL TESTS PASSED!")
        print("\nImplementation Summary:")
        print("- Stage 1: clone_repo_bounded() - WORKING")
        print("- Stage 2: extract_code_metrics() - WORKING")
        print("- Stage 3: AI review integration - WORKING")
        print("- Stage 4: Fallback behavior - WORKING")
        print(f"- CODE_ANALYSIS_AVAILABLE flag: {CODE_ANALYSIS_AVAILABLE}")
        return 0
    else:
        print("SOME TESTS FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(main())