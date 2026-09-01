import sys
sys.path.insert(0, '.')

from ai_review import clone_repo_bounded
import os
import tempfile
import shutil

def test_case(name, repo_full_name, size_kb, expect_success):
    print(f'\\n=== {name} ===')
    print(f'Repo: {repo_full_name}, size_kb: {size_kb}')
    result = clone_repo_bounded(repo_full_name, size_kb)
    if expect_success:
        if result is None:
            print('FAIL: Expected success but got None')
            return False
        else:
            print(f'SUCCESS: Got path {result}')
            # Check if directory exists
            if os.path.isdir(result):
                print('Directory exists.')
                # List a few files inside
                try:
                    entries = os.listdir(result)
                    print(f'Entries (first 5): {entries[:5]}')
                    # Check for .git directory
                    if '.git' in entries:
                        print('.git directory present, indicating a clone.')
                    else:
                        print('Warning: .git directory not found; maybe clone failed?')
                except Exception as e:
                    print(f'Error listing directory: {e}')
                # Cleanup
                try:
                    shutil.rmtree(result, ignore_errors=True)
                    print('Cleaned up.')
                except Exception as e:
                    print(f'Error during cleanup: {e}')
                # Verify gone
                if os.path.isdir(result):
                    print('WARNING: Directory still exists after cleanup!')
                else:
                    print('Confirmed removed.')
            else:
                print('WARNING: Path does not exist as directory.')
                return False
            return True
    else:
        if result is None:
            print('SUCCESS: Got None as expected.')
            return True
        else:
            print(f'FAIL: Expected None but got {result}')
            # Cleanup if needed
            if os.path.isdir(result):
                try:
                    shutil.rmtree(result, ignore_errors=True)
                    print('Cleaned up unexpected directory.')
                except Exception as e:
                    print(f'Error cleaning up: {e}')
            return False

# Case 1: small real Python repo (we'll use a known small repo)
# Use "octocat/Spoon-Knife" (tiny)
success1 = test_case('Small repo', 'octocat/Spoon-Knife', 100, True)

# Case 2: large repo (size exceeds cap) - we can use any repo but pass large size
success2 = test_case('Large size', 'octocat/Spoon-Knife', 60*1024, False)  # 60 MB > 50 MB

# Case 3: invalid/nonexistent repo
success3 = test_case('Nonexistent repo', 'this-repo-definitely-does-not-exist-12345/foo', 100, False)

print('\\n=== Summary ===')
print(f'Small repo test: {'PASS' if success1 else 'FAIL'}')
print(f'Large size test: {'PASS' if success2 else 'FAIL'}')
print(f'Nonexistent test: {'PASS' if success3 else 'FAIL'}')

if success1 and success2 and success3:
    print('All tests passed.')
else:
    print('Some tests failed.')
    sys.exit(1)
