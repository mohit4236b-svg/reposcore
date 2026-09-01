import sys
import os
import time
import shutil
import tempfile
import subprocess
import stat
import radon.complexity
import radon.raw
from multiprocessing.pool import ThreadPool
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- clone_repo_bounded (as intended) ----
MAX_REPO_SIZE_KB = 50 * 1024  # 50 MB

def clone_repo_bounded(repo_full_name: str, size_kb: int):
    """Clone a repo with size and timeout limits.
    Returns path to temp dir on success, None on failure.
    """
    def _remove_readonly(func, path, exc_info):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    if size_kb > MAX_REPO_SIZE_KB:
        logger.info(f"Repo {repo_full_name} size {size_kb}KB exceeds cap {MAX_REPO_SIZE_KB}KB")
        return None
    try:
        tmpdir = tempfile.mkdtemp(prefix=f"reposcore_{repo_full_name.replace(\'/\', \'_\")}_")
        repo_url = f"https://github.com/{repo_full_name}.git"
        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, tmpdir],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30
        )
        if result.returncode != 0:
            logger.warning(f"Git clone failed for {repo_full_name}: {result.stderr.decode()}")
            shutil.rmtree(tmpdir, onerror=_remove_readonly)
            return None
        return tmpdir
    except subprocess.TimeoutExpired:
        logger.warning(f"Git clone timeout for {repo_full_name}")
        if 'tmpdir' in locals():
            shutil.rmtree(tmpdir, onerror=_remove_readonly)
        return None
    except Exception as e:
        logger.warning(f"Error cloning {repo_full_name}: {e}")
        if 'tmpdir' in locals():
            shutil.rmtree(tmpdir, onerror=_remove_readonly)
        return None

# ---- extract_code_metrics (as intended) ----
def extract_code_metrics(repo_path: str):
    """Extract code metrics from a cloned repository.
    Returns dict with metrics or None on failure/timeout/no .py files.
    """
    TIMEOUT = 15
    SKIP_DIRS = {'.git', 'venv', '__pycache__', 'node_modules', '.kilo'}

    def _is_probably_worktree(path: str) -> bool:
        gitfile = os.path.join(path, '.git')
        if os.path.isfile(gitfile):
            try:
                with open(gitfile, 'r') as f:
                    line = f.read().strip()
                    if line.startswith('gitdir:'):
                        return True
            except Exception:
                pass
        return False

    def _should_skip(dirpath: str) -> bool:
        parts = os.path.normpath(dirpath).split(os.sep)
        for p in parts:
            if p in SKIP_DIRS:
                return True
            if _is_probably_worktree(os.path.join(*parts[:parts.index(p)+1])):
                return True
        return False

    py_files = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not _should_skip(os.path.join(root, d))]
        for f in files:
            if f.endswith('.py'):
                py_files.append(os.path.join(root, f))
    if not py_files:
        return None

    def _run_extraction():
        total_loc = 0
        complexity_sum = 0
        function_count = 0
        max_complexity = 0
        high_complexity = []  # list of (name, complexity)
        for py_file in py_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
            except Exception:
                continue
            raw = radon.raw.analyze(content)
            total_loc += raw.loc
            try:
                blocks = radon.complexity.cc_visit(content)
            except Exception:
                blocks = []
            for block in blocks:
                comp = block.complexity
                complexity_sum += comp
                function_count += 1
                if comp > max_complexity:
                    max_complexity = comp
                high_complexity.append((block.name, comp))
        high_complexity.sort(key=lambda x: x[1], reverse=True)
        top5 = high_complexity[:5]
        avg_complexity = complexity_sum / function_count if function_count > 0 else 0.0
        return {
            "file_count": len(py_files),
            "total_loc": total_loc,
            "avg_complexity": round(avg_complexity, 2),
            "max_complexity": max_complexity,
            "high_complexity_functions": [
                {"name": n, "complexity": c} for n, c in top5
            ]
        }

    pool = ThreadPool(processes=1)
    async_result = pool.apply_async(_run_extraction)
    try:
        result = async_result.get(timeout=TIMEOUT)
    except Exception as e:
        logger.warning(f"Code metrics extraction timed out after {TIMEOUT}s: {e}")
        result = None
    finally:
        pool.close()
        pool.join()
    return result

# ---- Test 1: extract_code_metrics on local reposcore repo ----
print("=== Testing extract_code_metrics on local reposcore repo ===")
start = time.time()
local_metrics = extract_code_metrics(".")
local_elapsed = time.time() - start
print(f"Elapsed time: {local_elapsed:.2f} seconds")
print("Metrics:")
import json
print(json.dumps(local_metrics, indent=2))

# ---- Test 2: clone_repo_bounded on a small Python repo, then extract ----
print("\n=== Testing clone_repo_bounded + extract_code_metrics on pallets/flask ===")
# We will guess size_kb = 20000 (20 MB) ; if too low, clone_repo_bounded will return None due to size check.
# We'll first try with a high guess, but the function will reject if size_kb > MAX_REPO_SIZE_KB (50 MB).
# So we need to pick a size_kb <= 50000. We'll try 30000.
repo = "pallets/flask"
size_guess = 30000  # KB
start = time.time()
tmpdir = clone_repo_bounded(repo, size_guess)
clone_elapsed = time.time() - start
if tmpdir is None:
    print(f"clone_repo_bounded returned None (size_guess={size_guess} KB). Trying with lower size guess?")
    # Maybe the repo is larger than 30 MB? Let's try 10 MB.
    size_guess2 = 10000
    start = time.time()
    tmpdir = clone_repo_bounded(repo, size_guess2)
    clone_elapsed = time.time() - start
    if tmpdir is None:
        print(f"clone_repo_bounded returned None again with size_guess={size_guess2} KB. Skipping clone test.")
        clone_metrics = None
        total_time = None
    else:
        print(f"Clone succeeded with size_guess={size_guess2} KB.")
        # proceed
else:
    print(f"Clone succeeded, path: {tmpdir}, clone time: {clone_elapsed:.2f} seconds")
    # extract metrics
    start2 = time.time()
    clone_metrics = extract_code_metrics(tmpdir)
    extract_elapsed = time.time() - start2
    total_time = clone_elapsed + extract_elapsed
    print(f"Extract time: {extract_elapsed:.2f} seconds")
    print(f"Total time: {total_time:.2f} seconds")
    print("Metrics:")
    print(json.dumps(clone_metrics, indent=2))
    # Cleanup
    try:
        shutil.rmtree(tmpdir, ignore_errors=True)
        print("Cleaned up temp directory.")
    except Exception as e:
        print(f"Error cleaning up: {e}")

# ---- Test 3: empty directory ----
print("\n=== Testing extract_code_metrics on empty directory ===")
empty_dir = os.path.join(os.getenv('TEMP'), "empty_test_dir")
if not os.path.exists(empty_dir):
    os.makedirs(empty_dir)
start = time.time()
empty_metrics = extract_code_metrics(empty_dir)
empty_elapsed = time.time() - start
print(f"Elapsed time: {empty_elapsed:.2f} seconds")
print(f"Metrics: {empty_metrics}")
# Cleanup
shutil.rmtree(empty_dir, ignore_errors=True)
print("Cleaned up empty directory.")

# ---- Final leftover check ----
print("\n=== Final leftover check ===")
temp_dir = os.getenv('TEMP')
leftover = [d for d in os.listdir(temp_dir) if d.startswith('reposcore_')]
if leftover:
    print(f"Leftover reposcore_* directories: {leftover}")
    for d in leftover:
        path = os.path.join(temp_dir, d)
        try:
            shutil.rmtree(path, ignore_errors=True)
            print(f"Removed {path}")
        except Exception as e:
            print(f"Error removing {path}: {e}")
else:
    print("No leftover reposcore_* directories.")

print("\n=== Done ===")
