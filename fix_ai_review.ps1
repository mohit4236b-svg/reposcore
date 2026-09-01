$path = "C:\Users\ASUS\OneDrive\Documents\GitHub\reposcore\ai_review.py"
$content = Get-Content -Raw -Path $path
# Split into lines
$lines = $content -split "`r?`n"
# We will rebuild the file.
# We know the original structure up to the point before we added our functions.
# Let's find the line that starts with "def format_ai_review_for_display"
$formatIdx = -1
for ($i=0; $i -lt $lines.Length; $i++) {
    if ($lines[$i].TrimStart().StartsWith("def format_ai_review_for_display")) {
        $formatIdx = $i
        break
    }
}
if ($formatIdx -eq -1) { throw "format_ai_review_for_display not found" }
# We want to keep everything up to the end of that function.
# We need to find the end of the function: look for the next line that starts with "def " or end of file.
$endFuncIdx = -1
for ($i=$formatIdx+1; $i -lt $lines.Length; $i++) {
    if ($lines[$i].TrimStart().StartsWith("def ")) {
        $endFuncIdx = $i
        break
    }
}
if ($endFuncIdx -eq -1) { $endFuncIdx = $lines.Length }
# So the original file is lines[0..$endFuncIdx]
$originalLines = $lines[0..$endFuncIdx]
# Now we want to append our two functions after the original file, but before the prompt line? Actually we inserted extract_code_metrics before the prompt line, and clone_repo_bounded after import stat? We want to have both functions present.
# Let's instead construct the desired final file:
# We'll keep the original file up to the end of format_ai_review_for_display, then add our two functions, then the prompt line and everything after.
# We need to find the prompt line index in the original lines.
$promptIdx = -1
for ($i=0; $i -lt $originalLines.Length; $i++) {
    if ($originalLines[$i].TrimStart().StartsWith("# Prompt for AI review")) {
        $promptIdx = $i
        break
    }
}
if ($promptIdx -eq -1) { throw "Prompt line not found in original lines" }
# We will keep lines[0..$promptIdx] (up to but not including the prompt line)
# Then insert our two functions, then the prompt line and after.
$prePrompt = $originalLines[0..$promptIdx]
$postPrompt = $originalLines[$promptIdx..($originalLines.Length-1)]
# Now define our two functions as strings.
$cloneFunc = @"
def clone_repo_bounded(repo_full_name: str, size_kb: int) -> Optional[str]:
    \"\"\"Clone a repo with size and timeout limits.
    Returns path to temp dir on success, None on failure.
    \"\"\"
    import tempfile
    import subprocess
    import os
    import shutil
    import stat

    def _remove_readonly(func, path, exc_info):
        os.chmod(path, stat.S_IWRITE)
        func(path)

    if size_kb > MAX_REPO_SIZE_KB:
        logger.info(f"Repo {repo_full_name} size {size_kb}KB exceeds cap {MAX_REPO_SIZE_KB}KB")
        return None
    try:
        tmpdir = tempfile.mkdtemp(prefix=f"reposcore_{repo_full_name.replace(\'/\', \'_\")}_")
        repo_url = f"https://github.com/{repo_full_name}.git"
        # shallow clone with depth 1
        # timeout 30 seconds
        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, tmpdir],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30
        )
        if result.returncode != 0:
            logger.warning(f"Git clone failed for {repo_full_name}: {result.stderr.decode()}")
            # cleanup – handle read‑only files on Windows
            shutil.rmtree(tmpdir, onerror=_remove_readonly)
            return None
        return tmpdir
    except subprocess.TimeoutExpired:
        logger.warning(f"Git clone timeout for {repo_full_name}")
        # cleanup if tmpdir was created
        if 'tmpdir' in locals():
            shutil.rmtree(tmpdir, onerror=_remove_readonly)
        return None
    except Exception as e:
        logger.warning(f"Error cloning {repo_full_name}: {e}")
        if 'tmpdir' in locals():
            shutil.rmtree(tmpdir, onerror=_remove_readonly)
        return None
"@
$extractFunc = @"
def extract_code_metrics(repo_path: str) -> Optional[dict]:
    \"\"\"Extract code metrics from a cloned repository.
    Returns dict with metrics or None on failure/timeout/no .py files.
    \"\"\"
    import os
    import radon.complexity
    import radon.raw
    from multiprocessing.pool import ThreadPool
    import logging
    import stat

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

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
        # Modify dirs in-place to skip unwanted directories
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
"
# Now combine
$newLines = $prePrompt
$newLines += $cloneFunc -split "`r?`n"
$newLines += $extractFunc -split "`r?`n"
$newLines += $postPrompt
# Write back
Set-Content -Path $path -Value ($newLines -join "`r`n") -Encoding UTF8
Write-Host "File rewritten successfully."
