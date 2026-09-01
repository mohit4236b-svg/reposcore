$path = "C:\Users\ASUS\OneDrive\Documents\GitHub\reposcore\ai_review.py"
$lines = Get-Content $path
$startIdx = 19  # zero-indexed line of def clone_repo_bounded
$endIdx = 65    # zero-indexed line of # Prompt for AI review
# Build new function lines
$newFunc = @(
    'def clone_repo_bounded(repo_full_name: str, size_kb: int) -> Optional[str]:',
    '    """Clone a repo with size and timeout limits.',
    '    Returns path to temp dir on success, None on failure.',
    '    """',
    '    import tempfile',
    '    import subprocess',
    '    import os',
    '    import shutil',
    '    import stat',
    '    def _remove_readonly(func, path, exc_info):',
    '        os.chmod(path, stat.S_IWRITE)',
    '        func(path)',
    '    if size_kb > MAX_REPO_SIZE_KB:',
    '        logger.info(f"Repo {repo_full_name} size {size_kb}KB exceeds cap {MAX_REPO_SIZE_KB}KB")',
    '        return None',
    '    try:',
    '        tmpdir = tempfile.mkdtemp(prefix=f"reposcore_{repo_full_name.replace(\'/\', \'_\")}_")',
    '        repo_url = f"https://github.com/{repo_full_name}.git"',
    '        # shallow clone with depth 1',
    '        # timeout 30 seconds',
    '        result = subprocess.run(',
    '            ["git", "clone", "--depth", "1", repo_url, tmpdir],',
    '            stdout=subprocess.PIPE,',
    '            stderr=subprocess.PIVE,',
    '            timeout=30',
    '        )',
    '        if result.returncode != 0:',
    '            logger.warning(f"Git clone failed for {repo_full_name}: {result.stderr.decode()}")',
    '            # cleanup',
    '            shutil.rmtree(tmpdir, onerror=_remove_readonly)',
    '            return None',
    '    except subprocess.TimeoutExpired:',
    '        logger.warning(f"Git clone timeout for {repo_full_name}")',
    '        # cleanup if tmpdir was created',
    '        if ''tmpdir'' in locals():',
    '            shutil.rmtree(tmpdir, onerror=_remove_readonly)',
    '        return None',
    '    except Exception as e:',
    '        logger.warning(f"Error cloning {repo_full_name}: {e}")',
    '        if ''tmpdir'' in locals():',
    '            shutil.rmtree(tmpdir, onerror=_remove_readonly)',
    '        return None'
)
# Note: there is a typo above: subPIVE should be subPIPE. Fix later.
# Let's write a helper to build the lines correctly to avoid typos.
# We'll instead build the lines using a more reliable method: we can copy the original function lines and modify them.
# But given time, let's just output the script and see if it works; we can fix typos if any.
# Actually we can avoid building the whole function by editing the existing lines as previously intended but using the correct startIdx and endIdx.
# Let's do that instead: we will keep the existing lines and modify the specific lines we need.
