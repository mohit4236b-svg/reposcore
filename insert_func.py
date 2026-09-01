import sys

def insert_function(source_path, function_code, marker="/* INSERT_EXTRACT_CODE_METRICS_HERE */"):
    with open(source_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # Find line with marker
    insert_idx = None
    for i, line in enumerate(lines):
        if marker in line:
            insert_idx = i
            break
    if insert_idx is None:
        # If marker not found, insert before the line that starts with "# Prompt for AI review"
        for i, line in enumerate(lines):
            if line.strip().startswith("# Prompt for AI review"):
                insert_idx = i
                break
    if insert_idx is None:
        raise ValueError("Could not find insertion point")
    # Insert function lines (each line should end with newline)
    for i, line in enumerate(function_code):
        lines.insert(insert_idx + i, line + "\n")
    with open(source_path, "w", encoding="utf-8") as f:
        f.writelines(lines)

if __name__ == "__main__":
    source = r"C:\Users\ASUS\OneDrive\Documents\GitHub\reposcore\ai_review.py"
    func = '''def extract_code_metrics(repo_path: str) -> Optional[dict]:
    \"\"\"Extract code metrics from a cloned repository.
    Returns dict with metrics or None on failure/timeout/no .py files.
    \"\"\"
    import os
    import radon.complexity
    import radon.raw
    from multiprocessing.pool import ThreadPool
    import stat
    import logging
    logger = logging.getLogger(__name__)
    # Timeout in seconds
    TIMEOUT = 15
    # Directories to skip
    SKIP_DIRS = {'.git', 'venv', '__pycache__', 'node_modules', '.kilo'}
    def _is_probably_worktree(path):
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
    def _should_skip(dirpath):
        parts = os.path.normpath(dirpath).split(os.sep)
        for p in parts:
            if p in SKIP_DIRS:
                return True
            if _is_probably_worktree(os.path.join(*parts[:parts.index(p)+1])):
                return True
        return False
    # Collect .py files
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
            "high_complexity_functions": [{"name": n, "complexity": c} for n, c in top5]
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
    return result'''
    insert_function(source, func.splitlines())
    print("Function inserted successfully.")
