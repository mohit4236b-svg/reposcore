"""
Security patches for subprocess calls to prevent injection and path traversal.
This module provides validated wrappers for subprocess operations.
"""

import re
import os
import tempfile
import shutil
import subprocess
from typing import Optional, List, Tuple


REPO_NAME_PATTERN = re.compile(r'^[a-zA-Z0-9_.-]+$')


def validate_repo_name(owner: str, repo: str) -> Tuple[bool, str]:
    """
    Validate repository owner and name to prevent injection attacks.
    
    Args:
        owner: Repository owner/organization name
        repo: Repository name
        
    Returns:
        (is_valid, error_message)
    """
    if not owner or not repo:
        return False, "Owner and repo must not be empty"
    
    if len(owner) > 100 or len(repo) > 100:
        return False, "Owner or repo name too long"
    
    if not REPO_NAME_PATTERN.match(owner):
        return False, "Invalid owner name. Use only alphanumeric, hyphens, underscores, and dots."
    
    if not REPO_NAME_PATTERN.match(repo):
        return False, "Invalid repo name. Use only alphanumeric, hyphens, underscores, and dots."
    
    return True, ""


def validate_clone_path(path: str, temp_base: Optional[str] = None) -> bool:
    """
    Validate that a path is within an allowed directory.
    
    Args:
        path: Path to validate
        temp_base: Optional base directory to restrict to
        
    Returns:
        True if path is safe, False otherwise
    """
    if not path:
        return False
    
    # Prevent path traversal
    if '..' in path or path.startswith('/'):
        return False
    
    # If temp_base is provided, ensure path is within it
    if temp_base:
        try:
            abs_path = os.path.abspath(path)
            abs_base = os.path.abspath(temp_base)
            return abs_path.startswith(abs_base)
        except Exception:
            return False
    
    return True


def safe_git_clone(repo_owner: str, repo_name: str, size_kb: int, 
                   max_size_kb: int = 50 * 1024, timeout: int = 30) -> Optional[str]:
    """
    Safely clone a GitHub repository with validation and bounds checking.
    
    Args:
        repo_owner: Repository owner name
        repo_name: Repository name
        size_kb: Repository size in KB from GitHub API
        max_size_kb: Maximum allowed size in KB (default 50MB)
        timeout: Clone timeout in seconds
        
    Returns:
        Path to cloned repo on success, None on failure
    """
    # Validate input
    is_valid, error = validate_repo_name(repo_owner, repo_name)
    if not is_valid:
        return None
    
    # Size check
    if size_kb > max_size_kb:
        return None
    
    # Create temp directory in a controlled location
    temp_base = tempfile.gettempdir()
    temp_dir = tempfile.mkdtemp(prefix="reposcore_clone_", dir=temp_base)
    
    # Double-check temp_dir is valid
    if not validate_clone_path(temp_dir, temp_base):
        return None
    
    full_name = f"{repo_owner}/{repo_name}"
    
    try:
        # Build clone URL (no user input in shell)
        clone_url = f"https://github.com/{full_name}.git"
        
        result = subprocess.run(
            ["git", "clone", "--depth", "1", clone_url, temp_dir],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            return temp_dir
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None
            
    except subprocess.TimeoutExpired:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None


def safe_subprocess_run(command: List[str], cwd: str, timeout: int = 60) -> Optional[str]:
    """
    Safely run a subprocess command with path validation.
    
    Args:
        command: Command and arguments as list
        cwd: Working directory for command
        timeout: Command timeout in seconds
        
    Returns:
        stdout on success, None on failure
    """
    # Validate cwd is within allowed paths
    if not validate_clone_path(cwd, tempfile.gettempdir()):
        return None
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd
        )
        
        if result.returncode in (0, 1):  # Some tools return 1 for "issues found"
            return result.stdout
        return None
    except Exception:
        return None
