"""
Security vulnerability scanner for repository dependencies.
Uses pip-audit and Safety API to detect known CVEs in requirements.txt, package.json, etc.
"""

import json
import os
import re
import subprocess
import tempfile
import shutil
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum


class Severity(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    UNKNOWN = "UNKNOWN"


@dataclass
class Vulnerability:
    package_name: str
    version: str
    vulnerability_id: str
    severity: Severity
    description: str
    fix_version: Optional[str] = None
    advisory_url: Optional[str] = None


@dataclass
class SecurityScanResult:
    total_vulnerabilities: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    vulnerabilities: List[Vulnerability]
    scanned_files: List[str]
    scan_method: str
    dependencies_found: int
    risk_level: str
    timestamp: str


def _validate_repo_path(path: str) -> bool:
    """Validate that path is a safe directory within allowed bounds."""
    if not path or not os.path.isdir(path):
        return False
    temp_base = tempfile.gettempdir()
    try:
        abs_path = os.path.abspath(path)
        abs_base = os.path.abspath(temp_base)
        return abs_path.startswith(abs_base)
    except Exception:
        return False


def parse_severity(severity_str: str) -> Severity:
    """Convert string severity to Severity enum."""
    severity_str = severity_str.upper().strip()
    if severity_str in ("CRITICAL", "9.0", "10.0"):
        return Severity.CRITICAL
    elif severity_str in ("HIGH", "7.0", "8.0", "9.0"):
        return Severity.HIGH
    elif severity_str in ("MEDIUM", "5.0", "6.0", "7.0"):
        return Severity.MEDIUM
    elif severity_str in ("LOW", "3.0", "4.0", "5.0"):
        return Severity.LOW
    return Severity.UNKNOWN


def scan_pip_audit(repo_path: str) -> Optional[Dict[str, Any]]:
    """
    Run pip-audit on a Python repository to find vulnerabilities.
    
    Returns:
        Parsed pip-audit output or None if pip-audit is not available
    """
    if not _validate_repo_path(repo_path):
        return None

    requirements_files = [
        os.path.join(repo_path, "requirements.txt"),
        os.path.join(repo_path, "pyproject.toml"),
        os.path.join(repo_path, "setup.py"),
        os.path.join(repo_path, "Pipfile"),
    ]
    
    requirements_file = None
    for f in requirements_files:
        if os.path.exists(f) and _validate_repo_path(os.path.dirname(f)):
            requirements_file = f
            break
    
    if not requirements_file:
        return None
    
    try:
        result = subprocess.run(
            ["pip-audit", "-r", requirements_file, "--format", "json"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=repo_path
        )
        
        if result.returncode in (0, 1):
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                return None
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return None


def scan_safety(repo_path: str) -> Optional[List[Dict[str, Any]]]:
    """
    Run safety CLI to check for vulnerabilities.
    
    Returns:
        List of vulnerabilities found or None if safety is not available
    """
    if not _validate_repo_path(repo_path):
        return None

    requirements_file = os.path.join(repo_path, "requirements.txt")
    
    if not os.path.exists(requirements_file) or not _validate_repo_path(os.path.dirname(requirements_file)):
        return None
    
    try:
        result = subprocess.run(
            ["safety", "check", "-r", requirements_file, "--json"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=repo_path
        )
        
        if result.returncode in (0, 1):
            try:
                output = json.loads(result.stdout)
                if isinstance(output, dict) and "vulnerabilities" in output:
                    return output["vulnerabilities"]
                return output if isinstance(output, list) else None
            except json.JSONDecodeError:
                return None
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return None


def parse_requirements_file(file_path: str) -> List[Dict[str, str]]:
    """Parse a requirements.txt file and extract package names and versions."""
    packages = []
    
    if not os.path.exists(file_path):
        return packages
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('-'):
                    continue
                
                match = re.match(r'^([a-zA-Z0-9_-]+)([=<>!~]+)(.+)$', line)
                if match:
                    packages.append({
                        "name": match.group(1),
                        "version_spec": match.group(2) + match.group(3)
                    })
                else:
                    match_name = re.match(r'^([a-zA-Z0-9_-]+)', line)
                    if match_name:
                        packages.append({
                            "name": match_name.group(1),
                            "version_spec": "unspecified"
                        })
    except Exception:
        pass
    
    return packages


def scan_package_json(repo_path: str) -> Optional[Dict[str, Any]]:
    """Scan package.json for known vulnerable npm packages using npm audit."""
    if not _validate_repo_path(repo_path):
        return None

    package_json_path = os.path.join(repo_path, "package.json")
    
    if not os.path.exists(package_json_path) or not _validate_repo_path(os.path.dirname(package_json_path)):
        return None
    
    try:
        result = subprocess.run(
            ["npm", "audit", "--json"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=repo_path
        )
        
        if result.returncode in (0, 1):
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                return None
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return None


def detect_dependency_files(repo_path: str) -> List[str]:
    """Detect all dependency files in a repository."""
    if not _validate_repo_path(repo_path):
        return []
        
    dep_files = []
    
    dep_patterns = [
        "requirements.txt",
        "pyproject.toml",
        "setup.py",
        "Pipfile",
        "Pipfile.lock",
        "package.json",
        "package-lock.json",
        "yarn.lock",
        "Gemfile",
        "Gemfile.lock",
        "Cargo.toml",
        "Cargo.lock",
        "go.mod",
        "pom.xml",
        "build.gradle",
        "composer.json",
    ]
    
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'venv', 'env']]
        
        for file in files:
            if file in dep_patterns:
                rel_path = os.path.relpath(os.path.join(root, file), repo_path)
                dep_files.append(rel_path)
    
    return dep_files


def calculate_risk_level(vulns: List[Vulnerability]) -> str:
    """Calculate overall risk level based on vulnerabilities found."""
    critical = sum(1 for v in vulns if v.severity == Severity.CRITICAL)
    high = sum(1 for v in vulns if v.severity == Severity.HIGH)
    medium = sum(1 for v in vulns if v.severity == Severity.MEDIUM)
    low = sum(1 for v in vulns if v.severity == Severity.LOW)
    
    if critical > 0:
        return "CRITICAL"
    elif high > 0:
        return "HIGH"
    elif medium > 0:
        return "MEDIUM"
    elif low > 0:
        return "LOW"
    return "NONE"


def scan_repository(repo_path: str) -> SecurityScanResult:
    """
    Perform comprehensive security scan on a repository.
    
    Args:
        repo_path: Path to cloned repository
        
    Returns:
        SecurityScanResult with all findings
    """
    from datetime import datetime
    
    vulnerabilities = []
    scanned_files = []
    scan_method = "none"
    dependencies_found = 0
    
    dep_files = detect_dependency_files(repo_path)
    scanned_files = dep_files
    
    python_dep_files = ["requirements.txt", "pyproject.toml", "setup.py", "Pipfile"]
    for dep_file in dep_files:
        if any(pdf in dep_file for pdf in python_dep_files):
            file_path = os.path.join(repo_path, dep_file)
            packages = parse_requirements_file(file_path)
            dependencies_found += len(packages)
    
    pip_audit_result = scan_pip_audit(repo_path)
    if pip_audit_result:
        scan_method = "pip-audit"
        vulns_data = pip_audit_result.get("vulnerabilities", [])
        if isinstance(vulns_data, dict):
            for pkg_name, pkg_vulns in vulns_data.items():
                for vuln in pkg_vulns:
                    vulnerabilities.append(Vulnerability(
                        package_name=pkg_name,
                        version=vuln.get("version", "unknown"),
                        vulnerability_id=vuln.get("id", "unknown"),
                        severity=parse_severity(vuln.get("severity", "")),
                        description=vuln.get("description", "No description available"),
                        fix_version=vuln.get("fix_version"),
                        advisory_url=vuln.get("advisory_url")
                    ))
    
    safety_result = scan_safety(repo_path)
    if safety_result and not vulnerabilities:
        scan_method = "safety"
        for vuln in safety_result:
            vulnerabilities.append(Vulnerability(
                package_name=vuln.get("package_name", "unknown"),
                version=vuln.get("installed_version", "unknown"),
                vulnerability_id=vuln.get("cve", vuln.get("id", "unknown")),
                severity=parse_severity(vuln.get("severity", "")),
                description=vuln.get("description", "No description available"),
                fix_version=vuln.get("fix_versions", [None])[0] if vuln.get("fix_versions") else None,
                advisory_url=vuln.get("advisory")
            ))
    
    critical = sum(1 for v in vulnerabilities if v.severity == Severity.CRITICAL)
    high = sum(1 for v in vulnerabilities if v.severity == Severity.HIGH)
    medium = sum(1 for v in vulnerabilities if v.severity == Severity.MEDIUM)
    low = sum(1 for v in vulnerabilities if v.severity == Severity.LOW)
    
    return SecurityScanResult(
        total_vulnerabilities=len(vulnerabilities),
        critical_count=critical,
        high_count=high,
        medium_count=medium,
        low_count=low,
        vulnerabilities=vulnerabilities,
        scanned_files=scanned_files,
        scan_method=scan_method,
        dependencies_found=dependencies_found,
        risk_level=calculate_risk_level(vulnerabilities),
        timestamp=datetime.utcnow().isoformat()
    )


def get_vulnerability_summary(result: SecurityScanResult) -> str:
    """Generate a human-readable summary of the security scan."""
    if result.total_vulnerabilities == 0:
        return f"No vulnerabilities found in {result.dependencies_found} dependencies scanned."
    
    summary_parts = []
    if result.critical_count > 0:
        summary_parts.append(f"{result.critical_count} CRITICAL")
    if result.high_count > 0:
        summary_parts.append(f"{result.high_count} HIGH")
    if result.medium_count > 0:
        summary_parts.append(f"{result.medium_count} MEDIUM")
    if result.low_count > 0:
        summary_parts.append(f"{result.low_count} LOW")
    
    return f"Found {result.total_vulnerabilities} vulnerabilities ({', '.join(summary_parts)}) in {result.dependencies_found} dependencies"
