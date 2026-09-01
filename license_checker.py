"""
License compliance checker for repositories.
Detects license type and flags incompatible licenses for commercial use.
"""

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class LicenseCategory(Enum):
    PERMISSIVE = "PERMISSIVE"
    COPYLEFT_WEAK = "COPYLEFT_WEAK"
    COPYLEFT_STRONG = "COPYLEFT_STRONG"
    PROPRIETARY = "PROPRIETARY"
    UNKNOWN = "UNKNOWN"
    NONE = "NONE"


LICENSE_PATTERNS = {
    "MIT": {
        "pattern": r"MIT License",
        "spdx_id": "MIT",
        "category": LicenseCategory.PERMISSIVE,
        "commercial_use": True,
        "modifications": True,
        "liability": False,
        "patent_use": True,
    },
    "Apache-2.0": {
        "pattern": r"Apache License, Version 2\.0",
        "spdx_id": "Apache-2.0",
        "category": LicenseCategory.PERMISSIVE,
        "commercial_use": True,
        "modifications": True,
        "liability": False,
        "patent_use": True,
    },
    "BSD-3-Clause": {
        "pattern": r"BSD 3-Clause|BSD 3-Clause License",
        "spdx_id": "BSD-3-Clause",
        "category": LicenseCategory.PERMISSIVE,
        "commercial_use": True,
        "modifications": True,
        "liability": False,
        "patent_use": True,
    },
    "BSD-2-Clause": {
        "pattern": r"BSD 2-Clause|BSD 2-Clause License",
        "spdx_id": "BSD-2-Clause",
        "category": LicenseCategory.PERMISSIVE,
        "commercial_use": True,
        "modifications": True,
        "liability": False,
        "patent_use": True,
    },
    "ISC": {
        "pattern": r"ISC License",
        "spdx_id": "ISC",
        "category": LicenseCategory.PERMISSIVE,
        "commercial_use": True,
        "modifications": True,
        "liability": False,
        "patent_use": True,
    },
    "Unlicense": {
        "pattern": r"Unlicense",
        "spdx_id": "Unlicense",
        "category": LicenseCategory.PERMISSIVE,
        "commercial_use": True,
        "modifications": True,
        "liability": False,
        "patent_use": True,
    },
    "GPL-3.0": {
        "pattern": r"GNU General Public License v3|GPLv3|GPL-3.0",
        "spdx_id": "GPL-3.0",
        "category": LicenseCategory.COPYLEFT_STRONG,
        "commercial_use": True,
        "modifications": True,
        "liability": False,
        "patent_use": True,
    },
    "GPL-2.0": {
        "pattern": r"GNU General Public License v2|GPLv2|GPL-2.0",
        "spdx_id": "GPL-2.0",
        "category": LicenseCategory.COPYLEFT_STRONG,
        "commercial_use": True,
        "modifications": True,
        "liability": False,
        "patent_use": True,
    },
    "LGPL-3.0": {
        "pattern": r"GNU Lesser General Public License v3|LGPLv3|LGPL-3.0",
        "spdx_id": "LGPL-3.0",
        "category": LicenseCategory.COPYLEFT_WEAK,
        "commercial_use": True,
        "modifications": True,
        "liability": False,
        "patent_use": True,
    },
    "LGPL-2.1": {
        "pattern": r"GNU Lesser General Public License v2|LGPLv2|LGPL-2.1",
        "spdx_id": "LGPL-2.1",
        "category": LicenseCategory.COPYLEFT_WEAK,
        "commercial_use": True,
        "modifications": True,
        "liability": False,
        "patent_use": True,
    },
    "MPL-2.0": {
        "pattern": r"Mozilla Public License 2\.0|MPL-2.0",
        "spdx_id": "MPL-2.0",
        "category": LicenseCategory.COPYLEFT_WEAK,
        "commercial_use": True,
        "modifications": True,
        "liability": False,
        "patent_use": True,
    },
    "AGPL-3.0": {
        "pattern": r"GNU AFFERO General Public License v3|AGPLv3|AGPL-3.0",
        "spdx_id": "AGPL-3.0",
        "category": LicenseCategory.COPYLEFT_STRONG,
        "commercial_use": True,
        "modifications": True,
        "liability": False,
        "patent_use": True,
    },
    "EPL-2.0": {
        "pattern": r"Eclipse Public License 2\.0|EPL-2.0",
        "spdx_id": "EPL-2.0",
        "category": LicenseCategory.COPYLEFT_WEAK,
        "commercial_use": True,
        "modifications": True,
        "liability": False,
        "patent_use": True,
    },
    "CC0-1.0": {
        "pattern": r"CC0|CC0 1\.0 Universal",
        "spdx_id": "CC0-1.0",
        "category": LicenseCategory.PERMISSIVE,
        "commercial_use": True,
        "modifications": True,
        "liability": False,
        "patent_use": True,
    },
    "CC-BY-4.0": {
        "pattern": r"Creative Commons Attribution 4\.0|CC-BY-4.0",
        "spdx_id": "CC-BY-4.0",
        "category": LicenseCategory.PERMISSIVE,
        "commercial_use": True,
        "modifications": True,
        "liability": False,
        "patent_use": False,
    },
    "CC-BY-SA-4.0": {
        "pattern": r"Creative Commons Attribution Share Alike 4\.0|CC-BY-SA-4.0",
        "spdx_id": "CC-BY-SA-4.0",
        "category": LicenseCategory.COPYLEFT_WEAK,
        "commercial_use": True,
        "modifications": True,
        "liability": False,
        "patent_use": False,
    },
}


COMMERCIAL_USE_CONCERNS = {
    LicenseCategory.COPYLEFT_STRONG: [
        "Strong copyleft - derivative works must be released under same license",
        "May require releasing proprietary source code",
        "Incompatible with proprietary software distribution",
    ],
    LicenseCategory.COPYLEFT_WEAK: [
        "Weak copyleft - may have linking exceptions",
        "Check specific license terms for commercial use",
        "Static linking may trigger copyleft requirements",
    ],
    LicenseCategory.UNKNOWN: [
        "License could not be identified",
        "Manual review required before commercial use",
    ],
    LicenseCategory.NONE: [
        "No license detected - all rights reserved by default",
        "Cannot legally use, modify, or distribute",
        "Contact owner for explicit permission",
    ],
}


@dataclass
class LicenseInfo:
    name: str
    spdx_id: str
    category: LicenseCategory
    commercial_use: bool
    modifications: bool
    liability: bool
    patent_use: bool
    warnings: List[str]
    source: str


@dataclass
class LicenseCheckResult:
    has_license: bool
    license_info: Optional[LicenseInfo]
    compliance_score: int
    risk_level: str
    warnings: List[str]
    commercial_compatible: bool


def detect_license_from_file(repo_path: str) -> Optional[str]:
    """Detect license from LICENSE or COPYING file in repo."""
    license_files = ["LICENSE", "LICENSE.txt", "LICENSE.md", "COPYING", "COPYING.txt", "LICENSE-APACHE", "LICENSE-MIT", "LICENSE-BSD"]
    
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'venv']]
        
        for file in files:
            if file.upper() in [lf.upper() for lf in license_files]:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read(10000)
                    return content
                except Exception:
                    continue
    
    return None


def parse_license_from_github(repo_data: dict) -> Optional[dict]:
    """Extract license info from GitHub API response."""
    if not repo_data.get("license"):
        return None
    
    license_data = repo_data["license"]
    return {
        "key": license_data.get("key"),
        "name": license_data.get("name"),
        "spdx_id": license_data.get("spdx_id"),
        "url": license_data.get("url"),
    }


def match_license_pattern(text: str) -> Optional[Dict]:
    """Match license text against known patterns."""
    if not text:
        return None
    
    text_lower = text.lower()
    
    for lic_name, lic_info in LICENSE_PATTERNS.items():
        pattern = lic_info["pattern"]
        if re.search(pattern, text, re.IGNORECASE):
            return {
                "name": lic_name,
                **lic_info
            }
    
    return None


def check_license_from_repo(repo_path: str, github_license: Optional[dict] = None) -> LicenseCheckResult:
    """
    Check repository license compliance.
    
    Args:
        repo_path: Path to cloned repository
        github_license: License data from GitHub API (optional)
        
    Returns:
        LicenseCheckResult with compliance information
    """
    warnings = []
    
    if github_license and github_license.get("spdx_id"):
        spdx_id = github_license["spdx_id"]
        for lic_name, lic_info in LICENSE_PATTERNS.items():
            if lic_info["spdx_id"] == spdx_id:
                license_info = LicenseInfo(
                    name=lic_name,
                    spdx_id=lic_info["spdx_id"],
                    category=lic_info["category"],
                    commercial_use=lic_info["commercial_use"],
                    modifications=lic_info["modifications"],
                    liability=lic_info["liability"],
                    patent_use=lic_info["patent_use"],
                    warnings=[],
                    source="github_api"
                )
                
                if lic_info["category"] in COMMERCIAL_USE_CONCERNS:
                    license_info.warnings = COMMERCIAL_USE_CONCERNS[lic_info["category"]]
                
                return _build_result(license_info)
    
    license_file_content = detect_license_from_file(repo_path)
    if license_file_content:
        matched = match_license_pattern(license_file_content)
        if matched:
            license_info = LicenseInfo(
                name=matched["name"],
                spdx_id=matched["spdx_id"],
                category=matched["category"],
                commercial_use=matched["commercial_use"],
                modifications=matched["modifications"],
                liability=matched["liability"],
                patent_use=matched["patent_use"],
                warnings=[],
                source="license_file"
            )
            
            if matched["category"] in COMMERCIAL_USE_CONCERNS:
                license_info.warnings = COMMERCIAL_USE_CONCERNS[matched["category"]]
            
            return _build_result(license_info)
    
    license_info = LicenseInfo(
        name="No License Detected",
        spdx_id="NOASSERTION",
        category=LicenseCategory.NONE,
        commercial_use=False,
        modifications=False,
        liability=False,
        patent_use=False,
        warnings=COPYRIGHT_WARNINGS if False else [],
        source="none"
    )
    
    return _build_result(license_info)


def _build_result(license_info: LicenseInfo) -> LicenseCheckResult:
    """Build LicenseCheckResult from LicenseInfo."""
    warnings = license_info.warnings.copy() if license_info.warnings else []
    commercial_compatible = license_info.commercial_use
    risk_level = "NONE"
    
    if license_info.category == LicenseCategory.NONE:
        risk_level = "HIGH"
        warnings.append("No license - default copyright applies. Cannot legally use without permission.")
        commercial_compatible = False
    elif license_info.category == LicenseCategory.COPYLEFT_STRONG:
        risk_level = "MEDIUM"
    elif license_info.category == LicenseCategory.COPYLEFT_WEAK:
        risk_level = "LOW"
    elif license_info.category == LicenseCategory.UNKNOWN:
        risk_level = "MEDIUM"
    else:
        risk_level = "NONE"
    
    if not license_info.liability:
        warnings.append("License provides NO WARRANTY or liability protection.")
    
    if not license_info.patent_use:
        warnings.append("License does NOT grant patent rights - verify patent status independently.")
    
    compliance_score = 100
    if not license_info.commercial_use:
        compliance_score -= 30
    if not license_info.modifications:
        compliance_score -= 20
    if not license_info.liability:
        compliance_score -= 10
    if not license_info.patent_use:
        compliance_score -= 10
    if license_info.category == LicenseCategory.NONE:
        compliance_score = 0
    elif license_info.category == LicenseCategory.COPYLEFT_STRONG:
        compliance_score -= 15
    
    return LicenseCheckResult(
        has_license=license_info.category != LicenseCategory.NONE,
        license_info=license_info,
        compliance_score=max(0, compliance_score),
        risk_level=risk_level,
        warnings=warnings,
        commercial_compatible=commercial_compatible
    )


COPYRIGHT_WARNINGS = [
    "No license file detected in repository",
    "Default copyright law applies - all rights reserved",
    "Cannot legally use, modify, or distribute without explicit permission",
    "For open source use, owner must provide explicit license",
]
