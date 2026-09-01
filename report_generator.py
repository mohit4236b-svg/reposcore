"""
Report generator for PDF and HTML quality reports.
Generates comprehensive repository analysis reports with all scores, charts, and AI review.
"""

import io
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any


@dataclass
class ReportData:
    full_name: str
    html_url: str
    features: Dict[str, Any]
    ml_prediction: float
    ml_probability: float
    heuristic_score: Dict[str, Any]
    combined_score: float
    trend_analysis: Optional[Dict[str, Any]]
    security_scan: Optional[Dict[str, Any]]
    license_check: Optional[Dict[str, Any]]
    ai_review: Optional[Dict[str, Any]]
    code_metrics: Optional[Dict[str, Any]]
    generated_at: str


def generate_html_report(report_data: ReportData) -> str:
    """Generate comprehensive HTML report."""
    
    scores = report_data.heuristic_score
    components = scores.get("components", {}) if scores else {}
    
    vulnerability_html = ""
    if report_data.security_scan:
        scan = report_data.security_scan
        vuln_count = scan.get("total_vulnerabilities", 0)
        risk_level = scan.get("risk_level", "UNKNOWN")
        
        if vuln_count > 0:
            vulnerability_html = f"""
            <div class="section vulnerability">
                <h2>Security Analysis</h2>
                <div class="alert alert-{'critical' if risk_level == 'CRITICAL' else 'warning'}">
                    <strong>{vuln_count} vulnerabilities found</strong> - Risk Level: {risk_level}
                </div>
                <table class="vuln-table">
                    <tr><th>Package</th><th>Severity</th><th>ID</th><th>Description</th></tr>
            """
            for vuln in scan.get("vulnerabilities", [])[:10]:
                vulnerability_html += f"""
                    <tr>
                        <td>{vuln.get('package_name', 'unknown')}</td>
                        <td><span class="severity severity-{vuln.get('severity', 'UNKNOWN').lower()}">{vuln.get('severity', 'UNKNOWN')}</span></td>
                        <td>{vuln.get('vulnerability_id', 'N/A')}</td>
                        <td>{vuln.get('description', 'No description')[:100]}...</td>
                    </tr>
                """
            vulnerability_html += "</table></div>"
        else:
            vulnerability_html = """
            <div class="section vulnerability">
                <h2>Security Analysis</h2>
                <div class="alert alert-success">No vulnerabilities detected</div>
            </div>
            """
    
    license_html = ""
    if report_data.license_check:
        lic = report_data.license_check
        license_html = f"""
        <div class="section license">
            <h2>License Compliance</h2>
            <div class="license-info">
                <p><strong>License:</strong> {lic.get('license_info', {}).get('name', 'Unknown')}</p>
                <p><strong>SPDX ID:</strong> {lic.get('license_info', {}).get('spdx_id', 'NOASSERTION')}</p>
                <p><strong>Compliance Score:</strong> {lic.get('compliance_score', 'N/A')}/100</p>
                <p><strong>Commercial Compatible:</strong> {'Yes' if lic.get('commercial_compatible') else 'No'}</p>
            </div>
        """
        if lic.get('warnings'):
            license_html += "<div class='warnings'><strong>Warnings:</strong><ul>"
            for warning in lic.get('warnings', []):
                license_html += f"<li>{warning}</li>"
            license_html += "</ul></div>"
        license_html += "</div>"
    
    trend_html = ""
    if report_data.trend_analysis:
        trend = report_data.trend_analysis
        trend_html = f"""
        <div class="section trend">
            <h2>Trend Analysis</h2>
            <div class="metrics-grid">
                <div class="metric">
                    <span class="metric-label">Star Growth (30d)</span>
                    <span class="metric-value">{trend.get('star_growth_rate_30d', 0):.1f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Star Growth (90d)</span>
                    <span class="metric-value">{trend.get('star_growth_rate_90d', 0):.1f}%</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Commits (90d)</span>
                    <span class="metric-value">{trend.get('commit_activity_90d', 0)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Health Status</span>
                    <span class="metric-value">{trend.get('health_status', 'unknown').upper()}</span>
                </div>
            </div>
            <p><strong>Activity Trend:</strong> {trend.get('activity_trend', 'unknown').title()}</p>
            <p><strong>Commit Frequency:</strong> {trend.get('commit_frequency', 'unknown').replace('_', ' ').title()}</p>
        </div>
        """
    
    code_metrics_html = ""
    if report_data.code_metrics:
        metrics = report_data.code_metrics
        code_metrics_html = f"""
        <div class="section code-metrics">
            <h2>Code Metrics</h2>
            <div class="metrics-grid">
                <div class="metric">
                    <span class="metric-label">Files</span>
                    <span class="metric-value">{metrics.get('file_count', 0)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Total LOC</span>
                    <span class="metric-value">{metrics.get('total_loc', 0):,}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Avg Complexity</span>
                    <span class="metric-value">{metrics.get('avg_complexity', 0):.1f}</span>
                </div>
            </div>
        </div>
        """
    
    ai_review_html = ""
    if report_data.ai_review and report_data.ai_review.get("status") == "success":
        review = report_data.ai_review
        ai_review_html = f"""
        <div class="section ai-review">
            <h2>AI Review</h2>
            <div class="review-content">
                {review.get('review', 'No review available')}
            </div>
        </div>
        """
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RepoScore Report - {report_data.full_name}</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0d1117; color: #c9d1d9; line-height: 1.6; }}
        .container {{ max-width: 1000px; margin: 0 auto; padding: 2rem; }}
        .header {{ text-align: center; margin-bottom: 2rem; padding-bottom: 1rem; border-bottom: 1px solid #30363d; }}
        .header h1 {{ color: #58a6ff; font-size: 2rem; margin-bottom: 0.5rem; }}
        .header .subtitle {{ color: #8b949e; }}
        .score-hero {{ background: linear-gradient(135deg, #161b22 0%, #21262d 100%); border-radius: 12px; padding: 2rem; text-align: center; margin-bottom: 2rem; border: 1px solid #30363d; }}
        .score-hero .score {{ font-size: 4rem; font-weight: 700; color: {'#4CAF50' if report_data.combined_score >= 70 else '#d4a017' if report_data.combined_score >= 40 else '#f44336'}; }}
        .score-hero .label {{ color: #8b949e; font-size: 1.2rem; }}
        .score-hero .breakdown {{ display: flex; justify-content: center; gap: 2rem; margin-top: 1rem; color: #8b949e; }}
        .section {{ background: #161b22; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; border: 1px solid #30363d; }}
        .section h2 {{ color: #58a6ff; margin-bottom: 1rem; font-size: 1.3rem; border-bottom: 1px solid #30363d; padding-bottom: 0.5rem; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem; }}
        .metric {{ background: #21262d; padding: 1rem; border-radius: 8px; text-align: center; }}
        .metric-label {{ display: block; color: #8b949e; font-size: 0.85rem; margin-bottom: 0.25rem; }}
        .metric-value {{ display: block; font-size: 1.5rem; font-weight: 600; color: #c9d1d9; }}
        .component-scores {{ margin-top: 1rem; }}
        .component {{ margin-bottom: 0.75rem; }}
        .component-header {{ display: flex; justify-content: space-between; margin-bottom: 0.25rem; }}
        .component-bar {{ background: #21262d; height: 8px; border-radius: 4px; overflow: hidden; }}
        .component-fill {{ height: 100%; border-radius: 4px; transition: width 0.3s; }}
        .alert {{ padding: 1rem; border-radius: 8px; margin: 1rem 0; }}
        .alert-success {{ background: rgba(46, 160, 67, 0.15); border: 1px solid #2ea043; color: #3fb950; }}
        .alert-warning {{ background: rgba(212, 160, 23, 0.15); border: 1px solid #d4a017; color: #e0c26e; }}
        .alert-critical {{ background: rgba(248, 81, 73, 0.15); border: 1px solid #f85149; color: #f85149; }}
        .vuln-table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
        .vuln-table th, .vuln-table td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid #30363d; }}
        .vuln-table th {{ color: #8b949e; font-weight: 500; }}
        .severity {{ padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.8rem; font-weight: 600; }}
        .severity-critical {{ background: rgba(248, 81, 73, 0.2); color: #f85149; }}
        .severity-high {{ background: rgba(248, 81, 73, 0.15); color: #ff7b72; }}
        .severity-medium {{ background: rgba(212, 160, 23, 0.2); color: #d4a017; }}
        .severity-low {{ background: rgba(46, 160, 67, 0.15); color: #3fb950; }}
        .license-info p {{ margin-bottom: 0.5rem; }}
        .warnings ul {{ margin-left: 1.5rem; color: #d4a017; }}
        .review-content {{ white-space: pre-wrap; line-height: 1.8; }}
        .footer {{ text-align: center; color: #8b949e; font-size: 0.85rem; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #30363d; }}
        @media print {{ body {{ background: white; color: black; }} .section {{ border: 1px solid #ddd; }} }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>⭐ RepoScore Report</h1>
            <p class="subtitle">{report_data.full_name}</p>
            <p class="subtitle"><a href="{report_data.html_url}" style="color: #58a6ff;">{report_data.html_url}</a></p>
        </div>
        
        <div class="score-hero">
            <div class="score">{report_data.combined_score:.1f}</div>
            <div class="label">Combined Quality Score</div>
            <div class="breakdown">
                <span>ML: {report_data.ml_probability * 100:.1f}%</span>
                <span>Heuristic: {scores.get('total_score', 0) if scores else 0:.1f}</span>
            </div>
        </div>
        
        <div class="section">
            <h2>Component Scores</h2>
            <div class="component-scores">
                <div class="component">
                    <div class="component-header"><span>Maintenance</span><span>{components.get('maintenance', 0):.1f}/100</span></div>
                    <div class="component-bar"><div class="component-fill" style="width: {components.get('maintenance', 0)}%; background: {'#4CAF50' if components.get('maintenance', 0) >= 70 else '#d4a017' if components.get('maintenance', 0) >= 40 else '#f44336'};"></div></div>
                </div>
                <div class="component">
                    <div class="component-header"><span>Community</span><span>{components.get('community', 0):.1f}/100</span></div>
                    <div class="component-bar"><div class="component-fill" style="width: {components.get('community', 0)}%; background: {'#4CAF50' if components.get('community', 0) >= 70 else '#d4a017' if components.get('community', 0) >= 40 else '#f44336'};"></div></div>
                </div>
                <div class="component">
                    <div class="component-header"><span>Documentation</span><span>{components.get('documentation', 0):.1f}/100</span></div>
                    <div class="component-bar"><div class="component-fill" style="width: {components.get('documentation', 0)}%; background: {'#4CAF50' if components.get('documentation', 0) >= 70 else '#d4a017' if components.get('documentation', 0) >= 40 else '#f44336'};"></div></div>
                </div>
                <div class="component">
                    <div class="component-header"><span>Contributors</span><span>{components.get('contributors', 0):.1f}/100</span></div>
                    <div class="component-bar"><div class="component-fill" style="width: {components.get('contributors', 0)}%; background: {'#4CAF50' if components.get('contributors', 0) >= 70 else '#d4a017' if components.get('contributors', 0) >= 40 else '#f44336'};"></div></div>
                </div>
            </div>
        </div>
        
        {license_html}
        {vulnerability_html}
        {trend_html}
        {code_metrics_html}
        {ai_review_html}
        
        <div class="footer">
            <p>Generated by RepoScore on {report_data.generated_at}</p>
            <p>This report is for informational purposes only.</p>
        </div>
    </div>
</body>
</html>"""


def generate_json_report(report_data: ReportData) -> str:
    """Generate JSON report data."""
    return json.dumps({
        "repository": {
            "full_name": report_data.full_name,
            "html_url": report_data.html_url,
        },
        "scores": {
            "combined": report_data.combined_score,
            "ml_probability": report_data.ml_probability,
            "heuristic": report_data.heuristic_score,
        },
        "features": {
            "stars": report_data.features.get("stars", 0),
            "forks": report_data.features.get("forks", 0),
            "open_issues": report_data.features.get("open_issues", 0),
            "total_contributors": report_data.features.get("total_contributors", 0),
            "has_tests": report_data.features.get("has_tests", False),
            "has_ci": report_data.features.get("has_ci", False),
            "has_license": report_data.features.get("has_license", False),
        },
        "security": report_data.security_scan,
        "license": report_data.license_check,
        "trends": report_data.trend_analysis,
        "code_metrics": report_data.code_metrics,
        "ai_review": report_data.ai_review,
        "generated_at": report_data.generated_at,
    }, indent=2)


def generate_report(
    full_name: str,
    html_url: str,
    features: Dict[str, Any],
    ml_probability: float,
    heuristic_score: Dict[str, Any],
    combined_score: float,
    trend_analysis: Optional[Dict[str, Any]] = None,
    security_scan: Optional[Dict[str, Any]] = None,
    license_check: Optional[Dict[str, Any]] = None,
    ai_review: Optional[Dict[str, Any]] = None,
    code_metrics: Optional[Dict[str, Any]] = None,
    format: str = "html"
) -> str:
    """
    Generate a comprehensive repository quality report.
    
    Args:
        full_name: Repository full name
        html_url: Repository URL
        features: Repository features
        ml_probability: ML model probability
        heuristic_score: Heuristic scoring result
        combined_score: Combined score
        trend_analysis: Trend analysis data
        security_scan: Security scan result
        license_check: License check result
        ai_review: AI review data
        code_metrics: Code metrics data
        format: Output format ("html" or "json")
        
    Returns:
        Report content as string
    """
    report_data = ReportData(
        full_name=full_name,
        html_url=html_url,
        features=features,
        ml_prediction=1 if ml_probability >= 0.5 else 0,
        ml_probability=ml_probability,
        heuristic_score=heuristic_score,
        combined_score=combined_score,
        trend_analysis=trend_analysis,
        security_scan=security_scan,
        license_check=license_check,
        ai_review=ai_review,
        code_metrics=code_metrics,
        generated_at=datetime.utcnow().isoformat()
    )
    
    if format.lower() == "json":
        return generate_json_report(report_data)
    else:
        return generate_html_report(report_data)
