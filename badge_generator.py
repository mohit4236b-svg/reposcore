"""
SVG badge generation for repository quality scores.
"""

from typing import Dict, Any, Optional
import json


def generate_svg_badge(score_data: Dict[str, Any], width: int = 220) -> str:
    """
    Generate an SVG badge for a repository's quality score.
    
    Args:
        score_data: Dictionary containing total_score, tier, and optional color
        width: Badge width in pixels (default 220)
    
    Returns:
        SVG string
    """
    score = score_data.get("total_score", 0)
    tier = score_data.get("tier", "F")
    color = score_data.get("color", "#F44336")
    emoji = score_data.get("tier_emoji", "❌")
    
    # Badge SVG using shields.io style
    svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="30">
  <linearGradient id="b" x2="0" y2="100%" x1="0" y1="0">
    <stop offset="0" stop-color="{color}" stop-opacity="100"/>
    <stop offset="100" stop-color="{color}" stop-opacity="83"/>
  </linearGradient>
  <rect width="{width}" height="30" rx="3" fill="#000000"/>
  <rect x="{width//2}" width="{width//2}" height="30" fill="#555555"/>
  <rect x="0" width="{width}" height="30" fill="url(#b)" rx="3" fill-opacity="100"/>
  <g fill="#FFFFFF" font-family="Verdana,a,sans-serif" font-size="11">
    <text x="{width//4}" y="20" text-anchor="middle">{emoji} RepoScore</text>
    <text x="{width//2 + width//4}" y="20" text-anchor="middle">{score:.0f}% ({tier})</text>
  </g>
</svg>'''
    
    return svg


def generate_shields_io_badge_url(score: float, label: str = "reposcore") -> str:
    """
    Generate a shields.io compatible badge URL.
    
    Args:
        score: Score between 0-100
        label: Badge label (default "reposcore")
    
    Returns:
        shields.io badge URL
    """
    if score >= 90:
        color = "brightgreen"
    elif score >= 80:
        color = "green"
    elif score >= 70:
        color = "yellowgreen"
    elif score >= 60:
        color = "yellow"
    elif score >= 50:
        color = "orange"
    else:
        color = "red"
    
    return f"https://img.shields.io/badge/{label}-{score:.0f}%25-{color}.svg"


def get_tier_from_score(score: float) -> Dict[str, str]:
    """Get tier info for a given score."""
    if score >= 90:
        return {"tier": "A+", "emoji": "🏆", "color": "#4CAF50"}
    elif score >= 80:
        return {"tier": "A", "emoji": "⭐", "color": "#8BC34A"}
    elif score >= 70:
        return {"tier": "B", "emoji": "👍", "color": "#CDDC39"}
    elif score >= 60:
        return {"tier": "C", "emoji": "👌", "color": "#FF9800"}
    else:
        return {"tier": "F", "emoji": "❌", "color": "#F44336"}


def create_badge_response(score_data: Dict[str, Any], 
                          format: str = "svg") -> Dict[str, Any]:
    """
    Create a response suitable for FastAPI/Sanic response.
    
    Args:
        score_data: Score data from scorer
        format: "svg" for inline, "url" for shields.io URL
    
    Returns:
        Dict with content and metadata
    """
    if format == "url":
        return {
            "badge_url": generate_shields_io_badge_url(score_data.get("total_score", 0)),
            "content_type": "application/json"
        }
    else:
        svg = generate_svg_badge(score_data)
        return {
            "content": svg,
            "content_type": "image/svg+xml",
            "headers": {
                "Cache-Control": "public, max-age=3600",
                "Content-Disposition": "inline; filename=badge.svg",
                "Access-Control-Allow-Origin": "*"
            }
        }