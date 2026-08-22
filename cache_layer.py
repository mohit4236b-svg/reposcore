"""
Intelligent Redis caching layer with dynamic TTL based on repository activity.
"""

import redis
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")


class DynamicCache:
    """
    Redis cache with activity-based TTL for repository scores.
    
    TTL Strategy:
    - Active repos (<30 days since commit): 6 hours
    - Moderately active (30-90 days): 12 hours
    - Inactive (90-365 days): 24 hours
    - Inactive (>365 days): 48 hours
    - Archived: 72 hours
    """
    
    def __init__(self, redis_url: str = REDIS_URL):
        self.redis = redis.Redis.from_url(redis_url, decode_responses=True)
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get value from cache."""
        try:
            data = self.redis.get(key)
            return json.loads(data) if data else None
        except (redis.RedisError, json.JSONDecodeError):
            return None
    
    def set(self, key: str, value: Dict[str, Any], ttl_seconds: int) -> bool:
        """Set value with TTL in seconds."""
        try:
            return self.redis.setex(key, ttl_seconds, json.dumps(value, default=str))
        except redis.RedisError:
            return False
    
    def set_with_activity_ttl(self, key: str, value: Dict[str, Any],
                              last_commit_days: int = 999999,
                              is_archived: bool = False) -> bool:
        """Set with TTL based on repository activity level."""
        
        if is_archived:
            ttl_hours = 72
        elif last_commit_days < 30:
            ttl_hours = 6
        elif last_commit_days < 90:
            ttl_hours = 12
        elif last_commit_days < 365:
            ttl_hours = 24
        else:
            ttl_hours = 48
        
        return self.set(key, value, ttl_hours * 3600)
    
    def store_etag(self, etag_key: str, etag: str, ttl_hours: int = 24) -> None:
        """Store ETag for conditional requests."""
        try:
            self.redis.setex(f"etag:{etag_key}", ttl_hours * 3600, etag)
        except redis.RedisError:
            pass
    
    def get_etag(self, etag_key: str) -> Optional[str]:
        """Get stored ETag."""
        try:
            return self.redis.get(f"etag:{etag_key}")
        except redis.RedisError:
            return None
    
    def delete(self, key: str) -> bool:
        """Delete a key from cache."""
        try:
            return bool(self.redis.delete(key))
        except redis.RedisError:
            return False
    
    def flush_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern."""
        try:
            keys = self.redis.keys(pattern)
            if keys:
                return self.redis.delete(*keys)
        except redis.RedisError:
            pass
        return 0


# Global cache instance
cache = DynamicCache()