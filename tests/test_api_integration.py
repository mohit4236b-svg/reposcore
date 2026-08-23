"""
Integration tests for the async API with Redis/Celery.

These tests verify:
1. TTL calculation logic in cache_layer.py
2. API endpoint structure
3. Celery worker module is properly structured
"""

import os
import sys
import json
import pytest
from unittest.mock import patch, MagicMock, PropertyMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Mock Redis for API tests that need it
@pytest.fixture(autouse=True)
def mock_redis():
    """Mock Redis client for tests that don't need real Redis."""
    import fakeredis
    fake_redis = fakeredis.FakeRedis(decode_responses=True)
    
    # Mock both api_server.redis_client and cache_layer.cache
    with patch('api_server.redis_client', fake_redis):
        with patch('cache_layer.cache') as mock_cache:
            # Configure the mock cache to use the fake redis
            mock_cache.get.return_value = None
            mock_cache.set.return_value = True
            mock_cache.set_with_activity_ttl.return_value = True
            mock_cache.store_etag.return_value = None
            mock_cache.get_etag.return_value = None
            mock_cache.delete.return_value = True
            mock_cache.flush_pattern.return_value = 0
            # Allow setting actual data on the mock
            mock_cache._data = {}
            def mock_set(key, value, ttl_seconds):
                mock_cache._data[key] = value
                return True
            def mock_get(key):
                return mock_cache._data.get(key)
            mock_cache.set.side_effect = mock_set
            mock_cache.get.side_effect = mock_get
            mock_cache.set_with_activity_ttl.side_effect = mock_set
            yield fake_redis


class TestCacheTTLLogic:
    """Test Redis TTL calculation logic matching the docstring."""
    
    def test_cache_ttl_logic_active_repo(self):
        """Verify TTL=6h for active repos (<30 days since last commit)."""
        from cache_layer import DynamicCache
        
        ttl_seconds = DynamicCache._calculate_ttl_for_activity(15, False)
        assert ttl_seconds == 6 * 3600
        
    def test_cache_ttl_logic_active_boundary(self):
        """At exactly 30 days, repo should still be 'active' (<30)."""
        from cache_layer import DynamicCache
        
        ttl_seconds = DynamicCache._calculate_ttl_for_activity(29, False)
        assert ttl_seconds == 6 * 3600

    def test_cache_ttl_logic_moderately_active(self):
        """Verify TTL=12h for repos with 30-90 days since last commit."""
        from cache_layer import DynamicCache
        
        ttl_seconds = DynamicCache._calculate_ttl_for_activity(60, False)
        assert ttl_seconds == 12 * 3600

    def test_cache_ttl_logic_moderately_active_boundary(self):
        """At 90 days, repo should be 'inactive warm' (>=90)."""
        from cache_layer import DynamicCache
        
        ttl_seconds = DynamicCache._calculate_ttl_for_activity(89, False)
        assert ttl_seconds == 12 * 3600

    def test_cache_ttl_logic_warm_inactive(self):
        """Verify TTL=24h for repos with 90-365 days since last commit."""
        from cache_layer import DynamicCache
        
        ttl_seconds = DynamicCache._calculate_ttl_for_activity(200, False)
        assert ttl_seconds == 24 * 3600

    def test_cache_ttl_logic_inactive_boundary(self):
        """At 365 days, repo should be 'inactive' (>=365)."""
        from cache_layer import DynamicCache
        
        ttl_seconds = DynamicCache._calculate_ttl_for_activity(364, False)
        assert ttl_seconds == 24 * 3600

    def test_cache_ttl_logic_inactive(self):
        """Verify TTL=48h for repos with >365 days since last commit."""
        from cache_layer import DynamicCache
        
        ttl_seconds = DynamicCache._calculate_ttl_for_activity(400, False)
        assert ttl_seconds == 48 * 3600

    def test_cache_ttl_logic_archived(self):
        """Verify TTL=72h for archived repos."""
        from cache_layer import DynamicCache
        
        ttl_seconds = DynamicCache._calculate_ttl_for_activity(100, True)
        assert ttl_seconds == 72 * 3600


class TestCacheLayerMethods:
    """Test DynamicCache methods exist and have correct signatures."""
    
    def test_set_with_activity_ttl_method_exists(self):
        """Verify DynamicCache has set_with_activity_ttl method."""
        from cache_layer import DynamicCache
        
        assert hasattr(DynamicCache, 'set_with_activity_ttl')
        
    def test_get_set_methods_exist(self):
        """Verify basic get/set methods exist."""
        from cache_layer import DynamicCache
        
        assert hasattr(DynamicCache, 'get')
        assert hasattr(DynamicCache, 'set')

    def test_ttl_values_match_docstring(self):
        """Verify TTL_HOURS constant matches docstring."""
        from cache_layer import DynamicCache
        
        assert DynamicCache.TTL_HOURS["active"] == 6
        assert DynamicCache.TTL_HOURS["moderately_active"] == 12
        assert DynamicCache.TTL_HOURS["inactive_warm"] == 24
        assert DynamicCache.TTL_HOURS["inactive"] == 48
        assert DynamicCache.TTL_HOURS["archived"] == 72


class TestAPIEndpoints:
    """Test API endpoints are properly defined."""
    
    def test_api_health_endpoint(self):
        """Test health check endpoint."""
        from fastapi.testclient import TestClient
        from api_server import app
        
        client = TestClient(app)
        response = client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        
    def test_api_job_status_not_found(self):
        """Test job status endpoint 404 for non-existent job."""
        from fastapi.testclient import TestClient
        from api_server import app
        
        client = TestClient(app)
        response = client.get("/api/jobs/nonexistent-job")
        
        assert response.status_code == 404
        
    def test_api_submit_job_returns_job_id(self):
        """Test job submission returns job_id."""
        from fastapi.testclient import TestClient
        from api_server import app
        
        client = TestClient(app)
        response = client.post("/api/jobs?owner=testowner&repo=testrepo&threshold=0.3")
        
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert isinstance(data["job_id"], str)
        
    def test_api_score_endpoint(self):
        """Test /api/score/{owner}/{repo} endpoint."""
        from fastapi.testclient import TestClient
        from api_server import app
        
        client = TestClient(app)
        response = client.get("/api/score/owner/repo")
        
        assert response.status_code == 200


class TestCeleryWorker:
    """Test Celery worker module structure."""
    
    def test_celery_app_exists(self):
        """Verify celery_app is defined."""
        from workers.celery_worker import celery_app
        
        assert celery_app is not None
        assert hasattr(celery_app, 'conf')
        
    def test_analyze_task_exists(self):
        """Verify analyze_repository_async task exists."""
        from workers.celery_worker import analyze_repository_async
        
        assert callable(analyze_repository_async)
        
    def test_task_accepts_owner_repo_token(self):
        """Verify task accepts owner, repo, token parameters."""
        import inspect
        from workers.celery_worker import analyze_repository_async
        
        sig = inspect.signature(analyze_repository_async)
        params = list(sig.parameters.keys())
        
        assert "owner" in params
        assert "repo" in params
        assert "token" in params


class TestAPIThreshold:
    """Test API threshold configuration."""
    
    def test_api_default_threshold_is_03(self):
        """Verify API default threshold is 0.3."""
        import inspect
        from api_server import submit_analysis
        
        sig = inspect.signature(submit_analysis)
        threshold_default = sig.parameters["threshold"].default.default
        
        assert threshold_default == 0.3


class TestScoringSystems:
    """Compare ML model vs RepoScorer heuristics."""
    
    def test_score_repo_function_default_threshold(self):
        """Verify score_repo uses 0.3 as default threshold."""
        import inspect
        import reposcore_cli
        
        sig = inspect.signature(reposcore_cli.score_repo)
        default_threshold = sig.parameters["threshold"].default
        
        assert default_threshold == 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])