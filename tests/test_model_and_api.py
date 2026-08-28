"""
Tests for model loading, prediction, and FastAPI endpoints.

These tests verify:
1. The ML model loads correctly from its saved file
2. A single prediction call returns a valid score/confidence in expected range
3. FastAPI endpoints return expected status codes and response shapes
"""
import os
import sys
import pytest
import joblib
import numpy as np
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "..", "models")


class TestModelLoading:
    """Test that ML model artifacts load correctly."""

    def test_rf_model_loads(self):
        """RandomForest model loads without error."""
        model_path = os.path.join(MODELS_DIR, "rf_model.pkl")
        assert os.path.exists(model_path), f"Model file not found: {model_path}"
        
        model = joblib.load(model_path)
        assert model is not None
        assert hasattr(model, "predict_proba")
        assert hasattr(model, "predict")

    def test_tfidf_readme_loads(self):
        """TF-IDF vectorizer for README loads without error."""
        path = os.path.join(MODELS_DIR, "tfidf_readme.pkl")
        assert os.path.exists(path)
        
        vectorizer = joblib.load(path)
        assert vectorizer is not None
        assert hasattr(vectorizer, "transform")

    def test_tfidf_topics_loads(self):
        """TF-IDF vectorizer for topics loads without error."""
        path = os.path.join(MODELS_DIR, "tfidf_topics.pkl")
        assert os.path.exists(path)
        
        vectorizer = joblib.load(path)
        assert vectorizer is not None
        assert hasattr(vectorizer, "transform")

    def test_scaler_loads(self):
        """StandardScaler loads without error."""
        path = os.path.join(MODELS_DIR, "scaler.pkl")
        assert os.path.exists(path)
        
        scaler = joblib.load(path)
        assert scaler is not None
        assert hasattr(scaler, "transform")


class TestModelPrediction:
    """Test that model predictions work and return valid ranges."""

    @pytest.fixture
    def models(self):
        """Load all model artifacts."""
        return (
            joblib.load(os.path.join(MODELS_DIR, "rf_model.pkl")),
            joblib.load(os.path.join(MODELS_DIR, "tfidf_readme.pkl")),
            joblib.load(os.path.join(MODELS_DIR, "tfidf_topics.pkl")),
            joblib.load(os.path.join(MODELS_DIR, "scaler.pkl")),
        )

    def test_predict_proba_returns_valid_probability(self, models):
        """predict_proba returns a probability in [0, 1]."""
        rf_model, tfidf_readme, tfidf_topics, scaler = models
        
        from reposcore_utils import featurize
        
        features = {
            "full_name": "test/repo",
            "html_url": "https://github.com/test/repo",
            "topics": ["python", "ml"],
            "stars": 100,
            "forks": 10,
            "open_issues": 5,
            "repo_age_days": 365,
            "last_commit_days": 7,
            "total_contributors": 5,
            "has_tests": True,
            "has_ci": True,
            "has_license": True,
            "readme_size": 5000,
            "readme_text_clean": "This is a test readme with documentation.",
            "commit_count_90d": 20,
            "issue_response_hours": 24,
            "pr_merge_time_days": 3,
            "contributor_commits_last_90d": 50,
            "is_archived": False,
            "is_fork": False,
            "stars_per_month": 2.5,
            "has_readme": 1,
        }
        
        X = featurize(features, tfidf_readme, tfidf_topics, scaler)
        proba = rf_model.predict_proba(X)[0][1]
        
        assert proba is not None
        assert 0.0 <= proba <= 1.0
        assert isinstance(proba, (float, np.floating))

    def test_predict_returns_valid_class(self, models):
        """predict returns a valid class (0 or 1)."""
        rf_model, tfidf_readme, tfidf_topics, scaler = models
        
        from reposcore_utils import featurize
        
        features = {
            "full_name": "test/repo",
            "html_url": "https://github.com/test/repo",
            "topics": ["python", "ml"],
            "stars": 100,
            "forks": 10,
            "open_issues": 5,
            "repo_age_days": 365,
            "last_commit_days": 7,
            "total_contributors": 5,
            "has_tests": True,
            "has_ci": True,
            "has_license": True,
            "readme_size": 5000,
            "readme_text_clean": "This is a test readme with documentation.",
            "commit_count_90d": 20,
            "issue_response_hours": 24,
            "pr_merge_time_days": 3,
            "contributor_commits_last_90d": 50,
            "is_archived": False,
            "is_fork": False,
            "stars_per_month": 2.5,
            "has_readme": 1,
        }
        
        X = featurize(features, tfidf_readme, tfidf_topics, scaler)
        prediction = rf_model.predict(X)[0]
        
        assert prediction in (0, 1)


class TestFastAPIEndpoints:
    """Test FastAPI endpoints using TestClient."""

    @pytest.fixture
    def client(self):
        """Create TestClient for FastAPI app."""
        from fastapi.testclient import TestClient
        from api_server import app
        return TestClient(app)

    def test_health_endpoint(self, client):
        """GET /api/health returns 200 with expected shape."""
        response = client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert isinstance(data["timestamp"], str)

    def test_submit_job_returns_job_id(self, client):
        """POST /api/jobs returns 200 with job_id."""
        response = client.post("/api/jobs?owner=testowner&repo=testrepo&threshold=0.3")
        
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert isinstance(data["job_id"], str)
        assert len(data["job_id"]) > 0
        assert data["status"] == "completed"
        assert "result_key" in data

    def test_get_repo_score_endpoint(self, client):
        """GET /api/score/{owner}/{repo} returns 200 with score data."""
        response = client.get("/api/score/testowner/testrepo")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_score" in data
        assert "components" in data
        assert "tier" in data
        assert "probability" in data
        assert "prediction" in data

    def test_get_job_status_endpoint(self, client):
        """GET /api/jobs/{job_id} returns 404 for non-existent job."""
        response = client.get("/api/jobs/nonexistent-job-id")
        
        assert response.status_code == 404

    def test_badge_endpoint(self, client):
        """GET /api/badge/{owner}/{repo} returns SVG."""
        response = client.get("/api/badge/testowner/testrepo")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/svg+xml"
        assert "svg" in response.text.lower()

    def test_shields_badge_endpoint(self, client):
        """GET /api/badges/{owner}/{repo} returns badge URL."""
        response = client.get("/api/badges/testowner/testrepo")
        
        assert response.status_code == 200
        data = response.json()
        assert "badge_url" in data
        assert "shields.io" in data["badge_url"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])