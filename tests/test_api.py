"""
Tests for the Churn Prediction REST API.

Covers all 6 endpoints with valid requests, edge cases, and error handling.
Uses FastAPI's TestClient for synchronous testing without running a server.
"""

import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)


class TestHealth:
    def test_health_returns_200(self):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_includes_version(self):
        data = client.get("/health").json()
        assert "version" in data
        assert data["status"] == "healthy"


class TestPredict:
    def test_predict_default(self):
        resp = client.post("/predict", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert len(data["results"]) == 3

    def test_predict_custom_samples(self):
        resp = client.post("/predict", json={"n_samples": 200})
        assert resp.status_code == 200
        data = resp.json()
        assert data["n_samples"] == 200
        assert "churn_rate" in data

    def test_predict_custom_test_size(self):
        resp = client.post("/predict", json={"n_samples": 200, "test_size": 0.3})
        assert resp.status_code == 200
        assert resp.json()["test_size"] == 0.3

    def test_predict_too_few_samples(self):
        resp = client.post("/predict", json={"n_samples": 10})
        assert resp.status_code == 422  # validation error

    def test_predict_all_metrics_present(self):
        data = client.post("/predict", json={"n_samples": 200}).json()
        expected_metrics = {"accuracy", "precision", "recall", "f1", "roc_auc"}
        for model_results in data["results"].values():
            assert set(model_results.keys()) == expected_metrics


class TestPredictCustomer:
    @pytest.fixture
    def valid_customer(self):
        return {
            "age": 30, "tenure": 6, "monthly_charges": 80,
            "total_charges": 480, "contract_type": "month",
            "internet_service": "Fiber", "payment_method": "electronic",
            "model_name": "gradient_boosting", "n_training_samples": 200,
        }

    def test_valid_prediction(self, valid_customer):
        resp = client.post("/predict/customer", json=valid_customer)
        assert resp.status_code == 200
        data = resp.json()
        assert "churn_probability" in data
        assert "risk_level" in data
        assert data["risk_level"] in ("LOW", "MEDIUM", "HIGH")

    def test_invalid_model(self, valid_customer):
        valid_customer["model_name"] = "neural_net"
        resp = client.post("/predict/customer", json=valid_customer)
        assert resp.status_code == 400

    def test_invalid_contract_type(self, valid_customer):
        valid_customer["contract_type"] = "invalid"
        resp = client.post("/predict/customer", json=valid_customer)
        assert resp.status_code == 400


class TestFeatureImportance:
    def test_default_request(self):
        resp = client.post("/feature-importance", json={})
        assert resp.status_code == 200
        data = resp.json()
        assert "importance" in data
        assert len(data["importance"]) == 7

    def test_logistic_regression(self):
        resp = client.post(
            "/feature-importance",
            json={"model_name": "logistic_regression", "n_samples": 200},
        )
        assert resp.status_code == 200
        importance = resp.json()["importance"]
        # Values should sum to ~1.0
        assert abs(sum(importance.values()) - 1.0) < 0.01

    def test_invalid_model(self):
        resp = client.post(
            "/feature-importance", json={"model_name": "invalid"}
        )
        assert resp.status_code == 400


class TestCrossValidate:
    def test_default_request(self):
        resp = client.post("/cross-validate", json={"n_samples": 200})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 3

    def test_custom_splits(self):
        resp = client.post(
            "/cross-validate", json={"n_samples": 200, "n_splits": 3}
        )
        assert resp.status_code == 200
        assert resp.json()["n_splits"] == 3


class TestConfusionMatrix:
    def test_default_request(self):
        resp = client.post("/confusion-matrix", json={"n_samples": 200})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["results"]) == 3

    def test_matrix_structure(self):
        data = client.post(
            "/confusion-matrix", json={"n_samples": 300}
        ).json()
        for model_name, result in data["results"].items():
            assert "matrix" in result
            assert "tn" in result
            assert "fp" in result
            assert "fn" in result
            assert "tp" in result
            # All values non-negative
            assert result["tn"] >= 0
            assert result["fp"] >= 0


class TestCompareModels:
    def test_default_request(self):
        resp = client.post("/compare", json={"n_samples": 200})
        assert resp.status_code == 200
        data = resp.json()
        assert "recommended_model" in data
        assert "reason" in data
        assert "holdout_results" in data
        assert "cv_results" in data
        assert "rankings" in data

    def test_recommended_model_is_valid(self):
        data = client.post("/compare", json={"n_samples": 300}).json()
        valid = {"logistic_regression", "random_forest", "gradient_boosting"}
        assert data["recommended_model"] in valid

    def test_rankings_cover_all_metrics(self):
        data = client.post("/compare", json={"n_samples": 200}).json()
        expected = {"accuracy", "precision", "recall", "f1", "roc_auc"}
        assert set(data["rankings"].keys()) == expected


class TestOptimalThreshold:
    def test_default_request(self):
        resp = client.post("/optimal-threshold", json={"n_samples": 200})
        assert resp.status_code == 200
        data = resp.json()
        assert "optimal_threshold" in data
        assert "best_score" in data
        assert 0.05 <= data["optimal_threshold"] <= 0.95

    def test_custom_metric(self):
        resp = client.post(
            "/optimal-threshold",
            json={"n_samples": 200, "metric": "recall"},
        )
        assert resp.status_code == 200
        assert resp.json()["metric"] == "recall"

    def test_invalid_model(self):
        resp = client.post(
            "/optimal-threshold",
            json={"model_name": "invalid"},
        )
        assert resp.status_code == 400

    def test_invalid_metric(self):
        resp = client.post(
            "/optimal-threshold",
            json={"metric": "rmse"},
        )
        assert resp.status_code == 400


class TestLearningCurve:
    def test_default_request(self):
        resp = client.post("/learning-curve", json={"n_samples": 200})
        assert resp.status_code == 200
        data = resp.json()
        assert "train_sizes" in data
        assert "train_scores" in data
        assert "test_scores" in data

    def test_custom_points(self):
        resp = client.post(
            "/learning-curve",
            json={"n_samples": 300, "n_points": 5},
        )
        data = resp.json()
        assert len(data["train_sizes"]) == 5
        assert len(data["train_scores"]) == 5

    def test_invalid_model(self):
        resp = client.post(
            "/learning-curve",
            json={"model_name": "invalid"},
        )
        assert resp.status_code == 400
