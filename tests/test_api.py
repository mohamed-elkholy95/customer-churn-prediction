import pytest
from fastapi.testclient import TestClient
from src.api.main import app
client = TestClient(app)

class TestAPI:
    def test_health(self): assert client.get("/health").status_code == 200
    def test_predict(self):
        resp = client.post("/predict", json={"n_samples": 200})
        assert resp.status_code == 200
        assert "results" in resp.json()
