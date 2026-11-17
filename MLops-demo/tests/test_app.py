import pytest
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200


def test_predict_valid_input():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
        "petal_length": 1.4,
        "petal_width": 0.2,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code in [200, 503]

    if response.status_code == 200:
        assert "prediction" in response.json()
        assert response.json()["prediction"] in [0, 1, 2]


def test_predict_invalid_schema():
    payload = {
        "sepal_length": 5.1,
        "sepal_width": 3.5,
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
