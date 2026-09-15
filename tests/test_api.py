"""Tests for the FastAPI application, using FastAPI's TestClient."""

import pytest
from fastapi.testclient import TestClient

from api.main import app
from src import config

client = TestClient(app)


def _valid_payload(**overrides):
    payload = {
        "bedrooms": 3,
        "bathrooms": 2.0,
        "living_area": 1500,
        "lot_area": 8000,
        "floors": 1.0,
        "garage_cars": 2,
        "year_built": 1998,
        "year_remodeled": 2005,
        "neighborhood": "NAmes",
        "bldg_type": "1Fam",
        "overall_qual": 6,
        "overall_cond": 5,
        "total_rooms": 6,
        "fireplaces": 1,
        "central_air": "Y",
        "has_deck_or_porch": True,
    }
    payload.update(overrides)
    return payload


def test_root_endpoint():
    res = client.get("/")
    assert res.status_code == 200
    body = res.json()
    assert "name" in body
    assert "endpoints" in body


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "healthy"}


@pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="Model must be trained first")
def test_model_info_endpoint():
    res = client.get("/model-info")
    assert res.status_code == 200
    body = res.json()
    assert "model_name" in body
    assert "supported_features" in body


@pytest.mark.skipif(not config.MODEL_PATH.exists(), reason="Model must be trained first")
def test_predict_endpoint_returns_price():
    res = client.post("/predict", json=_valid_payload())
    assert res.status_code == 200
    body = res.json()
    assert body["predicted_price"] > 0
    assert body["currency"] == "USD"
    assert "disclaimer" in body


def test_predict_endpoint_rejects_invalid_neighborhood():
    res = client.post("/predict", json=_valid_payload(neighborhood="Atlantis"))
    assert res.status_code == 422


def test_predict_endpoint_rejects_negative_bedrooms():
    res = client.post("/predict", json=_valid_payload(bedrooms=-1))
    assert res.status_code == 422


def test_predict_endpoint_rejects_missing_field():
    payload = _valid_payload()
    del payload["living_area"]
    res = client.post("/predict", json=payload)
    assert res.status_code == 422


def test_predict_endpoint_rejects_bad_central_air_value():
    res = client.post("/predict", json=_valid_payload(central_air="Maybe"))
    assert res.status_code == 422
