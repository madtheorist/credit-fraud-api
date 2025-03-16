import pytest
from fastapi.testclient import TestClient
from src.api import app

# Create a test client
client = TestClient(app)

def test_predict_valid_input():
    """Test FastAPI prediction endpoint with valid input."""

    data = [
        {"amt": 1000, "hour": 23, "time_since_last_minutes": 10, "category": "shopping_net"},
        {"amt": 50, "hour": 15, "time_since_last_minutes": 5, "category": "shopping_pos"},
    ]
    
    # Send POST request and check response status
    response = client.post("/predict", json=data)
    assert response.status_code == 200

    # Parse JSON response
    # Ensure response contains "predictions"
    json_data = response.json()
    assert "predictions" in json_data
    assert isinstance(json_data["predictions"], list)

def test_predict_invalid_input():
    """Test FastAPI prediction endpoint with invalid input."""
    # Send an empty list (invalid)
    response = client.post("/predict", json=[])
    assert response.status_code == 400

def test_predict_non_json_input():
    """Test FastAPI with non-JSON input."""
    response = client.post("/predict", data="not a json")
    # Ensure response is 422 (Unprocessable Entity)
    assert response.status_code == 422