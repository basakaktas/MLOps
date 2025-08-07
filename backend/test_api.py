from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/")
    assert response.status_code == 200

def test_prediction_valid():
    response = client.get("/predict/?movie_title=Toy")
    assert response.status_code == 200
    assert "recommendations" in response.json()

def test_prediction_invalid():
    response = client.get("/predict/?movie_title=asldkfj")
    assert response.status_code == 200
    assert "recommendations" in response.json() or "error" in response.json()
