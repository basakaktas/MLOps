import pytest
from fastapi.testclient import TestClient
from main import app
import pandas as pd
import os

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def test_data():
    test_movies = pd.DataFrame({
        'movieId': [1, 2, 3],
        'title': ['Toy Story', 'Toy Story 2', 'Jurassic Park']
    })
    return test_movies

def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "endpoints" in response.json()
    assert "/predict/" in str(response.json()["endpoints"])

def test_predict_endpoint_with_valid_movie(client, monkeypatch, test_data):
    def mock_get_movies():
        return test_data
    
    def mock_kneighbors(*args, **kwargs):
        return ([0.1, 0.2, 0.3], [[0, 1, 2]])  
    
    monkeypatch.setattr("main.movies", test_data)
    monkeypatch.setattr("main.knn.kneighbors", mock_kneighbors)
    
    response = client.get("/predict/", params={"movie_title": "Toy Story"})
    data = response.json()
    
    assert response.status_code == 200
    assert data["status"] == "success"
    assert "Toy Story 2" in data["recommendations"]  
    assert data["match"] == "Toy Story"

def test_predict_endpoint_with_unknown_movie(client, monkeypatch, test_data):
    monkeypatch.setattr("main.movies", test_data)
    
    response = client.get("/predict/", params={"movie_title": "Non-existent Movie"})
    data = response.json()
    
    assert response.status_code == 200
    assert data["status"] == "error"
    assert "Movie not found" in data["error"]
    assert len(data["suggestions"]) > 0  

def test_health_endpoint(client):
    response = client.get("/health")
    data = response.json()
    
    assert response.status_code == 200
    assert data["status"] == "healthy"
    assert isinstance(data["model_version"], str)
    assert isinstance(data["loaded"], bool)