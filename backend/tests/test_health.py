
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["project"] == "TradingArena"
