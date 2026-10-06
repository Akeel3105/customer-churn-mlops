from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health():
    response = client.get("/")

    assert response.status_code == 200
    # assert response.status_code == 500   # intentionally wrong to check CI CD pipeline on render
    assert response.json() == {"status": "healthy"}


def test_predict():
    features = [0.0] * 30

    response = client.post(
        "/predict",
        json={"features": features}
    )

    assert response.status_code == 200
    assert response.json()["prediction"] in [0, 1]