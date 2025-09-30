import json
import pytest
from api import app  # importa la app Flask desde api.py


@pytest.fixture
def client():
    app.testing = True
    with app.test_client() as client:
        yield client


def test_predict_endpoint(client):
    # ejemplo de payload con las 30 features del modelo
    payload = {
        "features": [
            14.5, 20.5, 95.0, 600.0, 0.1,
            0.15, 0.2, 0.1, 0.25, 0.07,
            0.5, 1.2, 3.5, 45.0, 0.01,
            0.04, 0.05, 0.02, 0.04, 0.009,
            16.0, 25.0, 110.0, 800.0, 0.13,
            0.23, 0.3, 0.14, 0.29, 0.08
        ]
    }

    response = client.post(
        "/predict",
        data=json.dumps(payload),
        content_type="application/json"
    )

    # Verifica que el endpoint responde bien
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data
    assert isinstance(data["prediction"], int) or isinstance(data["prediction"], float)
