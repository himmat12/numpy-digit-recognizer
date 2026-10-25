import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app
from digit_recognizer.network import NeuralNetwork


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A test client serving a tiny untrained model saved in a temp folder."""
    model_path = tmp_path / "model.npz"
    NeuralNetwork(784, 8, 10).save(model_path)
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    monkeypatch.setenv("MODEL_VERSION", "test-model")

    with TestClient(app) as test_client:  # "with" runs the startup (model loading)
        yield test_client


def assert_valid_prediction(body):
    assert 0 <= body["digit"] <= 9
    assert len(body["probabilities"]) == 10
    assert np.isclose(sum(body["probabilities"]), 1.0)
    assert body["model_version"] == "test-model"


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_version": "test-model"}


def test_predict_pixels(client):
    response = client.post("/predict/pixels", json={"pixels": [0.5] * 784})
    assert response.status_code == 200
    assert_valid_prediction(response.json())


def test_predict_pixels_rejects_wrong_length(client):
    response = client.post("/predict/pixels", json={"pixels": [0.5] * 10})
    assert response.status_code == 422


def test_predict_image(client):
    buffer = io.BytesIO()
    Image.new("L", (100, 100), color=255).save(buffer, format="PNG")

    response = client.post("/predict", files={"file": ("digit.png", buffer.getvalue(), "image/png")})
    assert response.status_code == 200
    assert_valid_prediction(response.json())


def test_predict_rejects_non_image(client):
    response = client.post("/predict", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400
