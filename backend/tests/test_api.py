import io
import base64
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app, service


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(service, "load", lambda: None)
    monkeypatch.setattr(service, "model", None)
    monkeypatch.setattr(service, "error", "Custom model file is missing.")
    with TestClient(app) as test_client:
        yield test_client


def png_bytes(color=(230, 230, 230)):
    image = Image.new("RGB", (80, 60), color)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_health_without_model(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is False
    assert response.json()["prediction_available"] is False


def test_classes_mapping(client):
    classes = client.get("/api/classes").json()["classes"]
    assert [item["class_id"] for item in classes] == list(range(6))
    assert [item["model_name"] for item in classes] == [
        "Missing_hole", "Mouse_bite", "Open_circuit", "Short", "Spur", "Spurious_copper"
    ]


def test_invalid_image_rejected(client):
    response = client.post("/api/predict", files={"file": ("board.png", b"not an image", "image/png")})
    assert response.status_code == 400


def test_unsupported_extension_rejected(client):
    response = client.post("/api/predict", files={"file": ("board.gif", b"anything", "image/gif")})
    assert response.status_code == 400


def test_invalid_threshold_rejected(client):
    response = client.post("/api/predict", files={"file": ("board.png", png_bytes(), "image/png")}, data={"confidence_threshold": "1.5"})
    assert response.status_code == 422


def test_prediction_requires_custom_model(client):
    response = client.post("/api/predict", files={"file": ("board.png", png_bytes(), "image/png")})
    assert response.status_code == 503


def test_oversized_upload_rejected(client, monkeypatch):
    from app import main
    monkeypatch.setattr(main, "MAX_UPLOAD_SIZE_BYTES", 32)
    response = client.post("/api/predict", files={"file": ("board.png", png_bytes(), "image/png")})
    assert response.status_code == 413


def test_prediction_response_counts_and_annotated_image(client, monkeypatch):
    def mocked_predict(_image, _confidence):
        return {
            "image_width": 80, "image_height": 60, "total_defects": 2,
            "confidence_threshold": 0.25, "processing_time_ms": 9.1,
            "class_counts": {"Missing Hole": 1, "Mouse Bite": 1, "Open Circuit": 0, "Short": 0, "Spur": 0, "Spurious Copper": 0},
            "detections": [
                {"class_id": 0, "class_name": "Missing Hole", "confidence": 0.94, "bbox": {"x1": 1, "y1": 2, "x2": 20, "y2": 25}},
                {"class_id": 1, "class_name": "Mouse Bite", "confidence": 0.84, "bbox": {"x1": 30, "y1": 10, "x2": 45, "y2": 31}},
            ],
            "annotated_image": "data:image/jpeg;base64,dGVzdA==",
        }
    monkeypatch.setattr(service, "model", object())
    monkeypatch.setattr(service, "error", None)
    monkeypatch.setattr(service, "predict", mocked_predict)
    response = client.post("/api/predict", files={"file": ("board.png", png_bytes(), "image/png")})
    assert response.status_code == 200
    payload = response.json()
    assert payload["total_defects"] == len(payload["detections"]) == 2
    assert sum(payload["class_counts"].values()) == 2
    assert payload["annotated_image"].startswith("data:image/jpeg;base64,")


def test_zero_detection_response(client, monkeypatch):
    def mocked_predict(_image, confidence):
        return {
            "image_width": 80, "image_height": 60, "total_defects": 0,
            "confidence_threshold": confidence, "processing_time_ms": 1.0,
            "class_counts": {"Missing Hole": 0, "Mouse Bite": 0, "Open Circuit": 0, "Short": 0, "Spur": 0, "Spurious Copper": 0},
            "detections": [], "annotated_image": "data:image/jpeg;base64,dGVzdA==",
        }
    monkeypatch.setattr(service, "model", object())
    monkeypatch.setattr(service, "error", None)
    monkeypatch.setattr(service, "predict", mocked_predict)
    response = client.post("/api/predict", files={"file": ("board.png", png_bytes(), "image/png")})
    assert response.status_code == 200
    assert response.json()["total_defects"] == 0
    assert sum(response.json()["class_counts"].values()) == 0
    assert response.json()["detections"] == []


def test_model_output_is_converted_to_response_and_annotated_image(monkeypatch):
    class Tensor:
        def __init__(self, value):
            self.value = value

        def cpu(self):
            return self

        def tolist(self):
            return self.value

    class Boxes:
        xyxy = Tensor([[8.0, 9.0, 40.0, 44.0]])
        cls = Tensor([5.0])
        conf = Tensor([0.88])

        def __len__(self):
            return 1

    class Model:
        def predict(self, **kwargs):
            assert kwargs["conf"] == 0.4
            return [type("Result", (), {"boxes": Boxes()})()]

    from app.inference import ModelService
    inference = ModelService()
    inference.model = Model()
    image_response = inference.predict(png_bytes(), 0.4)
    assert image_response["image_width"] == 80
    assert image_response["total_defects"] == 1
    assert image_response["class_counts"]["Spurious Copper"] == 1
    assert image_response["detections"][0]["bbox"] == {"x1": 8.0, "y1": 9.0, "x2": 40.0, "y2": 44.0}
    encoded = image_response["annotated_image"].split(",", 1)[1]
    assert base64.b64decode(encoded).startswith(b"\xff\xd8")


def test_startup_with_missing_model_reports_unavailable(monkeypatch, tmp_path):
    monkeypatch.setattr(service, "model_path", tmp_path / "missing.pt")
    with TestClient(app) as test_client:
        payload = test_client.get("/api/health").json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is False
    assert payload["prediction_available"] is False


def test_real_model_smoke_when_model_and_sample_are_available():
    from app.config import MODEL_PATH
    from app.inference import ModelService

    sample = os.getenv("PCB_SMOKE_IMAGE")
    if not MODEL_PATH.is_file() or not sample or not Path(sample).is_file():
        pytest.skip("Requires backend/models/pcb_defect_best.pt and PCB_SMOKE_IMAGE pointing to a sample image.")
    real_service = ModelService(MODEL_PATH)
    real_service.load()
    assert real_service.ready, real_service.error
    prediction = real_service.predict(Path(sample).read_bytes(), 0.25)
    assert prediction["total_defects"] == len(prediction["detections"])
    assert prediction["annotated_image"].startswith("data:image/jpeg;base64,")
