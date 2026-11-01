"""YOLO model lifecycle and in-memory PCB image inference."""

from __future__ import annotations

import base64
import io
import logging
import threading
import time
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from .config import CLASS_NAMES, FRIENDLY_CLASS_NAMES, MODEL_PATH, SUPPORTED_PIL_FORMATS

logger = logging.getLogger(__name__)


class ModelUnavailableError(Exception):
    """Raised when prediction is requested without a valid custom model."""


class InvalidImageError(Exception):
    """Raised when uploaded bytes cannot be decoded as a supported image."""


class ModelService:
    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        self.model_path = model_path
        self.model: Any | None = None
        self.error: str | None = None
        self._inference_lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.model is not None and self.error is None

    def load(self) -> None:
        self.model = None
        self.error = None
        if not self.model_path.is_file():
            self.error = "Custom model file is missing."
            logger.info("Custom YOLO model not found at configured model location.")
            return
        try:
            from ultralytics import YOLO

            model = YOLO(str(self.model_path))
            actual_names = self._normalize_names(getattr(model, "names", None))
            if actual_names != list(CLASS_NAMES):
                self.error = "Model class metadata does not match the required six classes and order."
                logger.error("Custom model class metadata does not match the required six-class mapping.")
                return
            self.model = model
            logger.info("Custom YOLO model loaded and class metadata validated.")
        except Exception as exc:
            self.error = "The custom model could not be loaded. Check the backend logs."
            logger.exception("Failed to load or validate custom YOLO model.")

    @staticmethod
    def _normalize_names(names: Any) -> list[str] | None:
        if isinstance(names, dict):
            try:
                return [str(names[index]) for index in range(len(names))]
            except (KeyError, TypeError):
                return None
        if isinstance(names, (list, tuple)):
            return [str(name) for name in names]
        return None

    def predict(self, image_bytes: bytes, confidence: float) -> dict[str, Any]:
        try:
            image = Image.open(io.BytesIO(image_bytes))
            if image.width * image.height > 50_000_000:
                raise InvalidImageError("Image dimensions exceed the 50 megapixel processing limit.")
            image.load()
            if image.format not in SUPPORTED_PIL_FORMATS:
                raise InvalidImageError("Unsupported image format. Upload a JPEG, PNG, or WEBP image.")
            image = image.convert("RGB")
        except InvalidImageError:
            raise
        except (UnidentifiedImageError, Image.DecompressionBombError, OSError, ValueError) as exc:
            raise InvalidImageError("The uploaded file is not a readable image.") from exc

        if not self.ready:
            raise ModelUnavailableError("The trained PCB defect model is not available. Add the model file and restart the backend.")

        width, height = image.size
        started = time.perf_counter()
        try:
            with self._inference_lock:
                results = self.model.predict(source=image, conf=confidence, verbose=False)
            result = results[0]
            boxes = result.boxes
            detections: list[dict[str, Any]] = []
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.cpu().tolist()
                class_ids = boxes.cls.cpu().tolist()
                confidences = boxes.conf.cpu().tolist()
                for coords, class_value, confidence_value in zip(xyxy, class_ids, confidences):
                    class_id = int(class_value)
                    if not 0 <= class_id < len(FRIENDLY_CLASS_NAMES):
                        raise ValueError("Prediction contained a class ID outside the configured class mapping.")
                    x1, y1, x2, y2 = coords
                    detections.append({
                        "class_id": class_id,
                        "class_name": FRIENDLY_CLASS_NAMES[class_id],
                        "confidence": float(confidence_value),
                        "bbox": {
                            "x1": round(max(0.0, min(float(x1), width)), 2),
                            "y1": round(max(0.0, min(float(y1), height)), 2),
                            "x2": round(max(0.0, min(float(x2), width)), 2),
                            "y2": round(max(0.0, min(float(y2), height)), 2),
                        },
                    })
            elapsed_ms = (time.perf_counter() - started) * 1000
        except Exception as exc:
            logger.exception("Custom YOLO inference failed.")
            raise RuntimeError("Inference failed while processing the image.") from exc

        detections.sort(key=lambda item: item["confidence"], reverse=True)
        counts = {name: 0 for name in FRIENDLY_CLASS_NAMES}
        for detection in detections:
            counts[detection["class_name"]] += 1
        try:
            annotated = self._annotate(image, detections)
        except Exception as exc:
            logger.exception("Failed to prepare annotated inference result.")
            raise RuntimeError("Inference succeeded but the annotated image could not be prepared.") from exc
        encoded = base64.b64encode(annotated).decode("ascii")
        return {
            "image_width": width,
            "image_height": height,
            "total_defects": len(detections),
            "confidence_threshold": confidence,
            "processing_time_ms": round(elapsed_ms, 2),
            "class_counts": counts,
            "detections": detections,
            "annotated_image": f"data:image/jpeg;base64,{encoded}",
        }

    @staticmethod
    def _annotate(image: Image.Image, detections: list[dict[str, Any]]) -> bytes:
        canvas = image.copy()
        draw = ImageDraw.Draw(canvas)
        width, height = canvas.size
        stroke = max(2, round(min(width, height) / 400))
        try:
            font = ImageFont.truetype("arial.ttf", max(12, round(min(width, height) / 45)))
        except OSError:
            font = ImageFont.load_default()
        colors = [(44, 201, 177), (255, 177, 74), (88, 157, 255), (255, 103, 117), (190, 132, 255), (112, 208, 130)]
        for item in detections:
            box = item["bbox"]
            color = colors[item["class_id"] % len(colors)]
            bounds = (box["x1"], box["y1"], box["x2"], box["y2"])
            draw.rectangle(bounds, outline=color, width=stroke)
            label = f'{item["class_name"]} {item["confidence"]:.0%}'
            text_box = draw.textbbox((0, 0), label, font=font)
            label_height = text_box[3] - text_box[1] + 8
            label_top = max(0, box["y1"] - label_height)
            draw.rectangle((box["x1"], label_top, box["x1"] + text_box[2] + 12, label_top + label_height), fill=color)
            draw.text((box["x1"] + 6, label_top + 3), label, fill=(15, 25, 39), font=font)
        output = io.BytesIO()
        canvas.save(output, format="JPEG", quality=92)
        return output.getvalue()
