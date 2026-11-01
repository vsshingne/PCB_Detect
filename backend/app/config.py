"""Application configuration loaded from environment variables."""

import os
from pathlib import Path

from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".env")


def _csv_env(name: str, default: str) -> list[str]:
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


def _positive_int_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
        return value if value > 0 else default
    except ValueError:
        return default


def _confidence_env() -> float:
    try:
        value = float(os.getenv("DEFAULT_CONFIDENCE_THRESHOLD", "0.25"))
        return value if 0 <= value <= 1 else 0.25
    except ValueError:
        return 0.25


MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/pcb_defect_best.pt"))
if not MODEL_PATH.is_absolute():
    MODEL_PATH = BACKEND_DIR / MODEL_PATH
MODEL_PATH = MODEL_PATH.resolve()
MAX_UPLOAD_SIZE_BYTES = _positive_int_env("MAX_UPLOAD_SIZE_MB", 10) * 1024 * 1024
DEFAULT_CONFIDENCE_THRESHOLD = _confidence_env()
ALLOWED_ORIGINS = _csv_env(
    "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
)
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SUPPORTED_PIL_FORMATS = {"JPEG", "PNG", "WEBP"}
CLASS_NAMES = (
    "Missing_hole",
    "Mouse_bite",
    "Open_circuit",
    "Short",
    "Spur",
    "Spurious_copper",
)
FRIENDLY_CLASS_NAMES = (
    "Missing Hole",
    "Mouse Bite",
    "Open Circuit",
    "Short",
    "Spur",
    "Spurious Copper",
)
