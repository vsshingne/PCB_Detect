from pydantic import BaseModel, ConfigDict, Field


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    prediction_available: bool
    model_error: str | None = None


class ClassInfo(BaseModel):
    class_id: int
    model_name: str
    display_name: str


class ClassesResponse(BaseModel):
    classes: list[ClassInfo]


class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float


class Detection(BaseModel):
    class_id: int
    class_name: str
    confidence: float = Field(ge=0, le=1)
    bbox: BoundingBox


class PredictionResponse(BaseModel):
    image_width: int
    image_height: int
    total_defects: int
    confidence_threshold: float
    processing_time_ms: float
    class_counts: dict[str, int]
    detections: list[Detection]
    annotated_image: str


class ErrorResponse(BaseModel):
    detail: str
