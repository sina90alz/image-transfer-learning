from pydantic import BaseModel
from typing import List

class Prediction(BaseModel):
    label: str
    confidence: float

class PredictResponse(BaseModel):
    top_k: int
    predictions: List[Prediction]

class MetadataResponse(BaseModel):
    model_name: str
    model_version: str
    classes: List[str]
    metrics_summary: dict
