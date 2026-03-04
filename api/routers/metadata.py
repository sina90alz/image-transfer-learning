from fastapi import APIRouter, HTTPException, Request
from app.schemas import MetadataResponse
from app.config import settings

router = APIRouter(tags=["metadata"])

METRICS_SUMMARY = {
    "test_accuracy": 0.7914,
    "macro_f1": 0.8020,
    "weighted_f1": 0.7911,
    "notes": "Main failure mode: TB predicted as NORMAL. See confusion matrix in README."
}

@router.get("/metadata", response_model=MetadataResponse)
def metadata(request: Request):
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return MetadataResponse(
        model_name=settings.model_name,
        model_version=settings.model_version,
        classes=model.classes,
        metrics_summary=METRICS_SUMMARY,
    )