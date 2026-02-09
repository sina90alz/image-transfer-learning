from fastapi import FastAPI, UploadFile, File, HTTPException
from app.schemas import PredictResponse, Prediction, MetadataResponse
from app.model import InferenceModel
from app.config import settings

app = FastAPI(title="Image Transfer Learning Inference", version="1.0.0")

# Hardcode your best-known metrics
METRICS_SUMMARY = {
    "test_accuracy": 0.7914,
    "macro_f1": 0.8020,
    "weighted_f1": 0.7911,
    "notes": "Main failure mode: TB predicted as NORMAL. See confusion matrix in README."
}

_model: InferenceModel | None = None

@app.on_event("startup")
def load_model():
    global _model
    _model = InferenceModel(
        model_path=settings.model_path,
        image_size=settings.image_size,
        device=settings.device,
    )

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/metadata", response_model=MetadataResponse)
def metadata():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return MetadataResponse(
        model_name=settings.model_name,
        model_version=settings.model_version,
        classes=_model.classes,
        metrics_summary=METRICS_SUMMARY,
    )

@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    image_bytes = await file.read()
    preds = _model.predict_topk(image_bytes, k=3)

    return PredictResponse(
        top_k=3,
        predictions=[Prediction(label=lbl, confidence=round(conf, 6)) for lbl, conf in preds],
    )
