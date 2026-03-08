import time
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException, Request, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from api.core.deps import get_api_client
from api.db.session import get_db
from api.db.models import PredictionLog
from api.schemas import PredictResponse, Prediction

router = APIRouter(tags=["predict"])

@router.post("/predict", response_model=PredictResponse)
async def predict(
    request: Request,
    file: UploadFile = File(...),
    _client = Depends(get_api_client),
    db: AsyncSession = Depends(get_db),
):
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    t0 = time.perf_counter()
    image_bytes = await file.read()
    preds = model.predict_topk(image_bytes, k=3)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    # log top-1
    top1_label, top1_conf = preds[0]
    db.add(PredictionLog(
        client_id=_client.id,
        endpoint="/predict",
        latency_ms=latency_ms,
        top1_label=top1_label,
        top1_confidence=float(top1_conf),
    ))
    await db.commit()

    return PredictResponse(
        top_k=3,
        predictions=[Prediction(label=lbl, confidence=round(conf, 6)) for lbl, conf in preds],
    )

@router.post("/predict_batch")
async def predict_batch(
    request: Request,
    files: List[UploadFile] = File(...),
    _client = Depends(get_api_client),
    db: AsyncSession = Depends(get_db),
):
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    t0_req = time.perf_counter()

    outputs = []
    for f in files:
        if f.content_type is None or not f.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail=f"Invalid file type: {f.filename}")

        image_bytes = await f.read()
        preds = model.predict_topk(image_bytes, k=3)
        outputs.append({
            "filename": f.filename,
            "predictions": [{"label": lbl, "confidence": round(conf, 6)} for lbl, conf in preds],
        })

    latency_ms = (time.perf_counter() - t0_req) * 1000.0

    # log batch summary as one row (top-1 of first image, or change if you prefer)
    if outputs:
        top1 = outputs[0]["predictions"][0]
        db.add(PredictionLog(
            client_id=_client.id,
            endpoint="/predict_batch",
            latency_ms=latency_ms,
            top1_label=top1["label"],
            top1_confidence=float(top1["confidence"]),
        ))
        await db.commit()

    return {"count": len(outputs), "results": outputs}