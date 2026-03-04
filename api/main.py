from fastapi import FastAPI

from api.model import InferenceModel
from api.config import settings
from api.db.init_db import init_db
from api.routers import health, metadata, predict, admin

def create_app() -> FastAPI:
    app = FastAPI(title="Image Transfer Learning Inference", version="1.0.0")

    # keep your startup logic (we can refine later)
    @app.on_event("startup")
    async def startup():
        await init_db()
        app.state.model = InferenceModel(
            model_path=settings.model_path,
            image_size=settings.image_size,
            device=settings.device,
        )

    app.include_router(health.router)
    app.include_router(metadata.router)
    app.include_router(predict.router)
    app.include_router(admin.router)
    return app

app = create_app()