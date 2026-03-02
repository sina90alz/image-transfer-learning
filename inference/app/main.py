from fastapi import FastAPI

from app.model import InferenceModel
from app.config import settings

from inference.app.db.init_db import init_db
from inference.app.routers import health, metadata, predict, admin

def create_app() -> FastAPI:
    app = FastAPI(title="Image Transfer Learning Inference", version="1.0.0")

    @app.lifespan("startup")
    async def startup():
        # init DB tables
        await init_db()

        # load model once
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