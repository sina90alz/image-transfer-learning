from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        protected_namespaces=(),
        env_file=".env",
        extra="ignore",
    )

    # Inference/runtime config
    model_path: str = "artifacts/model.pt"
    image_size: int = 160
    model_name: str = "efficientnet_b0"
    model_version: str = "best_efficientnet_b0"
    device: str = "cpu"

    # API/DB config
    DATABASE_URL: str = "sqlite+aiosqlite:///./inference/app.db"
    ADMIN_KEY: str = "change-me"


settings = Settings()
