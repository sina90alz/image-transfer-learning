from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite+aiosqlite:///./inference/app.db"
    ADMIN_KEY: str = "change-me"

settings = Settings()