from inference.app.db.session import engine
from inference.app.db.models import Base

async def init_db() -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)