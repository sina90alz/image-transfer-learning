from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from inference.app.core.config import settings
from inference.app.core.security import generate_api_key, hash_api_key
from inference.app.db.session import get_db
from inference.app.db.models import ApiClient

router = APIRouter(prefix="/admin", tags=["admin"])

def require_admin(x_admin_key: str | None = Header(default=None, alias="X-Admin-Key")):
    if x_admin_key != settings.ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized admin")

@router.post("/api-keys")
async def create_api_key(payload: dict, _: None = Depends(require_admin), db: AsyncSession = Depends(get_db)):
    name = payload.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="name is required")

    exists = await db.execute(select(ApiClient).where(ApiClient.name == name))
    if exists.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Client name already exists")

    raw = generate_api_key()
    client = ApiClient(name=name, key_hash=hash_api_key(raw), is_active=True)
    db.add(client)
    await db.commit()

    return {"name": name, "api_key": raw}  # one-time reveal