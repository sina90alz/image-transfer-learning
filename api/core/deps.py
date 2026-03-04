from fastapi import Header, HTTPException, status, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from api.db.session import get_db
from api.db.models import ApiClient
from api.core.security import hash_api_key

async def get_api_client(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    db: AsyncSession = Depends(get_db),
) -> ApiClient:
    if not x_api_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing X-API-Key")

    key_hash = hash_api_key(x_api_key)
    res = await db.execute(select(ApiClient).where(ApiClient.key_hash == key_hash))
    client = res.scalar_one_or_none()

    if not client or not client.is_active:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")

    return client