import secrets
import hashlib

def generate_api_key(prefix: str = "ml") -> str:
    return f"{prefix}_live_{secrets.token_urlsafe(32)}"

def hash_api_key(raw: str) -> str:
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()