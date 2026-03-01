from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String, DateTime, Boolean, Integer, ForeignKey, Float, func

class Base(DeclarativeBase):
    pass

class ApiClient(Base):
    __tablename__ = "api_clients"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now())

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    client_id: Mapped[int] = mapped_column(ForeignKey("api_clients.id"), index=True)

    endpoint: Mapped[str] = mapped_column(String(50))
    latency_ms: Mapped[float] = mapped_column(Float)

    top1_label: Mapped[str] = mapped_column(String(50))
    top1_confidence: Mapped[float] = mapped_column(Float)

    created_at: Mapped[str] = mapped_column(DateTime(timezone=True), server_default=func.now())