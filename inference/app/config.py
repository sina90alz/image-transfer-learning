from pydantic import BaseModel, ConfigDict

class Settings(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    
    model_path: str = "app/model.pt"
    image_size: int = 160
    model_name: str = "efficientnet_b0"
    model_version: str = "best_efficientnet_b0"
    device: str = "cpu"

settings = Settings()
