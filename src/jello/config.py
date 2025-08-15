from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    model_id: str | None = None
    dtype: str = "auto"       # auto|float32|bfloat16|float16
    device_map: str = "auto"  # auto|cpu|cuda|mps
    max_new_tokens: int = 512
    temperature: float = 0.2
    top_p: float = 0.95
    seed: int = 42
