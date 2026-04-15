from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List

class Settings(BaseSettings):
    # ========================= AH Config =========================
    APP_NAME: str
    APP_VERSION: str

    # ========================= LLM Config ============================
    GENERATION_BACKEND: str

    OPENAI_API_KEY: str
    OPENAI_BASE_URL: str
    GOOGLE_API_KEYS: List[str]

    GENERATION_MODEL_ID: str

    INPUT_DEFAULT_MAX_CHARACTERS: int
    GENERATION_DEFAULT_MAX_TOKENS: int
    GENERATION_DEFAULT_TEMPERATURE: float

    model_config = SettingsConfigDict(
        env_file=".env", 
        env_file_encoding="utf-8",
        case_sensitive=True
    )


def get_settings() -> Settings:
    return Settings()