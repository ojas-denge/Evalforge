from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "EvalForge"
    app_version: str = "0.1.0"
    app_env: str = "development"
    log_level: str = "INFO"

    openai_api_key: str | None = None
    llm_model: str = "gpt-4.1-mini"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    chroma_host: str = "localhost"
    chroma_port: int = 8000

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
