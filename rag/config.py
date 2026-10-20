"""Configuration loaded from .env and validated at import time."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from rag import __version__

BASE_DIR = Path(__file__).resolve().parents[1]
VERSION = __version__


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM (OpenAI-compatible)
    llm_base_url: str = "https://api.siliconflow.cn/v1"
    llm_api_key: str = ""
    llm_model: str = "deepseek-ai/DeepSeek-V3"
    temperature: float = 0.2
    max_tokens: int = 1024
    llm_timeout: int = 90

    # PDF parsing
    scan_page_char_threshold: int = 200
    table_column_ratio: float = 0.4
    max_context_chars: int = 12000

    # Reserved for V2
    chunk_strategy: str = "semantic"
    chunk_size: int = 500
    chunk_overlap: int = 80

    @property
    def upload_dir(self) -> Path:
        path = BASE_DIR / "data" / "uploads"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key.strip())

    @property
    def chat_completions_url(self) -> str:
        return self.llm_base_url.rstrip("/") + "/chat/completions"

    @property
    def models_url(self) -> str:
        return self.llm_base_url.rstrip("/") + "/models"


settings = Settings()
