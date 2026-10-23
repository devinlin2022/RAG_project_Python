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
    # Below this many chars a page dense in vector shapes is treated as a diagram
    figure_text_max: int = 1500

    # Public tunnel (optional)
    ngrok_enabled: bool = False
    ngrok_authtoken: str = ""

    # OCR / figure recognition. Reuses the chat key by default.
    ocr_enabled: bool = False
    ocr_base_url: str = ""
    ocr_api_key: str = ""
    ocr_model: str = "PaddlePaddle/PaddleOCR-VL-1.5"
    vision_model: str = "Qwen/Qwen2.5-VL-72B-Instruct"
    ocr_render_dpi: int = 150
    ocr_max_pages: int = 20
    ocr_timeout: int = 120
    enhance_max_seconds: int = 120

    # Embeddings (semantic chunking). bge-m3 is free on SiliconFlow.
    embed_base_url: str = ""
    embed_api_key: str = ""
    embed_model: str = "BAAI/bge-m3"
    embed_batch: int = 32

    # Chunking
    chunk_strategy: str = "auto"
    chunk_size: int = 500
    chunk_overlap: int = 80
    chunk_min_chars: int = 120
    semantic_percentile: float = 85.0

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

    # OCR and embeddings fall back to the chat key so one .env is enough.
    @property
    def ocr_base_url_resolved(self) -> str:
        return (self.ocr_base_url or self.llm_base_url).rstrip("/")

    @property
    def ocr_api_key_resolved(self) -> str:
        return (self.ocr_api_key or self.llm_api_key).strip()

    @property
    def ocr_ready(self) -> bool:
        return self.ocr_enabled and bool(self.ocr_api_key_resolved)

    @property
    def embed_base_url_resolved(self) -> str:
        return (self.embed_base_url or self.llm_base_url).rstrip("/")

    @property
    def embed_api_key_resolved(self) -> str:
        return (self.embed_api_key or self.llm_api_key).strip()

    @property
    def embed_ready(self) -> bool:
        return bool(self.embed_api_key_resolved)


settings = Settings()
