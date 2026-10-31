"""Configuration loaded from .env and validated at import time.

Chat, embedding, rerank and vision read the same key unless one is
deliberately overridden.
"""

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from rag import __version__

BASE_DIR = Path(__file__).resolve().parents[1]
VERSION = __version__

DEFAULT_BASE_URL = "https://api.siliconflow.cn/v1"
DEFAULT_LLM = "deepseek-ai/DeepSeek-V3.2"
DEFAULT_EMBED = "BAAI/bge-m3"
DEFAULT_RERANK = "BAAI/bge-reranker-v2-m3"
# Scanned pages only need faithful transcription, and 8B does that as accurately
# as 32B while spending a third of the time (measured on the demo policy: same
# 272 chars, 7.7s vs 17.5s). Figures get the 32B model, which reasons about arrows.
DEFAULT_OCR = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_VISION = "Qwen/Qwen3-VL-32B-Instruct"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # ── one entry for every capability ───────────────────────────────
    # SILICONFLOW_API_KEY / SILICONFLOW_BASE_URL are the documented names;
    # the LLM_* names are kept so an older .env keeps working.
    llm_api_key: str = Field(
        "",
        validation_alias=AliasChoices("SILICONFLOW_API_KEY", "LLM_API_KEY", "llm_api_key"),
    )
    llm_base_url: str = Field(
        DEFAULT_BASE_URL,
        validation_alias=AliasChoices("SILICONFLOW_BASE_URL", "LLM_BASE_URL", "llm_base_url"),
    )

    # ── models: one id per capability, all on the same platform ─────
    llm_model: str = Field(
        DEFAULT_LLM,
        validation_alias=AliasChoices("LLM_MODEL", "SILICONFLOW_MODEL_NAME", "llm_model"),
    )
    embed_model: str = Field(
        DEFAULT_EMBED, validation_alias=AliasChoices("EMBED_MODEL", "embed_model")
    )
    rerank_model: str = Field(
        DEFAULT_RERANK, validation_alias=AliasChoices("RERANK_MODEL", "rerank_model")
    )
    # OCR is a prompt over a vision model, not a separate service. Two ids on
    # purpose: transcription on scanned pages wants the cheap model, figures
    # want the one that reasons about what the arrows mean.
    ocr_model: str = Field(
        DEFAULT_OCR, validation_alias=AliasChoices("OCR_MODEL", "ocr_model")
    )
    vision_model: str = Field(
        DEFAULT_VISION, validation_alias=AliasChoices("VISION_MODEL", "vision_model")
    )

    temperature: float = 0.2
    max_tokens: int = 1024
    llm_timeout: int = 90

    # ── parsing ─────────────────────────────────────────────────────
    scan_page_char_threshold: int = 200
    table_column_ratio: float = 0.4
    max_context_chars: int = 12000
    # Below this many chars a page dense in vector shapes is treated as a diagram
    figure_text_max: int = 1500
    # Parsers that have no real pages (docx / md / xlsx) are cut into virtual
    # pages of this size so citations, deletion and evaluation still work.
    native_page_chars: int = 1800
    # Files parsed in parallel during one upload
    parse_workers: int = 4
    # pdfplumber's ruled-table pass is the slowest part of ingest; only pages
    # that already look tabular are scanned, up to this many.
    table_scan_max_pages: int = 40

    # ── public tunnel ───────────────────────────────────────────────
    ngrok_enabled: bool = False
    ngrok_authtoken: str = ""

    # ── vision rails (OCR / figure caption) ─────────────────────────
    # On by default: the moment a key resolves, scanned pages should be read.
    # Set OCR_ENABLED=false to keep the vision model out of the loop entirely.
    ocr_enabled: bool = True
    ocr_base_url: str = ""
    ocr_api_key: str = ""
    ocr_render_dpi: int = 150
    ocr_max_pages: int = 20
    ocr_timeout: int = 120
    # Concurrent vision calls. 1 restores the old serial behaviour.
    ocr_concurrency: int = 3
    ocr_min_interval: float = 0.0
    http_retries: int = 3
    # Seconds between any two platform calls, across every thread. The embed,
    # rerank and vision paths each run a thread pool; without a shared gate
    # three at once looks like a burst and comes back 429. 0 disables pacing.
    http_min_interval: float = 0.12
    enhance_max_seconds: int = 600

    # ── embeddings ──────────────────────────────────────────────────
    embed_base_url: str = ""
    embed_api_key: str = ""
    embed_batch: int = 32
    # Batches sent in parallel while indexing a document
    embed_concurrency: int = 3

    # ── chunking ────────────────────────────────────────────────────
    chunk_strategy: str = "auto"
    chunk_size: int = 500
    chunk_overlap: int = 80
    chunk_min_chars: int = 120
    semantic_percentile: float = 85.0

    # ── retrieval ───────────────────────────────────────────────────
    retrieve_top_k: int = 6
    rerank_candidates: int = 24
    rrf_k: int = 60
    retrieve_strategy: str = "rrf"  # dense | sparse | rrf
    rerank_enabled: bool = True
    rerank_timeout: int = 60
    rerank_floor: float = 0.0

    # ── multi-turn and streaming ────────────────────────────────────
    history_turns: int = 3
    rewrite_enabled: bool = True

    # ── storage ─────────────────────────────────────────────────────
    db_filename: str = "rag.sqlite"
    max_upload_mb: int = 60

    # ── cost display (0 = hide the money line) ──────────────────────
    price_in_per_m: float = 0.0
    price_out_per_m: float = 0.0

    # ── hardening ───────────────────────────────────────────────────
    app_host: str = "127.0.0.1"
    app_port: int = 7860
    app_api_key: str = ""
    ui_username: str = "admin"
    ui_password: str = ""
    rate_limit_per_min: int = 30
    log_json: bool = True

    # ── paths ───────────────────────────────────────────────────────

    @property
    def upload_dir(self) -> Path:
        path = BASE_DIR / "data" / "uploads"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def db_path(self) -> Path:
        path = BASE_DIR / "data" / self.db_filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # ── capability gates ────────────────────────────────────────────

    @property
    def api_key(self) -> str:
        return self.llm_api_key.strip()

    @property
    def llm_configured(self) -> bool:
        return bool(self.api_key)

    @property
    def rerank_ready(self) -> bool:
        return self.rerank_enabled and self.llm_configured

    @property
    def embed_ready(self) -> bool:
        return bool(self.embed_api_key_resolved)

    @property
    def ocr_ready(self) -> bool:
        return self.ocr_enabled and bool(self.ocr_api_key_resolved)

    @property
    def vision_state(self) -> tuple[bool, str]:
        """Why the vision rail is (not) available: no key, disabled, or ready."""
        if not self.ocr_enabled:
            return False, "已按 OCR_ENABLED=false 关闭"
        if not self.ocr_api_key_resolved:
            return False, "未配置 API Key（SILICONFLOW_API_KEY）"
        return True, "就绪"

    @property
    def auth_required(self) -> bool:
        """Protection for the HTTP API itself, unrelated to the model key."""
        return bool(self.app_api_key.strip())

    # ── URLs ────────────────────────────────────────────────────────

    @property
    def llm_url(self) -> str:
        return self.llm_base_url.rstrip("/")

    @property
    def chat_completions_url(self) -> str:
        return self.llm_url + "/chat/completions"

    @property
    def models_url(self) -> str:
        return self.llm_url + "/models"

    @property
    def rerank_url(self) -> str:
        return self.llm_url + "/rerank"

    @property
    def embed_url(self) -> str:
        return self.embed_base_url_resolved + "/embeddings"

    @property
    def ocr_base_url_resolved(self) -> str:
        return (self.ocr_base_url or self.llm_base_url).rstrip("/")

    @property
    def ocr_api_key_resolved(self) -> str:
        return (self.ocr_api_key or self.llm_api_key).strip()

    @property
    def embed_base_url_resolved(self) -> str:
        return (self.embed_base_url or self.llm_base_url).rstrip("/")

    @property
    def embed_api_key_resolved(self) -> str:
        return (self.embed_api_key or self.llm_api_key).strip()

    @property
    def ui_auth(self) -> tuple[str, str] | None:
        """Gradio login, enabled only when a password is configured."""
        return (self.ui_username, self.ui_password) if self.ui_password.strip() else None


settings = Settings()
