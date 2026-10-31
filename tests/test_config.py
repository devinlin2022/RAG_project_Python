"""Configuration switches and the messages that explain them."""

import pytest

pytest.importorskip("pydantic_settings")

from rag.config import Settings  # noqa: E402

ENV_KEYS = (
    "SILICONFLOW_API_KEY",
    "LLM_API_KEY",
    "SILICONFLOW_BASE_URL",
    "LLM_BASE_URL",
    "OCR_ENABLED",
    "OCR_MODEL",
    "VISION_MODEL",
    "APP_API_KEY",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def make(**kwargs) -> Settings:
    """A Settings object that ignores the developer's own .env."""
    return Settings(_env_file=None, **kwargs)


def test_vision_is_on_as_soon_as_a_key_exists():
    """The default must not depend on a second switch nobody knows about."""
    settings = make(SILICONFLOW_API_KEY="sk-test")
    assert settings.ocr_ready is True
    assert settings.vision_state == (True, "就绪")


def test_vision_reports_switched_off_differently_from_missing_key():
    with_key = make(SILICONFLOW_API_KEY="sk-test", OCR_ENABLED="false")
    without_key = make()

    assert with_key.ocr_ready is False
    assert "OCR_ENABLED" in with_key.vision_state[1]
    assert without_key.ocr_ready is False
    assert "SILICONFLOW_API_KEY" in without_key.vision_state[1]


def test_ocr_and_figure_models_are_separate_knobs():
    """Transcription wants the cheap model, figures want the reasoning one."""
    settings = make()
    assert settings.ocr_model == "Qwen/Qwen3-VL-8B-Instruct"
    assert settings.vision_model == "Qwen/Qwen3-VL-32B-Instruct"
    assert settings.ocr_model != settings.vision_model


def test_access_protection_is_unrelated_to_the_model_key():
    """A configured model key must not make the UI claim the API is protected."""
    open_api = make(SILICONFLOW_API_KEY="sk-test")
    guarded = make(SILICONFLOW_API_KEY="sk-test", APP_API_KEY="secret")

    assert open_api.auth_required is False
    assert guarded.auth_required is True


def test_401_message_names_the_only_variable_a_user_has_to_set(monkeypatch):
    from rag.generate import llm as llm_module
    from rag.generate.llm import LLMError

    class _Response:
        status_code = 401
        text = '{"code":401,"message":"Invalid token"}'

    # `_explain` reads the process-wide settings, so patch that object directly.
    monkeypatch.setattr(llm_module.settings, "llm_api_key", "sk-abcdefgh1234")
    monkeypatch.setattr(llm_module.settings, "llm_base_url", "https://api.siliconflow.cn/v1")
    error = llm_module._explain(_Response())

    assert isinstance(error, LLMError)
    message = str(error)
    assert "SILICONFLOW_API_KEY" in message
    assert "LLM_API_KEY" not in message
    assert "1234" in message  # the tail is how a user tells two keys apart
    assert llm_module.settings.llm_model in message
