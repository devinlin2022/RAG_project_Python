"""Model chains: which errors move to the next id and which do not."""

from rag.models import chain, is_model_error


def test_chain_deduplicates_and_keeps_order():
    assert chain("a", "b, a ,,c") == ["a", "b", "c"]


def test_chain_without_fallbacks_is_one_entry():
    assert chain("a", "") == ["a"]


def test_missing_model_is_a_model_error():
    assert is_model_error(404, "")
    assert is_model_error(400, '{"code":20012,"message":"Model does not exist."}')


def test_auth_and_quota_are_not_model_errors():
    assert not is_model_error(401, '{"code":30014,"message":"Token is invalid."}')
    assert not is_model_error(402, "insufficient balance")
    assert not is_model_error(429, "rate limited")
    assert not is_model_error(400, '{"code":20015,"message":"invalid parameter"}')


def test_settings_expose_one_chain_per_rail():
    from rag.config import settings

    assert settings.llm_chain[0] == settings.llm_model
    assert settings.rerank_chain[0] == settings.rerank_model
    assert settings.ocr_chain[0] == settings.ocr_model
    assert settings.vision_chain[0] == settings.vision_model
    assert settings.embed_chain[0] == settings.embed_model
    assert len(settings.llm_chain) >= 3
    assert len(settings.embed_chain) >= 3
