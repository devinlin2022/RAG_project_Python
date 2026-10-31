"""Colab bootstrap: .env handling and notebook structure."""

import json
from pathlib import Path

import pytest

from tools import colab

EXAMPLE = """# keep this comment
SILICONFLOW_API_KEY=
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
OCR_ENABLED=true
"""


def test_parse_env_skips_comments_and_quotes():
    values = colab.parse_env('A=1\n# B=2\n\nC="x y"\nD=\'z\'\n')
    assert values == {"A": "1", "C": "x y", "D": "z"}


def test_patch_env_replaces_value_in_place():
    text = colab.patch_env(EXAMPLE, {"SILICONFLOW_API_KEY": "sk-abc"})
    assert "SILICONFLOW_API_KEY=sk-abc" in text
    assert "# keep this comment" in text
    assert text.index("SILICONFLOW_API_KEY") < text.index("SILICONFLOW_BASE_URL")
    assert text.count("SILICONFLOW_API_KEY") == 1


def test_patch_env_appends_missing_key_once():
    text = colab.patch_env(EXAMPLE, {"NGROK_AUTHTOKEN": "tok"})
    assert text.rstrip().endswith("NGROK_AUTHTOKEN=tok")
    assert text.count("NGROK_AUTHTOKEN=tok") == 1
    again = colab.patch_env(text, {"NGROK_AUTHTOKEN": "tok2"})
    assert again.count(colab.ENV_MARKER) == 1
    assert "NGROK_AUTHTOKEN=tok2" in again and "NGROK_AUTHTOKEN=tok\n" not in again


def test_patch_env_noop_keeps_text():
    assert colab.patch_env(EXAMPLE, {}) == EXAMPLE


def test_patch_env_empty_text():
    assert colab.patch_env("", {"A": "1"}) == "A=1\n"


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    example = tmp_path / ".env.example"
    example.write_text(EXAMPLE, encoding="utf-8")
    monkeypatch.setattr(colab, "EXAMPLE_PATH", example)
    monkeypatch.setattr(colab, "ENV_PATH", tmp_path / ".env")
    return tmp_path


def test_write_env_creates_file_from_example(sandbox):
    info = colab.write_env(api_key="sk-new")
    assert info["created"] and info["has_key"] and not info["has_token"]
    text = (sandbox / ".env").read_text(encoding="utf-8")
    assert "SILICONFLOW_API_KEY=sk-new" in text and "OCR_ENABLED=true" in text


def test_write_env_does_not_overwrite_existing_key(sandbox):
    env = sandbox / ".env"
    env.write_text(colab.patch_env(EXAMPLE, {"SILICONFLOW_API_KEY": "sk-old"}), encoding="utf-8")

    info = colab.write_env(api_key="sk-from-secret")
    assert info["changed"] == []
    assert "sk-old" in env.read_text(encoding="utf-8")

    info = colab.write_env(api_key="sk-from-secret", force=True)
    assert info["changed"] == ["SILICONFLOW_API_KEY"]
    assert "sk-from-secret" in env.read_text(encoding="utf-8")


def test_write_env_without_any_value_does_not_touch_file(sandbox):
    info = colab.write_env()
    assert not info["created"] and info["changed"] == []
    assert not (sandbox / ".env").exists()


def test_notebook_is_valid_and_calls_boot():
    path = Path(__file__).resolve().parents[1] / "colab.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
    assert "from tools.colab import boot" in code
    assert "boot()" in code
    assert "if not in_repo and not root.exists()" in code
