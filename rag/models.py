"""Model chains: move to the next id when the platform says this one is gone.

A 404 or a 400 with code 20012 ("Model does not exist") means the model was
retired or is not enabled for this account. Auth, billing and rate limits are
not model problems, so they must surface instead of triggering a switch.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

MODEL_ERROR_MARKERS = (
    "model does not exist",
    "model not exist",
    "model not found",
    "invalid model",
    "20012",
)


class ModelUnavailable(Exception):
    """The platform does not recognise this model id."""

    def __init__(self, model: str, detail: str):
        super().__init__(f"{model}: {detail}")
        self.model = model
        self.detail = detail


def is_model_error(status: int, body: str) -> bool:
    """True only for "this model id is not usable here"."""
    if status == 404:
        return True
    if status != 400:
        return False
    text = (body or "").lower()
    return any(marker in text for marker in MODEL_ERROR_MARKERS)


def chain(primary: str, extra: str) -> list[str]:
    """Deduplicated list, primary first."""
    names: list[str] = []
    for raw in [primary, *str(extra or "").split(",")]:
        name = (raw or "").strip()
        if name and name not in names:
            names.append(name)
    return names
