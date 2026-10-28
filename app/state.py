"""Process-wide service instance. One index, shared by UI and API."""

from rag.service import Service

_service: Service | None = None


def get_service() -> Service:
    global _service
    if _service is None:
        _service = Service()
    return _service


def set_service(service: Service | None) -> None:
    """Test hook: replace or reset the singleton."""
    global _service
    _service = service
