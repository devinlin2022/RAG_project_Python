"""Shared POST with a minimum call interval and backoff on 429 / 5xx."""

import logging
import time

import requests

log = logging.getLogger(__name__)

_last_call = 0.0


def _retry_after(response) -> float:
    try:
        return max(float(response.headers.get("Retry-After", "")), 1.0)
    except (TypeError, ValueError):
        return 0.0


def post_json(
    url: str,
    headers: dict,
    payload: dict,
    timeout: int,
    min_interval: float = 0.0,
    retries: int = 3,
):
    """Throttle, then POST. Retries only on 429 and 5xx; other codes pass through."""
    global _last_call
    wait = min_interval - (time.time() - _last_call)
    if wait > 0:
        time.sleep(wait)

    response = None
    for attempt in range(retries + 1):
        _last_call = time.time()
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=timeout)
        except requests.Timeout:
            if attempt >= retries:
                raise
            time.sleep(2**attempt)
            continue

        if response.status_code == 429 or response.status_code >= 500:
            if attempt >= retries:
                return response
            delay = _retry_after(response) or 2**attempt
            log.warning(
                "HTTP %s, retry %s/%s in %.0fs",
                response.status_code,
                attempt + 1,
                retries,
                delay,
            )
            time.sleep(delay)
            continue
        return response
    return response
