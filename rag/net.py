"""Shared HTTP layer: pace every call, retry only what can succeed, fail fast on the rest.

Three things live here, all of them learned the hard way:

1. **Pacing.** A single global interval between any two requests. The embedding,
   rerank and vision paths all fire from thread pools (`EMBED_CONCURRENCY`,
   `OCR_CONCURRENCY`), so without a shared gate a 20-page scan leaves as three
   simultaneous bursts and the platform answers 429 on half of them. The lock is
   held *across* the sleep on purpose: that is what makes concurrent callers
   queue behind each other instead of all waking at once.
2. **Jitter.** Plain `2**n` backoff makes every retrying thread wake at the same
   millisecond and re-collide. A random share of the delay spreads them out.
3. **Classification.** 401/402/403/404 and 400 will not fix themselves. Retrying
   them three times turns a clear "your key is rejected" into a nine-second
   pause followed by the same message, and tempts people to blame rate limits.
"""

import logging
import random
import socket
import threading
import time

import requests

log = logging.getLogger(__name__)

# Permanent: the request or the credential is wrong, repeating it cannot help.
FATAL_CODES = {400, 401, 402, 403, 404, 405, 413, 422}
# Transient: the same request may succeed in a second.
RETRY_CODES = {408, 409, 425, 429}

_pace_lock = threading.Lock()
_last_call = 0.0

JITTER_RATIO = 0.4  # delay is stretched by a random 0..40%


def reset_pacing() -> None:
    """Forget the last-call stamp. Used by tests and by the restart button."""
    global _last_call
    with _pace_lock:
        _last_call = 0.0


def _pace(min_interval: float) -> None:
    """Block until this caller's reserved slot, then return ready to send.

    Reserve the slot under the lock, sleep *outside* it. Holding the lock across
    the sleep looks equivalent but is not: the thread that wakes up has to fight
    for the GIL before it reaches `requests.post`, and a thread that finished its
    sleep earlier can overtake it — the server then sees two arrivals 5ms apart
    even though we waited 120ms. Reserving an absolute slot and sleeping to it
    keeps arrival order aligned with the schedule.
    """
    global _last_call
    if min_interval <= 0:
        return
    with _pace_lock:
        slot = max(time.monotonic(), _last_call + min_interval)
        _last_call = slot
    wait = slot - time.monotonic()
    if wait > 0:
        time.sleep(wait)


def retry_after(response) -> float:
    """Seconds the server asked us to wait, capped so a bad header can't stall a minute."""
    raw = (response.headers.get("Retry-After") or "").strip()
    if not raw:
        return 0.0
    try:
        return min(max(float(raw), 0.0), 30.0)
    except ValueError:
        return 0.0


def backoff_delay(attempt: int, response=None, base: float = 1.0, cap: float = 20.0) -> float:
    """Exponential, capped, jittered; the server's Retry-After wins when larger."""
    delay = min(cap, base * (2**attempt))
    delay += random.uniform(0, delay * JITTER_RATIO)
    return max(delay, retry_after(response) if response is not None else 0.0)


def is_retryable(status_code: int) -> bool:
    return status_code in RETRY_CODES or status_code >= 500


def post_json(
    url: str,
    headers: dict,
    payload: dict,
    timeout: int,
    min_interval: float = 0.0,
    retries: int = 3,
    base_delay: float = 1.0,
    stream: bool = False,
):
    """Paced POST with bounded retries. Returns the response for the caller to explain.

    Non-retryable statuses are returned immediately; the caller owns the message
    because only it knows whether the call was chat, embed, rerank or vision.
    """
    response = None
    for attempt in range(max(0, retries) + 1):
        _pace(min_interval)
        try:
            response = requests.post(
                url, headers=headers, json=payload, timeout=timeout, stream=stream
            )
        except requests.Timeout:
            if attempt >= retries:
                raise
            delay = backoff_delay(attempt, base=base_delay)
            log.warning("timeout, retry %s/%s in %.1fs", attempt + 1, retries, delay)
            time.sleep(delay)
            continue
        except requests.ConnectionError as exc:
            if attempt >= retries:
                raise
            delay = backoff_delay(attempt, base=base_delay)
            log.warning("connection error (%s), retry %s/%s in %.1fs", exc, attempt + 1, retries, delay)
            time.sleep(delay)
            continue

        if not is_retryable(response.status_code) or attempt >= retries:
            return response

        delay = backoff_delay(attempt, response, base=base_delay)
        log.warning(
            "HTTP %s, retry %s/%s in %.1fs",
            response.status_code,
            attempt + 1,
            retries,
            delay,
        )
        # The body is not consumed yet (stream=True only reads headers), so this
        # response can be dropped without leaking a connection.
        response.close()
        time.sleep(delay)
    return response


def describe_status(status_code: int, body: str = "") -> str:
    """One short Chinese clause per status, so every call site explains itself the same way."""
    if status_code == 401:
        return "Key 被拒绝（401）"
    if status_code == 402:
        return "余额不足（402）"
    if status_code == 403:
        return "无权限（403）"
    if status_code == 404:
        return "接口或模型不存在（404）"
    if status_code == 429:
        return "触发限流（429）"
    if status_code >= 500:
        return f"服务端错误（{status_code}）"
    return f"HTTP {status_code}"


def port_ready(port: int, host: str = "127.0.0.1", timeout: float = 0.3) -> bool:
    """True when something is listening on the port. Used by colab boot and tests."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, int(port))) == 0
