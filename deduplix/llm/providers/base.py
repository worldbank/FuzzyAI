"""Provider protocol, response container and the shared retry / rate-limit / timeout logic."""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("deduplix.llm.providers")


@dataclass
class ProviderResponse:
    """What a provider returns for one call."""

    text: Optional[str] = None
    json: Optional[dict] = None  # structured output, when the provider returned one
    probabilities: Optional[dict] = None  # label -> probability, for classifier endpoints
    usage: Dict[str, int] = field(default_factory=dict)  # input_tokens, output_tokens
    raw: Any = None


def is_retryable(exc: BaseException) -> bool:
    """429, 5xx, timeouts and connection errors are retried; everything else is not."""
    for attr in ("status_code", "status", "code"):
        code = getattr(exc, attr, None)
        if isinstance(code, int):
            return code == 429 or code >= 500
    resp = getattr(exc, "response", None)
    code = getattr(resp, "status_code", None)
    if isinstance(code, int):
        return code == 429 or code >= 500
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    low = type(exc).__name__.lower()
    return "timeout" in low or "connection" in low


class BaseProvider:
    """Subclasses implement ``_complete``; ``complete`` adds rate limiting and retries.

    Backoff is exponential with full jitter: ``uniform(0, min(cap, base * 2**attempt))``.
    ``sleep`` and ``rng`` are injectable so tests never wait.
    """

    name = "base"
    model: Optional[str] = None

    def __init__(self, max_retries: int = 3, timeout: float = 60.0,
                 requests_per_minute: Optional[int] = None, backoff_base: float = 1.0,
                 backoff_cap: float = 30.0, sleep: Callable[[float], None] = time.sleep,
                 rng: Optional[Callable[[], float]] = None):
        self.max_retries = max_retries
        self.timeout = timeout
        self.requests_per_minute = requests_per_minute
        self.backoff_base = backoff_base
        self.backoff_cap = backoff_cap
        self._sleep = sleep
        self._rng = rng or random.random
        self._limiter = None
        if requests_per_minute:
            from ...security.rate_limiter import RateLimitConfig, RateLimiter

            self._limiter = RateLimiter(RateLimitConfig(
                requests_per_minute=requests_per_minute, requests_per_hour=10**9,
                requests_per_day=10**9, burst_limit=max(requests_per_minute, 1),
                burst_window=60.0))

    def _complete(self, messages: List[dict], *, schema: Optional[dict], temperature: float,
                  max_tokens: int) -> ProviderResponse:
        raise NotImplementedError

    def backoff_delay(self, attempt: int) -> float:
        return self._rng() * min(self.backoff_cap, self.backoff_base * (2 ** attempt))

    def _wait_for_slot(self) -> None:
        if self._limiter is None:
            return
        while not self._limiter.allow_request(self.name):
            self._sleep(max(self._limiter.get_retry_after(), 0.05))

    def complete(self, messages: List[dict], *, schema: Optional[dict] = None,
                 temperature: float = 0.0, max_tokens: int = 4096) -> ProviderResponse:
        attempt = 0
        while True:
            self._wait_for_slot()
            try:
                return self._complete(messages, schema=schema, temperature=temperature,
                                      max_tokens=max_tokens)
            except Exception as exc:
                if attempt >= self.max_retries or not is_retryable(exc):
                    raise
                delay = self.backoff_delay(attempt)
                logger.warning("%s call failed (%s); retry %d/%d in %.2fs", self.name,
                               type(exc).__name__, attempt + 1, self.max_retries, delay)
                self._sleep(delay)
                attempt += 1
