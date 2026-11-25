from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Self

import httpx
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})


class HttpClient:
    """Synchronous HTTP client with bounded, exponential retries."""

    def __init__(
        self,
        base_url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float = 30.0,
        max_retries: int = 5,
        transport: httpx.BaseTransport | None = None,
        retry_wait: Callable[[RetryCallState], float] | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {max_retries}.")
        self._http = httpx.Client(
            base_url=base_url,
            headers=headers,
            timeout=timeout,
            transport=transport,
        )
        self._retrying = Retrying(
            retry=retry_if_exception(_is_retryable),
            stop=stop_after_attempt(max_retries + 1),
            wait=retry_wait or wait_exponential(multiplier=2, max=30),
            reraise=True,
        )

    def get_json(self, path: str, *, params: Mapping[str, Any] | None = None) -> Any:
        return self._retrying(self._get, path, params).json()

    def get_text(self, path: str, *, params: Mapping[str, Any] | None = None) -> str:
        return self._retrying(self._get, path, params).text

    def _get(self, path: str, params: Mapping[str, Any] | None) -> httpx.Response:
        response = self._http.get(path, params=params)
        response.raise_for_status()
        return response

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.TransportError):
        return True
    return (
        isinstance(exc, httpx.HTTPStatusError)
        and exc.response.status_code in RETRYABLE_STATUS_CODES
    )