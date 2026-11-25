from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from typing import Any

import httpx
from tenacity import RetryCallState

from datasets.http_client import HttpClient
from datasets.utils import utc_window

FINGRID_API_BASE_URL = "https://data.fingrid.fi/api/"


class FingridClient(HttpClient):
    """Fingrid Open Data client for paginated time series."""

    MAX_PAGES = 1_000

    def __init__(
        self,
        api_key: str,
        *,
        timeout: float = 30.0,
        max_retries: int = 5,
        page_size: int = 20_000,
        transport: httpx.BaseTransport | None = None,
        retry_wait: Callable[[RetryCallState], float] | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("Fingrid API key is required; register one at data.fingrid.fi.")
        if page_size < 1:
            raise ValueError(f"page_size must be positive, got {page_size}.")
        self._page_size = page_size
        super().__init__(
            FINGRID_API_BASE_URL,
            headers={"x-api-key": api_key},
            timeout=timeout,
            max_retries=max_retries,
            transport=transport,
            retry_wait=retry_wait,
        )

    def iter_samples(
        self, dataset_id: int, start: datetime, end: datetime
    ) -> Iterator[tuple[datetime, float | None]]:
        """Yield ``(timestamp_utc, value)`` for every record Fingrid returns for the window."""
        start, end = utc_window(start, end)
        for page in range(1, self.MAX_PAGES + 1):
            records, last_page = self._get_page(dataset_id, start, end, page)
            yield from map(self._parse_record, records)

            if last_page is not None:
                if page >= last_page:
                    return
            elif len(records) < self._page_size:
                return

        raise RuntimeError(
            f"Fingrid pagination exceeded {self.MAX_PAGES} pages for dataset {dataset_id}."
        )

    def _get_page(
        self, dataset_id: int, start: datetime, end: datetime, page: int
    ) -> tuple[list[dict[str, Any]], int | None]:
        payload = self.get_json(
            f"datasets/{dataset_id}/data",
            params={
                "startTime": self._format_timestamp(start),
                "endTime": self._format_timestamp(end),
                "format": "json",
                "page": page,
                "pageSize": self._page_size,
                "sortBy": "startTime",
                "sortOrder": "asc",
            },
        )
        if isinstance(payload, list):  # legacy bare-list response
            return payload, None
        if not isinstance(payload, dict):
            raise ValueError(
                f"Unexpected Fingrid response type: {type(payload).__name__}."
            )
        pagination = payload.get("pagination") or {}
        return payload.get("data") or [], pagination.get("lastPage")

    @staticmethod
    def _format_timestamp(timestamp: datetime) -> str:
        return timestamp.astimezone(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")

    @staticmethod
    def _parse_record(record: dict[str, Any]) -> tuple[datetime, float | None]:
        raw_timestamp = record.get("startTime")
        if raw_timestamp is None:
            raise ValueError("Fingrid API record is missing 'startTime'.")
        timestamp = datetime.fromisoformat(str(raw_timestamp))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        raw_value = record.get("value")
        return timestamp.astimezone(UTC), None if raw_value is None else float(raw_value)