from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any, Callable

import pyarrow as pa

from datasets.api import DataSourceFetcher, FetchRequest

_API_BASE_URL = "https://data.fingrid.fi/api/datasets/{dataset_id}/data"
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_RESOLUTION_MICROSECONDS: dict[str, int] = {
    "15min": 15 * 60 * 1_000_000,
    "30min": 30 * 60 * 1_000_000,
    "1h": 60 * 60 * 1_000_000,
}
_MAX_PAGES = 1_000
_RETRYABLE_HTTP_STATUS = frozenset({429, 500, 502, 503, 504})
_RETRY_BASE_DELAY_SECONDS = 2.0
_RETRY_MAX_DELAY_SECONDS = 30.0

# Signature of urllib.request.urlopen; injectable so tests can run offline.
HttpOpener = Callable[..., Any]
Sleeper = Callable[[float], None]


class FingridWindFetcher:
    """
    Fetches Finnish wind power generation (MW) from Fingrid Open Data.

    Dataset 75 is Fingrid's 15-minute wind generation series; samples are
    averaged into hourly buckets unless ``FetchRequest.resolution`` asks for
    a finer resolution.
    """

    def __init__(
        self,
        api_key: str,
        dataset_id: int = 75,
        *,
        timeout: float = 30.0,
        page_size: int = 20_000,
        max_retries: int = 5,
        opener: HttpOpener | None = None,
        sleep: Sleeper | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("Fingrid API key is required; register one at data.fingrid.fi.")
        if page_size < 1:
            raise ValueError(f"page_size must be positive, got {page_size}.")
        if max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {max_retries}.")
        self._api_key = api_key
        self._dataset_id = dataset_id
        self._timeout = timeout
        self._page_size = page_size
        self._max_retries = max_retries
        self._opener: HttpOpener = opener or urllib.request.urlopen
        self._sleep: Sleeper = sleep or time.sleep

    @property
    def source_name(self) -> str:
        return f"fingrid_{self._dataset_id}"

    @property
    def expected_schema(self) -> pa.Schema:
        return pa.schema([
            pa.field("timestamp_utc", pa.timestamp("us", tz="UTC"), nullable=False),
            pa.field("wind_mw", pa.float32(), nullable=False),
        ])

    def fetch(self, request: FetchRequest) -> pa.Table:
        start = _require_utc(request.start_utc, "start_utc")
        end = _require_utc(request.end_utc, "end_utc")
        if start >= end:
            raise ValueError(
                f"[{self.source_name}] start_utc ({start.isoformat()}) must be "
                f"before end_utc ({end.isoformat()})."
            )
        step_us = _resolution_microseconds(request.resolution, self.source_name)

        sums: dict[int, float] = defaultdict(float)
        counts: dict[int, int] = defaultdict(int)
        for timestamp, value in self._iter_samples(start, end):
            if value is None or not start <= timestamp < end:
                continue
            bucket = (_to_epoch_microseconds(timestamp) // step_us) * step_us
            sums[bucket] += value
            counts[bucket] += 1

        table = self._build_table(sums, counts)
        if not table.schema.equals(self.expected_schema, check_metadata=False):
            raise TypeError(
                f"[{self.source_name}] Schema mismatch.\n"
                f"Expected: {self.expected_schema}\nGot: {table.schema}"
            )
        return table

    def _iter_samples(
        self, start: datetime, end: datetime
    ) -> Iterator[tuple[datetime, float | None]]:
        """Yield ``(timestamp_utc, wind_mw_or_None)`` records across all API pages."""
        page = 1
        while True:
            payload = self._get_json(self._build_url(start, end, page))
            records, last_page = _split_payload(payload, self.source_name)
            for record in records:
                timestamp = _parse_api_timestamp(record.get("startTime"), self.source_name)
                yield timestamp, _parse_value(record.get("value"), self.source_name)

            if last_page is not None:
                if page >= last_page:
                    return
            elif len(records) < self._page_size:
                return

            page += 1
            if page > _MAX_PAGES:
                raise RuntimeError(
                    f"[{self.source_name}] Exceeded {_MAX_PAGES} API pages; "
                    "refusing to keep paginating."
                )

    def _build_url(self, start: datetime, end: datetime, page: int) -> str:
        params = {
            "startTime": _format_api_timestamp(start),
            "endTime": _format_api_timestamp(end),
            "format": "json",
            "page": str(page),
            "pageSize": str(self._page_size),
            "sortBy": "startTime",
            "sortOrder": "asc",
        }
        base = _API_BASE_URL.format(dataset_id=self._dataset_id)
        return f"{base}?{urllib.parse.urlencode(params)}"

    def _get_json(self, url: str) -> Any:
        request = urllib.request.Request(
            url,
            headers={"Accept": "application/json", "x-api-key": self._api_key},
        )
        delay = _RETRY_BASE_DELAY_SECONDS
        for attempt in range(self._max_retries + 1):
            try:
                with self._opener(request, timeout=self._timeout) as response:
                    body = response.read()
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                if exc.code in _RETRYABLE_HTTP_STATUS and attempt < self._max_retries:
                    self._sleep(_retry_delay(exc, delay))
                    delay = min(delay * 2, _RETRY_MAX_DELAY_SECONDS)
                    continue
                raise RuntimeError(
                    f"[{self.source_name}] Fingrid API returned HTTP {exc.code}: {detail}"
                ) from exc
            except urllib.error.URLError as exc:
                raise RuntimeError(
                    f"[{self.source_name}] Could not reach Fingrid API: {exc.reason}"
                ) from exc

        try:
            return json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError(
                f"[{self.source_name}] Fingrid API returned invalid JSON: {body[:200]!r}"
            ) from exc

    def _build_table(
        self, sums: dict[int, float], counts: dict[int, int]
    ) -> pa.Table:
        timestamps: list[datetime] = []
        values: list[float] = []
        for bucket in sorted(sums):
            timestamps.append(_EPOCH + timedelta(microseconds=bucket))
            values.append(sums[bucket] / counts[bucket])
        return pa.Table.from_pydict(
            {"timestamp_utc": timestamps, "wind_mw": values},
            schema=self.expected_schema,
        )


def run_fetcher(fetcher: DataSourceFetcher, request: FetchRequest) -> pa.Table:
    table = fetcher.fetch(request)
    if not table.schema.equals(fetcher.expected_schema, check_metadata=False):
        raise TypeError(
            f"[{fetcher.source_name}] Schema mismatch.\n"
            f"Expected: {fetcher.expected_schema}\nGot: {table.schema}"
        )
    return table


def _require_utc(value: datetime, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"FetchRequest.{field} must be timezone-aware (UTC).")
    return value.astimezone(UTC)


def _resolution_microseconds(resolution: str, source_name: str) -> int:
    try:
        return _RESOLUTION_MICROSECONDS[resolution]
    except KeyError:
        supported = ", ".join(sorted(_RESOLUTION_MICROSECONDS))
        raise ValueError(
            f"[{source_name}] Unsupported resolution {resolution!r}; expected one of: {supported}."
        ) from None


def _to_epoch_microseconds(timestamp: datetime) -> int:
    return (timestamp - _EPOCH) // timedelta(microseconds=1)


def _format_api_timestamp(timestamp: datetime) -> str:
    return timestamp.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _retry_delay(exc: urllib.error.HTTPError, fallback: float) -> float:
    """Prefer the server's Retry-After header, floored at zero, else back off exponentially."""
    headers = exc.headers
    retry_after = headers.get("Retry-After") if headers is not None else None
    if retry_after is not None:
        try:
            return max(0.0, float(retry_after))
        except (TypeError, ValueError):
            pass
    return fallback


def _parse_api_timestamp(raw: Any, source_name: str) -> datetime:
    if raw is None:
        raise ValueError(f"[{source_name}] Fingrid API record is missing 'startTime'.")
    try:
        parsed = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"[{source_name}] Cannot parse Fingrid timestamp {raw!r}."
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _parse_value(raw: Any, source_name: str) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"[{source_name}] Cannot parse Fingrid value {raw!r} as a number."
        ) from exc


def _split_payload(payload: Any, source_name: str) -> tuple[list[dict[str, Any]], int | None]:
    """Normalize both the paginated object and the legacy bare-list API responses."""
    if isinstance(payload, list):
        return payload, None
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"[{source_name}] Unexpected Fingrid API payload type: {type(payload).__name__}."
        )

    records = payload.get("data") or []
    if not isinstance(records, list):
        raise RuntimeError(f"[{source_name}] Fingrid API 'data' field is not a list.")

    pagination = payload.get("pagination")
    last_page = pagination.get("lastPage") if isinstance(pagination, dict) else None
    if last_page is not None:
        try:
            last_page = int(last_page)
        except (TypeError, ValueError):
            last_page = None
    return records, last_page