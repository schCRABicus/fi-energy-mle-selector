#!/usr/bin/env python3
"""Append recent hours to an existing Phase 1 window and rebuild its outputs."""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from dotenv import load_dotenv

from fi_energy.datasets import (
    DataSourceFetcher,
    FetchRequest,
    FingridConsumptionFetcher,
    FingridEEFlowFetcher,
    FingridHydroFetcher,
    FingridNuclearFetcher,
    FingridSE1FlowFetcher,
    FingridSE3FlowFetcher,
    FingridWindCapacityFetcher,
    FingridWindFetcher,
    FmiWeatherFetcher,
    NordPoolSpotFetcher,
)
from fi_energy.datasets.alignment import (
    TIMESTAMP_COLUMN,
    align_hourly,
    drop_incomplete_boundary_rows,
    interpolate_short_gaps,
    validate_hourly_table,
)
from fi_energy.datasets.fmi.client import FmiClient

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_DIR = ROOT / "data" / "3y"

# Kaisaniemi's wind sensor stopped reporting on 2026-11-21. Harmaja is the
# nearest Helsinki station with a complete record, so it fills only the gaps.
HELSINKI_WIND_FALLBACK_FMISID = 100969
MAX_INTERPOLATION_GAP_HOURS = 2

_TIMESTAMP_TYPE = pa.timestamp("us", tz="UTC")
_WINDOW_START_KEY = b"fi_energy.window_start_utc"
_WINDOW_END_KEY = b"fi_energy.window_end_utc"
_REFRESHED_KEY = b"fi_energy.refreshed_at_utc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch hours missing after the last cleaned row, append them to "
            "an existing Phase 1 window, and rebuild the combined tables."
        )
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Existing Phase 1 window directory (default: {_display(DEFAULT_DATA_DIR)}).",
    )
    parser.add_argument(
        "--start",
        type=_timestamp,
        help=(
            "First hour to refresh; defaults to the hour after the last "
            "cleaned row. The refreshed range always overwrites existing rows."
        ),
    )
    parser.add_argument(
        "--end",
        type=_timestamp,
        help="Exclusive window end; defaults to the current UTC hour.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("FINGRID_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "FINGRID_API_KEY is missing; add it to .env or the environment."
        )

    data_dir = args.data_dir.expanduser().resolve()
    clean_path = data_dir / "combined" / "hourly_clean.parquet"
    if not clean_path.is_file():
        raise SystemExit(f"Cleaned Phase 1 table not found: {clean_path}")

    cleaned = pq.read_table(clean_path)
    previous_end = cleaned[TIMESTAMP_COLUMN][-1].as_py()
    end = args.end or datetime.now(UTC).replace(
        minute=0, second=0, microsecond=0
    )
    start = args.start or previous_end + timedelta(hours=1)
    if start >= end:
        print(
            f"nothing to refresh: cleaned data already ends at "
            f"{_iso(previous_end)}",
            flush=True,
        )
        return 0

    print(
        f"refresh   {_iso(start)} -> {_iso(end)} "
        f"({int((end - start) / timedelta(hours=1)):,} hours)",
        flush=True,
    )
    request = FetchRequest(start_utc=start, end_utc=end, resolution="1h")
    fetched: dict[str, tuple[str, pa.Table]] = {}
    fallback_filled = 0
    for group, fetcher in _fetchers(api_key):
        name = fetcher.dataset_name
        print(f"fetching  {name} ...", flush=True)
        with fetcher:
            table = fetcher.fetch(request)
        _require_expected_schema(fetcher, table)
        validate_hourly_table(table, start=start, end=end, label=name)
        if name == "helsinki_weather":
            table, fallback_filled = _fill_helsinki_wind_speed(
                table, start=start, end=end
            )
        fetched[name] = (group, table)

    refreshed_at = datetime.now(UTC)
    for name, (group, new_table) in fetched.items():
        path = data_dir / group / f"{name}.parquet"
        existing = pq.read_table(path)
        merged = _merge_existing(existing, new_table, cutoff=start)
        merged = _with_window_metadata(
            merged, start=merged[TIMESTAMP_COLUMN][0].as_py(), end=end,
            refreshed_at=refreshed_at,
        )
        _write_table(path, merged)
        print(
            f"appended  {group}/{path.name}  {merged.num_rows:,} rows  "
            f"last {_iso(merged[TIMESTAMP_COLUMN][-1].as_py())}",
            flush=True,
        )

    window_start = min(
        pq.read_table(path)[TIMESTAMP_COLUMN][0].as_py()
        for path in sorted(data_dir.glob("*/*.parquet"))
        if path.parent.name in {"fingrid", "fmi", "energy_charts"}
    )
    combined, cleaned = _rebuild_combined(
        data_dir, start=window_start, end=end
    )
    _write_manifest(
        data_dir,
        start=window_start,
        end=end,
        previous_end=previous_end,
        refresh_start=start,
        fallback_filled=fallback_filled,
        combined=combined,
        cleaned=cleaned,
        refreshed_at=refreshed_at,
    )
    print(
        f"cleaned   {cleaned.num_rows:,} rows  "
        f"{_iso(cleaned[TIMESTAMP_COLUMN][0].as_py())} -> "
        f"{_iso(cleaned[TIMESTAMP_COLUMN][-1].as_py())}",
        flush=True,
    )
    print(f"manifest  {_display(data_dir / 'manifest.json')}", flush=True)
    return 0


def _fetchers(api_key: str) -> list[tuple[str, DataSourceFetcher]]:
    return [
        ("fingrid", FingridWindFetcher(api_key)),
        ("fingrid", FingridWindCapacityFetcher(api_key)),
        ("fingrid", FingridConsumptionFetcher(api_key)),
        ("fingrid", FingridNuclearFetcher(api_key)),
        ("fingrid", FingridHydroFetcher(api_key)),
        ("fingrid", FingridSE1FlowFetcher(api_key)),
        ("fingrid", FingridSE3FlowFetcher(api_key)),
        ("fingrid", FingridEEFlowFetcher(api_key)),
        ("fmi", FmiWeatherFetcher("helsinki", show_progress=False)),
        ("fmi", FmiWeatherFetcher("vaasa", show_progress=False)),
        ("fmi", FmiWeatherFetcher("oulu", show_progress=False)),
        ("energy_charts", NordPoolSpotFetcher()),
    ]


def _fill_helsinki_wind_speed(
    table: pa.Table,
    *,
    start: datetime,
    end: datetime,
) -> tuple[pa.Table, int]:
    field = table.schema.field("wind_speed_ms")
    values = table["wind_speed_ms"].to_pylist()
    missing = [index for index, value in enumerate(values) if value is None]
    if not missing:
        return table, 0

    with FmiClient(timeout=60) as client:
        fallback = {
            timestamp: row.get("WS_PT1H_AVG")
            for timestamp, row in client.iter_weather(
                HELSINKI_WIND_FALLBACK_FMISID, start, end
            )
        }
    timestamps = table[TIMESTAMP_COLUMN].to_pylist()
    filled = 0
    for index in missing:
        value = fallback.get(timestamps[index])
        if value is not None:
            values[index] = value
            filled += 1
    if filled == 0:
        return table, 0
    return (
        table.set_column(
            table.schema.get_field_index("wind_speed_ms"),
            field,
            pa.array(values, type=field.type),
        ),
        filled,
    )


def _merge_existing(
    existing: pa.Table,
    new: pa.Table,
    *,
    cutoff: datetime,
) -> pa.Table:
    cutoff_scalar = pa.scalar(cutoff, type=_TIMESTAMP_TYPE)
    kept = existing.filter(
        pc.less(existing[TIMESTAMP_COLUMN], cutoff_scalar)
    )
    merged = pa.concat_tables([kept, new]).sort_by(
        [(TIMESTAMP_COLUMN, "ascending")]
    )
    timestamps = merged[TIMESTAMP_COLUMN].to_pylist()
    if len(timestamps) != len(set(timestamps)):
        raise RuntimeError("Refreshed source contains duplicate timestamps.")
    return merged


def _rebuild_combined(
    data_dir: Path,
    *,
    start: datetime,
    end: datetime,
) -> tuple[pa.Table, pa.Table]:
    tables: dict[str, pa.Table] = {}
    for path in sorted(data_dir.glob("*/*.parquet")):
        group = path.parent.name
        if group not in {"fingrid", "fmi", "energy_charts"}:
            continue
        table = pq.read_table(path)
        if group == "fmi":
            city = path.stem.removesuffix("_weather")
            table = table.rename_columns([
                name if name == TIMESTAMP_COLUMN else f"{city}_{name}"
                for name in table.column_names
            ])
        tables[path.stem] = table

    combined = align_hourly(tables, start=start, end=end)
    combined_path = data_dir / "combined" / "hourly.parquet"
    _write_table(combined_path, combined)

    trimmed = drop_incomplete_boundary_rows(combined)
    cleaned = interpolate_short_gaps(
        trimmed, max_gap=MAX_INTERPOLATION_GAP_HOURS
    )
    cleaned_path = data_dir / "combined" / "hourly_clean.parquet"
    _write_table(cleaned_path, cleaned)
    return combined, cleaned


def _write_manifest(
    data_dir: Path,
    *,
    start: datetime,
    end: datetime,
    previous_end: datetime,
    refresh_start: datetime,
    fallback_filled: int,
    combined: pa.Table,
    cleaned: pa.Table,
    refreshed_at: datetime,
) -> None:
    sources = []
    for path in sorted(data_dir.glob("*/*.parquet")):
        group = path.parent.name
        if group not in {"fingrid", "fmi", "energy_charts"}:
            continue
        table = pq.read_table(path)
        timestamps = table[TIMESTAMP_COLUMN].to_pylist()
        value_columns = [
            name for name in table.column_names if name != TIMESTAMP_COLUMN
        ]
        sources.append({
            "source": path.stem,
            "group": group,
            "path": f"{group}/{path.name}",
            "rows": table.num_rows,
            "first_utc": _iso(timestamps[0]),
            "last_utc": _iso(timestamps[-1]),
            "null_values": {
                name: table[name].null_count for name in value_columns
            },
        })

    expected_hours = int((end - start) / timedelta(hours=1))
    value_columns = [
        name for name in combined.column_names if name != TIMESTAMP_COLUMN
    ]
    cleaned_values = [
        name for name in cleaned.column_names if name != TIMESTAMP_COLUMN
    ]
    cleaned_timestamps = cleaned[TIMESTAMP_COLUMN].to_pylist()
    manifest = {
        "generated_at_utc": _iso(refreshed_at),
        "status": "ok",
        "window": {
            "label": data_dir.name,
            "start_utc": _iso(start),
            "end_utc": _iso(end),
            "expected_hours": expected_hours,
            "resolution": "1h",
        },
        "refresh": {
            "previous_cleaned_end_utc": _iso(previous_end),
            "refresh_start_utc": _iso(refresh_start),
            "refresh_end_utc": _iso(end),
            "helsinki_wind_fallback_fmisid": HELSINKI_WIND_FALLBACK_FMISID,
            "helsinki_wind_fallback_values": fallback_filled,
        },
        "sources": sources,
        "combined": {
            "path": "combined/hourly.parquet",
            "rows": combined.num_rows,
            "feature_columns": len(value_columns),
            "null_values": {
                name: combined[name].null_count for name in value_columns
            },
        },
        "cleaned": {
            "path": "combined/hourly_clean.parquet",
            "rows": cleaned.num_rows,
            "first_utc": _iso(cleaned_timestamps[0]),
            "last_utc": _iso(cleaned_timestamps[-1]),
            "rows_removed_at_boundaries": combined.num_rows - cleaned.num_rows,
            "null_values": {
                name: cleaned[name].null_count for name in cleaned_values
            },
        },
        "policy": {
            "interpolation": {
                "method": "linear",
                "max_gap_hours": MAX_INTERPOLATION_GAP_HOURS,
            },
            "nulls": "propagate; no imputation beyond short gaps",
        },
    }
    path = data_dir / "manifest.json"
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _with_window_metadata(
    table: pa.Table,
    *,
    start: datetime,
    end: datetime,
    refreshed_at: datetime,
) -> pa.Table:
    metadata = dict(table.schema.metadata or {})
    metadata[_WINDOW_START_KEY] = _iso(start).encode("ascii")
    metadata[_WINDOW_END_KEY] = _iso(end).encode("ascii")
    metadata[_REFRESHED_KEY] = _iso(refreshed_at).encode("ascii")
    return table.replace_schema_metadata(metadata)


def _require_expected_schema(
    fetcher: DataSourceFetcher, table: pa.Table
) -> None:
    if not table.schema.equals(fetcher.expected_schema, check_metadata=False):
        raise RuntimeError(
            f"{fetcher.source_name}: schema {table.schema} does not match "
            f"{fetcher.expected_schema}."
        )


def _write_table(path: Path, table: pa.Table) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(path)


def _timestamp(raw: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"invalid ISO timestamp {raw!r}: {exc}"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC).replace(minute=0, second=0, microsecond=0)


def _iso(timestamp: datetime) -> str:
    return timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())