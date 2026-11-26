#!/usr/bin/env python3
"""Fetch every Phase 1 series over one shared UTC hourly window."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pyarrow as pa
import pyarrow.parquet as pq
from dotenv import load_dotenv

from datasets import (
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
from datasets.alignment import (
    TIMESTAMP_COLUMN,
    align_hourly,
    hourly_index,
    trailing_window,
    validate_hourly_table,
    window_label,
)
from datasets.utils import utc_window

FMI_CITIES = ("helsinki", "vaasa", "oulu")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch all Phase 1 sources on one UTC hourly grid and save full "
            "Parquet files under data/."
        )
    )
    parser.add_argument(
        "--start",
        type=_timestamp,
        help=(
            "Window start as an ISO timestamp (rounded down to the UTC hour). "
            "Defaults to --years before the end."
        ),
    )
    parser.add_argument(
        "--end",
        type=_timestamp,
        help=(
            "Window end as an ISO timestamp (rounded down to the UTC hour). "
            "Defaults to the current UTC hour."
        ),
    )
    parser.add_argument(
        "--years",
        type=int,
        default=3,
        help="Trailing window length used when --start is omitted (default: 3).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data",
        help=(
            "Base output directory; each window is written under "
            "<output-dir>/<window>/ (default: data/)."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help=(
            "Reuse a source Parquet file that already exists for this window "
            "instead of fetching it again."
        ),
    )
    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="Skip the aligned combined/hourly.parquet table.",
    )
    return parser.parse_args()


def build_groups(
    api_key: str,
) -> list[tuple[str, list[DataSourceFetcher]]]:
    fingrid: list[DataSourceFetcher] = [
        FingridWindFetcher(api_key),
        FingridWindCapacityFetcher(api_key),
        FingridConsumptionFetcher(api_key),
        FingridNuclearFetcher(api_key),
        FingridHydroFetcher(api_key),
        FingridSE1FlowFetcher(api_key),
        FingridSE3FlowFetcher(api_key),
        FingridEEFlowFetcher(api_key),
    ]
    fmi: list[DataSourceFetcher] = [
        FmiWeatherFetcher(city, show_progress=True) for city in FMI_CITIES
    ]
    return [
        ("fingrid", fingrid),
        ("fmi", fmi),
        ("energy_charts", [NordPoolSpotFetcher()]),
    ]


def main() -> int:
    args = parse_args()
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("FINGRID_API_KEY", "").strip()
    if not api_key:
        raise SystemExit(
            "FINGRID_API_KEY is missing; add it to .env or the environment."
        )

    if args.years < 1:
        raise SystemExit("--years must be at least 1.")

    end = args.end or trailing_window(years=args.years)[1]
    start = args.start or trailing_window(end, years=args.years)[0]
    try:
        start, end = utc_window(start, end)
        index = hourly_index(start, end)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    output_dir = args.output_dir.expanduser().resolve()
    request = FetchRequest(start_utc=start, end_utc=end, resolution="1h")
    groups = build_groups(api_key)
    total = sum(len(fetchers) for _, fetchers in groups)
    label = window_label(start, end)
    window_dir = output_dir / label
    window_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"window {_iso(start)} -> {_iso(end)}  [{label}]  "
        f"({len(index):,} hourly slots, {total} series)",
        flush=True,
    )

    saved: list[dict[str, Any]] = []
    aligned_tables: dict[str, pa.Table] = {}
    failures: list[dict[str, str]] = []
    completed = 0

    for group, fetchers in groups:
        for fetcher in fetchers:
            completed += 1
            dataset_name = fetcher.dataset_name
            filename = f"{dataset_name}.parquet"
            path = window_dir / group / filename
            reused = args.skip_existing and path.exists()
            print(
                f"[{completed}/{total}] {dataset_name} "
                f"({'reusing' if reused else 'fetching'}) ...",
                flush=True,
            )
            began = time.monotonic()
            try:
                with fetcher:
                    if reused:
                        try:
                            table = pq.read_table(path)
                        except Exception:  # noqa: BLE001 - refetch a damaged file
                            table = fetcher.fetch(request)
                            reused = False
                    else:
                        table = fetcher.fetch(request)
                _require_expected_schema(fetcher, table)
                validate_hourly_table(
                    table, start=start, end=end, label=dataset_name
                )

                if not reused:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pq.write_table(table, path, compression="zstd")

                elapsed = time.monotonic() - began
                entry = {
                    "source": dataset_name,
                    "lineage": fetcher.source_name,
                    "group": group,
                    "path": f"{group}/{filename}",
                    "reused": reused,
                    "seconds": round(elapsed, 2),
                    **_coverage(table, expected_hours=len(index)),
                }
                saved.append(entry)
                aligned_tables[dataset_name] = _aligned_columns(
                    group, dataset_name, table
                )
                print(
                    f"  {entry['rows']:,} rows  "
                    f"missing {entry['missing_hours']:,}  "
                    f"nulls {_total_nulls(entry)}  "
                    f"{elapsed:.1f}s"
                    f"{' reused' if reused else ''}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001 - report every source and continue
                elapsed = time.monotonic() - began
                failures.append({
                    "source": dataset_name,
                    "lineage": fetcher.source_name,
                    "group": group,
                    "error": f"{type(exc).__name__}: {exc}",
                })
                print(f"  FAILED after {elapsed:.1f}s: {type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)

    combined_entry: dict[str, Any] | None = None
    if failures or args.no_combined:
        if failures:
            print(
                f"{len(failures)} series failed; skipping combined table.",
                file=sys.stderr,
                flush=True,
            )
    else:
        combined = align_hourly(aligned_tables, start=start, end=end)
        combined_path = window_dir / "combined" / "hourly.parquet"
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(combined, combined_path, compression="zstd")
        value_columns = [
            name for name in combined.column_names if name != TIMESTAMP_COLUMN
        ]
        combined_entry = {
            "path": "combined/hourly.parquet",
            "rows": combined.num_rows,
            "feature_columns": len(value_columns),
            "null_values": {
                name: combined[name].null_count for name in value_columns
            },
        }
        print(
            f"combined  {combined.num_rows:,} rows x "
            f"{len(value_columns)} feature columns",
            flush=True,
        )

    manifest = {
        "generated_at_utc": _iso(datetime.now(UTC)),
        "status": "partial" if failures else "ok",
        "output_dir": str(output_dir),
        "window_dir": str(window_dir),
        "window": {
            "label": label,
            "start_utc": _iso(start),
            "end_utc": _iso(end),
            "resolution": "1h",
            "expected_hours": len(index),
        },
        "sources": saved,
        "combined": combined_entry,
        "failures": failures,
    }
    manifest_path = window_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"manifest  {manifest_path}", flush=True)

    if failures:
        return 1
    print(f"saved {len(saved)} series under {window_dir}", flush=True)
    return 0


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


def _require_expected_schema(fetcher: DataSourceFetcher, table: pa.Table) -> None:
    if not table.schema.equals(fetcher.expected_schema, check_metadata=False):
        raise ValueError(
            f"{fetcher.source_name}: schema {table.schema} does not match "
            f"{fetcher.expected_schema}."
        )


def _coverage(table: pa.Table, *, expected_hours: int) -> dict[str, Any]:
    timestamps = table[TIMESTAMP_COLUMN].to_pylist()
    value_columns = [
        name for name in table.column_names if name != TIMESTAMP_COLUMN
    ]
    return {
        "rows": table.num_rows,
        "missing_hours": expected_hours - table.num_rows,
        "first_utc": _iso(timestamps[0]) if timestamps else None,
        "last_utc": _iso(timestamps[-1]) if timestamps else None,
        "null_values": {
            name: table[name].null_count for name in value_columns
        },
    }


def _total_nulls(entry: dict[str, Any]) -> int:
    return sum(entry["null_values"].values())


def _aligned_columns(group: str, dataset_name: str, table: pa.Table) -> pa.Table:
    if group != "fmi":
        return table
    city = dataset_name.removesuffix("_weather")
    return table.rename_columns([
        name if name == TIMESTAMP_COLUMN else f"{city}_{name}"
        for name in table.column_names
    ])


if __name__ == "__main__":
    raise SystemExit(main())