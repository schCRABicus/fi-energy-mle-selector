#!/usr/bin/env python3
"""Fetch every Phase 1 series over one shared UTC hourly window."""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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
    drop_incomplete_boundary_rows,
    hourly_index,
    interpolate_short_gaps,
    trailing_window,
    validate_hourly_table,
    window_label,
)
from datasets.utils import utc_window

FMI_CITIES = ("helsinki", "vaasa", "oulu")

# Fingrid's free tier rate-limits bursts, so keep its series serial; FMI
# handles the three city pulls comfortably, and Energy-Charts is one request.
GROUP_LIMITS = {"fingrid": 1, "fmi": 3, "energy_charts": 1}

# Linear interpolation is limited to runs of 1-2 missing hours (< 3 hours).
MAX_INTERPOLATION_GAP_HOURS = 2


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
        "--jobs",
        type=int,
        default=4,
        help=(
            "Maximum concurrent source fetches (default: 4). Per-group caps "
            "keep Fingrid serial for its free-tier rate limit and allow up to "
            "three FMI cities at once. Use 1 for a fully sequential pull with "
            "FMI progress bars."
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
    *,
    show_fmi_progress: bool,
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
        FmiWeatherFetcher(city, show_progress=show_fmi_progress)
        for city in FMI_CITIES
    ]
    return [
        ("fingrid", fingrid),
        ("fmi", fmi),
        ("energy_charts", [NordPoolSpotFetcher()]),
    ]


@dataclass(frozen=True)
class SourceTask:
    order: int
    group: str
    fetcher: DataSourceFetcher
    path: Path
    reused: bool


@dataclass(frozen=True)
class SourceResult:
    task: SourceTask
    table: pa.Table | None
    entry: dict[str, Any] | None
    error: str | None
    seconds: float


_PRINT_LOCK = threading.Lock()
_WINDOW_START_KEY = b"fi_energy.window_start_utc"
_WINDOW_END_KEY = b"fi_energy.window_end_utc"


def _file_matches_window(path: Path, start: datetime, end: datetime) -> bool:
    """A reused file must have been written for exactly this window."""
    if not path.exists():
        return False
    try:
        metadata = pq.read_schema(path).metadata or {}
    except Exception:  # noqa: BLE001 - treat an unreadable file as stale
        return False
    return (
        metadata.get(_WINDOW_START_KEY) == _iso(start).encode("ascii")
        and metadata.get(_WINDOW_END_KEY) == _iso(end).encode("ascii")
    )


def _with_window_metadata(
    table: pa.Table, start: datetime, end: datetime
) -> pa.Table:
    metadata = dict(table.schema.metadata or {})
    metadata[_WINDOW_START_KEY] = _iso(start).encode("ascii")
    metadata[_WINDOW_END_KEY] = _iso(end).encode("ascii")
    return table.replace_schema_metadata(metadata)


def _log(message: str, *, error: bool = False) -> None:
    with _PRINT_LOCK:
        print(message, file=sys.stderr if error else sys.stdout, flush=True)


def _fetch_source(
    task: SourceTask,
    *,
    request: FetchRequest,
    start: datetime,
    end: datetime,
    expected_hours: int,
    total: int,
    limits: dict[str, threading.Semaphore],
) -> SourceResult:
    dataset_name = task.fetcher.dataset_name
    with limits[task.group]:
        _log(
            f"[{task.order}/{total}] {dataset_name} "
            f"({'reusing' if task.reused else 'fetching'}) ..."
        )
        began = time.monotonic()
        reused = task.reused
        try:
            with task.fetcher:
                if reused:
                    try:
                        table = pq.read_table(task.path)
                    except Exception:  # noqa: BLE001 - refetch a damaged file
                        table = task.fetcher.fetch(request)
                        reused = False
                else:
                    table = task.fetcher.fetch(request)

            _require_expected_schema(task.fetcher, table)
            validate_hourly_table(table, start=start, end=end, label=dataset_name)

            if not reused:
                task.path.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(
                    _with_window_metadata(table, start, end),
                    task.path,
                    compression="zstd",
                )

            elapsed = time.monotonic() - began
            entry = {
                "source": dataset_name,
                "lineage": task.fetcher.source_name,
                "group": task.group,
                "path": f"{task.group}/{task.path.name}",
                "reused": reused,
                "seconds": round(elapsed, 2),
                **_coverage(table, expected_hours=expected_hours),
            }
            return SourceResult(
                task=task,
                table=table,
                entry=entry,
                error=None,
                seconds=round(elapsed, 2),
            )
        except Exception as exc:  # noqa: BLE001 - report every source and continue
            elapsed = time.monotonic() - began
            return SourceResult(
                task=task,
                table=None,
                entry=None,
                error=f"{type(exc).__name__}: {exc}",
                seconds=round(elapsed, 2),
            )


def _run_tasks(
    tasks: list[SourceTask],
    *,
    request: FetchRequest,
    start: datetime,
    end: datetime,
    expected_hours: int,
    workers: int,
) -> list[SourceResult]:
    total = len(tasks)
    limits: dict[str, threading.Semaphore] = {}
    for task in tasks:
        limits.setdefault(
            task.group,
            threading.Semaphore(GROUP_LIMITS.get(task.group, 1)),
        )

    # Start the independently slow groups first so a serial Fingrid pull does
    # not occupy the workers while the FMI cities are still queued.
    scheduled = sorted(
        tasks, key=lambda task: (task.group == "fingrid", task.order)
    )
    results: list[SourceResult] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                _fetch_source,
                task,
                request=request,
                start=start,
                end=end,
                expected_hours=expected_hours,
                total=total,
                limits=limits,
            )
            for task in scheduled
        ]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if result.error is None:
                entry = result.entry
                assert entry is not None
                _log(
                    f"  [{result.task.order}/{total}] {entry['source']}  "
                    f"{entry['rows']:,} rows  "
                    f"missing {entry['missing_hours']:,}  "
                    f"nulls {_total_nulls(entry)}  "
                    f"{result.seconds:.1f}s"
                    f"{' reused' if result.entry['reused'] else ''}"
                )
            else:
                _log(
                    f"  [{result.task.order}/{total}] "
                    f"{result.task.fetcher.dataset_name} FAILED after "
                    f"{result.seconds:.1f}s: {result.error}",
                    error=True,
                )
    results.sort(key=lambda result: result.task.order)
    return results


def _build_tasks(
    groups: list[tuple[str, list[DataSourceFetcher]]],
    window_dir: Path,
    *,
    skip_existing: bool,
    start: datetime,
    end: datetime,
) -> list[SourceTask]:
    tasks: list[SourceTask] = []
    for group, fetchers in groups:
        for fetcher in fetchers:
            path = window_dir / group / f"{fetcher.dataset_name}.parquet"
            tasks.append(SourceTask(
                order=len(tasks) + 1,
                group=group,
                fetcher=fetcher,
                path=path,
                reused=(
                    skip_existing
                    and _file_matches_window(path, start, end)
                ),
            ))
    return tasks


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
    if args.jobs < 1:
        raise SystemExit("--jobs must be at least 1.")

    end = args.end or trailing_window(years=args.years)[1]
    start = args.start or trailing_window(end, years=args.years)[0]
    try:
        start, end = utc_window(start, end)
        index = hourly_index(start, end)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    output_dir = args.output_dir.expanduser().resolve()
    request = FetchRequest(start_utc=start, end_utc=end, resolution="1h")
    groups = build_groups(api_key, show_fmi_progress=args.jobs == 1)
    total = sum(len(fetchers) for _, fetchers in groups)
    label = window_label(start, end)
    window_dir = output_dir / label
    window_dir.mkdir(parents=True, exist_ok=True)
    workers = min(args.jobs, total)
    print(
        f"window {_iso(start)} -> {_iso(end)}  [{label}]  "
        f"({len(index):,} hourly slots, {total} series, {workers} worker(s))",
        flush=True,
    )

    tasks = _build_tasks(
        groups,
        window_dir,
        skip_existing=args.skip_existing,
        start=start,
        end=end,
    )
    results = _run_tasks(
        tasks,
        request=request,
        start=start,
        end=end,
        expected_hours=len(index),
        workers=workers,
    )

    saved: list[dict[str, Any]] = []
    aligned_tables: dict[str, pa.Table] = {}
    failures: list[dict[str, str]] = []
    for result in results:
        task = result.task
        dataset_name = task.fetcher.dataset_name
        if result.error is not None or result.entry is None or result.table is None:
            failures.append({
                "source": dataset_name,
                "lineage": task.fetcher.source_name,
                "group": task.group,
                "error": result.error or "fetch produced no table",
            })
            continue
        saved.append(result.entry)
        aligned_tables[dataset_name] = _aligned_columns(
            task.group, dataset_name, result.table
        )

    combined_entry: dict[str, Any] | None = None
    cleaned_entry: dict[str, Any] | None = None
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

        trimmed = drop_incomplete_boundary_rows(combined)
        cleaned = interpolate_short_gaps(
            trimmed, max_gap=MAX_INTERPOLATION_GAP_HOURS
        )
        filled_values = {
            name: trimmed[name].null_count - cleaned[name].null_count
            for name in value_columns
        }
        cleaned_path = window_dir / "combined" / "hourly_clean.parquet"
        pq.write_table(cleaned, cleaned_path, compression="zstd")
        cleaned_timestamps = cleaned[TIMESTAMP_COLUMN].to_pylist()
        remaining_nulls = {
            name: cleaned[name].null_count for name in value_columns
        }
        cleaned_entry = {
            "path": "combined/hourly_clean.parquet",
            "rows": cleaned.num_rows,
            "feature_columns": len(value_columns),
            "first_utc": (
                _iso(cleaned_timestamps[0]) if cleaned_timestamps else None
            ),
            "last_utc": (
                _iso(cleaned_timestamps[-1]) if cleaned_timestamps else None
            ),
            "rows_removed_at_boundaries": combined.num_rows - trimmed.num_rows,
            "interpolation": {
                "method": "linear",
                "max_gap_hours": MAX_INTERPOLATION_GAP_HOURS,
                "filled_values": filled_values,
                "total_filled": sum(filled_values.values()),
            },
            "null_values": remaining_nulls,
        }
        print(
            f"cleaned   {cleaned.num_rows:,} rows; "
            f"filled {sum(filled_values.values()):,} short-gap values; "
            f"dropped {combined.num_rows - trimmed.num_rows:,} boundary rows; "
            f"{sum(remaining_nulls.values()):,} nulls remain",
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
        "cleaned": cleaned_entry,
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