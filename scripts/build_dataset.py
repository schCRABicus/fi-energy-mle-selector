#!/usr/bin/env python3
"""Build the complete 24-feature + next-hour-target training dataset."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

from fi_energy.datasets.alignment import TIMESTAMP_COLUMN
from fi_energy.features import (
    TARGET_COLUMN,
    DatasetBuild,
    build_training_dataset,
    feature_statistics,
)

ROOT = Path(__file__).resolve().parents[1]

_METADATA_PREFIX = "fi_energy."
_DEFAULT_INPUT = ROOT / "data" / "3y" / "combined" / "hourly_clean.parquet"
_DEFAULT_OUTPUT = ROOT / "data" / "3y" / "dataset" / "hourly_dataset.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Assemble the 24-feature training table and the next-hour spot "
            "price target, verify completeness and leakage, and save it."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_DEFAULT_INPUT,
        help=f"Cleaned Phase 1 Parquet table (default: {_display(_DEFAULT_INPUT)}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Dataset Parquet path (default: {_display(_DEFAULT_OUTPUT)}).",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also write a CSV copy beside the Parquet file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input table not found: {input_path}")
    if output_path == input_path:
        raise SystemExit("Output path must differ from the input path.")

    try:
        source = pq.read_table(input_path)
        build = build_training_dataset(source)
    except (OSError, pa.ArrowException, TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Dataset build failed: {exc}") from exc

    generated_at = datetime.now(UTC)
    table = build.table.replace_schema_metadata(
        _schema_metadata(
            build.table.schema.metadata,
            source_path=input_path,
            generated_at=generated_at,
            build=build,
        )
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_parquet(table, output_path)
    csv_path = output_path.with_suffix(".csv") if args.csv else None
    if csv_path is not None:
        _write_csv(table, csv_path)

    stats = feature_statistics(build)
    nulls = {
        name: stats[name]["null_count"]
        for name in (*build.feature_columns, build.target_column)
    }
    if any(nulls.values()):
        raise SystemExit(f"Dataset contains nulls: {nulls}")

    manifest = {
        "generated_at_utc": _iso(generated_at),
        "status": "ok",
        "source": {
            "path": str(input_path),
            "rows": build.source_rows,
            "first_utc": _iso(source[TIMESTAMP_COLUMN][0].as_py()),
            "last_utc": _iso(source[TIMESTAMP_COLUMN][-1].as_py()),
        },
        "output": {
            "path": str(output_path),
            "csv_path": None if csv_path is None else str(csv_path),
            "rows": table.num_rows,
            "columns": table.num_columns,
            "feature_columns": list(build.feature_columns),
            "target_column": build.target_column,
            "compression": "zstd",
        },
        "target": {
            "definition": (
                "spot_price_c_kwh at timestamp + 1 hour; every feature is "
                "observed at timestamp"
            ),
            "alignment": "next_hour",
            "unit": "c/kWh (VAT-exclusive day-ahead price)",
        },
        "rows": {
            "source": build.source_rows,
            "kept": table.num_rows,
            "dropped_incomplete": build.dropped_rows,
            "first_utc": _iso(table[TIMESTAMP_COLUMN][0].as_py()),
            "last_utc": _iso(table[TIMESTAMP_COLUMN][-1].as_py()),
        },
        "null_values": nulls,
        "nan_values": {
            name: _nan_count(table[name])
            for name in (*build.feature_columns, build.target_column)
        },
        "leakage_checks": build.checks,
        "statistics": stats,
        "policy": {
            "complete_cases": (
                "drop any row with a null or NaN feature or target; no "
                "imputation at this stage"
            ),
            "future_information": (
                "features use only the current and preceding hours; rolling "
                "windows are trailing and the target is never an input"
            ),
            "target": (
                "the label is the next hour's spot price; the final source "
                "row therefore has no label and is dropped"
            ),
        },
    }
    manifest_path = output_path.parent / "manifest.json"
    _write_manifest(manifest_path, manifest)

    _print_summary(build, stats, output_path, csv_path)
    print(f"manifest  {_display(manifest_path)}", flush=True)
    return 0


def _schema_metadata(
    existing: dict[bytes, bytes] | None,
    *,
    source_path: Path,
    generated_at: datetime,
    build: DatasetBuild,
) -> dict[bytes, bytes]:
    values = {
        "dataset_stage": "phase2",
        "generated_at_utc": _iso(generated_at),
        "source_table": str(source_path),
        "source_rows": str(build.source_rows),
        "rows": str(build.table.num_rows),
        "dropped_incomplete_rows": str(build.dropped_rows),
        "feature_columns": json.dumps(list(build.feature_columns)),
        "target_column": build.target_column,
        "target_alignment": "next_hour",
        "leakage_checks": json.dumps(build.checks),
        "complete_case_policy": "drop null or NaN feature/target rows",
    }
    metadata = dict(existing or {})
    metadata.update({
        f"{_METADATA_PREFIX}{key}".encode("ascii"): value.encode("utf-8")
        for key, value in values.items()
    })
    return metadata


def _write_parquet(table: pa.Table, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        pq.write_table(table, temporary, compression="zstd")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_csv(table: pa.Table, path: Path) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        pcsv.write_csv(table, temporary)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _print_summary(
    build: DatasetBuild,
    stats: dict[str, dict[str, Any]],
    output_path: Path,
    csv_path: Path | None,
) -> None:
    print(
        f"source    {build.source_rows:,} rows  "
        f"-> dataset {build.table.num_rows:,} rows "
        f"({build.dropped_rows:,} incomplete dropped)",
        flush=True,
    )
    print(
        f"features  {len(build.feature_columns)} + target "
        f"{build.target_column}",
        flush=True,
    )
    for name in (*build.feature_columns, build.target_column):
        entry = stats[name]
        print(
            f"{name:<28} nulls 0  "
            f"min {_format(entry['min'])}  "
            f"mean {_format(entry['mean'])}  "
            f"max {_format(entry['max'])}",
            flush=True,
        )
    print(
        "checks    " + ", ".join(
            f"{name}={'ok' if ok else 'FAILED'}"
            for name, ok in build.checks.items()
        ),
        flush=True,
    )
    print(
        f"wrote     {_display(output_path)}  "
        f"({build.table.num_rows:,} x {build.table.num_columns})",
        flush=True,
    )
    if csv_path is not None:
        print(f"csv       {_display(csv_path)}", flush=True)


def _nan_count(column: pa.ChunkedArray) -> int:
    if not pa.types.is_floating(column.type):
        return 0
    return int(
        pc.sum(
            pc.cast(
                pc.fill_null(pc.is_nan(column), False),
                pa.int64(),
            )
        ).as_py()
        or 0
    )


def _format(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6f}"


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _iso(timestamp: datetime) -> str:
    return timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())