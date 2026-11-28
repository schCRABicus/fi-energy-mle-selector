#!/usr/bin/env python3
"""Build the two standard Finnish hourly household load profiles."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

from fi_energy.datasets.alignment import TIMESTAMP_COLUMN
from fi_energy.profiles import (
    BASE_TEMPERATURE_C,
    DEFAULT_PROFILES,
    TEMPERATURE_SMOOTHING_HOURS,
    LoadProfileBuild,
    build_load_profile_table,
    profile_annual_totals,
)

ROOT = Path(__file__).resolve().parents[1]

_METADATA_PREFIX = "fi_energy."
_DEFAULT_INPUT = ROOT / "data" / "3y" / "dataset" / "hourly_dataset.parquet"
_DEFAULT_OUTPUT = (
    ROOT / "data" / "3y" / "profiles" / "hourly_load_profiles.parquet"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the 2,500 kWh/year apartment and 18,000 kWh/year "
            "electrically heated detached house profiles on the dataset's "
            "hourly grid."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=_DEFAULT_INPUT,
        help=(
            "Hourly table with timestamps and Helsinki temperature "
            f"(default: {_display(_DEFAULT_INPUT)})."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_DEFAULT_OUTPUT,
        help=f"Profile Parquet path (default: {_display(_DEFAULT_OUTPUT)}).",
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
        build = build_load_profile_table(source)
    except (OSError, pa.ArrowException, TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Load profile build failed: {exc}") from exc

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

    totals = profile_annual_totals(build)
    manifest = {
        "generated_at_utc": _iso(generated_at),
        "status": "ok",
        "source": {
            "path": str(input_path),
            "rows": source.num_rows,
            "first_utc": _iso(source[TIMESTAMP_COLUMN][0].as_py()),
            "last_utc": _iso(source[TIMESTAMP_COLUMN][-1].as_py()),
        },
        "output": {
            "path": str(output_path),
            "csv_path": None if csv_path is None else str(csv_path),
            "rows": table.num_rows,
            "columns": table.column_names,
            "unit": "hourly_kwh",
            "compression": "zstd",
        },
        "model": {
            "name": "finnish_household_hourly_v1",
            "local_timezone": "Europe/Helsinki",
            "temperature_column": build.temperature_column,
            "base_temperature_c": BASE_TEMPERATURE_C,
            "temperature_smoothing_hours": TEMPERATURE_SMOOTHING_HOURS,
            "space_heating": (
                "proportional to max(0, 17 C - T) after a six-hour trailing "
                "mean of the outdoor temperature"
            ),
            "daily_shape": (
                "local-time household and hot-water shapes, with a later and "
                "flatter weekend profile"
            ),
            "normalization": (
                "each component is scaled so a fully covered calendar year "
                "totals its target kWh, absorbing interior gaps; a partial "
                "year at the window edge is scaled by its share of the hours "
                "in that year"
            ),
        },
        "profiles": [profile.to_dict() for profile in build.profiles],
        "annual_totals_kwh": totals,
        "checks": build.checks,
    }
    manifest_path = output_path.parent / "manifest.json"
    _write_manifest(manifest_path, manifest)

    _print_summary(build, totals, output_path, csv_path)
    print(f"manifest  {_display(manifest_path)}", flush=True)
    return 0


def _schema_metadata(
    existing: dict[bytes, bytes] | None,
    *,
    source_path: Path,
    generated_at: datetime,
    build: LoadProfileBuild,
) -> dict[bytes, bytes]:
    values = {
        "profile_model": "finnish_household_hourly_v1",
        "generated_at_utc": _iso(generated_at),
        "source_table": str(source_path),
        "temperature_column": build.temperature_column,
        "base_temperature_c": str(BASE_TEMPERATURE_C),
        "temperature_smoothing_hours": str(TEMPERATURE_SMOOTHING_HOURS),
        "profiles": json.dumps(
            [profile.to_dict() for profile in build.profiles]
        ),
        "checks": json.dumps(build.checks),
        "unit": "hourly_kwh",
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
    build: LoadProfileBuild,
    totals: dict[str, dict[int, float]],
    output_path: Path,
    csv_path: Path | None,
) -> None:
    print(
        f"source    {build.table.num_rows:,} hourly rows",
        flush=True,
    )
    for profile in build.profiles:
        shares = profile.shares
        print(
            f"{profile.key:<24} {profile.annual_kwh:>10,.0f} kWh/yr  "
            f"household {shares['household']:.0%}  "
            f"hot water {shares['hot_water']:.0%}  "
            f"heating {shares['space_heating']:.0%}",
            flush=True,
        )
        for year, total in totals[profile.key].items():
            print(f"  {year}  {total:,.1f} kWh", flush=True)
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


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _iso(timestamp: datetime) -> str:
    return timestamp.astimezone(UTC).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    raise SystemExit(main())