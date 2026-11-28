#!/usr/bin/env python3
"""Build the Phase 2 feature table from the cleaned Phase 1 hourly dataset."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from fi_energy.features import build_feature_table, describe_features
from fi_energy.features.calendar import HOLIDAY_COUNTRY, LOCAL_TIMEZONE
from fi_energy.features.thermal_demand import HEATING_BASE_TEMPERATURE_C

ROOT = Path(__file__).resolve().parents[1]

_METADATA_PREFIX = "fi_energy."
_DEFAULT_INPUT = ROOT / "data" / "3y" / "combined" / "hourly_clean.parquet"
_DEFAULT_OUTPUT = ROOT / "data" / "3y" / "features" / "hourly_features.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the timestamped Phase 2 feature table from the cleaned "
            "Phase 1 hourly dataset."
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
        help=f"Feature Parquet table (default: {_display(_DEFAULT_OUTPUT)}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute features and print diagnostics without writing files.",
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
        features = build_feature_table(source)
        description = describe_features(source, features)
    except (OSError, pa.ArrowException, TypeError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"Feature build failed: {exc}") from exc

    generated_at = datetime.now(UTC)
    features = features.replace_schema_metadata(
        _schema_metadata(
            features.schema.metadata,
            source_path=input_path,
            generated_at=generated_at,
            description=description,
        )
    )

    _print_summary(input_path, description)
    if args.dry_run:
        print("dry run: no files written", flush=True)
        return 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_parquet(features, output_path)
    manifest_path = output_path.parent / "manifest.json"
    _write_manifest(
        manifest_path,
        input_path=input_path,
        output_path=output_path,
        generated_at=generated_at,
        description=description,
    )

    print(
        f"wrote     {_display(output_path)}  "
        f"({features.num_rows:,} x {features.num_columns})",
        flush=True,
    )
    print(f"manifest  {_display(manifest_path)}", flush=True)
    return 0


def _schema_metadata(
    existing: dict[bytes, bytes] | None,
    *,
    source_path: Path,
    generated_at: datetime,
    description: dict[str, Any],
) -> dict[bytes, bytes]:
    diagnostics = description["diagnostics"]
    wind = diagnostics["wind_ratios"]
    thermal = diagnostics["thermal_demand"]
    calendar = diagnostics["calendar"]
    negative_min = wind["negative_wind_min_mw"]
    values = {
        "feature_stage": "phase2",
        "generated_at_utc": _iso(generated_at),
        "source_table": str(source_path),
        "source_rows": str(description["rows"]),
        "feature_names": json.dumps(description["feature_columns"]),
        "feature_specs": json.dumps(description["features"]),
        "wind_negative_policy": "clip_to_zero",
        "wind_negative_count": str(wind["negative_wind_count"]),
        "wind_negative_min_mw": "" if negative_min is None else str(negative_min),
        "zero_denominator_policy": "null",
        "zero_denominators": json.dumps(wind["zero_denominators"]),
        "heating_base_temperature_c": str(HEATING_BASE_TEMPERATURE_C),
        "heating_hours": str(thermal["heating_hours"]),
        "temperature_nulls": str(thermal["temperature_nulls"]),
        "temperature_rolling_policy": (
            "trailing 24h/72h; nulls skipped; full window required"
        ),
        "calendar_timezone": LOCAL_TIMEZONE,
        "calendar_holiday_country": HOLIDAY_COUNTRY,
        "calendar_weekend_hours": str(calendar["weekend_hours"]),
        "calendar_holiday_hours": str(calendar["holiday_hours"]),
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


def _write_manifest(
    path: Path,
    *,
    input_path: Path,
    output_path: Path,
    generated_at: datetime,
    description: dict[str, Any],
) -> None:
    diagnostics = description["diagnostics"]
    wind = diagnostics["wind_ratios"]
    thermal = diagnostics["thermal_demand"]
    calendar = diagnostics["calendar"]
    manifest = {
        "generated_at_utc": _iso(generated_at),
        "status": "ok",
        "source": {
            "path": str(input_path),
            "rows": description["rows"],
            "input_null_counts": description["input_null_counts"],
        },
        "output": {
            "path": str(output_path),
            "rows": description["rows"],
            "columns": len(description["feature_columns"]) + 1,
            "feature_columns": description["feature_columns"],
            "compression": "zstd",
        },
        "features": description["features"],
        "policy": {
            "negative_wind": {
                "action": "clip_to_zero",
                "count": wind["negative_wind_count"],
                "min_mw": wind["negative_wind_min_mw"],
            },
            "zero_denominators": {
                "action": "null",
                "counts": wind["zero_denominators"],
            },
            "heating_demand": {
                "base_temperature_c": HEATING_BASE_TEMPERATURE_C,
                "heating_hours": thermal["heating_hours"],
                "max_heating_degree_hours": (
                    thermal["max_heating_degree_hours"]
                ),
            },
            "temperature_rolling": {
                "windows_hours": [24, 72],
                "alignment": "trailing; never uses future hours",
                "nulls": (
                    "skipped within the window; null until a full window of "
                    "history exists"
                ),
            },
            "calendar": {
                "timezone": LOCAL_TIMEZONE,
                "holiday_country": HOLIDAY_COUNTRY,
                "weekend_hours": calendar["weekend_hours"],
                "holiday_hours": calendar["holiday_hours"],
            },
            "nulls": "propagate; no imputation",
            "rows": "preserve every source row",
        },
        "null_values": description["null_values"],
        "statistics": description["statistics"],
    }
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _print_summary(
    input_path: Path,
    description: dict[str, Any],
) -> None:
    groups = ", ".join(description["groups"])
    print(
        f"source    {_display(input_path)}  {description['rows']:,} rows",
        flush=True,
    )
    print(
        f"features  {len(description['feature_columns'])}  "
        f"(groups: {groups})",
        flush=True,
    )

    diagnostics = description["diagnostics"]["wind_ratios"]
    negative_count = diagnostics["negative_wind_count"]
    if negative_count:
        print(
            f"wind_mw   clipped {negative_count:,} negative values to 0.0 "
            f"(min {diagnostics['negative_wind_min_mw']:.2f} MW)",
            flush=True,
        )
    else:
        print("wind_mw   no negative values to clip", flush=True)

    thermal = description["diagnostics"]["thermal_demand"]
    print(
        f"temp      {thermal['temperature_nulls']:,} null Helsinki readings; "
        f"{thermal['heating_hours']:,} heating hours "
        f"(max {_format(thermal['max_heating_degree_hours'])} degree-hours "
        f"below {HEATING_BASE_TEMPERATURE_C:g} C)",
        flush=True,
    )

    calendar = description["diagnostics"]["calendar"]
    print(
        f"calendar  {calendar['weekend_hours']:,} weekend hours; "
        f"{calendar['holiday_hours']:,} public-holiday hours "
        f"({calendar['distinct_holidays']} distinct holidays, "
        f"{LOCAL_TIMEZONE})",
        flush=True,
    )

    for name in description["feature_columns"]:
        stats = description["statistics"][name]
        print(
            f"{name:<24} nulls {stats['null_count']:,}  "
            f"min {_format(stats['min'])}  "
            f"mean {_format(stats['mean'])}  "
            f"max {_format(stats['max'])}",
            flush=True,
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