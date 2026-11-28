#!/usr/bin/env python3
"""Digitize curated retail offers from sahkonhinta.fi into JSON."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fi_energy.contracts import (
    SAHKONHINTA_API_URL,
    Kulutusvaikutus,
    OfferCategory,
    SelectedProduct,
    build_offer_catalog,
    save_offer_catalog,
)
from fi_energy.datasets.http_client import HttpClient

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT = ROOT / "configs" / "offers.json"
DEFAULT_POSTAL_CODE = "00100"

# Product ids are stable in sahkonhinta.fi's consumer API. Each selection is
# paired with the comparison category it represents; hybrid offers also carry
# the retailer's free-text kulutusvaikutus terms in structured form.
SELECTION = (
    SelectedProduct(
        "95b5e00e-7172-4c23-99aa-d55cbd854b47",  # Helen, fixed 12 kk
        OfferCategory.FIXED,
    ),
    SelectedProduct(
        "8a4df42f-6d84-45b8-bfea-cdde3d131b60",  # Korpelan, fixed 24 kk
        OfferCategory.FIXED,
    ),
    SelectedProduct(
        "8873bd98-9d7d-4c01-94e5-e33187a44b9a",  # Porvoon WIND, fixed 12 kk
        OfferCategory.WIND_FIXED,
    ),
    SelectedProduct(
        "bcbbd0bd-b7fb-4690-a468-57bc1df203d0",  # Vaasan tuulisähkö
        OfferCategory.WIND_FIXED,
    ),
    SelectedProduct(
        "167344b3-be87-493a-9040-b11ca5de9342",  # Helen, spot
        OfferCategory.SPOT,
    ),
    SelectedProduct(
        "95457cac-4a87-4244-a7e9-92bbf7ab4c80",  # Nurmijärven tuuli, spot
        OfferCategory.WIND_SPOT,
    ),
    SelectedProduct(
        "7468ad32-2f7b-4daa-bcb8-2cfb69a76384",  # Oomi Jousto 12 kk
        OfferCategory.HYBRID,
        Kulutusvaikutus(
            estimated_c_kwh=0.05,
            typical_range_c_kwh=(-3.9, 1.6),
        ),
    ),
    SelectedProduct(
        "57d99274-807e-4c1e-a9d1-a7fde923f327",  # Keravan Joustosähkö 12 kk
        OfferCategory.HYBRID,
        Kulutusvaikutus(typical_range_c_kwh=(-3.0, 3.0)),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Digitize curated sahkonhinta.fi retail offers into a validated "
            "JSON config."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        help=(
            "Saved /api/productlist JSON payload. When omitted, the payload "
            "is fetched live."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {_display(DEFAULT_OUTPUT)}).",
    )
    parser.add_argument(
        "--postal-code",
        default=DEFAULT_POSTAL_CODE,
        help=f"Finnish postal code (default: {DEFAULT_POSTAL_CODE}).",
    )
    parser.add_argument(
        "--retrieved-at",
        help=(
            "Retrieval timestamp in ISO 8601 form; defaults to the current "
            "UTC time."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_path = args.output.expanduser().resolve()
    retrieved_at = _parse_retrieved_at(args.retrieved_at)

    if args.input is None:
        products = _fetch_products(args.postal_code)
        source_label = f"live API ({args.postal_code})"
    else:
        input_path = args.input.expanduser().resolve()
        products = _read_payload(input_path)
        source_label = str(input_path)

    catalog = build_offer_catalog(
        products,
        SELECTION,
        postal_code=args.postal_code,
        retrieved_at=retrieved_at,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_offer_catalog(catalog, output_path)

    counts = {
        category.value: len(catalog.by_category(category))
        for category in catalog.categories
    }
    summary = ", ".join(
        f"{count} {category}" for category, count in counts.items()
    )
    print(f"source    {source_label}", flush=True)
    print(
        f"offers    {len(catalog.offers)} ({summary})",
        flush=True,
    )
    print(f"wrote     {_display(output_path)}", flush=True)
    return 0


def _fetch_products(postal_code: str) -> list[dict[str, Any]]:
    with HttpClient(SAHKONHINTA_API_URL) as client:
        payload = client.get_json(f"/api/productlist/{postal_code}")
    return _require_payload(payload)


def _read_payload(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise SystemExit(f"Cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise SystemExit(f"{path} is not valid JSON: {exc}") from exc
    return _require_payload(payload)


def _require_payload(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, list) or not all(
        isinstance(item, dict) for item in payload
    ):
        raise SystemExit(
            "Expected the sahkonhinta.fi product list to be a JSON array of "
            "objects."
        )
    return payload


def _parse_retrieved_at(value: str | None) -> datetime:
    if value is None:
        return datetime.now(UTC)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise SystemExit(
            "--retrieved-at must be an ISO 8601 timestamp."
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise SystemExit("--retrieved-at must include a timezone.")
    return parsed.astimezone(UTC)


def _display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())