from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from fi_energy.contracts import (
    Kulutusvaikutus,
    OfferCatalog,
    OfferCategory,
    SelectedProduct,
    build_offer_catalog,
    load_offer_catalog,
    offer_from_product,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "offers.json"
RETRIEVED_AT = datetime(2026, 11, 28, 12, 48, 50, tzinfo=UTC)


def make_product(
    *,
    product_id: str = "product-1",
    name: str = "Test Offer",
    provider: str = "Test Energy Oy",
    pricing_model: str = "FixedPrice",
    contract_type: str = "FixedTerm",
    fixed_range: str = "Fixed12",
    target_group: str = "Household",
    wind: float = 0.0,
    renewable: float = 0.0,
    components: list[tuple[str, float, str, bool]] | None = None,
    max_annual_consumption: float | None = 100000.0,
    product_url: str = "https://example.test/offer",
) -> dict[str, Any]:
    if components is None:
        if pricing_model == "Spot":
            components = [
                ("General", 0.5, "CentPerKiwattHour", False),
                ("Monthly", 4.9, "EurPerMonth", False),
            ]
        else:
            components = [
                ("Monthly", 4.9, "EurPerMonth", False),
                ("General", 10.0, "CentPerKiwattHour", False),
            ]
    return {
        "Id": product_id,
        "Name": name,
        "Company": {"Name": provider},
        "Details": {
            "PricingModel": pricing_model,
            "ContractType": contract_type,
            "FixedTimeRange": fixed_range,
            "TargetGroup": target_group,
            "ElectricitySource": {
                "Renewable": {"Wind": wind, "Total": renewable}
            },
            "Pricing": {
                "PriceComponents": [
                    {
                        "PriceComponentType": component_type,
                        "OriginalPayment": {
                            "Price": price,
                            "PaymentUnit": unit,
                        },
                        "HasDiscount": has_discount,
                    }
                    for component_type, price, unit, has_discount in components
                ]
            },
            "ConsumptionLimitation": {
                "MaxXKWhPerY": max_annual_consumption
            },
            "ProductLink": product_url,
        },
    }


def build_catalog(
    products: list[dict[str, Any]],
    selection: list[SelectedProduct],
) -> OfferCatalog:
    return build_offer_catalog(
        products,
        selection,
        postal_code="00100",
        retrieved_at=RETRIEVED_AT,
    )


def test_offer_from_fixed_product():
    offer = offer_from_product(
        make_product(), OfferCategory.FIXED
    )

    assert offer.id == "test-energy-oy-test-offer"
    assert offer.pricing_model.value == "fixed"
    assert offer.contract.type.value == "fixed_term"
    assert offer.contract.term_months == 12
    assert offer.prices.monthly_fee_eur == pytest.approx(4.9)
    assert offer.prices.energy_price_c_kwh == pytest.approx(10.0)
    assert offer.prices.spot_margin_c_kwh is None
    assert offer.vat_included is True
    assert offer.source.product_id == "product-1"
    assert offer.source.api_pricing_model == "FixedPrice"


def test_offer_from_spot_product_uses_the_general_component_as_margin():
    offer = offer_from_product(
        make_product(pricing_model="Spot", contract_type="OpenEnded"),
        OfferCategory.SPOT,
    )

    assert offer.contract.type.value == "open_ended"
    assert offer.contract.term_months is None
    assert offer.prices.energy_price_c_kwh is None
    assert offer.prices.spot_margin_c_kwh == pytest.approx(0.5)


def test_offer_from_hybrid_product_keeps_consumption_impact():
    impact = Kulutusvaikutus(estimated_c_kwh=0.05)

    offer = offer_from_product(
        make_product(pricing_model="Hybrid"),
        OfferCategory.HYBRID,
        consumption_impact=impact,
    )

    assert offer.consumption_impact == impact


def test_wind_category_requires_one_hundred_percent_wind():
    with pytest.raises(ValueError, match="100% wind category"):
        offer_from_product(
            make_product(wind=50.0, renewable=50.0),
            OfferCategory.WIND_FIXED,
        )


def test_wind_offer_requires_a_wind_category():
    with pytest.raises(ValueError, match="not in a wind category"):
        offer_from_product(
            make_product(wind=100.0, renewable=100.0),
            OfferCategory.FIXED,
        )


def test_rejects_a_discounted_component():
    product = make_product(
        components=[
            ("Monthly", 4.9, "EurPerMonth", True),
            ("General", 10.0, "CentPerKiwattHour", False),
        ]
    )

    with pytest.raises(ValueError, match="discount"):
        offer_from_product(product, OfferCategory.FIXED)


def test_rejects_company_target_group():
    with pytest.raises(ValueError, match="household"):
        offer_from_product(
            make_product(target_group="Company"),
            OfferCategory.FIXED,
        )


def test_rejects_time_of_day_pricing():
    product = make_product(
        components=[
            ("Monthly", 4.9, "EurPerMonth", False),
            ("DayTime", 10.0, "CentPerKiwattHour", False),
            ("NightTime", 8.0, "CentPerKiwattHour", False),
        ]
    )

    with pytest.raises(ValueError, match="Unsupported price component"):
        offer_from_product(product, OfferCategory.FIXED)


def test_rejects_an_unparseable_fixed_term():
    with pytest.raises(ValueError, match="FixedTimeRange"):
        offer_from_product(
            make_product(fixed_range="Other"),
            OfferCategory.FIXED,
        )


def test_hybrid_selection_requires_consumption_impact():
    with pytest.raises(ValueError, match="consumption_impact"):
        SelectedProduct("product-1", OfferCategory.HYBRID)


def test_build_catalog_preserves_selection_order():
    products = [
        make_product(product_id="a", name="Alpha"),
        make_product(
            product_id="b",
            name="Beta",
            pricing_model="Spot",
            contract_type="OpenEnded",
        ),
    ]

    catalog = build_catalog(
        products,
        [
            SelectedProduct("b", OfferCategory.SPOT),
            SelectedProduct("a", OfferCategory.FIXED),
        ],
    )

    assert [offer.id for offer in catalog.offers] == [
        "test-energy-oy-beta",
        "test-energy-oy-alpha",
    ]
    assert catalog.source.postal_code == "00100"
    assert catalog.source.retrieved_at == RETRIEVED_AT


def test_build_catalog_rejects_a_missing_product():
    with pytest.raises(ValueError, match="absent"):
        build_catalog(
            [make_product(product_id="a")],
            [SelectedProduct("missing", OfferCategory.FIXED)],
        )


def test_build_catalog_rejects_duplicate_ids_in_the_payload():
    with pytest.raises(ValueError, match="Duplicate product id"):
        build_catalog(
            [make_product(product_id="a"), make_product(product_id="a")],
            [SelectedProduct("a", OfferCategory.FIXED)],
        )


def test_shipped_catalog_covers_every_requested_category():
    catalog = load_offer_catalog(CONFIG_PATH)

    assert len(catalog.offers) == 8
    assert catalog.categories == (
        OfferCategory.FIXED,
        OfferCategory.WIND_FIXED,
        OfferCategory.SPOT,
        OfferCategory.WIND_SPOT,
        OfferCategory.HYBRID,
    )
    assert len(catalog.by_category(OfferCategory.FIXED)) == 2
    assert len(catalog.by_category(OfferCategory.WIND_FIXED)) == 2
    assert len(catalog.by_category(OfferCategory.SPOT)) == 1
    assert len(catalog.by_category(OfferCategory.WIND_SPOT)) == 1
    assert len(catalog.by_category(OfferCategory.HYBRID)) == 2

    for offer in catalog.offers:
        if offer.category in {
            OfferCategory.WIND_FIXED,
            OfferCategory.WIND_SPOT,
        }:
            assert offer.energy_source.wind_share_pct == 100.0
        if offer.category is OfferCategory.HYBRID:
            assert offer.consumption_impact is not None
        assert offer.source.product_id
        assert offer.vat_included


def test_shipped_catalog_round_trips_through_json():
    catalog = load_offer_catalog(CONFIG_PATH)

    restored = OfferCatalog.from_dict(
        json.loads(json.dumps(catalog.to_dict(), ensure_ascii=False))
    )

    assert restored == catalog


def test_load_rejects_unknown_schema_version():
    data = load_offer_catalog(CONFIG_PATH).to_dict()
    data["schema_version"] = 2

    with pytest.raises(ValueError, match="Unsupported schema_version"):
        OfferCatalog.from_dict(data)


def test_load_rejects_unknown_keys():
    data = load_offer_catalog(CONFIG_PATH).to_dict()
    data["offers"][0]["unexpected"] = True

    with pytest.raises(ValueError, match="unknown key"):
        OfferCatalog.from_dict(data)


def test_load_rejects_duplicate_offer_ids():
    catalog = load_offer_catalog(CONFIG_PATH)
    duplicated = replace(catalog.offers[0], id=catalog.offers[1].id)

    with pytest.raises(ValueError, match="Duplicate offer id"):
        OfferCatalog(
            schema_version=catalog.schema_version,
            source=catalog.source,
            offers=(*catalog.offers, duplicated),
        )


def test_fixed_offer_requires_an_energy_price():
    catalog = load_offer_catalog(CONFIG_PATH)
    offer = catalog.offers[0]

    with pytest.raises(ValueError, match="energy_price_c_kwh"):
        replace(offer, prices=replace(offer.prices, energy_price_c_kwh=None))


def test_spot_offer_requires_a_margin():
    catalog = load_offer_catalog(CONFIG_PATH)
    offer = catalog.by_category(OfferCategory.SPOT)[0]

    with pytest.raises(ValueError, match="spot_margin_c_kwh"):
        replace(offer, prices=replace(offer.prices, spot_margin_c_kwh=None))


def test_hybrid_offer_requires_consumption_impact():
    catalog = load_offer_catalog(CONFIG_PATH)
    offer = catalog.by_category(OfferCategory.HYBRID)[0]

    with pytest.raises(ValueError, match="consumption_impact"):
        replace(offer, consumption_impact=None)


def test_consumption_impact_rejects_an_inverted_range():
    with pytest.raises(ValueError, match="lower bound"):
        Kulutusvaikutus(typical_range_c_kwh=(2.0, -2.0))


def test_load_rejects_invalid_json(tmp_path: Path):
    path = tmp_path / "offers.json"
    path.write_text("{not json", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        load_offer_catalog(path)