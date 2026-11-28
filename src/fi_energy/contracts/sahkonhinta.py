"""Digitization adapter for the sahkonhinta.fi product-list API."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from fi_energy.contracts.offers import (
    CatalogSource,
    ContractTerms,
    ContractType,
    EnergySource,
    Kulutusvaikutus,
    OfferCatalog,
    OfferCategory,
    OfferPricing,
    OfferSource,
    PricingModel,
    RetailOffer,
    SCHEMA_VERSION,
)

__all__ = [
    "SAHKONHINTA_API_URL",
    "SAHKONHINTA_NAME",
    "SAHKONHINTA_OPERATOR",
    "SelectedProduct",
    "build_offer_catalog",
    "offer_from_product",
]

SAHKONHINTA_API_URL = (
    "https://ev-shv-prod-app-wa-consumerapi1.azurewebsites.net"
)
SAHKONHINTA_NAME = "sahkonhinta.fi"
SAHKONHINTA_OPERATOR = "Energiavirasto (Energy Authority of Finland)"

_PRICING_MODELS = {
    "FixedPrice": PricingModel.FIXED,
    "Spot": PricingModel.SPOT,
    "Hybrid": PricingModel.HYBRID,
}

_TERM_PATTERN = re.compile(r"Fixed(?P<months>\d+)")


@dataclass(frozen=True, slots=True)
class SelectedProduct:
    """
    A curated sahkonhinta.fi product id and its comparison category.

    ``consumption_impact`` carries the retailer's free-text kulutusvaikutus
    terms in structured form; it is required for hybrid offers and forbidden
    for every other category.
    """

    product_id: str
    category: OfferCategory
    consumption_impact: Kulutusvaikutus | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.product_id, str) or not self.product_id.strip():
            raise ValueError("SelectedProduct.product_id must be non-empty.")
        category = OfferCategory(self.category)
        object.__setattr__(self, "category", category)
        if category is OfferCategory.HYBRID:
            if self.consumption_impact is None:
                raise ValueError(
                    "A hybrid SelectedProduct must declare consumption_impact."
                )
        elif self.consumption_impact is not None:
            raise ValueError(
                "Only a hybrid SelectedProduct may declare "
                "consumption_impact."
            )


def build_offer_catalog(
    products: Sequence[Mapping[str, Any]],
    selection: Sequence[SelectedProduct],
    *,
    postal_code: str,
    retrieved_at: datetime,
    api_url: str = SAHKONHINTA_API_URL,
) -> OfferCatalog:
    """
    Builds a validated offer catalog from a raw product-list payload.

    ``selection`` fixes both the offers to keep and the category each one
    represents. Products are emitted in selection order.
    """
    if not selection:
        raise ValueError("At least one product must be selected.")
    by_id: dict[str, Mapping[str, Any]] = {}
    for product in products:
        product_id = _product_id(product)
        if product_id in by_id:
            raise ValueError(f"Duplicate product id {product_id!r} in payload.")
        by_id[product_id] = product

    selected_ids: set[str] = set()
    offers: list[RetailOffer] = []
    used_slugs: set[str] = set()
    for item in selection:
        if item.product_id in selected_ids:
            raise ValueError(
                f"Product {item.product_id!r} is selected more than once."
            )
        selected_ids.add(item.product_id)
        try:
            product = by_id[item.product_id]
        except KeyError:
            raise ValueError(
                f"Selected product {item.product_id!r} is absent from the "
                "sahkonhinta.fi payload."
            ) from None
        offer = offer_from_product(
            product,
            item.category,
            consumption_impact=item.consumption_impact,
        )
        offer = _with_unique_id(offer, used_slugs)
        offers.append(offer)

    return OfferCatalog(
        schema_version=SCHEMA_VERSION,
        source=CatalogSource(
            name=SAHKONHINTA_NAME,
            operator=SAHKONHINTA_OPERATOR,
            api_url=api_url,
            postal_code=postal_code,
            retrieved_at=retrieved_at,
        ),
        offers=tuple(offers),
    )


def offer_from_product(
    product: Mapping[str, Any],
    category: OfferCategory,
    *,
    consumption_impact: Kulutusvaikutus | None = None,
) -> RetailOffer:
    """Converts one sahkonhinta.fi product object into a ``RetailOffer``."""
    category = OfferCategory(category)
    _require_keys(
        product,
        {"Id", "Name", "Company", "Details"},
        "product",
    )
    company = _mapping(product["Company"], "product.Company")
    _require_keys(company, {"Name"}, "product.Company")
    details = _mapping(product["Details"], "product.Details")
    _require_keys(
        details,
        {
            "PricingModel",
            "ContractType",
            "FixedTimeRange",
            "ElectricitySource",
            "Pricing",
            "ConsumptionLimitation",
            "TargetGroup",
            "ProductLink",
        },
        "product.Details",
    )

    target_group = _string(details["TargetGroup"], "product.Details.TargetGroup")
    if target_group not in {"Household", "Both"}:
        raise ValueError(
            f"Only household offers can be digitized, got target group "
            f"{target_group!r}."
        )

    api_pricing_model = _string(
        details["PricingModel"], "product.Details.PricingModel"
    )
    try:
        pricing_model = _PRICING_MODELS[api_pricing_model]
    except KeyError:
        raise ValueError(
            f"Unsupported sahkonhinta.fi pricing model "
            f"{api_pricing_model!r}."
        ) from None
    if _expected_pricing_model(category) is not pricing_model:
        raise ValueError(
            f"Category {category.value!r} does not match the API pricing "
            f"model {api_pricing_model!r}."
        )

    contract = _contract_terms(details)
    energy_source = _energy_source(details)
    prices = _prices(details, pricing_model)

    if category is OfferCategory.HYBRID:
        if consumption_impact is None:
            raise ValueError(
                "A hybrid offer requires a structured consumption_impact."
            )
    elif consumption_impact is not None:
        raise ValueError(
            "Only hybrid offers may declare a consumption_impact."
        )

    provider = _string(company["Name"], "product.Company.Name")
    product_name = _string(product["Name"], "product.Name").strip()
    product_url = _optional_string(details["ProductLink"], "ProductLink")
    if product_url is not None and not product_url.strip():
        product_url = None

    return RetailOffer(
        id=_slug(f"{provider}-{product_name}"),
        provider=provider,
        product_name=product_name,
        category=category,
        pricing_model=pricing_model,
        contract=contract,
        energy_source=energy_source,
        prices=prices,
        consumption_impact=consumption_impact,
        max_annual_consumption_kwh=_max_annual_consumption(details),
        vat_included=True,
        source=OfferSource(
            product_id=_product_id(product),
            product_url=product_url,
            api_pricing_model=api_pricing_model,
        ),
    )


def _product_id(product: Mapping[str, Any]) -> str:
    _require_keys(product, {"Id"}, "product")
    product_id = _string(product["Id"], "product.Id").strip()
    if not product_id:
        raise ValueError("product.Id must not be empty.")
    return product_id


def _expected_pricing_model(category: OfferCategory) -> PricingModel:
    return {
        OfferCategory.FIXED: PricingModel.FIXED,
        OfferCategory.WIND_FIXED: PricingModel.FIXED,
        OfferCategory.SPOT: PricingModel.SPOT,
        OfferCategory.WIND_SPOT: PricingModel.SPOT,
        OfferCategory.HYBRID: PricingModel.HYBRID,
    }[category]


def _contract_terms(details: Mapping[str, Any]) -> ContractTerms:
    api_type = _string(details["ContractType"], "product.Details.ContractType")
    if api_type == "OpenEnded":
        return ContractTerms(type=ContractType.OPEN_ENDED, term_months=None)
    if api_type == "FixedTerm":
        fixed_range = _string(
            details["FixedTimeRange"], "product.Details.FixedTimeRange"
        )
        match = _TERM_PATTERN.fullmatch(fixed_range)
        if match is None:
            raise ValueError(
                f"Cannot derive a term in months from FixedTimeRange "
                f"{fixed_range!r}."
            )
        return ContractTerms(
            type=ContractType.FIXED_TERM,
            term_months=int(match.group("months")),
        )
    raise ValueError(f"Unsupported sahkonhinta.fi contract type {api_type!r}.")


def _energy_source(details: Mapping[str, Any]) -> EnergySource:
    source = _mapping(
        details["ElectricitySource"], "product.Details.ElectricitySource"
    )
    _require_keys(source, {"Renewable"}, "product.Details.ElectricitySource")
    renewable = _mapping(
        source["Renewable"],
        "product.Details.ElectricitySource.Renewable",
    )
    _require_keys(
        renewable,
        {"Wind", "Total"},
        "product.Details.ElectricitySource.Renewable",
    )
    return EnergySource(
        wind_share_pct=_number(
            renewable["Wind"],
            "product.Details.ElectricitySource.Renewable.Wind",
        ),
        renewable_share_pct=_number(
            renewable["Total"],
            "product.Details.ElectricitySource.Renewable.Total",
        ),
    )


def _prices(
    details: Mapping[str, Any],
    pricing_model: PricingModel,
) -> OfferPricing:
    pricing = _mapping(details["Pricing"], "product.Details.Pricing")
    _require_keys(
        pricing,
        {"PriceComponents"},
        "product.Details.Pricing",
    )
    components = pricing["PriceComponents"]
    if not isinstance(components, Sequence) or isinstance(
        components, (str, bytes)
    ):
        raise ValueError(
            "product.Details.Pricing.PriceComponents must be an array."
        )

    by_type: dict[str, Mapping[str, Any]] = {}
    for index, raw_component in enumerate(components):
        path = f"product.Details.Pricing.PriceComponents[{index}]"
        component = _mapping(raw_component, path)
        _require_keys(
            component,
            {"PriceComponentType", "OriginalPayment", "HasDiscount"},
            path,
        )
        if component["HasDiscount"]:
            raise ValueError(
                f"{path} carries a discount; select an undiscounted listing "
                "or extend the schema to model the discount."
            )
        component_type = _string(
            component["PriceComponentType"], f"{path}.PriceComponentType"
        )
        if component_type not in {"Monthly", "General"}:
            raise ValueError(
                f"Unsupported price component {component_type!r}; only "
                "Monthly and General components can be digitized."
            )
        if component_type in by_type:
            raise ValueError(
                f"Multiple {component_type!r} price components are not "
                "supported."
            )
        by_type[component_type] = component

    monthly = by_type.get("Monthly")
    general = by_type.get("General")
    if monthly is None or general is None:
        raise ValueError(
            "An offer must declare exactly one Monthly and one General "
            "price component."
        )

    monthly_price, monthly_unit = _payment(monthly, "Monthly")
    energy_price, energy_unit = _payment(general, "General")
    if monthly_unit != "EurPerMonth":
        raise ValueError(
            f"Monthly fee unit must be EurPerMonth, got {monthly_unit!r}."
        )
    if energy_unit != "CentPerKiwattHour":
        raise ValueError(
            f"Energy price unit must be CentPerKiwattHour, got "
            f"{energy_unit!r}."
        )

    if pricing_model is PricingModel.SPOT:
        # The General component of a spot offer is the retailer margin.
        return OfferPricing(
            monthly_fee_eur=monthly_price,
            energy_price_c_kwh=None,
            spot_margin_c_kwh=energy_price,
        )
    return OfferPricing(
        monthly_fee_eur=monthly_price,
        energy_price_c_kwh=energy_price,
        spot_margin_c_kwh=None,
    )


def _payment(
    component: Mapping[str, Any], path: str
) -> tuple[float, str]:
    payment = _mapping(
        component["OriginalPayment"], f"{path}.OriginalPayment"
    )
    _require_keys(
        payment,
        {"Price", "PaymentUnit"},
        f"{path}.OriginalPayment",
    )
    return (
        _number(payment["Price"], f"{path}.OriginalPayment.Price"),
        _string(
            payment["PaymentUnit"], f"{path}.OriginalPayment.PaymentUnit"
        ),
    )


def _max_annual_consumption(details: Mapping[str, Any]) -> float | None:
    limitation = _mapping(
        details["ConsumptionLimitation"],
        "product.Details.ConsumptionLimitation",
    )
    _require_keys(
        limitation,
        {"MaxXKWhPerY"},
        "product.Details.ConsumptionLimitation",
    )
    value = limitation["MaxXKWhPerY"]
    return None if value is None else _number(
        value, "product.Details.ConsumptionLimitation.MaxXKWhPerY"
    )


def _with_unique_id(offer: RetailOffer, used: set[str]) -> RetailOffer:
    if offer.id not in used:
        used.add(offer.id)
        return offer
    suffix = 2
    while f"{offer.id}-{suffix}" in used:
        suffix += 1
    unique_id = f"{offer.id}-{suffix}"
    used.add(unique_id)
    return RetailOffer(
        id=unique_id,
        provider=offer.provider,
        product_name=offer.product_name,
        category=offer.category,
        pricing_model=offer.pricing_model,
        contract=offer.contract,
        energy_source=offer.energy_source,
        prices=offer.prices,
        consumption_impact=offer.consumption_impact,
        max_annual_consumption_kwh=offer.max_annual_consumption_kwh,
        vat_included=offer.vat_included,
        source=offer.source,
    )


def _slug(value: str) -> str:
    ascii_value = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
        .lower()
    )
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_value).strip("-")
    return slug or "offer"


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a JSON object.")
    return value


def _require_keys(
    obj: Mapping[str, Any], required: set[str], path: str
) -> None:
    missing = sorted(required - set(obj))
    if missing:
        raise ValueError(f"{path} is missing key(s): {', '.join(missing)}.")


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a string.")
    return value


def _optional_string(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, path)


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a number.")
    return float(value)