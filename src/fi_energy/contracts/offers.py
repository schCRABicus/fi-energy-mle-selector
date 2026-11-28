"""Structured retail electricity offers digitized from comparison sites."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

__all__ = [
    "CatalogSource",
    "ConsumptionImpactDefinition",
    "ContractTerms",
    "ContractType",
    "EnergySource",
    "Kulutusvaikutus",
    "OfferCatalog",
    "OfferCategory",
    "OfferPricing",
    "OfferSource",
    "PricingModel",
    "RetailOffer",
    "load_offer_catalog",
    "save_offer_catalog",
]

SCHEMA_VERSION = 1


class ContractType(StrEnum):
    """Whether an offer binds for a fixed term or continues indefinitely."""

    FIXED_TERM = "fixed_term"
    OPEN_ENDED = "open_ended"


class PricingModel(StrEnum):
    """How the energy component of the bill is calculated."""

    FIXED = "fixed"
    SPOT = "spot"
    HYBRID = "hybrid"


class OfferCategory(StrEnum):
    """The comparison category an offer was digitized from."""

    FIXED = "fixed"
    WIND_FIXED = "wind_fixed"
    SPOT = "spot"
    WIND_SPOT = "wind_spot"
    HYBRID = "hybrid"


class ConsumptionImpactDefinition(StrEnum):
    """The formula used to calculate a hybrid offer's kulutusvaikutus."""

    SPOT_TIMING_DEVIATION = "spot_timing_deviation"


@dataclass(frozen=True, slots=True)
class ContractTerms:
    """The commitment length of an offer."""

    type: ContractType
    term_months: int | None

    def __post_init__(self) -> None:
        contract_type = ContractType(self.type)
        object.__setattr__(self, "type", contract_type)
        if contract_type is ContractType.FIXED_TERM:
            if self.term_months is None:
                raise ValueError("A fixed-term contract must declare term_months.")
            _require_positive_int(self.term_months, "contract.term_months")
        elif self.term_months is not None:
            raise ValueError(
                "An open-ended contract must not declare term_months."
            )

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type.value, "term_months": self.term_months}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ContractTerms:
        obj = _object(data, "contract")
        _check_keys(obj, {"type", "term_months"}, "contract")
        return cls(
            type=_enum(ContractType, obj["type"], "contract.type"),
            term_months=_optional_positive_int(
                obj["term_months"], "contract.term_months"
            ),
        )


@dataclass(frozen=True, slots=True)
class EnergySource:
    """The reported production mix of an offer, in percent."""

    wind_share_pct: float
    renewable_share_pct: float

    def __post_init__(self) -> None:
        wind = _require_percentage(self.wind_share_pct, "energy_source.wind_share_pct")
        renewable = _require_percentage(
            self.renewable_share_pct, "energy_source.renewable_share_pct"
        )
        if wind > renewable:
            raise ValueError(
                "energy_source.wind_share_pct cannot exceed "
                "energy_source.renewable_share_pct."
            )
        object.__setattr__(self, "wind_share_pct", wind)
        object.__setattr__(self, "renewable_share_pct", renewable)

    def to_dict(self) -> dict[str, Any]:
        return {
            "wind_share_pct": self.wind_share_pct,
            "renewable_share_pct": self.renewable_share_pct,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EnergySource:
        obj = _object(data, "energy_source")
        _check_keys(
            obj,
            {"wind_share_pct", "renewable_share_pct"},
            "energy_source",
        )
        return cls(
            wind_share_pct=_number(
                obj["wind_share_pct"], "energy_source.wind_share_pct"
            ),
            renewable_share_pct=_number(
                obj["renewable_share_pct"],
                "energy_source.renewable_share_pct",
            ),
        )


@dataclass(frozen=True, slots=True)
class OfferPricing:
    """
    The fixed parts of an offer's price, including VAT.

    ``energy_price_c_kwh`` is the per-kWh price of a fixed or hybrid offer.
    ``spot_margin_c_kwh`` is the retailer margin added to the Nord Pool spot
    price for a spot offer. Exactly one of the two is set, depending on the
    offer's pricing model.
    """

    monthly_fee_eur: float
    energy_price_c_kwh: float | None
    spot_margin_c_kwh: float | None

    def __post_init__(self) -> None:
        monthly_fee = _require_non_negative(
            self.monthly_fee_eur, "prices.monthly_fee_eur"
        )
        energy_price = _optional_non_negative(
            self.energy_price_c_kwh, "prices.energy_price_c_kwh"
        )
        spot_margin = _optional_non_negative(
            self.spot_margin_c_kwh, "prices.spot_margin_c_kwh"
        )
        object.__setattr__(self, "monthly_fee_eur", monthly_fee)
        object.__setattr__(self, "energy_price_c_kwh", energy_price)
        object.__setattr__(self, "spot_margin_c_kwh", spot_margin)

    def to_dict(self) -> dict[str, Any]:
        return {
            "monthly_fee_eur": self.monthly_fee_eur,
            "energy_price_c_kwh": self.energy_price_c_kwh,
            "spot_margin_c_kwh": self.spot_margin_c_kwh,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OfferPricing:
        obj = _object(data, "prices")
        _check_keys(
            obj,
            {"monthly_fee_eur", "energy_price_c_kwh", "spot_margin_c_kwh"},
            "prices",
        )
        return cls(
            monthly_fee_eur=_number(
                obj["monthly_fee_eur"], "prices.monthly_fee_eur"
            ),
            energy_price_c_kwh=_optional_number(
                obj["energy_price_c_kwh"], "prices.energy_price_c_kwh"
            ),
            spot_margin_c_kwh=_optional_number(
                obj["spot_margin_c_kwh"], "prices.spot_margin_c_kwh"
            ),
        )


@dataclass(frozen=True, slots=True)
class Kulutusvaikutus:
    """
    A hybrid offer's monthly consumption-impact adjustment.

    The adjustment is a per-kWh spot-timing deviation: the difference between
    the consumption-weighted spot price and the monthly average spot price,
    multiplied by ``factor`` and clipped to ``cap_c_kwh`` when present.
    ``estimated_c_kwh`` and ``typical_range_c_kwh`` are retailer-provided
    indicative values, not guaranteed bounds.
    """

    definition: ConsumptionImpactDefinition = (
        ConsumptionImpactDefinition.SPOT_TIMING_DEVIATION
    )
    factor: float = 1.0
    estimated_c_kwh: float | None = None
    typical_range_c_kwh: tuple[float, float] | None = None
    cap_c_kwh: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        definition = ConsumptionImpactDefinition(self.definition)
        object.__setattr__(self, "definition", definition)

        if isinstance(self.factor, bool) or not isinstance(
            self.factor, (int, float)
        ):
            raise ValueError("consumption_impact.factor must be a number.")
        factor = float(self.factor)
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError(
                "consumption_impact.factor must be a positive finite number."
            )
        object.__setattr__(self, "factor", factor)
        object.__setattr__(
            self,
            "estimated_c_kwh",
            _optional_number(
                self.estimated_c_kwh, "consumption_impact.estimated_c_kwh"
            ),
        )
        object.__setattr__(
            self,
            "typical_range_c_kwh",
            _optional_range(
                self.typical_range_c_kwh,
                "consumption_impact.typical_range_c_kwh",
            ),
        )
        object.__setattr__(
            self,
            "cap_c_kwh",
            _optional_range(self.cap_c_kwh, "consumption_impact.cap_c_kwh"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "definition": self.definition.value,
            "factor": self.factor,
            "estimated_c_kwh": self.estimated_c_kwh,
            "typical_range_c_kwh": _range_to_list(self.typical_range_c_kwh),
            "cap_c_kwh": _range_to_list(self.cap_c_kwh),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Kulutusvaikutus:
        obj = _object(data, "consumption_impact")
        _check_keys(
            obj,
            {
                "definition",
                "factor",
                "estimated_c_kwh",
                "typical_range_c_kwh",
                "cap_c_kwh",
            },
            "consumption_impact",
        )
        return cls(
            definition=_enum(
                ConsumptionImpactDefinition,
                obj["definition"],
                "consumption_impact.definition",
            ),
            factor=_number(obj["factor"], "consumption_impact.factor"),
            estimated_c_kwh=_optional_number(
                obj["estimated_c_kwh"],
                "consumption_impact.estimated_c_kwh",
            ),
            typical_range_c_kwh=_optional_range(
                obj["typical_range_c_kwh"],
                "consumption_impact.typical_range_c_kwh",
            ),
            cap_c_kwh=_optional_range(
                obj["cap_c_kwh"], "consumption_impact.cap_c_kwh"
            ),
        )


@dataclass(frozen=True, slots=True)
class OfferSource:
    """Lineage for one offer in the comparison-site payload."""

    product_id: str
    product_url: str | None
    api_pricing_model: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "product_id",
            _require_non_empty_string(self.product_id, "source.product_id"),
        )
        product_url = self.product_url
        if product_url is not None:
            product_url = _require_non_empty_string(
                product_url, "source.product_url"
            )
        object.__setattr__(self, "product_url", product_url)
        object.__setattr__(
            self,
            "api_pricing_model",
            _require_non_empty_string(
                self.api_pricing_model, "source.api_pricing_model"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "product_id": self.product_id,
            "product_url": self.product_url,
            "api_pricing_model": self.api_pricing_model,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OfferSource:
        obj = _object(data, "source")
        _check_keys(
            obj,
            {"product_id", "product_url", "api_pricing_model"},
            "source",
        )
        return cls(
            product_id=_string(obj["product_id"], "source.product_id"),
            product_url=_optional_string(
                obj["product_url"], "source.product_url"
            ),
            api_pricing_model=_string(
                obj["api_pricing_model"], "source.api_pricing_model"
            ),
        )


@dataclass(frozen=True, slots=True)
class RetailOffer:
    """One digitized retail electricity offer."""

    id: str
    provider: str
    product_name: str
    category: OfferCategory
    pricing_model: PricingModel
    contract: ContractTerms
    energy_source: EnergySource
    prices: OfferPricing
    consumption_impact: Kulutusvaikutus | None
    max_annual_consumption_kwh: float | None
    vat_included: bool
    source: OfferSource

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "id", _require_non_empty_string(self.id, "offer.id")
        )
        object.__setattr__(
            self,
            "provider",
            _require_non_empty_string(self.provider, "offer.provider"),
        )
        object.__setattr__(
            self,
            "product_name",
            _require_non_empty_string(self.product_name, "offer.product_name"),
        )

        category = OfferCategory(self.category)
        pricing_model = PricingModel(self.pricing_model)
        object.__setattr__(self, "category", category)
        object.__setattr__(self, "pricing_model", pricing_model)
        _validate_pricing_model(self)

        max_consumption = _optional_non_negative(
            self.max_annual_consumption_kwh,
            "offer.max_annual_consumption_kwh",
        )
        object.__setattr__(
            self, "max_annual_consumption_kwh", max_consumption
        )
        if not isinstance(self.vat_included, bool):
            raise ValueError("offer.vat_included must be a boolean.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "provider": self.provider,
            "product_name": self.product_name,
            "category": self.category.value,
            "pricing_model": self.pricing_model.value,
            "contract": self.contract.to_dict(),
            "energy_source": self.energy_source.to_dict(),
            "prices": self.prices.to_dict(),
            "consumption_impact": (
                None
                if self.consumption_impact is None
                else self.consumption_impact.to_dict()
            ),
            "max_annual_consumption_kwh": self.max_annual_consumption_kwh,
            "vat_included": self.vat_included,
            "source": self.source.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RetailOffer:
        obj = _object(data, "offer")
        _check_keys(
            obj,
            {
                "id",
                "provider",
                "product_name",
                "category",
                "pricing_model",
                "contract",
                "energy_source",
                "prices",
                "consumption_impact",
                "max_annual_consumption_kwh",
                "vat_included",
                "source",
            },
            "offer",
        )
        impact_data = obj["consumption_impact"]
        impact = (
            None
            if impact_data is None
            else Kulutusvaikutus.from_dict(
                _object(impact_data, "offer.consumption_impact")
            )
        )
        return cls(
            id=_string(obj["id"], "offer.id"),
            provider=_string(obj["provider"], "offer.provider"),
            product_name=_string(obj["product_name"], "offer.product_name"),
            category=_enum(OfferCategory, obj["category"], "offer.category"),
            pricing_model=_enum(
                PricingModel, obj["pricing_model"], "offer.pricing_model"
            ),
            contract=ContractTerms.from_dict(
                _object(obj["contract"], "offer.contract")
            ),
            energy_source=EnergySource.from_dict(
                _object(obj["energy_source"], "offer.energy_source")
            ),
            prices=OfferPricing.from_dict(
                _object(obj["prices"], "offer.prices")
            ),
            consumption_impact=impact,
            max_annual_consumption_kwh=_optional_number(
                obj["max_annual_consumption_kwh"],
                "offer.max_annual_consumption_kwh",
            ),
            vat_included=obj["vat_included"],
            source=OfferSource.from_dict(
                _object(obj["source"], "offer.source")
            ),
        )


@dataclass(frozen=True, slots=True)
class CatalogSource:
    """Provenance for a whole digitized offer catalog."""

    name: str
    operator: str
    api_url: str
    postal_code: str
    retrieved_at: datetime

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "name", _require_non_empty_string(self.name, "source.name")
        )
        object.__setattr__(
            self,
            "operator",
            _require_non_empty_string(self.operator, "source.operator"),
        )
        object.__setattr__(
            self,
            "api_url",
            _require_non_empty_string(self.api_url, "source.api_url"),
        )
        postal_code = _require_non_empty_string(
            self.postal_code, "source.postal_code"
        )
        if len(postal_code) != 5 or not postal_code.isdigit():
            raise ValueError(
                "source.postal_code must be a five-digit Finnish postal code."
            )
        object.__setattr__(self, "postal_code", postal_code)

        retrieved_at = _require_aware_datetime(
            self.retrieved_at, "source.retrieved_at"
        )
        object.__setattr__(self, "retrieved_at", retrieved_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "operator": self.operator,
            "api_url": self.api_url,
            "postal_code": self.postal_code,
            "retrieved_at": _iso(self.retrieved_at),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> CatalogSource:
        obj = _object(data, "source")
        _check_keys(
            obj,
            {"name", "operator", "api_url", "postal_code", "retrieved_at"},
            "source",
        )
        return cls(
            name=_string(obj["name"], "source.name"),
            operator=_string(obj["operator"], "source.operator"),
            api_url=_string(obj["api_url"], "source.api_url"),
            postal_code=_string(obj["postal_code"], "source.postal_code"),
            retrieved_at=_parse_datetime(
                obj["retrieved_at"], "source.retrieved_at"
            ),
        )


@dataclass(frozen=True, slots=True)
class OfferCatalog:
    """A versioned, validated set of digitized retail offers."""

    schema_version: int
    source: CatalogSource
    offers: tuple[RetailOffer, ...]

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or not isinstance(
            self.schema_version, int
        ):
            raise ValueError("schema_version must be an integer.")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema_version {self.schema_version}; "
                f"expected {SCHEMA_VERSION}."
            )
        if not isinstance(self.source, CatalogSource):
            raise TypeError("source must be a CatalogSource.")
        if isinstance(self.offers, (str, bytes)):
            raise ValueError("offers must be a sequence of RetailOffer values.")
        offers = tuple(self.offers)
        if not offers:
            raise ValueError("An offer catalog must contain at least one offer.")
        if not all(isinstance(offer, RetailOffer) for offer in offers):
            raise TypeError("offers must contain only RetailOffer values.")

        seen: set[str] = set()
        for offer in offers:
            if offer.id in seen:
                raise ValueError(f"Duplicate offer id {offer.id!r}.")
            seen.add(offer.id)
        object.__setattr__(self, "offers", offers)

    @property
    def categories(self) -> tuple[OfferCategory, ...]:
        """Returns the categories present, in first-appearance order."""
        return tuple(dict.fromkeys(offer.category for offer in self.offers))

    def by_category(self, category: OfferCategory) -> tuple[RetailOffer, ...]:
        """Returns the offers in ``category``, in file order."""
        category = OfferCategory(category)
        return tuple(
            offer for offer in self.offers if offer.category is category
        )

    def offer(self, offer_id: str) -> RetailOffer:
        """Returns the offer with ``offer_id``, or raises ``KeyError``."""
        for offer in self.offers:
            if offer.id == offer_id:
                return offer
        raise KeyError(f"No offer with id {offer_id!r}.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": self.source.to_dict(),
            "offers": [offer.to_dict() for offer in self.offers],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OfferCatalog:
        obj = _object(data, "catalog")
        _check_keys(obj, {"schema_version", "source", "offers"}, "catalog")
        offers_data = obj["offers"]
        if not isinstance(offers_data, Sequence) or isinstance(
            offers_data, (str, bytes)
        ):
            raise ValueError("catalog.offers must be a JSON array.")
        return cls(
            schema_version=obj["schema_version"],
            source=CatalogSource.from_dict(
                _object(obj["source"], "catalog.source")
            ),
            offers=tuple(
                RetailOffer.from_dict(_object(item, f"offers[{index}]"))
                for index, item in enumerate(offers_data)
            ),
        )


def load_offer_catalog(path: str | Path) -> OfferCatalog:
    """Loads and validates an offer catalog from a JSON file."""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    return OfferCatalog.from_dict(data)


def save_offer_catalog(catalog: OfferCatalog, path: str | Path) -> None:
    """Writes ``catalog`` as UTF-8 JSON, replacing ``path`` atomically."""
    path = Path(path)
    payload = json.dumps(
        catalog.to_dict(), indent=2, ensure_ascii=False
    ) + "\n"
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _validate_pricing_model(offer: RetailOffer) -> None:
    expected = {
        OfferCategory.FIXED: PricingModel.FIXED,
        OfferCategory.WIND_FIXED: PricingModel.FIXED,
        OfferCategory.SPOT: PricingModel.SPOT,
        OfferCategory.WIND_SPOT: PricingModel.SPOT,
        OfferCategory.HYBRID: PricingModel.HYBRID,
    }[offer.category]
    if offer.pricing_model is not expected:
        raise ValueError(
            f"Offer {offer.id!r} category {offer.category.value!r} requires "
            f"pricing_model {expected.value!r}, got "
            f"{offer.pricing_model.value!r}."
        )

    is_wind_category = offer.category in {
        OfferCategory.WIND_FIXED,
        OfferCategory.WIND_SPOT,
    }
    if is_wind_category and offer.energy_source.wind_share_pct != 100.0:
        raise ValueError(
            f"Offer {offer.id!r} is in a 100% wind category but reports "
            f"{offer.energy_source.wind_share_pct:g}% wind."
        )
    if (
        offer.category is OfferCategory.FIXED
        or offer.category is OfferCategory.SPOT
    ) and offer.energy_source.wind_share_pct == 100.0:
        raise ValueError(
            f"Offer {offer.id!r} reports 100% wind but is not in a wind "
            "category."
        )

    prices = offer.prices
    impact = offer.consumption_impact
    if offer.pricing_model is PricingModel.FIXED:
        if prices.energy_price_c_kwh is None:
            raise ValueError(
                f"Fixed offer {offer.id!r} must declare energy_price_c_kwh."
            )
        if prices.spot_margin_c_kwh is not None:
            raise ValueError(
                f"Fixed offer {offer.id!r} must not declare spot_margin_c_kwh."
            )
        if impact is not None:
            raise ValueError(
                f"Fixed offer {offer.id!r} must not declare "
                "consumption_impact."
            )
    elif offer.pricing_model is PricingModel.SPOT:
        if prices.spot_margin_c_kwh is None:
            raise ValueError(
                f"Spot offer {offer.id!r} must declare spot_margin_c_kwh."
            )
        if prices.energy_price_c_kwh is not None:
            raise ValueError(
                f"Spot offer {offer.id!r} must not declare "
                "energy_price_c_kwh."
            )
        if impact is not None:
            raise ValueError(
                f"Spot offer {offer.id!r} must not declare "
                "consumption_impact."
            )
    else:
        if prices.energy_price_c_kwh is None:
            raise ValueError(
                f"Hybrid offer {offer.id!r} must declare energy_price_c_kwh."
            )
        if prices.spot_margin_c_kwh is not None:
            raise ValueError(
                f"Hybrid offer {offer.id!r} must not declare "
                "spot_margin_c_kwh."
            )
        if impact is None:
            raise ValueError(
                f"Hybrid offer {offer.id!r} must declare consumption_impact."
            )


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a JSON object.")
    return value


def _check_keys(
    obj: Mapping[str, Any],
    required: set[str],
    path: str,
) -> None:
    missing = sorted(required - set(obj))
    unknown = sorted(set(obj) - required)
    if missing:
        raise ValueError(f"{path} is missing key(s): {', '.join(missing)}.")
    if unknown:
        raise ValueError(f"{path} has unknown key(s): {', '.join(unknown)}.")


def _string(value: Any, path: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a string.")
    return value


def _optional_string(value: Any, path: str) -> str | None:
    if value is None:
        return None
    return _string(value, path)


def _require_non_empty_string(value: Any, path: str) -> str:
    text = _string(value, path)
    if not text.strip():
        raise ValueError(f"{path} must not be empty.")
    return text


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite.")
    return result


def _optional_number(value: Any, path: str) -> float | None:
    if value is None:
        return None
    return _number(value, path)


def _require_non_negative(value: Any, path: str) -> float:
    number = _number(value, path)
    if number < 0:
        raise ValueError(f"{path} must be non-negative.")
    return number


def _optional_non_negative(value: Any, path: str) -> float | None:
    if value is None:
        return None
    return _require_non_negative(value, path)


def _require_percentage(value: Any, path: str) -> float:
    number = _number(value, path)
    if not 0 <= number <= 100:
        raise ValueError(f"{path} must be between 0 and 100.")
    return number


def _require_positive_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer.")
    if value <= 0:
        raise ValueError(f"{path} must be positive.")
    return value


def _optional_positive_int(value: Any, path: str) -> int | None:
    if value is None:
        return None
    return _require_positive_int(value, path)


def _optional_range(
    value: Any, path: str
) -> tuple[float, float] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)) or not isinstance(
        value, Sequence
    ) or len(value) != 2:
        raise ValueError(f"{path} must be a two-number array.")
    low = _number(value[0], f"{path}[0]")
    high = _number(value[1], f"{path}[1]")
    if low > high:
        raise ValueError(f"{path} lower bound must not exceed its upper bound.")
    return (low, high)


def _range_to_list(value: tuple[float, float] | None) -> list[float] | None:
    return None if value is None else [value[0], value[1]]


def _enum(enum_type: type[StrEnum], value: Any, path: str) -> Any:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a string.")
    try:
        return enum_type(value)
    except ValueError:
        supported = ", ".join(member.value for member in enum_type)
        raise ValueError(
            f"{path} must be one of: {supported}."
        ) from None


def _parse_datetime(value: Any, path: str) -> datetime:
    text = _string(value, path)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(
            f"{path} must be an ISO 8601 timestamp, got {text!r}."
        ) from exc
    return _require_aware_datetime(parsed, path)


def _require_aware_datetime(value: Any, path: str) -> datetime:
    if not isinstance(value, datetime):
        raise ValueError(f"{path} must be a datetime.")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{path} must be timezone-aware.")
    return value.astimezone(UTC)


def _iso(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")