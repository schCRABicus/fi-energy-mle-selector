"""Retail electricity contract offers and their digitization adapters."""

from fi_energy.contracts.offers import (
    CatalogSource,
    ConsumptionImpactDefinition,
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
    load_offer_catalog,
    save_offer_catalog,
)
from fi_energy.contracts.sahkonhinta import (
    SAHKONHINTA_API_URL,
    SAHKONHINTA_NAME,
    SAHKONHINTA_OPERATOR,
    SelectedProduct,
    build_offer_catalog,
    offer_from_product,
)

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
    "SAHKONHINTA_API_URL",
    "SAHKONHINTA_NAME",
    "SAHKONHINTA_OPERATOR",
    "SelectedProduct",
    "build_offer_catalog",
    "load_offer_catalog",
    "offer_from_product",
    "save_offer_catalog",
]