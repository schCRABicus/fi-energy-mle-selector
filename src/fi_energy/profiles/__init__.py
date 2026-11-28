"""Standard Finnish household hourly load profiles."""

from fi_energy.profiles.load_profile import (
    APARTMENT_2500,
    BASE_TEMPERATURE_C,
    DEFAULT_PROFILES,
    DETACHED_HOUSE_18000,
    HOURLY_KWH,
    LOCAL_TIMEZONE,
    TEMPERATURE_SMOOTHING_HOURS,
    LoadProfileBuild,
    LoadProfileSpec,
    build_load_profile_table,
    profile_annual_totals,
    profile_table,
)

__all__ = [
    "APARTMENT_2500",
    "BASE_TEMPERATURE_C",
    "DEFAULT_PROFILES",
    "DETACHED_HOUSE_18000",
    "HOURLY_KWH",
    "LOCAL_TIMEZONE",
    "TEMPERATURE_SMOOTHING_HOURS",
    "LoadProfileBuild",
    "LoadProfileSpec",
    "build_load_profile_table",
    "profile_annual_totals",
    "profile_table",
]