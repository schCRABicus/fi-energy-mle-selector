# Finnish Retail Electricity Contract Selector via Probabilistic Deep Learning (MLE)

A probabilistic deep learning framework that forecasts hourly Finnish electricity spot price distributions using 
**Maximum Likelihood Estimation (MLE)** and recommends optimal retail electricity contracts (Fixed, 100% Wind Fixed, 
Spot, Wind Spot, and Hybrid/*Kulutusvaikutus*) under user-defined risk preferences.

## Mathematical Formulation

Instead of deterministic point predictions, a Heteroscedastic Multi-Layer Perceptron (MLP) predicts both the 
conditional mean $\mu_\theta(x_t)$ and variance $\sigma_\theta^2(x_t)$ of hourly Nord Pool FI spot prices:

$$y_t \mid x_t \sim \mathcal{N}\big(\mu_\theta(x_t),\, \sigma_\theta^2(x_t)\big)$$

The network is trained by minimizing the exact **Gaussian Negative Log-Likelihood (NLL)**:

$$\mathcal{L}_{\text{NLL}}(\theta) = \frac{1}{N} \sum_{t=1}^{N} \left( \frac{\big(y_t - \mu_\theta(x_t)\big)^2}{2\sigma_\theta^2(x_t)} + \frac{1}{2}\ln \sigma_\theta^2(x_t) \right)$$

## Data Sources (Finland)
- **Grid Supply, Wind, Nuclear, Hydro & Cross-Border Flows:** Fingrid Open Data (`data.fingrid.fi`)
- **Regional Weather (Temperature, Wind Speed, Radiation):** FMI Open Data (`en.ilmatieteenlaitos.fi/open-data`)
- **Day-Ahead Spot Prices (FI Area):** Nord Pool / ENTSO-E data via the Energy-Charts API (`api.energy-charts.info`); `sahkonhintatanaan.fi` is no longer online
- **Retail Contract Offers:** Finnish Energy Authority (`sahkonhinta.fi`)

## Repository Structure
```text
├── src/
│   └── fi_energy/    # Installable package, imported as `fi_energy` (editable via `uv sync`)
│       ├── datasets/ # Data-source adapters: shared contract + per-source fetchers
│       │   ├── api.py        # FetchRequest and DataSourceFetcher protocol
│       │   ├── http_client.py# HTTP client with retries (httpx + tenacity)
│       │   ├── utils.py      # Window validation, resampling, local Parquet loading
│       │   ├── alignment.py  # UTC hourly validation, outer-join, gap filling
│       │   ├── fingrid/      # Fingrid client, shared series fetcher, per-dataset fetchers
│       │   │   └── data/     # Committed 1-day Parquet snapshots for offline runs
│       │   ├── fmi/          # FMI WFS client and city weather fetcher
│       │   │   └── data/     # Committed 1-day Parquet snapshots for offline runs
│       │   └── energy_charts/# Energy-Charts client and Nord Pool spot-price fetcher
│       │       └── data/     # Committed 1-day Parquet snapshots for offline runs
│       └── features/         # Phase 2 feature engineering (models/simulator planned)
├── tests/            # Offline unit tests and live API integration tests
├── scripts/          # Reproducible pipelines (Phase 1 pull, Phase 2 feature build)
├── configs/          # (planned) retail contract definitions & household load profiles
├── data/             # Hourly datasets, feature tables, and manifests (git-ignored)
├── pyproject.toml    # Project metadata, dependencies, packaging, pytest configuration
├── uv.lock           # Locked dependency versions
├── README.md
└── TODO.md           # Project roadmap and implementation checklist
```

## Local Development

### Prerequisites
- [`uv`](https://docs.astral.sh/uv/) — manages the Python 3.14 interpreter and the virtualenv.
- A free Fingrid Open Data API key from [`data.fingrid.fi`](https://data.fingrid.fi) — only needed for live Fingrid data and integration tests.
- FMI Open Data and the Energy-Charts spot-price API need no credentials.

### 1. Install dependencies

```bash
git clone <repo-url>
cd fi-energy-mle-selector
uv sync
```

`uv sync` creates `.venv/`, installs the locked runtime and `dev` dependencies (including pytest), and installs this project itself in editable mode, so `import fi_energy` works from anywhere inside `uv run`. Add or change dependencies with `uv add <package>` / `uv add --dev <package>` — never edit `uv.lock` by hand. Prefix every command with `uv run` so it executes inside `.venv/`.

### 2. Configure credentials

Create a `.env` file in the project root (it is git-ignored):

```bash
printf 'FINGRID_API_KEY=your-api-key\n' > .env
```

Fetchers receive the key as a constructor argument; the live integration tests load `.env` through `python-dotenv`.

### 3. Run the tests

```bash
uv run pytest                       # full suite; live tests skip without FINGRID_API_KEY
uv run pytest -m "not integration"  # offline unit tests only
uv run pytest -m integration -rs    # all live tests (network; Fingrid also needs the key)
```

Run a single file or test:

```bash
uv run pytest tests/test_fingrid_wind_fetcher.py -q
uv run pytest tests/test_fingrid_wind_fetcher.py::test_fetch_from_local_reads_bundled_sample
```

Unit tests never touch the network: fetcher tests read the committed snapshots under `src/fi_energy/datasets/fingrid/data/`, `src/fi_energy/datasets/fmi/data/`, and `src/fi_energy/datasets/energy_charts/data/` (all covering 2024-03-01), and client tests serve scripted responses through `httpx.MockTransport`. The Fingrid snapshots hold the raw cadence: 96 quarter-hour samples for wind generation, consumption, and each cross-border flow, 24 hourly samples for wind capacity, and 958 roughly-90-second samples for nuclear and hydro. Each FMI snapshot holds 24 hourly rows for one city, and the spot-price snapshot holds 24 hourly FI prices. Only the integration tests hit the real APIs (a handful of requests, with automatic retry/backoff), so avoid running them in tight loops against the free-tier quota.

### 4. Fetch data from a specific fetcher

Each dataset has a fetcher under `src/fi_energy/datasets/<source>/`. The Fingrid fetchers all share the same constructor and request API; swap the class name in the snippet below.

| Fetcher | Dataset | Series | Output column | Saved name |
| --- | --- | --- | --- | --- |
| `FingridWindFetcher` | 75 | Wind power generation | `wind_mw` | `wind_production` |
| `FingridWindCapacityFetcher` | 268 | Installed wind capacity | `wind_capacity_mw` | `wind_capacity` |
| `FingridConsumptionFetcher` | 124 | National electricity consumption (MWh/h ≈ MW) | `consumption_mw` | `electricity_consumption` |
| `FingridNuclearFetcher` | 188 | Nuclear power production | `nuclear_mw` | `nuclear_production` |
| `FingridHydroFetcher` | 191 | Hydro power production | `hydro_mw` | `hydro_production` |
| `FingridSE1FlowFetcher` | 60 | Finland ↔ Northern Sweden flow (positive = export) | `se1_flow_mw` | `se1_flow` |
| `FingridSE3FlowFetcher` | 61 | Finland ↔ Central Sweden flow (positive = export) | `se3_flow_mw` | `se3_flow` |
| `FingridEEFlowFetcher` | 55 | Finland ↔ Estonia flow (positive = export) | `ee_flow_mw` | `ee_flow` |

```bash
uv run python - <<'PY'
import os
from datetime import UTC, datetime

from dotenv import load_dotenv

from fi_energy.datasets import FetchRequest, FingridWindFetcher

# Explicit path: load_dotenv() cannot introspect a file path when run via stdin.
load_dotenv(".env")

fetcher = FingridWindFetcher(os.environ["FINGRID_API_KEY"])
request = FetchRequest(
    start_utc=datetime(2024, 3, 1, tzinfo=UTC),
    end_utc=datetime(2024, 3, 2, tzinfo=UTC),
    resolution="1h",  # "15min", "30min", or "1h"
)
table = fetcher.fetch(request)

print(table)
print(table.slice(0, 3).to_pylist())
PY
```

Notes:
- Windows are half-open `[start_utc, end_utc)` and must be timezone-aware.
- Fingrid serves 15-minute consumption, wind generation, and cross-border flow samples after 13 June 2023 (hourly before that), and roughly 90-second (formerly 3-minute) real-time nuclear and hydro samples. `resolution="1h"` returns hourly means, `resolution="15min"` aggregates everything into quarter-hours, and the raw cadence is never exposed below `15min`.
- Flow values follow Fingrid's sign convention: positive is transmission from Finland to the neighbouring zone, negative is import into Finland.
- Multi-year ranges are paginated automatically and retried on Fingrid rate limits, so a 3-year window only needs one request object.
- `fetch` always calls the remote API. `fetch_from_local` returns data from the bundled snapshot when it exists and falls back to `fetch` otherwise, so the same request works offline.
- The bundled snapshots live at `src/fi_energy/datasets/fingrid/data/` (`wind_sample.parquet`, `wind_capacity_sample.parquet`, `consumption_sample.parquet`, `nuclear_sample.parquet`, `hydro_sample.parquet`, `se1_flow_sample.parquet`, `se3_flow_sample.parquet`, `ee_flow_sample.parquet`) and cover 2024-03-01 only. Override any of them with `local_path="path/to/snapshot.parquet"` or disable it with `local_path=None`.
- Output tables are built against `fetcher.expected_schema`, so `table.schema == fetcher.expected_schema` holds for every fetch.
- To persist a table, write it to Parquet with pyarrow: `import pyarrow.parquet as pq; pq.write_table(table, "data/wind.parquet")` (create `data/` first).

#### FMI weather (Helsinki, Vaasa, Oulu)

`FmiWeatherFetcher` takes a city name instead of an API key and returns `timestamp_utc`, `temperature_c`, `wind_speed_ms`, and `radiation_wm2`.

| City | Weather station | Radiation station |
| --- | --- | --- |
| `helsinki` (alias `kaisaniemi`) | Helsinki Kaisaniemi (100971) | Helsinki Kumpula (101004) |
| `vaasa` | Vaasa Klemettilä (101485) | Seinäjoki Pelmaa (101486) |
| `oulu` | Oulu Vihreäsaari (101794) | Oulu Vihreäsaari (101794) |

```bash
uv run python - <<'PY'
from datetime import UTC, datetime

from fi_energy.datasets import FetchRequest, FmiWeatherFetcher

request = FetchRequest(
    start_utc=datetime(2024, 3, 1, tzinfo=UTC),
    end_utc=datetime(2024, 3, 2, tzinfo=UTC),
)
for city in ("helsinki", "vaasa", "oulu"):
    with FmiWeatherFetcher(city) as fetcher:
        table = fetcher.fetch(request)
    print(fetcher.source_name, table.num_rows)
    print(table.slice(0, 3).to_pylist())
PY
```

Notes:
- FMI Open Data requires no API key. Only `resolution="1h"` is supported.
- Radiation comes from the nearest station with a multi-year record, which is why Vaasa uses Seinäjoki Pelmaa rather than a station inside the city. Values are FMI's one-minute global-radiation sample at the top of each hour (`GLOB_1MIN`, `timestep=60`), not an hourly mean. Small negative values at night are sensor offset and are preserved.
- FMI caps one WFS request at 31 days for hourly weather and 7 days for radiation, so `FmiClient` chunks long windows automatically. A 3-year radiation pull issues about 157 requests per city and can take minutes.
- Long downloads show a status tracker with one progress bar per stage (`weather`, `radiation`) on stderr: chunk count, percent, elapsed time, ETA, and observations so far. It is enabled automatically on an interactive terminal; pass `show_progress=True` to force it in logs or `show_progress=False` to silence it. `scripts/fetch_all.py` enables progress bars only in sequential mode (`--jobs 1`), because parallel bars would interleave.
- Missing observations become null values in the output table.
- `fetch_from_local` reads `src/fi_energy/datasets/fmi/data/<city>_sample.parquet` and falls back to the remote API when that snapshot is missing. Override it with `local_path="path/to/snapshot.parquet"` or disable it with `local_path=None`.

#### Nord Pool FI spot prices

`NordPoolSpotFetcher` returns hourly day-ahead spot prices as `timestamp_utc` and `spot_price_c_kwh`. It reads the FI bidding zone by default (`bzn="FI"`).

```bash
uv run python - <<'PY'
from datetime import UTC, datetime

from fi_energy.datasets import FetchRequest, NordPoolSpotFetcher

request = FetchRequest(
    start_utc=datetime(2024, 3, 1, tzinfo=UTC),
    end_utc=datetime(2024, 3, 2, tzinfo=UTC),
)
with NordPoolSpotFetcher() as fetcher:
    table = fetcher.fetch(request)
print(fetcher.source_name, table.num_rows)
print(table.slice(0, 3).to_pylist())
PY
```

Notes:
- `sahkonhintatanaan.fi` is offline, so prices come from the Energy-Charts API (`api.energy-charts.info`), which mirrors the ENTSO-E / Nord Pool day-ahead series and needs no API key. Prices are converted from EUR/MWh to `c/kWh` (`EUR/MWh ÷ 10`) and exclude VAT; a consumer spot contract adds VAT and the retailer margin.
- Any window is a single Energy-Charts request: a 3-year pull returns 26,304 hourly values in roughly two seconds.
- Finland switched its day-ahead market to 15-minute time units on 2025-10-01, and Energy-Charts returns quarter-hour prices from then on. The fetcher averages each hour's quarters into one `c/kWh` value (a missing quarter is ignored; an hour with no prices stays null), so the output remains hourly across the whole window.
- Energy-Charts data is licensed for private and internal use; see the `license_info` field in API responses.
- Missing prices become null values in the output table.
- `fetch_from_local` reads `src/fi_energy/datasets/energy_charts/data/fi_spot_sample.parquet` and falls back to the remote API when that snapshot is missing. Override it with `local_path="path/to/snapshot.parquet"` or disable it with `local_path=None`.

#### Full Phase 1 pull

[`scripts/fetch_all.py`](scripts/fetch_all.py) fetches every series over one shared UTC hourly window (three years ending at the current complete hour by default), verifies that each table sits on the same hourly grid, and saves each run under `data/<window>/`:

```text
data/
└── 3y/
    ├── manifest.json
    ├── fingrid/
    │   ├── wind_production.parquet
    │   ├── wind_capacity.parquet
    │   ├── electricity_consumption.parquet
    │   ├── nuclear_production.parquet
    │   ├── hydro_production.parquet
    │   ├── se1_flow.parquet
    │   ├── se3_flow.parquet
    │   └── ee_flow.parquet
    ├── fmi/
    │   ├── helsinki_weather.parquet
    │   ├── vaasa_weather.parquet
    │   └── oulu_weather.parquet
    ├── energy_charts/spot_price_fi.parquet
    └── combined/
        ├── hourly.parquet        # raw outer join: every window hour
        └── hourly_clean.parquet  # short gaps filled, boundary rows trimmed
```

The window directory is `3y` for a three-calendar-year window, or the exact hour count such as `24h` for other ranges. Dataset filenames are purpose-based; the manifest keeps the fetcher lineage (e.g. `fingrid_75`) alongside the clear name. `combined/hourly_clean.parquet` is the finalized Phase 1 table: leading and trailing rows where any feature is null are dropped, then interior runs of one or two missing hours (a gap shorter than three hours) are filled by linear interpolation. Longer interior gaps stay null; `manifest.json` records the filled counts, boundary rows removed, and remaining nulls per column.

```bash
uv run python scripts/fetch_all.py
uv run python scripts/fetch_all.py --skip-existing            # resume
uv run python scripts/fetch_all.py --jobs 1                   # sequential
uv run python scripts/fetch_all.py --start 2023-01-01 --end 2026-01-01
uv run python scripts/fetch_all.py --years 2 --output-dir /tmp/fi-energy
```

Bounds are ISO timestamps, interpreted as UTC when no offset is given, and rounded down to the hour. `--end` defaults to the current UTC hour, `--start` defaults to `--years` (3) before the end, and the window is half-open `[start_utc, end_utc)`. `--output-dir` is the base directory; each run writes to `<output-dir>/<window>/`. Sources are fetched concurrently with up to `--jobs` workers (default 4), with per-group caps: Fingrid series stay serial to respect the free-tier rate limit, up to three FMI cities run at once, and Energy-Charts is a single request. The FMI cities are scheduled first, so their multi-minute pulls overlap the Fingrid leg. `--jobs 1` restores a fully sequential pull and re-enables the FMI progress bars, which are suppressed when several sources run at once. `--skip-existing` reuses a source Parquet file that was written for the exact same window bounds, which resumes an interrupted pull without mixing in stale hours from a different trailing window; `--no-combined` skips the combined and cleaned outputs.

The script exits immediately if `FINGRID_API_KEY` is missing. Individual source failures are recorded and the run continues; a partial run exits with status 1, writes `manifest.json` with the `failures` list, and skips the combined/cleaned tables. Every source table is checked for exact UTC hourly timestamps, duplicate timestamps, and rows outside the window. `combined/hourly.parquet` contains every hour in the window; missing observations become nulls for that source's columns, and FMI columns are prefixed by city (`helsinki_temperature_c`, etc.). `src/fi_energy/datasets/alignment.py` provides the reusable `trailing_window`, `hourly_index`, `validate_hourly_table`, `align_hourly`, `interpolate_short_gaps`, and `drop_incomplete_boundary_rows` helpers.

The FMI leg is slower because FMI caps radiation requests at seven days; a 3-year city takes about five minutes (measured 5.2 minutes for Helsinki over 2022–2024), so with several workers the total run is roughly bounded by the slowest FMI city rather than the sum of all series. Fingrid series are fetched one at a time so the free-tier rate limit is not exceeded; a higher-tier key can raise `GROUP_LIMITS["fingrid"]` in the script. The spot leg is a single request and finishes in seconds.

#### Phase 2 feature table

[`scripts/build_features.py`](scripts/build_features.py) derives model features from the cleaned Phase 1 table and writes a timestamped feature table under `data/<window>/features/`:

```text
data/3y/features/
├── hourly_features.parquet  # timestamp_utc + registered derived features
└── manifest.json            # lineage, policy, null counts, and statistics
```

```bash
uv run python scripts/build_features.py
uv run python scripts/build_features.py --dry-run
uv run python scripts/build_features.py --input data/3y/combined/hourly_clean.parquet \
    --output data/3y/features/hourly_features.parquet
```

The wind group contains `wind_penetration_ratio` (`wind_mw / consumption_mw`, the hourly share of consumption met by wind) and `wind_capacity_factor` (`wind_mw / wind_capacity_mw`). Negative wind readings are clipped to zero for the ratios because generation cannot be negative; the Phase 1 `wind_mw` column is left untouched and the correction count is recorded in the manifest. Null inputs propagate, and a zero denominator becomes null instead of infinity. Source columns are not duplicated — only `timestamp_utc` plus derived features are stored, so later feature groups extend the same table.

The thermal group derives demand features from `helsinki_temperature_c`. `heating_degree_hours` is `max(0, 17 - T)`: each row covers one hour, so the value is also the degree-hours accumulated in that hour, zero on mild hours and null when the temperature is missing. `temp_rolling_24h` and `temp_rolling_72h` are trailing means of the Helsinki temperature (the current hour plus the preceding 23 or 71), so they capture thermal inertia without looking into the future. Null readings inside a window are skipped, and the rolling features stay null until a full window of history exists. The 17 °C base temperature and the rolling policy are recorded in both `manifest.json` and the Parquet schema metadata.

The interaction group multiplies `wind_penetration_ratio` by `heating_degree_hours` to produce `wind_temp_interaction`, which is high only on cold hours with little wind and therefore isolates the scarcity conditions that drive price spikes. It is declared after both producer groups; the catalog rejects any ordering or `subset(...)` that drops one of its inputs. A null in either input yields a null interaction.

The calendar group converts each UTC timestamp to `Europe/Helsinki` local time before extracting calendar parts, because Finnish demand and day-ahead prices follow local wall-clock time and public holidays are local calendar days. `sin_hour`/`cos_hour` encode hour of day on a 24-hour circle (midnight at angle zero), `sin_month`/`cos_month` encode month on a 12-month circle (January at angle zero), `is_weekend` flags Saturday and Sunday, and `is_fi_public_holiday` flags Finnish public holidays from the `holidays` package (`country="FI"`). All six are stored as `float32` 0/1 flags or sine/cosine values; the timezone and holiday country are recorded in the manifest and Parquet metadata.

Feature groups live in `src/fi_energy/features/groups/`; the spec/group/catalog DSL and build helpers stay at the package root. A `FeatureSpec` carries per-feature metadata (name, inputs, description) and a concrete `FeatureGroup` declares its `name` plus a tuple of specs, implements `_compute(table)`, and inherits `compute()`, which validates the inputs, enforces an append-only nullable-`float32` output contract, and owns the group's diagnostics. Group membership is not duplicated on the spec; `FeatureCatalog` in `src/fi_energy/features/catalog.py` owns it. The catalog is an ordered, validated set of groups: it rejects duplicate group or feature names and rejects a group that consumes a feature produced later, so a group may consume features from any earlier group. `DEFAULT_CATALOG` lists the groups used by the default build, `catalog.compute(table)` applies them in order, and `catalog.subset(...)` selects groups for Phase 3's ablation tiers while re-checking dependencies. Adding a group means adding one module and one entry to `DEFAULT_CATALOG`.

### 5. Add a new fetcher

1. For another Fingrid numeric series, subclass `FingridSeriesFetcher` and supply the dataset id, value column, schema, and local snapshot path (see `FingridNuclearFetcher`); it already implements `source_name`, `expected_schema`, `fetch`, and `fetch_from_local`. For other sources, create `src/fi_energy/datasets/<source>/<source>_fetcher.py` implementing the `DataSourceFetcher` protocol from `src/fi_energy/datasets/api.py`; reuse `src/fi_energy/datasets/utils.py` for window validation, resolution lookup, bucket averaging, and local Parquet loading.
2. Export it from the subpackage's `__init__.py` and from `src/fi_energy/datasets/__init__.py`.
3. Add an offline Parquet fixture for fetcher tests, serve scripted HTTP responses through `httpx.MockTransport` for client tests, and add a live test marked `@pytest.mark.integration` that skips when its credentials are missing (see `tests/test_fingrid_wind_fetcher.py`, `tests/test_fingrid_wind_capacity_fetcher.py`, `tests/test_fingrid_client.py`, and the `tests/test_fingrid_*_integration.py` files).