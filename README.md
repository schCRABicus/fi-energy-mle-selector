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
- **Day-Ahead Spot Prices (FI Area):** Nord Pool / ENTSO-E / `sahkonhintatanaan.fi`
- **Retail Contract Offers:** Finnish Energy Authority (`sahkonhinta.fi`)

## Repository Structure
```text
├── datasets/         # Data-source adapters: shared contract + per-source fetchers
│   ├── api.py        # FetchRequest and DataSourceFetcher protocol
│   ├── http_client.py# HTTP client with retries (httpx + tenacity)
│   ├── utils.py      # Window validation, resampling, local Parquet loading
│   ├── fingrid/      # Fingrid client, shared series fetcher, per-dataset fetchers
│   │   └── data/     # Committed 1-day Parquet snapshots for offline runs
│   └── fmi/          # FMI WFS client and city weather fetcher
│       └── data/     # Committed 1-day Parquet snapshots for offline runs
├── tests/            # Offline unit tests and live API integration tests
├── configs/          # (planned) retail contract definitions & household load profiles
├── data/             # (planned) raw and processed hourly datasets (git-ignored)
├── src/              # (planned) feature engineering, PyTorch MLE models, simulator
├── pyproject.toml    # Project metadata, dependencies, pytest configuration
├── uv.lock           # Locked dependency versions
├── README.md
└── TODO.md           # Project roadmap and implementation checklist
```

## Local Development

### Prerequisites
- [`uv`](https://docs.astral.sh/uv/) — manages the Python 3.14 interpreter and the virtualenv.
- A free Fingrid Open Data API key from [`data.fingrid.fi`](https://data.fingrid.fi) — only needed for live Fingrid data and integration tests.
- FMI Open Data needs no credentials.

### 1. Install dependencies

```bash
git clone <repo-url>
cd fi-energy-mle-selector
uv sync
```

`uv sync` creates `.venv/` and installs the locked runtime and `dev` dependencies (including pytest). Add or change dependencies with `uv add <package>` / `uv add --dev <package>` — never edit `uv.lock` by hand. Prefix every command with `uv run` so it executes inside `.venv/`.

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

Unit tests never touch the network: fetcher tests read the committed snapshots under `datasets/fingrid/data/` and `datasets/fmi/data/` (all covering 2024-03-01), and client tests serve scripted responses through `httpx.MockTransport`. The Fingrid snapshots hold the raw cadence: 96 quarter-hour samples for wind generation, consumption, and each cross-border flow, 24 hourly samples for wind capacity, and 958 roughly-90-second samples for nuclear and hydro. Each FMI snapshot holds 24 hourly rows for one city. Only the integration tests hit the real APIs (a handful of requests, with automatic retry/backoff), so avoid running them in tight loops against the free-tier quota.

### 4. Fetch data from a specific fetcher

Each dataset has a fetcher under `datasets/<source>/`. The Fingrid fetchers all share the same constructor and request API; swap the class name in the snippet below.

| Fetcher | Dataset | Series | Output column |
| --- | --- | --- | --- |
| `FingridWindFetcher` | 75 | Wind power generation | `wind_mw` |
| `FingridWindCapacityFetcher` | 268 | Installed wind capacity | `wind_capacity_mw` |
| `FingridConsumptionFetcher` | 124 | National electricity consumption (MWh/h ≈ MW) | `consumption_mw` |
| `FingridNuclearFetcher` | 188 | Nuclear power production | `nuclear_mw` |
| `FingridHydroFetcher` | 191 | Hydro power production | `hydro_mw` |
| `FingridSE1FlowFetcher` | 60 | Finland ↔ Northern Sweden flow (positive = export) | `se1_flow_mw` |
| `FingridSE3FlowFetcher` | 61 | Finland ↔ Central Sweden flow (positive = export) | `se3_flow_mw` |
| `FingridEEFlowFetcher` | 55 | Finland ↔ Estonia flow (positive = export) | `ee_flow_mw` |

```bash
uv run python - <<'PY'
import os
from datetime import UTC, datetime

from dotenv import load_dotenv

from datasets import FetchRequest, FingridWindFetcher

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
- The bundled snapshots live at `datasets/fingrid/data/` (`wind_sample.parquet`, `wind_capacity_sample.parquet`, `consumption_sample.parquet`, `nuclear_sample.parquet`, `hydro_sample.parquet`, `se1_flow_sample.parquet`, `se3_flow_sample.parquet`, `ee_flow_sample.parquet`) and cover 2024-03-01 only. Override any of them with `local_path="path/to/snapshot.parquet"` or disable it with `local_path=None`.
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

from datasets import FetchRequest, FmiWeatherFetcher

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
- Missing observations become null values in the output table.
- `fetch_from_local` reads `datasets/fmi/data/<city>_sample.parquet` and falls back to the remote API when that snapshot is missing. Override it with `local_path="path/to/snapshot.parquet"` or disable it with `local_path=None`.

The Phase 1 historical pull (3 years, hourly) loops over the fetchers and writes one Parquet file per series:

```bash
uv run python - <<'PY'
import os
from datetime import UTC, datetime
from pathlib import Path

import pyarrow.parquet as pq
from dotenv import load_dotenv

from datasets import (
    FetchRequest,
    FingridConsumptionFetcher,
    FingridEEFlowFetcher,
    FingridHydroFetcher,
    FingridNuclearFetcher,
    FingridSE1FlowFetcher,
    FingridSE3FlowFetcher,
    FingridWindCapacityFetcher,
    FingridWindFetcher,
    FmiWeatherFetcher,
)

load_dotenv(".env")

request = FetchRequest(
    start_utc=datetime(2022, 1, 1, tzinfo=UTC),
    end_utc=datetime(2025, 1, 1, tzinfo=UTC),
    resolution="1h",
)
fingrid_fetchers = (
    FingridWindFetcher(os.environ["FINGRID_API_KEY"]),
    FingridWindCapacityFetcher(os.environ["FINGRID_API_KEY"]),
    FingridConsumptionFetcher(os.environ["FINGRID_API_KEY"]),
    FingridNuclearFetcher(os.environ["FINGRID_API_KEY"]),
    FingridHydroFetcher(os.environ["FINGRID_API_KEY"]),
    FingridSE1FlowFetcher(os.environ["FINGRID_API_KEY"]),
    FingridSE3FlowFetcher(os.environ["FINGRID_API_KEY"]),
    FingridEEFlowFetcher(os.environ["FINGRID_API_KEY"]),
)
fmi_fetchers = tuple(FmiWeatherFetcher(city) for city in ("helsinki", "vaasa", "oulu"))

for source, fetchers in (("fingrid", fingrid_fetchers), ("fmi", fmi_fetchers)):
    output_dir = Path("data") / source
    output_dir.mkdir(parents=True, exist_ok=True)
    for fetcher in fetchers:
        with fetcher:
            table = fetcher.fetch(request)
        pq.write_table(
            table,
            output_dir / f"{fetcher.source_name}.parquet",
            compression="zstd",
        )
        print(fetcher.source_name, table.num_rows)
PY
```

The FMI leg is slower because FMI caps radiation requests at seven days; expect several minutes and keep the free-tier request volume in mind.

### 5. Add a new fetcher

1. For another Fingrid numeric series, subclass `FingridSeriesFetcher` and supply the dataset id, value column, schema, and local snapshot path (see `FingridNuclearFetcher`); it already implements `source_name`, `expected_schema`, `fetch`, and `fetch_from_local`. For other sources, create `datasets/<source>/<source>_fetcher.py` implementing the `DataSourceFetcher` protocol from `datasets/api.py`; reuse `datasets/utils.py` for window validation, resolution lookup, bucket averaging, and local Parquet loading.
2. Export it from the subpackage's `__init__.py` and from `datasets/__init__.py`.
3. Add an offline Parquet fixture for fetcher tests, serve scripted HTTP responses through `httpx.MockTransport` for client tests, and add a live test marked `@pytest.mark.integration` that skips when its credentials are missing (see `tests/test_fingrid_wind_fetcher.py`, `tests/test_fingrid_wind_capacity_fetcher.py`, `tests/test_fingrid_client.py`, and the `tests/test_fingrid_*_integration.py` files).