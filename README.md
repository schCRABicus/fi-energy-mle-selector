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
│   ├── http_client.py# JSON HTTP client with retries (httpx + tenacity)
│   └── fingrid_wind/ # Fingrid client + wind power fetcher (dataset 75)
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
- A free Fingrid Open Data API key from [`data.fingrid.fi`](https://data.fingrid.fi) — only needed for live data and integration tests.

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
uv run pytest -m integration -rs    # live Fingrid tests only (network + key required)
```

Run a single file or test:

```bash
uv run pytest tests/test_fingrid_wind_fetcher.py -q
uv run pytest tests/test_fingrid_wind_fetcher.py::test_fetch_resamples_quarter_hourly_samples_to_hourly_means
```

The integration tests hit the real API (a handful of requests, with automatic 429 retry/backoff), so avoid running them in tight loops against the free-tier quota.

### 4. Fetch data from a specific fetcher

Each dataset has a fetcher under `datasets/<source>/`. Today there is one: `FingridWindFetcher` (Fingrid dataset 75, wind generation in MW).

```bash
uv run python - <<'PY'
import os
from datetime import UTC, datetime

from dotenv import load_dotenv

from datasets import FetchRequest, FingridWindFetcher, run_fetcher

# Explicit path: load_dotenv() cannot introspect a file path when run via stdin.
load_dotenv(".env")

fetcher = FingridWindFetcher(os.environ["FINGRID_API_KEY"])
request = FetchRequest(
    start_utc=datetime(2024, 3, 1, tzinfo=UTC),
    end_utc=datetime(2024, 3, 2, tzinfo=UTC),
    resolution="1h",  # "15min", "30min", or "1h"
)
table = run_fetcher(fetcher, request)

print(table)
print(table.slice(0, 3).to_pylist())
PY
```

Notes:
- Windows are half-open `[start_utc, end_utc)` and must be timezone-aware.
- Fingrid serves 15-minute samples after June 2023; `resolution="1h"` returns hourly means, `resolution="15min"` leaves the raw cadence.
- Multi-year ranges are paginated automatically and retried on Fingrid rate limits.
- `run_fetcher` re-validates the output against `fetcher.expected_schema`; call `fetcher.fetch(...)` directly to skip that extra check.
- To persist a table, write it to Parquet with pyarrow: `import pyarrow.parquet as pq; pq.write_table(table, "data/wind.parquet")` (create `data/` first).

### 5. Add a new fetcher

1. Create `datasets/<source>/<source>_fetcher.py` with a class implementing `source_name`, `expected_schema`, and `fetch` (see the `DataSourceFetcher` protocol in `datasets/api.py`).
2. Export it from the subpackage's `__init__.py` and from `datasets/__init__.py`.
3. Add offline unit tests that serve scripted responses through an `httpx.MockTransport` and inject the resulting client, plus a live test marked `@pytest.mark.integration` that skips when its credentials are missing (see `tests/test_fingrid_wind_fetcher.py` and `tests/test_fingrid_wind_integration.py`).