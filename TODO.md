# Project Action Plan & Checklist

## Phase 1: Data Acquisition & Merging
- [x] Register a free API key on Fingrid Open Data (`data.fingrid.fi`).
- [x] Fetch 3 years of wind power generation (Dataset `75` for 15-min/hourly historical series, or Dataset `181` for real-time measurements) and wind production capacity (Dataset `268`) via the Fingrid API.
- [x] Fetch 3 years of national electricity consumption (Dataset `124` for historical 15-min/hourly series, or Dataset `193`), nuclear power (Dataset `188`), and hydro power (Dataset `191`) from Fingrid.
- [x] Fetch 3 years of cross-border transmission flows between Finland and `SE1`, `SE3`, and `EE` from Fingrid.
- [x] Download 3 years of hourly temperature, wind speed, and radiation data from FMI Open Data (`en.ilmatieteenlaitos.fi/open-data`) for Helsinki (Kaisaniemi), Vaasa, and Oulu.
- [x] Download 3 years of hourly Nord Pool FI spot prices (`c/kWh`) via the Energy-Charts API (ENTSO-E / Nord Pool data); `sahkonhintatanaan.fi` is offline and ENTSO-E's own API needs a token.
- [x] Align and merge all Fingrid (15-minute series post-June 2023 resampled to hourly means), FMI, and spot price time series onto a single UTC hourly index and save the full pull via `scripts/fetch_all.py` (gaps preserved as nulls).
- [x] Interpolate short gaps (< 3 hours) by linear interpolation and drop incomplete boundary rows; save `combined/hourly_clean.parquet`. Longer interior gaps remain null and are recorded in the manifest.

## Phase 2: Feature Engineering (24-Feature Pipeline)
- [x] Compute grid ratio features: `wind_penetration_ratio` (`wind_mw / consumption_mw`) and `wind_capacity_factor` (`wind_mw / wind_capacity_mw`).
- [ ] Compute thermal demand features: `heating_degree_hours` ($\max(0, 17^\circ\text{C} - T_{\text{helsinki}})$), `temp_rolling_24h`, and `temp_rolling_72h`.
- [ ] Compute interaction feature `wind_temp_interaction` (`wind_penetration_ratio * heating_degree_hours`) to isolate cold, windless scarcity hours.
- [ ] Compute cyclical time encodings (`sin_hour`, `cos_hour`, `sin_month`, `cos_month`), `is_weekend`, and `is_fi_public_holiday` using the Python `holidays` package (`country='FI'`).
- [ ] Save the complete 24-feature + target dataset to a clean Parquet/CSV file and verify zero NaNs and zero future-target leakage.
- [ ] Digitize 6–8 real retail offers from `sahkonhinta.fi` (Fixed, 100% Wind Fixed, Spot, 100% Wind Spot, and Hybrid/*kulutusvaikutus*) into a structured JSON config.
- [ ] Create two standard Finnish hourly load profiles (`hourly_kwh`): a 2,500 kWh/yr apartment and an 18,000 kWh/yr electrically heated detached house.

## Phase 3: Model Training & Feature Ablation Study
- [ ] Implement a chronological split (Years 1–2 for training, first 2 months of Year 3 for validation/early stopping, remaining 10–12 months for held-out testing) and fit `StandardScaler` strictly on training data.
- [ ] Implement the baseline deterministic MLP (`nn.MSELoss()`) in PyTorch.
- [ ] Implement `HeteroscedasticPriceNet` in PyTorch with dual outputs ($\mu_\theta(x)$ and $\sigma_\theta^2(x) = \text{Softplus}(\cdot) + 10^{-4}$) and the custom `gaussian_nll_loss` function.
- [ ] Add weight decay (`1e-4`) and dropout (`0.1`) to prevent variance head ($\sigma_\theta^2$) overfitting across the 24 features.
- [ ] Run a 3-tier feature ablation experiment comparing test Gaussian NLL and RMSE across: (1) 6 basic features, (2) the 24 domain-engineered features, and (3) an expanded noisy feature set (adding raw lags/polynomials) to prove why domain-curated features generalize best.

## Phase 4: Contract Simulator & Evaluation
- [ ] Implement the analytical annual cost calculator for Fixed-price offers (`12 * base_fee + sum(fixed_rate * kwh)`).
- [ ] Implement the analytical expected cost and standard deviation ($\Sigma_{\text{spot}} = \sqrt{\sum (c_t \sigma_t)^2}$) calculator for Spot offers.
- [ ] Implement the monthly consumption-impact (*kulutusvaikutus*) cost calculator for Hybrid offers.
- [ ] Evaluate contract rankings across risk-aversion weights ($\lambda \in \{0, 0.5, 1.0, 1.645\}$) on the held-out test year and compare recommended offers against true realized annual costs.

## Phase 5: Diagnostics & Final Write-Up
- [ ] Plot predicted spot price $\mu_\theta(x)$ with $\pm 1.96\sigma_\theta(x)$ uncertainty bands during a freezing, low-wind winter week versus a windy summer week.
- [ ] Plot the empirical calibration curve (predicted confidence level vs. actual fraction of realized spot prices falling inside the interval) and feature ablation table.
- [ ] Plot expected annual cost vs. 95% worst-case cost across all contracts for both household profiles.
- [ ] Write the final report covering the Gaussian MLE derivation, Finnish grid feature engineering rationale, ablation benchmarks, and contract backtest results.