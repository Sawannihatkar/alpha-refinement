# Alpha Refinement: Indian Banking, FMCG & IT Stocks

A research repo documenting and implementing the **methods used to refine raw alpha forecasts (ξ)** in the systematic investment process — from the initial signal through shrinkage, beta/idio-vol adjustment, Black-Litterman blending, transaction-cost aware sizing, and final portfolio construction.

## Why this repo?

A raw alpha forecast (ξ) is noisy, biased, and rarely usable as-is. Before it can drive a position it must be **refined** through a stack of statistical and portfolio techniques. This repo:

1. Builds a **realistic sample dataset** (long-form panel) for N Indian stocks × T daily dates with the columns practitioners actually need:
   - `Date`, `Ticker`, `Close`, `Sector`, `Benchmark Return (Rm)`, `Factor Beta (β)`, `Raw Forecast (ξ)`, `Idiosyncratic Vol (σᵢ)`.
2. Pulls **live data via `yfinance`** (NSE tickers for Banking, FMCG, IT) and the NIFTY 50 benchmark.
3. Demonstrates the **refinement stack**: signal denoising → cross-sectional winsorisation → beta/idio-vol scaling → Bayesian shrinkage → Black-Litterman → portfolio optimisation with costs.

## Project layout

```
alpha-refinement/
├── data/                  # raw & processed CSVs (gitignored at scale)
├── notebooks/             # exploratory notebooks
│   └── 01_data_construction.ipynb
├── src/                   # library code
│   ├── __init__.py
│   ├── data.py            # yfinance + Fama-French loaders
│   ├── factors.py         # beta, idiosyncratic vol, factor returns
│   ├── alpha_refine.py    # the refinement methods
│   └── portfolio.py       # Black-Litterman + constrained optimisation
├── tests/                 # pytest unit tests
├── docs/                  # methodology notes
├── requirements.txt
└── README.md
```

## The data contract (one row per asset-day)

| Column | Meaning |
|---|---|
| `Date` | Trading date |
| `Ticker` | NSE symbol (e.g. `HDFCBANK.NS`) |
| `Close` | Adjusted close price (₹) |
| `Sector` | `Banking`, `FMCG`, or `IT` |
| `Benchmark Return (Rm)` | NIFTY 50 daily return (common to all rows on the date) |
| `Factor Beta (β)` | Stock's rolling beta to NIFTY 50 (e.g. 252d OLS) |
| `Raw Forecast (ξ)` | The pre-refinement alpha — e.g. an 8% annualised expected return |
| `Idiosyncratic Vol (σᵢ)` | Std. dev. of the residual from the market-model regression |

This mirrors the table you sketched (AAPL/MSFT example), but populated for real Indian names.

## Quickstart

```bash
pip install -r requirements.txt
python -c "from src.data import build_sample_panel; df = build_sample_panel(); print(df.head())"
```

## Methods covered

See [`docs/methods.md`](docs/methods.md) for the full write-up. Summary:

1. **Denoising the raw signal** — exponential smoothing, Kalman filter, wavelet shrinkage.
2. **Cross-sectional winsorisation & z-scoring** — kill outliers that dominate optimisation.
3. **Volatility-targeted scaling** — divide ξ by σᵢ so forecasts are expressed in risk-adjusted units.
4. **Beta-neutralisation** — strip the benchmark-driven part so the residual is genuine alpha.
5. **Bayesian shrinkage** — pull each ξ toward the sector (or grand) mean: `ξ̃ = (1−w)·ξ + w·ξ̄_sector`.
6. **Black-Litterman blending** — treat ξ as the "view" and combine with CAPIM prior.
7. **Transaction-cost aware sizing** — penalise turnover using the half-spread + impact model.
8. **Final constrained optimisation** — mean-variance with L2 costs, sector caps, beta & vol targets.

## Indian sector tickers used (NSE)

| Sector | Tickers (`.NS` suffix) |
|---|---|
| Banking | HDFCBANK, ICICIBANK, KOTAKBANK, SBIN, AXISBANK, INDUSINDBK |
| FMCG | HINDUNILVR, ITC, NESTLEIND, BRITANNIA, DABUR, MARICO |
| IT | TCS, INFY, WIPRO, HCLTECH, TECHM, LTIM |
| Benchmark | `^NSEI` (NIFTY 50) |

## License

MIT. Data via Yahoo Finance — for research/education only.
