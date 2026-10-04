"""Data construction: yfinance + Fama-French loaders.

The single public entry point is :func:`build_sample_panel`, which returns a
long-form DataFrame matching the contract documented in the README.

Columns returned (per asset-day):

* ``Date``            – trading date
* ``Ticker``          – NSE symbol
* ``Close``           – adjusted close (₹)
* ``Sector``          – Banking | FMCG | IT
* ``Benchmark Return`` – NIFTY 50 daily return (constant across rows on a date)
* ``Factor Beta``     – rolling 252-day OLS beta to NIFTY 50
* ``Raw Forecast``    – ξ, a synthetic raw alpha signal (see ``_synthesise_raw_forecast``)
* ``Idiosyncratic Vol`` – σᵢ, std. dev. of OLS residual (rolling)
"""
from __future__ import annotations

import warnings
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf

from .factors import rolling_beta_idio_vol

# ---------------------------------------------------------------------------
# Universe
# ---------------------------------------------------------------------------
TICKERS: dict[str, list[str]] = {
    "Banking": [
        "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS",
        "SBIN.NS", "AXISBANK.NS", "INDUSINDBK.NS",
    ],
    "FMCG": [
        "HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS",
        "BRITANNIA.NS", "DABUR.NS", "MARICO.NS",
    ],
    "IT": [
        "TCS.NS", "INFY.NS", "WIPRO.NS",
        "HCLTECH.NS", "TECHM.NS", "LTTS.NS",
    ],
}
BENCHMARK = "^NSEI"  # NIFTY 50


def _flatten(tickers: Iterable[str] | None = None) -> list[str]:
    if tickers is None:
        return [BENCHMARK] + [t for sector in TICKERS.values() for t in sector]
    return list(tickers)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def fetch_prices(
    tickers: Iterable[str] | None = None,
    start: str = "2023-01-01",
    end: str | None = None,
) -> pd.DataFrame:
    """Return adjusted close prices for the given NSE tickers (wide form)."""
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    tickers = list(tickers) if tickers is not None else _flatten()
    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,   # adjusted close
        progress=False,
        threads=True,
    )
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        # Single ticker case
        close = data[["Close"]].rename(columns={"Close": tickers[0]})
    close = close.dropna(how="all").ffill().dropna(how="all")
    return close


def fetch_benchmark(start: str = "2023-01-01", end: str | None = None) -> pd.Series:
    """Return the NIFTY 50 adjusted close series."""
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    px = fetch_prices([BENCHMARK], start=start, end=end)
    return px[BENCHMARK]


# ---------------------------------------------------------------------------
# Raw forecast synthesis
# ---------------------------------------------------------------------------
def _synthesise_raw_forecast(
    returns: pd.DataFrame, sector_map: dict[str, str], seed: int = 7
) -> pd.Series:
    """Generate a noisy, biased raw alpha signal ξ for prototyping.

    The signal mixes a sector-mean drift, a momentum term (12-1), and Gaussian
    noise.  It is intentionally imperfect so the refinement stack has something
    to chew on.
    """
    rng = np.random.default_rng(seed)

    def _momentum(r: pd.Series) -> pd.Series:
        # 12-1 momentum: 12m return minus most-recent month
        if len(r.dropna()) < 252:
            return pd.Series(np.nan, index=r.index)
        return r.rolling(252).sum().shift(21)

    mom = returns.apply(_momentum)
    sector_mu = {s: 0.06 for s in set(sector_map.values())}  # 6% annualised base
    rows = []
    for tkr, sec in sector_map.items():
        if tkr not in returns.columns:
            continue
        s = returns[tkr]
        m = mom[tkr]
        noise = pd.Series(rng.normal(0, 0.04, len(s)), index=s.index, name=tkr)
        xi = sector_mu[sec] + 0.5 * m.fillna(0) + noise
        xi.name = tkr
        rows.append(xi)
    return pd.concat(rows, axis=1)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def build_sample_panel(
    start: str = "2023-01-01",
    end: str | None = None,
    window: int = 252,
) -> pd.DataFrame:
    """Build the long-form sample panel described in the README."""
    end = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    sector_map = {t: s for s, lst in TICKERS.items() for t in lst}
    stock_tickers = list(sector_map)

    # 1. Prices
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bench_close = fetch_benchmark(start=start, end=end)
        stock_close = fetch_prices(stock_tickers, start=start, end=end)

    # 2. Returns
    bench_ret = bench_close.pct_change().rename("Benchmark Return")
    stock_ret = stock_close.pct_change()

    # 3. Rolling beta & idio vol
    beta, idio = rolling_beta_idio_vol(stock_ret, bench_ret, window=window)

    # 4. Raw forecast
    xi = _synthesise_raw_forecast(stock_ret, sector_map)

    # 5. Assemble long-form panel
    panels = []
    for tkr in stock_tickers:
        if tkr not in stock_close.columns:
            continue
        df = pd.DataFrame(
            {
                "Date": stock_close.index,
                "Ticker": tkr.replace(".NS", ""),
                "Close": stock_close[tkr].values,
                "Sector": sector_map[tkr],
                "Benchmark Return": bench_ret.reindex(stock_close.index).values,
                "Factor Beta": beta[tkr].reindex(stock_close.index).values,
                "Raw Forecast": xi[tkr].reindex(stock_close.index).values,
                "Idiosyncratic Vol": idio[tkr].reindex(stock_close.index).values,
            }
        )
        panels.append(df)
    panel = pd.concat(panels, ignore_index=True)
    panel = panel.dropna(subset=["Factor Beta", "Idiosyncratic Vol", "Raw Forecast"])
    return panel.reset_index(drop=True)
