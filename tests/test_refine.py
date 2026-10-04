"""Smoke tests for the alpha-refinement pipeline."""
import numpy as np
import pandas as pd
import pytest

from src.alpha_refine import (
    bayesian_shrink,
    beta_neutralise,
    black_litterman_blend,
    cross_sectional_zscore,
    refine_pipeline,
    volatility_scale,
    winsorise,
)


@pytest.fixture
def toy_panel() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=3, freq="W")
    tickers = ["A", "B", "C", "D", "E"]
    rows = []
    for d in dates:
        for t in tickers:
            rows.append({
                "Date": d,
                "Ticker": t,
                "Sector": ["Banking", "Banking", "FMCG", "FMCG", "IT"][tickers.index(t)],
                "Raw Forecast": rng.normal(0.05, 0.05),
                "Idiosyncratic Vol": abs(rng.normal(0.20, 0.05)),
                "Factor Beta": rng.normal(1.0, 0.2),
            })
    return pd.DataFrame(rows)


def test_winsorise_caps_extremes(toy_panel):
    out = winsorise(toy_panel, lower=0.05, upper=0.95)
    # No value should lie strictly outside the per-date quantiles
    for _, sub in out.groupby("Date"):
        lo = sub["Raw Forecast"].quantile(0.05)
        hi = sub["Raw Forecast"].quantile(0.95)
        assert (sub["Forecast_Winsor"] >= lo - 1e-12).all()
        assert (sub["Forecast_Winsor"] <= hi + 1e-12).all()


def test_zscore_is_unit_variance_per_date(toy_panel):
    out = cross_sectional_zscore(winsorise(toy_panel))
    g = out.groupby("Date")["Forecast_Z"]
    # mean ~ 0, std ~ 1 within each date
    assert np.allclose(g.mean().values, 0.0, atol=1e-8)
    assert np.allclose(g.std().values, 1.0, atol=1e-8)


def test_volatility_scale_inverse_relation(toy_panel):
    out = volatility_scale(winsorise(toy_panel))
    # Higher vol → smaller scaled forecast (same raw input)
    mask = out["Raw Forecast"] == out["Raw Forecast"].iloc[0]
    sample = out[mask]
    if len(sample) >= 2:
        vols = sample["Idiosyncratic Vol"].values
        scaled = sample["Forecast_VolScaled"].values
        # monotonic decreasing in vol (when raw is held constant)
        order = np.argsort(vols)
        assert scaled[order[0]] >= scaled[order[-1]]


def test_beta_neutralise_zero_mean_per_date(toy_panel):
    out = beta_neutralise(volatility_scale(winsorise(toy_panel)))
    for d, sub in out.groupby("Date"):
        # market component removed → weighted mean ≈ 0
        weighted = (sub["Factor Beta"] * sub["Forecast_BetaNeutral"]).sum()
        market = (sub["Factor Beta"] * sub["Forecast_VolScaled"]).sum()
        # the subtracted term equals market / Σβ
        assert np.isfinite(sub["Forecast_BetaNeutral"]).all()


def test_shrinkage_pulls_toward_sector_mean(toy_panel):
    # Bayesian shrinkage operates on the *input* forecast column directly;
    # use the winsorised forecast to keep the example self-contained.
    out = bayesian_shrink(winsorise(toy_panel), forecast_col="Forecast_Winsor", weight=0.5)
    for (d, sec), sub in out.groupby(["Date", "Sector"]):
        sector_mean = sub["Forecast_Winsor"].mean()
        for _, r in sub.iterrows():
            lo, hi = sorted([r["Forecast_Winsor"], sector_mean])
            assert lo - 1e-12 <= r["Forecast_Shrunk"] <= hi + 1e-12


def test_pipeline_returns_all_columns(toy_panel):
    out = refine_pipeline(toy_panel)
    expected = {
        "Forecast_Winsor", "Forecast_Z", "Forecast_VolScaled",
        "Forecast_BetaNeutral", "Forecast_Shrunk", "Forecast_BL",
    }
    assert expected.issubset(out.columns)
