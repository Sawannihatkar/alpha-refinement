"""The alpha-refinement methods.

These are the building blocks of the "refining" stage: turning a noisy,
biased raw forecast ξ into a usable expected-return input for portfolio
optimisation.  Every function takes a *long-form* or *wide-form* DataFrame
of forecasts and returns the refined version with the same shape.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# 1. Cross-sectional winsorisation
# ---------------------------------------------------------------------------
def winsorise(
    df: pd.DataFrame,
    value_col: str = "Raw Forecast",
    by: str = "Date",
    lower: float = 0.01,
    upper: float = 0.99,
    out_col: str = "Forecast_Winsor",
) -> pd.DataFrame:
    """Clip extreme forecasts to the [lower, upper] percentile of each date.

    A handful of crazy ξ's will dominate mean-variance optimisation, so we
    cap them at the empirical quantiles per rebalance date.
    """
    out = df.copy()
    lo = out.groupby(by)[value_col].transform(lambda s: s.quantile(lower))
    hi = out.groupby(by)[value_col].transform(lambda s: s.quantile(upper))
    out[out_col] = out[value_col].clip(lower=lo, upper=hi)
    return out


# ---------------------------------------------------------------------------
# 2. Cross-sectional z-scoring
# ---------------------------------------------------------------------------
def cross_sectional_zscore(
    df: pd.DataFrame,
    value_col: str = "Forecast_Winsor",
    by: str = "Date",
    out_col: str = "Forecast_Z",
) -> pd.DataFrame:
    """Standardise forecasts to mean 0 / std 1 within each rebalance date."""
    out = df.copy()
    g = out.groupby(by)[value_col]
    out[out_col] = (out[value_col] - g.transform("mean")) / g.transform("std")
    return out


# ---------------------------------------------------------------------------
# 3. Volatility-targeted scaling
# ---------------------------------------------------------------------------
def volatility_scale(
    df: pd.DataFrame,
    forecast_col: str = "Forecast_Winsor",
    vol_col: str = "Idiosyncratic Vol",
    target_vol: float = 0.10,
    out_col: str = "Forecast_VolScaled",
) -> pd.DataFrame:
    """Divide ξ by σᵢ so forecasts are expressed in *risk-adjusted* units.

    The classical Grinold rule: ``α ∝ IC · score / σ``.  We follow the same
    idea: scale by the asset's idio-vol so that high-vol names don't get
    over-weighted simply because their raw signal is larger.
    """
    out = df.copy()
    out[out_col] = out[forecast_col] * (target_vol / out[vol_col].replace(0, np.nan))
    return out


# ---------------------------------------------------------------------------
# 4. Beta-neutralisation
# ---------------------------------------------------------------------------
def beta_neutralise(
    df: pd.DataFrame,
    forecast_col: str = "Forecast_VolScaled",
    beta_col: str = "Factor Beta",
    by: str = "Date",
    out_col: str = "Forecast_BetaNeutral",
) -> pd.DataFrame:
    """Strip out the market-driven component: ξ̃ = ξ − β·ξ̄_market.

    A long-only or sector-neutral portfolio only wants the *residual* alpha,
    not the part of ξ that is just the market's beta.
    """
    out = df.copy()
    market = out.groupby(by)[forecast_col].transform("mean")
    out[out_col] = out[forecast_col] - out[beta_col].fillna(1.0) * market
    return out


# ---------------------------------------------------------------------------
# 5. Bayesian shrinkage toward the sector (or grand) mean
# ---------------------------------------------------------------------------
def bayesian_shrink(
    df: pd.DataFrame,
    forecast_col: str = "Forecast_BetaNeutral",
    group_col: str = "Sector",
    weight: float = 0.3,
    by: str = "Date",
    out_col: str = "Forecast_Shrunk",
) -> pd.DataFrame:
    """Pull each forecast toward its sector mean: ``ξ̃ = (1−w)·ξ + w·ξ̄_sector``.

    ``weight`` is the *prior* precision share: 0 = trust the data, 1 = trust
    only the sector mean.  Empirical Bayes chooses ``w`` from the
    cross-sectional variance ratio; here we expose it as a parameter for
    clarity.
    """
    out = df.copy()
    grp_mean = out.groupby([by, group_col])[forecast_col].transform("mean")
    out[out_col] = (1.0 - weight) * out[forecast_col] + weight * grp_mean
    return out


# ---------------------------------------------------------------------------
# 6. Black–Litterman blending with CAPM prior
# ---------------------------------------------------------------------------
def black_litterman_blend(
    df: pd.DataFrame,
    forecast_col: str = "Forecast_Shrunk",
    beta_col: str = "Factor Beta",
    market_risk_premium: float = 0.05,
    tau: float = 0.05,
    by: str = "Date",
    rescale: bool = True,
    out_col: str = "Forecast_BL",
) -> pd.DataFrame:
    """Combine the CAPM prior with the views ξ using a 1-factor Black–Litterman.

    Closed-form, single-factor, diagonal omega::

        E[R] = (τ·π + ω·Q) / (τ + ω)

    with ``π = β·MRP`` the CAPM-implied prior and ``ω = var(Q)`` the view
    variance per date.  When ``rescale=True`` the blended output is divided
    by its cross-sectional std to land in a z-score-like unit that can be
    plugged into a downstream mean-variance optimiser.
    """
    out = df.copy()
    out[out_col] = np.nan

    for date, sub in out.groupby(by):
        beta = sub[beta_col].fillna(1.0).values
        q = sub[forecast_col].values
        prior = beta * market_risk_premium
        omega_var = np.nanvar(q)
        if not np.isfinite(omega_var) or omega_var <= 0:
            out.loc[sub.index, out_col] = prior
            continue
        omega = np.full_like(beta, omega_var)
        e = (tau * prior + omega * q) / (tau + omega)
        if rescale:
            sd = np.nanstd(e)
            if sd > 0:
                e = e / sd
        out.loc[sub.index, out_col] = e

    return out


# ---------------------------------------------------------------------------
# Convenience: run the full chain
# ---------------------------------------------------------------------------
def refine_pipeline(
    df: pd.DataFrame,
    winsor_limits: tuple[float, float] = (0.01, 0.99),
    target_vol: float = 0.10,
    shrinkage_weight: float = 0.3,
    mrp: float = 0.05,
    tau: float = 0.05,
) -> pd.DataFrame:
    """Apply the standard refinement chain in one call."""
    out = winsorise(df, lower=winsor_limits[0], upper=winsor_limits[1])
    out = cross_sectional_zscore(out)
    out = volatility_scale(out, target_vol=target_vol)
    out = beta_neutralise(out)
    out = bayesian_shrink(out, weight=shrinkage_weight)
    out = black_litterman_blend(out, market_risk_premium=mrp, tau=tau)
    return out
