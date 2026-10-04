"""Transaction-cost aware portfolio construction."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def cost_aware_weights(
    expected_returns: pd.Series,
    cov: pd.DataFrame,
    current_weights: pd.Series | None = None,
    sector_caps: dict[str, float] | None = None,
    sectors: pd.Series | None = None,
    risk_aversion: float = 2.5,
    tcost_bps: float = 10.0,
    long_only: bool = True,
    gross_cap: float = 1.0,
) -> pd.Series:
    """Mean-variance with L2 turnover cost.

    Minimise::

        w' Σ w − (1/λ) w' μ + γ ||w − w₀||₁

    subject to ``0 ≤ w ≤ 1`` (long-only), ``Σw ≤ gross_cap``, and optional
    per-sector caps.
    """
    n = len(expected_returns)
    w0 = (current_weights.reindex(expected_returns.index).fillna(0.0).values
          if current_weights is not None else np.zeros(n))
    mu = expected_returns.values
    sigma = cov.values

    gamma = (tcost_bps / 1e4) * risk_aversion  # rough L1→L2 proxy
    bounds = [(0.0, 1.0) if long_only else (-1.0, 1.0)] * n
    constraints = [{"type": "ineq", "fun": lambda w: gross_cap - w.sum()}]

    if sector_caps is not None and sectors is not None:
        sectors = sectors.reindex(expected_returns.index)
        for s, cap in sector_caps.items():
            mask = (sectors == s).values.astype(float)
            if mask.sum() == 0:
                continue
            def _cap(w: np.ndarray, m: np.ndarray = mask, c: float = cap) -> float:
                return c - float((m * w).sum())
            constraints.append({"type": "ineq", "fun": _cap})

    def obj(w: np.ndarray) -> float:
        port_var = w @ sigma @ w
        port_ret = w @ mu
        turnover = np.abs(w - w0).sum()
        return port_var - port_ret / risk_aversion + gamma * turnover

    w0_init = w0 / w0.sum() if w0.sum() > 0 else np.ones(n) / n
    res = minimize(
        obj,
        w0_init,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 300, "ftol": 1e-9},
    )
    w = pd.Series(res.x, index=expected_returns.index).clip(lower=0.0)
    w = w / w.sum() if w.sum() > 0 else w
    return w
