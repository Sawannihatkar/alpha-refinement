"""Factor helpers — rolling OLS beta and idiosyncratic volatility."""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_beta_idio_vol(
    stock_returns: pd.DataFrame,
    benchmark_returns: pd.Series,
    window: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return rolling OLS beta and idio-vol (residual std) for each stock.

    For each date t, the regression on the trailing ``window`` observations is

        r_i,τ − r_f,τ = α_i + β_i (r_m,τ − r_f,τ) + ε_i,τ

    with r_f = 0 (daily approximation).  Idio-vol is the std. dev. of the
    residuals from that regression.
    """
    if benchmark_returns.name is None:
        benchmark_returns = benchmark_returns.rename("benchmark")
    aligned = stock_returns.join(benchmark_returns, how="inner").dropna()
    y = aligned[benchmark_returns.name]

    betas = pd.DataFrame(index=aligned.index, columns=stock_returns.columns, dtype=float)
    idios = pd.DataFrame(index=aligned.index, columns=stock_returns.columns, dtype=float)

    for tkr in stock_returns.columns:
        x = aligned[tkr].values
        m = y.values
        n = len(x)
        betas_t = np.full(n, np.nan)
        idio_t = np.full(n, np.nan)
        for i in range(window, n + 1):
            xs = x[i - window : i]
            ms = m[i - window : i]
            if np.isnan(xs).any() or np.isnan(ms).any():
                continue
            xm = xs.mean()
            mm = ms.mean()
            cov = ((xs - xm) * (ms - mm)).sum()
            var = ((ms - mm) ** 2).sum()
            if var <= 0:
                continue
            b = cov / var
            a = xm - b * mm
            resid = xs - (a + b * ms)
            betas_t[i - 1] = b
            idio_t[i - 1] = resid.std(ddof=2)
        betas[tkr] = betas_t
        idios[tkr] = idio_t

    return betas.astype(float), idios.astype(float)
