# Methods for Refining Alphas in the Implementation Process

A raw alpha forecast `ξ` is the output of a *signal* model.  Before it can
drive capital allocation it must be **refined** through a deterministic,
auditable pipeline.  This document describes every step implemented in
`src/alpha_refine.py`.

## 1. The starting object

A long-form panel indexed by `(Date, Ticker)` with at minimum:

| column | meaning |
|---|---|
| `Raw Forecast` | ξ — expected excess return, *before* refinement |
| `Idiosyncratic Vol` | σᵢ — std. dev. of the market-model residual |
| `Factor Beta` | β — exposure to the benchmark |
| `Sector` | group label used for shrinkage and caps |

## 2. The refinement chain

```
ξ  ──►  winsorise  ──►  z-score  ──►  vol-scale  ──►  β-neutralise
                                                  │
                                                  ▼
                                       Bayesian shrinkage (sector)
                                                  │
                                                  ▼
                                       Black–Litterman blend
                                                  │
                                                  ▼
                                          refined α  →  optimiser
```

### 2.1 Winsorisation

**Why.** A single stock's signal can be 10× the cross-sectional median
(think: a one-off restructuring event).  That name will then dominate a
mean-variance optimiser even after risk scaling.  Capping at the 1st/99th
percentile per rebalance date keeps the long tail from breaking the chain.

```
ξ_w = clip(ξ, q_{1%}(ξ_t), q_{99%}(ξ_t))
```

### 2.2 Cross-sectional z-score

**Why.** Different signals live on different scales (return %, rank score,
sentiment z).  Z-scoring per date gives us a *common currency* and makes
the next scaling step interpretable.

```
z_t = (ξ_w − μ_t) / σ_t        (per rebalance date t)
```

### 2.3 Volatility-targeted scaling

**Why.** Grinold's fundamental law: ``α ∝ IC · score / σ``.  Dividing by
idiosyncratic vol converts forecasts to *risk-adjusted* units, so a 5%
forecast on a 40%-vol stock is shrunk relative to the same 5% on a 15%-vol
stock.

```
ξ_v = ξ_w × (σ_target / σᵢ)        with σ_target = 10% annualised
```

### 2.4 Beta-neutralisation

**Why.** In a market-neutral or sector-neutral book, the *market* component
of the forecast is not alpha.  Strip it out by regressing ξ on β within
each cross-section:

```
ξ_β = ξ_v − β · ξ̄_market
```

The residual is the genuinely idiosyncratic, market-orthogonal alpha.

### 2.5 Bayesian shrinkage (empirical prior)

**Why.** With ~18 names per sector, the per-name estimate of α is noisy.
A standard regularisation is to pull each forecast toward its sector mean:

```
ξ̃ = (1 − w) · ξ_β + w · ξ̄_sector
```

The shrinkage weight `w` can be set by *empirical Bayes* as

```
w = σ²_signal / (σ²_signal + σ²_noise / n_sector)
```

We expose `w` as a parameter; in practice a value of 0.2–0.4 works well
for monthly rebalances.

### 2.6 Black–Litterman blending

**Why.** The shrunk forecast `ξ̃` is one *view*.  Black–Litterman lets us
blend it with the CAPM-implied prior so we never bet *too* aggressively
against the market equilibrium:

```
E[R] = (τ·π + ω·ξ̃) / (τ + ω)
```

with `π = β · MRP` the CAPM-implied excess return and `ω` the view
variance (set here to the cross-sectional variance of `ξ̃` per date).

### 2.7 Cost-aware portfolio construction

**Why.** The refined α is the *objective function* of the optimiser, but
ignoring trading costs produces a portfolio that churns.  We add a
linear-in-turnover penalty (γ = ½ · tcost · |Δw|) inside the mean-variance
objective; full source in `src/portfolio.py`.

## 3. Sector caps & constraints (Indian market specifics)

Indian equity benchmarks have **constituent caps** (e.g. NIFTY 50 caps
any single stock at ~12%).  We enforce the analogous constraint in the
optimiser:

* per-stock cap: 10%
* per-sector cap: 40% (Banking), 30% (FMCG), 35% (IT)
* gross exposure ≤ 100%, long-only.

## 4. Diagnostic checks after refinement

The implementation should be paired with these checks at every rebalance:

1. **Rank correlation** of `ξ` vs refined α should be high but not 1.0
   (otherwise refinement did nothing).
2. **Cross-sectional IC** (Spearman) of refined α with next-period
   returns should exceed that of the raw ξ.
3. **Turnover vs γ** plot — the elbow of the cost/return curve.
4. **Realised ex-ante tracking error** vs constraint.

## 5. References

* Grinold & Kahn, *Active Portfolio Management*, 2nd ed.
* Black & Litterman (1992), *Asset Allocation: Combining Investor Views
  with Market Equilibrium*.
* Idzorek, *A Step-by-Step Guide to the Black–Litterman Model*.
* Novy-Marx & Velikov, *Taxes, Shorting, and the Cost of Capital*.
