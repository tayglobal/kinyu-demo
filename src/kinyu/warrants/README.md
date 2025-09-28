# Exotic Warrant Pricing Overview

This document explains how `price_exotic_warrant` in `src/lib.rs` prices a callable
warrant whose strike is reset weekly at a discount to the spot price and that is
subject to issuer buybacks and default risk. The pricing engine combines
correlated equity/credit Monte Carlo simulation with Longstaff–Schwartz least
squares regression to approximate the continuation value of the security.

## Model Inputs

The pricing function exposes the following key parameters:

- `s0`: initial stock price.
- `strike_discount`: fraction applied to the current spot when the strike is reset.
- `buyback_price`: price at which the issuer can buy back the warrant (call feature).
- `t`: time to maturity (in years).
- `r`: risk-free drift used under the risk-neutral measure.
- `sigma`: annualised equity volatility.
- `credit_spreads`: term structure of hazard rates expressed as `(time, spread)` pairs.
- `equity_credit_corr`: correlation between equity shocks and credit shocks.
- `recovery_rate`: payoff if the issuer defaults before maturity.
- `n_paths`, `n_steps`: number of Monte Carlo paths and time steps.
- `poly_degree`: degree of the polynomial basis for the regression.
- `seed`: RNG seed for reproducibility.

These parameters are passed to `price_exotic_warrant`, which orchestrates the
simulation and backward induction steps described below.【F:src/kinyu/warrants/src/lib.rs†L106-L210】

## Credit Curve Interpolation

Credit spreads are provided as discrete term points and interpolated linearly to
obtain the hazard rate at any simulation time. For a curve defined by points
`(t_i, λ_i)`, the interpolated hazard at time $t$ is

$$
\lambda(t) = \lambda_i + (\lambda_{i+1} - \lambda_i) \frac{t - t_i}{t_{i+1} - t_i}
$$

for $t \in [t_i, t_{i+1}]$. Values outside the supplied range clamp to the
nearest endpoint.【F:src/kinyu/warrants/src/lib.rs†L8-L34】

## Correlated Equity and Credit Simulation

`simulate_correlated_paths` generates joint equity and credit scenarios over the
lattice of `n_steps` with step size $\Delta t = T / n_{steps}$.【F:src/kinyu/warrants/src/lib.rs†L36-L81】 The procedure is:

1. Build a 2×2 correlation matrix and take its Cholesky factor to couple a pair
   of standard normal draws $Z_1, Z_2$. The resulting correlated shocks are
   $ε^S = L_{00} Z_1$ for equity and $ε^C = L_{10} Z_1 + L_{11} Z_2$ for credit.【F:src/kinyu/warrants/src/lib.rs†L52-L66】
2. Simulate the stock price with a geometric Brownian motion under the
   risk-neutral measure:

   $$
   S_{t+Δ t} = S_t \exp\left( (r - \tfrac{1}{2} σ^2)Δ t + σ ε^S \sqrt{Δ t} \right)
   $$

   This evolves each path column in the `paths` matrix.【F:src/kinyu/warrants/src/lib.rs†L59-L68】
3. At each step, compute the default probability over $Δ t$ using the
   interpolated hazard rate: $p_{default} = 1 - e^{-λ(t) Δ t}$. Draw a
   correlated uniform variate via the standard normal CDF, $U = Φ(ε^C)$, and
   register the first step where $U < p_{default}$ as the default time for the path.【F:src/kinyu/warrants/src/lib.rs†L69-L76】

Paths that never default are assigned a default time later than maturity to keep
post-processing simple.【F:src/kinyu/warrants/src/lib.rs†L56-L80】

## Weekly Strike Resets

After simulating stock paths, the code constructs a strike matrix whose entries
reflect the weekly reset rule. The strike at the start is `s0 * strike_discount`,
and every time the simulation crosses into a new (discrete) week the strike is
updated to the previous step's spot price multiplied by the discount factor.【F:src/kinyu/warrants/src/lib.rs†L127-L140】

This implements a piecewise-constant strike process $K_t$ defined by

$$
K_t = \mathtt{strike\_discount} \times S_{t^-}
$$


whenever $t$ hits a new week boundary.

## Payoff Structure

At maturity, each path's payoff is determined by

$$
P_T = 
\begin{cases}
\mathtt{recovery\_rate}, & \tau \leq T, \\
\max(S_T - K_T, 0), & \mathtt{otherwise},
\end{cases}
$$

where $\tau$ is the default time. This seeds the vector of terminal warrant
values for the backward induction.【F:src/kinyu/warrants/src/lib.rs†L143-L150】

## Least Squares Monte Carlo Backward Induction

The algorithm then rolls back from the penultimate time step to the origin.
During each step:

1. Gather in-the-money, surviving paths ($S_t > K_t$ and $\tau > t$). For
   these paths, compute discounted future values $Y_j = V_{t+Δ t}^{(j)} e^{-r Δ t}$.
   These serve as the dependent variable in a polynomial regression on the spot
   price $X_j = S_t^{(j)}$.【F:src/kinyu/warrants/src/lib.rs†L152-L175】
2. Fit a least-squares polynomial of degree `poly_degree` to approximate the
   conditional expectation $E[V_{t+Δ t} e^{-r Δ t} \,|\, S_t]$. This is solved by
   building a Vandermonde matrix and applying the normal equations
   $(X^\top X) \beta = X^\top Y$.【F:src/kinyu/warrants/src/lib.rs†L84-L104】
3. For each surviving path, discount the current continuation value and evaluate
   the regression to obtain an estimate of the continuation value: $C(S_t) = \sum_{d=0}^{D} \beta_d S_t^d$

   If the path is in the money, compare $C(S_t)$ with the issuer's buyback
   price. Whenever $C(S_t) > \mathtt{buyback\_price}$, the warrant value is
   capped at the buyback level to reflect the issuer exercising its call right; otherwise
   the value simply becomes the discounted continuation. Paths that are out of the
   money, or that have defaulted, also take the discounted continuation or the
   recovery payoff, respectively【F:src/kinyu/warrants/src/lib.rs†L177-L205】

The regression is recalibrated at every step to capture the path-dependent
strike and correlated credit dynamics.

## Final Price Estimate

After stepping back to time zero, the Monte Carlo price is the average of the
pathwise values $V_0^{(j)}$:

$$
\text{Price} = \frac{1}{N} \sum_{j=1}^{N} V_0^{(j)}
$$

The function returns this mean as the estimated fair value of the warrant.【F:src/kinyu/warrants/src/lib.rs†L206-L209】

## Summary of Risk Features

- **Equity dynamics:** risk-neutral GBM with volatility $σ$.
- **Credit risk:** reduced-form default with stochastic hazard interpolated from
  `credit_spreads` and correlated to equity shocks.
- **Strike path dependence:** weekly reset proportional to the most recent spot.
- **Issuer optionality:** buyback feature embedded via Longstaff–Schwartz.
- **Recovery:** constant recovery payoff applied immediately upon default.

Together, these components allow the module to capture a complex, callable
warrant structure with both market and credit risk drivers.

## Building and Running the Module

### Prerequisites

- Python 3.8 or higher
- Rust toolchain (install from https://rustup.rs/)
- maturin (Python package for building Rust extensions)

### Building the Extension

Install maturin:

```bash
pip3 install maturin
```

Build the Rust extension:

```bash
cd src/kinyu/warrants
maturin build
```

Install the built wheel:

```bash
pip3 install target/wheels/warrants-0.1.0-cp312-cp312-*.whl
```

### Running the Tests

After building and installing the module, you can run the integration tests:

```bash
cd src/kinyu/warrants
python3 test.py
```

The test suite includes:
- **Test 1**: Pricing with low credit risk
- **Test 2**: Pricing with high credit risk (should be lower than low risk)
- **Test 3**: Pricing with zero correlation (should differ from correlated case)
- **Test 4**: Pricing with high buyback price (should be higher than normal case)

Expected output:
```
--- Running Integration Tests for Exotic Warrant Pricer ---

[Test 1] Price with LOW credit risk: 9.894630
[Test 2] Price with HIGH credit risk: 8.249698
[Test 3] Price with ZERO correlation: 9.900788
[Test 4] Price with HIGH buyback price: 9.976867

--- All integration tests passed successfully! ---
```

### Troubleshooting

- If you encounter a "lock file version" error, delete `Cargo.lock` and rebuild
- Ensure you have the correct Python version (3.8+) and Rust toolchain installed
- On macOS, the wheel will be built for your specific architecture (arm64 or x86_64)
