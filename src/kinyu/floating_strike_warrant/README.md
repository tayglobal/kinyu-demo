# Floating Strike Warrant Pricer

This crate implements a Longstaff–Schwartz Monte Carlo (LSMC) pricer for a floating-strike warrant with periodic strike resets, issuer buybacks, holder put protection, and exercise quota limits.

## Contract Overview

A floating-strike warrant entitles the holder to purchase the underlying equity at a strike that is periodically reset to a fixed discount of the prevailing spot. Let

- $S_t$ be the underlying stock price at time $t$,
- $d \in (0,1]$ be the strike discount,
- $\tau^K$ be the strike reset interval (e.g., one week), and
- $K_t$ be the strike immediately after the reset at time $t$.

At each reset date $t \in \mathcal{T}^K$ we set

$$
K_t = d \cdot S_t.
$$

The holder can exercise any fraction $x_t \in [0, q_t]$ of the warrant inventory during a trading step, producing an immediate payoff of

$$
\text{payoff}(t, x_t) = x_t \cdot \max(S_t - K_t, 0).
$$

The remaining exercise allowance $q_t$ is capped. Define

- $\bar{q}$ as the per-period exercise limit fraction,
- $\tau^Q$ as the quota reset interval (e.g., monthly),
- $t^Q$ the next quota reset time, and
- $\Delta q_t$ the quantity already exercised within the current quota period.

Between quota resets we have

$$
q_t = \max\bigl(0, \bar{q} - \Delta q_t\bigr),
$$

and at a quota reset time $t \in \mathcal{T}^Q$ the allowance is replenished to $\bar{q}$.

The issuer may call back the warrant at a buyback price $B$; the holder's value is therefore truncated at $B$ whenever the call is available:

$$
V_t^{\text{post-call}} = \min\left(V_t^{\text{holder}}, B\right).
$$

When the underlying trades below a contractual trigger level $S^\text{put}$, the warrant holder can demand an early buyback at a guaranteed price $P^\text{put}$. This creates an additional cash flow opportunity that competes with the continuation and call choices in the valuation:

$$
\text{HolderPutPayoff}_t =
\begin{cases}
P^\text{put}, & S_t \leq S^\text{put}, \\
0, & \text{otherwise}.
\end{cases}
$$

## Pricing Methodology

We estimate the arbitrage-free price via an LSMC backward induction:

1. **Forward simulation**: Generate $N$ paths of the geometric Brownian motion underlying, applying strike resets and quota refills deterministically along each path.
2. **Continuation regression**: At each exercise step $t$, regress the discounted value $V_{t+\Delta}$ onto a feature basis $\Phi_t(S_t, K_t, q_t, \ldots)$ using singular value decomposition (SVD) to obtain a stable least-squares fit.
3. **Exercise policy**: Compare the intrinsic value $g_t = \max(S_t - K_t, 0)$ with the estimated marginal continuation value. The optimal exercise fraction $x_t^*$ satisfies

$$
x_t^* =
\begin{cases}
0, & g_t \leq \lambda_t, \\
\min(q_t, x_{\max}), & g_t > \lambda_t,
\end{cases}
$$

   where the shadow price of quota $\lambda_t$ is approximated by a finite difference of the continuation estimate:

$$
\lambda_t \approx \frac{\widehat{C}(q_t) - \widehat{C}(q_t - \delta q)}{\delta q}.
$$

4. **Discounting and aggregation**: Cash flows are discounted at the risk-free rate and averaged across paths to produce the warrant price.

This approach captures both the floating strike mechanics and the binding exercise caps by embedding the remaining quota directly into the regression features.

## Installation

The pricer is packaged as an internal Rust crate. To build it you will need the Rust toolchain (recommended via [`rustup`](https://rustup.rs/)). Once Rust is installed, fetch the crate dependencies:

```bash
cd /workspace/kinyu-demo/src/kinyu/floating_strike_warrant
cargo fetch
```

## Running the Tests

Execute the unit test suite with:

```bash
cd /workspace/kinyu-demo
cargo test -p floating_strike_warrant
```

or run all workspace tests:

```bash
cargo test
```

Both commands should finish without failures and validate the pricing routines, path mechanics, and stress scenarios described above.

## Python Binding Demo

The crate exposes an optional [PyO3](https://pyo3.rs/) binding (behind the `python` feature flag) that makes the Monte Carlo pricer available from Python. A minimal end-to-end workflow is:

```bash
cd /workspace/kinyu-demo/src/kinyu/floating_strike_warrant
python -m venv .venv
source .venv/bin/activate
pip install maturin
maturin build --release --features python
pip install target/wheels/floating_strike_warrant-*.whl
python python_demo.py
```

The `python_demo.py` script calls the Rust pricer for a baseline set of parameters and several perturbations to illustrate how the value reacts to key risk drivers. Because the pricer now tracks inventory depletion, quota refills, and the holder put option explicitly, changes to volatility, interest rates, the issuer buyback cap, and the put terms all propagate through the valuation. Running the script produces a Markdown table such as the one below (generated with 3,000 Monte Carlo paths per scenario and a fixed RNG seed):

| Scenario | Key value | Price |
| --- | --- | --- |
| Baseline | - | 3.139163 |
| Spot -5% | 95.0000 | 3.253824 |
| Spot +5% | 105.0000 | 3.091662 |
| Volatility 15% | 0.1500 | 2.620668 |
| Volatility 35% | 0.3500 | 3.665311 |
| Rate 0% | 0.0000 | 3.242602 |
| Rate 4% | 0.0400 | 3.061998 |
| Buyback 2 | 2.0000 | 1.886489 |
| Buyback 3 | 3.0000 | 2.515118 |
| Discount 85% | 0.8500 | 4.220395 |
| Discount 95% | 0.9500 | 2.096318 |
| Quota 10% | 0.1000 | 2.007972 |
| Quota 40% | 0.4000 | 2.621859 |
| Put trigger 85 | 85.0000 | 3.694463 |
| Put trigger 70 | 70.0000 | 2.912962 |
| Put price 4 | 4.0000 | 2.825968 |
| Put price 8 | 8.0000 | 3.452860 |

## Further Reading

For more background on LSMC pricing of American-style derivatives, see Longstaff & Schwartz (2001).

