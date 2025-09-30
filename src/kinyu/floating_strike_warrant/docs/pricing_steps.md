
# 1) State, controls, and timelines

**State at time $t$** (per path):

* $S_t$: underlying
* $K_t$: weekly-reset strike (e.g., $K_t=S_{t_\text{last reset}}$)
* $q_t \in [0,\bar q]$: remaining monthly exercise allowance (e.g., $\bar q=10\%$ of total shares you hold/are allowed to exercise; measure in "fraction of total")
* $\tau_t$: time to maturity
* Optional: $\sigma_\text{inst}$ (pathwise vol proxy), week-day indicator, days-to-next-reset $d^{(w)}_t$, days-to-month-end $d^{(m)}_t$

**Control at time $t$**:

* Exercise **fraction** $x_t \in [0, q_t]$ (partial exercise allowed).
* If your contract is pure "now-or-later" without partial exercise, discretize $x_t\in\{0, \min(q_t, \Delta q)\}$ with a small exercise quantum $\Delta q$. Partial exercise makes this cleaner and more realistic with volume caps.

**Cashflow at time $t$** (per unit exercised):

* $g_t=\max(S_t-K_t,0)$ (or your exact payoff definition under weekly reset).
* Issuer **buyback/call** at price $B_t$: at call times, holder value becomes $\min\{V^\text{holder}_t,B_t\}$ (game option approximation; see §5).

# 2) Backward induction with an inventory (quota) state

You want the *continuation value* as a function of both price **and remaining quota**:

$$V(t, S_t,K_t,q_t) = \max_{x_t\in[0,q_t]} \left\{ x_t \cdot g_t + \mathbb{E}_t\left[V(t+\Delta, S_{t+\Delta},K_{t+\Delta},q_t-x_t+\text{refill}_t)\right]\right\}$$

* At each **month boundary**, set $q_{t^+} = \bar q$ (refill).
* At each **week boundary**, reset $K_{t^+} = S_{t}$.

LSMC approximates the conditional expectation $C(t,S,K,q)=\mathbb{E}_t[V(t+\Delta,\cdot)]$.

# 3) Two practical implementation routes

## (A) Grid the quota

Discretize $q$ into $J$ levels: $\{0, \bar q/J, 2\bar q/J,\dots,\bar q\}$.

* In the **forward pass**, carry the *discrete* $q_t$ (after applying exercised amount).
* In the **backward pass**, **run a separate regression for each quota bin** (or a pooled regression with $q$ features, see (B)).
* At time $t$, for each path with quota bin $j$, compare:

  * **Immediate exercise of $x=\min(q_t,\Delta q)$**: value $x \cdot g_t + \widehat C(t; q_t-x)$
  * **Continue**: $\widehat C(t; q_t)$
  * If partial exercise is allowed continuously, solve bang-bang by marginal value (see below), or evaluate a few candidate $x$'s $(0, \Delta q, \text{all})$.

Pros: simple, robust. Cons: more regressions (one per $q$-bin).

## (B) Single regression with a *quota-aware* basis (recommended)

Keep $q_t$ continuous and regress $\widehat C(t,S,K,q)$ **once per time step** on a basis that includes $q$ and interactions (next section). This is faster and captures smooth dependence on remaining allowance.

# 4) Basis functions that work well

Let $m_t=\frac{S_t}{K_t}$ (moneyness under weekly reset). Suggested basis $\Phi$:

**Core terms**

* Polynomials in moneyness: $1, m, m^2, (m-1)^+, [(m-1)^+]^2$
* Time: $\tau, \sqrt{\tau}$, day-to-next-reset $d^{(w)}$, day-to-month-end $d^{(m)}$
* Quota: $q, q^2, \sqrt{q}$
* Cross terms: $m\cdot q, (m-1)^+\cdot q, m\cdot q^2, (m-1)^+\cdot \sqrt{q}$
* Vol proxy: $\sigma_\text{inst}, m\cdot \sigma_\text{inst}, q\cdot \sigma_\text{inst}$

**Indicators (piecewise structure)**

* $\mathbf{1}_{m>1}, \mathbf{1}_{q\approx 0}$ (e.g., $q<0.01\bar q$), $\mathbf{1}_{d^{(m)}\le 3}, \mathbf{1}_{d^{(w)}\le 1}$

**Why these help**

* The **cap binds** precisely when intrinsic is juicy $(m>1)$ and **quota is scarce**; cross-terms let the regression learn that **continuation value rises** as quota tightens in-the-money.
* $d^{(w)}$ and $d^{(m)}$ handle impending **reset** and **quota refill** cliffs.

# 5) Exercise rule via “marginal value of quota”

After you regress $\widehat C(t,S,K,q)$, compute a **finite-difference marginal value of quota** (a shadow price):

$$\lambda(t,S,K,q) \approx \frac{\widehat C(t,S,K,q) - \widehat C(t,S,K,q-\delta q)}{\delta q}$$

Intuition: $\lambda$ is the continuation value of **keeping** 1 more unit of exercise capacity.

**Bang-bang partial exercise** at time $t$:

* If **intrinsic** $g_t \le \lambda$: do **not** exercise (continuation is better).
* If $g_t > \lambda$: exercise up to the cap available **today**.

  * With only a monthly cap (no per-day cap), this means $x_t = q_t$ (exercise all remaining) is optimal in this myopic rule. To smooth, cap $x_t$ by a per-step maximum $x^{\max}$ or choose $x_t$ s.t. $g_t \approx \lambda$ if you prefer a throttled policy.

**Issuer buyback (call) right**:

* At call dates (or continuously if callable anytime), **truncate** the holder value:

  $$V^\text{post-call}(t, S,K,q)=\min\big\{V^\text{holder}(t,S,K,q), B_t\big\}$$

* In the backward pass: after computing $\widehat C$ and applying the exercise decision, replace pathwise value with $\min(\cdot, B_t)$ at call times. (For true *game options*, iterate a few times: holder's best response then issuer's, or use a small penalty toward the min operator.)

# 6) Month & week boundaries (mechanics)

* **Week reset:** at reset times, set $K_{t^+}=S_t$. Include a basis indicator $\mathbf{1}_{d^{(w)}=0}$.
* **Month refill:** at month-end, set $q_{t^+}=\bar q$. Include $\mathbf{1}_{d^{(m)}=0}$.
* Keep these transformations **inside the simulated state transition** so the regression "sees" post-jump states during training.

# 7) Practical LSMC loop (pseudo-code)

```
# Forward simulate paths of S, store S_t, and deterministically update K_t (weekly) and q_t (monthly refill, minus exercises).
for path in 1..N:
  simulate S_{t0..T}
  init K_{t0} = S_{t0}
  init q_{t0} = \bar q

# Backward pass
set V_T = 0 for all paths (or terminal payoff if special)
for t in {T-Δ, ..., t0}:
  # 1) Build features Φ_t from (m=S/K, q, τ, d^(w), d^(m), σ_inst, indicators)
  # 2) Regression: fit C_hat = E[ V_{t+Δ} | Φ_t ]
  # 3) Compute λ ≈ (C_hat(q) - C_hat(q-δq)) / δq
  # 4) Exercise policy:
      if g_t > λ:
          x_t = min(q_t, x_max_step)   # or q_t if no per-step cap
      else:
          x_t = 0
  # 5) Cashflow and next value:
      V_t_path = x_t * g_t + C_hat(t, S,K, q_t - x_t)
  # 6) Apply issuer buyback if applicable at t:
      if is_call_time(t): V_t_path = min(V_t_path, B_t)
  # 7) Update q for bookkeeping used in earlier times:
      q_t_effective = q_t - x_t
  # 8) At boundaries:
      if week_reset_today: K_{t^-} -> K_{t^+} = S_t
      if month_refill_today: q_{t^-} -> q_{t^+} = \bar q
```

# 8) Diagnostics & variance control

* **Sanity bounds:**
  $\text{Price(with cap)} \le \text{Price(without cap)}$
  $\text{Price(with issuer call)} \le \text{Price(no call)}$
* **Control variates:**
  Use the unconstrained American (no cap, no call) as CV to reduce variance.
* **Stress knobs:**
  Vary $\bar q$ (5–20%), vol, and per-step $x^{\max}$ to see binding frequency. Track $\Pr[g_t>\lambda]$ and quota utilization distribution.
* **Basis parsimony:**
  Start with ~10–20 basis terms, expand only if needed. Regularize (ridge) to stabilize $\lambda$ estimates.

# 9) Notes on impact

* The cap **always weakly reduces** value; the hit is larger when: high vol, deep ITM spikes coincide with **low remaining $q$** and buyback is **not** overly dilutive (i.e., you'd like to exercise but can't).
* With weekly resets, moneyness reverts each week, **increasing** the chance that delaying exercise wastes ITM periods—this makes the cap more costly than in fixed-strike cases.
