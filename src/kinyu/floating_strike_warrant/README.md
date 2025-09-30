# Floating Strike Warrant Pricing Library

This directory contains a Rust-based Python library for pricing a complex, non-standard floating strike warrant using a Least-Squares Monte Carlo (LSMC) simulation.

**WARNING:** This implementation is for demonstration purposes and is known to be flawed. The core financial logic does not produce a correct price, and the `test_buyback_cap` test case fails. It should not be used for any real financial application.

## 1. Warrant Features

The warrant has the following key features, which are configurable through the `WarrantParams` class:

*   **Floating Strike:** The strike price (`K`) is not fixed. It resets periodically (e.g., weekly) to a specified discount of the underlying asset's price (`S`).
    *   *Parameters:* `strike_reset_period_days`, `strike_discount`.

*   **Issuer Buyback (Call) Right:** The issuer has the right to buy back the warrant from the holder at a predetermined price. This acts as a cap on the warrant's value.
    *   *Parameter:* `buyback_price`.

*   **Capped Exercise Quota:** There is a limit on the volume of warrants that can be exercised within a given period (e.g., monthly). This quota resets at the end of each period.
    *   *Parameters:* `exercise_limit_percentage`, `exercise_limit_period_days`, `next_exercise_reset_date`, `exercised_this_period_percentage`.

## 2. Pricing Model: Least-Squares Monte Carlo (LSMC)

The warrant is priced using an LSMC simulation, which is well-suited for path-dependent and American-style options.

### Core Logic

The model works via backward induction:

1.  **Path Generation:** We simulate thousands of possible price paths for the underlying stock (`S_t`) from today until maturity using Geometric Brownian Motion (GBM):

    `dS_t = r * S_t * dt + vol * S_t * dZ_t`

    Along each path, the state of the warrant (strike price, available quota) is updated according to the specified reset rules.

2.  **Backward Induction:** Starting at maturity (`T`), we step backward in time to today (`t=0`). At each step `t`:

    a.  **Estimate Continuation Value:** We determine the expected value of holding the warrant, `C(t)`, if it is not exercised. This is done by regressing the discounted future per-unit values of the warrant (`V(t+dt)`) against a set of basis functions of the current state (e.g., moneyness `S/K`).

        `V(t+dt) ~ B_0 + B_1*(S/K) + B_2*(S/K)^2 + ...`

    b.  **Holder's Optimal Decision:** The holder decides whether to exercise or hold by comparing the immediate exercise (intrinsic) value with the continuation value:

        `Holder's Optimal Value = max(Intrinsic Value, Continuation Value)`

        where `Intrinsic Value = max(S_t - K_t, 0)`.

    c.  **Issuer's Capping Action:** The issuer then applies their buyback right, capping the holder's optimal value at the buyback price.

        `Final Per-Unit Value V(t) = min(Holder's Optimal Value, Buyback Price)`

    d.  **Determine Cash Flow:** A cash flow is generated *only if* the holder's optimal decision was to exercise. The cash flow is the final, capped per-unit value multiplied by the available exercise quota for that path.

        `CashFlow(t) = V(t) * remaining_quota` (if exercise optimal), `0` (otherwise).

3.  **Final Price:** The final price of the warrant is the average of the sum of all discounted cash flows across every simulated path.

    `Price = E[ Sum_t( e^(-r*t) * CashFlow(t) ) ]`

## 3. Installation and Testing

### Installation

The library is built as a Rust extension to a Python package.

1.  Ensure you have a recent version of Rust and Python installed.
2.  From the root of the project (`/app`), run the following command to compile the Rust code and install the package in editable mode:
    ```bash
    pip install -e .
    ```

### Running Tests

The test suite validates the implementation with sanity checks and tests for sensitivity to key financial parameters.

1.  Make sure `pytest` is installed: `pip install pytest`.
2.  From the root of the project (`/app`), run the tests using the following command:
    ```bash
    PYTHONPATH=src python3 -m pytest src/kinyu/floating_strike_warrant/tests/test_pricing.py
    ```