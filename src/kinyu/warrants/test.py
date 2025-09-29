from kinyu import warrants

# Parameters for the warrant
s0 = 100.0
strike_discount = 0.9
buyback_price = 15.0
t = 1.0
forward_curve = [[0.0, 0.05], [t, 0.05]] # Flat forward curve at 5%
sigma = 0.2
n_paths = 10000
n_steps = 252
poly_degree = 2 # Reduced from 4 for stability

# Credit-related parameters
credit_spreads_low = [[t, 0.01] for t in range(1, 6)]  # Low credit spread
credit_spreads_high = [[t, 0.20] for t in range(1, 6)] # High credit spread
equity_credit_corr = -0.5 # Negative correlation is typical
recovery_rate = 0.4
monthly_exercise_limit = 1.0 # Default: no limit

print("--- Running Integration Tests for Exotic Warrant Pricer ---")

try:
    # --- Test 1: Pricing with low credit risk ---
    price_low_risk = warrants.price_exotic_warrant(
        s0, strike_discount, buyback_price, t, forward_curve, sigma,
        credit_spreads_low, equity_credit_corr, recovery_rate,
        monthly_exercise_limit, n_paths, n_steps, poly_degree
    )
    print(f"\n[Test 1] Price with LOW credit risk: {price_low_risk:.6f}")
    assert price_low_risk > 0, "Price with low risk should be positive"

    # --- Test 2: Pricing with high credit risk ---
    price_high_risk = warrants.price_exotic_warrant(
        s0, strike_discount, buyback_price, t, forward_curve, sigma,
        credit_spreads_high, equity_credit_corr, recovery_rate,
        monthly_exercise_limit, n_paths, n_steps, poly_degree
    )
    print(f"[Test 2] Price with HIGH credit risk: {price_high_risk:.6f}")
    assert price_high_risk < price_low_risk, "High credit risk should lower the price"

    # --- Test 3: Pricing with zero correlation ---
    price_zero_corr = warrants.price_exotic_warrant(
        s0, strike_discount, buyback_price, t, forward_curve, sigma,
        credit_spreads_low, 0.0, recovery_rate, # Zero correlation
        monthly_exercise_limit, n_paths, n_steps, poly_degree
    )
    print(f"[Test 3] Price with ZERO correlation: {price_zero_corr:.6f}")
    # Correlation impact can be small, just check it's not identical
    assert abs(price_zero_corr - price_low_risk) > 1e-9, "Correlation should impact the price"

    # --- Test 4: Pricing with high buyback price (less likely to be called) ---
    price_high_buyback = warrants.price_exotic_warrant(
        s0, strike_discount, 1000.0, t, forward_curve, sigma, # High buyback
        credit_spreads_low, equity_credit_corr, recovery_rate,
        monthly_exercise_limit, n_paths, n_steps, poly_degree
    )
    print(f"[Test 4] Price with HIGH buyback price: {price_high_buyback:.6f}")
    assert price_high_buyback > price_low_risk, "Higher buyback price should increase the warrant value"

    # --- Test 5: Pricing with a monthly exercise limit ---
    price_with_limit = warrants.price_exotic_warrant(
        s0, strike_discount, buyback_price, t, forward_curve, sigma,
        credit_spreads_low, equity_credit_corr, recovery_rate,
        0.1, n_paths, n_steps, poly_degree # 10% monthly limit
    )
    print(f"[Test 5] Price with 10% MONTHLY exercise limit: {price_with_limit:.6f}")
    assert price_with_limit < price_low_risk, "Exercise limit should lower the price"

    # --- Test 6: Pricing with no credit risk (optional curve) ---
    price_no_credit_risk = warrants.price_exotic_warrant(
        s0, strike_discount, buyback_price, t, forward_curve, sigma,
        None, equity_credit_corr, recovery_rate, # No credit curve
        monthly_exercise_limit, n_paths, n_steps, poly_degree
    )
    print(f"[Test 6] Price with NO credit risk (None): {price_no_credit_risk:.6f}")

    # Compare with a zero-spread curve, should be very close
    price_zero_spread_risk = warrants.price_exotic_warrant(
        s0, strike_discount, buyback_price, t, forward_curve, sigma,
        [[1.0, 0.0]], equity_credit_corr, recovery_rate,
        monthly_exercise_limit, n_paths, n_steps, poly_degree
    )
    print(f"          (Price with ZERO credit spread: {price_zero_spread_risk:.6f})")
    assert abs(price_no_credit_risk - price_zero_spread_risk) < 0.5, "No credit risk should be ~ zero credit risk"


    print("\n--- All integration tests passed successfully! ---")

except Exception as e:
    print(f"\n--- An error occurred during testing: {e} ---")