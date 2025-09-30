import kinyu.floating_strike_warrant as fsw
from datetime import date, timedelta
import pytest

@pytest.fixture
def warrant_params():
    """Provides a standard set of warrant parameters for testing."""
    today = date.today()
    return {
        "spot": 100.0,
        "vol": 0.3,
        "risk_free_rate": 0.05,
        "maturity_date": today + timedelta(days=365),
        "strike_reset_period_days": 30,
        "strike_discount": 0.1,
        "buyback_price": 25.0,
        "exercise_limit_percentage": 0.1,
        "exercise_limit_period_days": 90,
        "next_exercise_reset_date": today + timedelta(days=60),
        "exercised_this_period_percentage": 0.02,
        "num_paths": 5000,
        "num_steps": 52,
        "seed": 42
    }

def test_price_sanity_check(warrant_params):
    """
    Tests if the calculated warrant price is within a reasonable range.
    """
    params = fsw.WarrantParams(**warrant_params)
    initial_strike = params.spot * (1.0 - params.strike_discount)
    initial_intrinsic = max(0, params.spot - initial_strike)
    initial_quota = params.exercise_limit_percentage - params.exercised_this_period_percentage
    immediate_exercise_value = initial_intrinsic * initial_quota
    price = fsw.price_warrant(params)
    print(f"\nCalculated Warrant Price: {price:.4f}")
    print(f"Initial Spot: {params.spot:.4f}")
    print(f"Immediate Exercise Value: {immediate_exercise_value:.4f}")
    assert price > 0
    assert price < params.spot
    assert price >= immediate_exercise_value - 1e-9
    print("\nPrice sanity checks passed.")

def test_volatility_sensitivity(warrant_params):
    """
    Tests that the warrant price increases with volatility.
    """
    base_params = fsw.WarrantParams(**warrant_params)
    base_price = fsw.price_warrant(base_params)
    high_vol_params_dict = warrant_params.copy()
    high_vol_params_dict["vol"] = warrant_params["vol"] + 0.1
    high_vol_params = fsw.WarrantParams(**high_vol_params_dict)
    high_vol_price = fsw.price_warrant(high_vol_params)
    print(f"\nBase Price (vol={base_params.vol:.2f}): {base_price:.4f}")
    print(f"High Vol Price (vol={high_vol_params.vol:.2f}): {high_vol_price:.4f}")
    assert high_vol_price > base_price
    print("\nVolatility sensitivity test passed.")

def test_buyback_cap(warrant_params):
    """
    Tests that a low buyback price correctly caps the warrant's value.
    """
    params_dict = warrant_params.copy()
    # Deep in-the-money, low vol to make exercise highly likely
    params_dict["spot"] = 150.0
    params_dict["vol"] = 0.1
    # Very low buyback price
    params_dict["buyback_price"] = 5.0
    params = fsw.WarrantParams(**params_dict)

    price = fsw.price_warrant(params)
    initial_quota = params.exercise_limit_percentage - params.exercised_this_period_percentage
    expected_capped_value = params.buyback_price * initial_quota

    print(f"\nCalculated Price with low buyback: {price:.4f}")
    print(f"Expected Capped Value: {expected_capped_value:.4f}")

    # The price should be very close to the capped value
    assert abs(price - expected_capped_value) < 0.1
    print("\nBuyback cap test passed.")

def test_quota_refill_value(warrant_params):
    """
    Tests that the option to have a quota refill has value.
    """
    today = date.today()
    params_dict = warrant_params.copy()

    # Case 1: Matures just BEFORE the quota reset
    params_dict["maturity_date"] = params_dict["next_exercise_reset_date"] - timedelta(days=1)
    params_no_refill = fsw.WarrantParams(**params_dict)
    price_no_refill = fsw.price_warrant(params_no_refill)

    # Case 2: Matures just AFTER the quota reset
    params_dict["maturity_date"] = params_dict["next_exercise_reset_date"] + timedelta(days=5)
    params_with_refill = fsw.WarrantParams(**params_dict)
    price_with_refill = fsw.price_warrant(params_with_refill)

    print(f"\nPrice (matures before refill): {price_no_refill:.4f}")
    print(f"Price (matures after refill): {price_with_refill:.4f}")

    assert price_with_refill > price_no_refill, "Value should increase when a quota refill is included."
    print("\nQuota refill value test passed.")