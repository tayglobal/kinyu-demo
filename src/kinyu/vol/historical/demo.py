import numpy as np
from kinyu_historical import HistoricalVolatility

def generate_ohlc_data(n_days=250, s0=100.0, sigma=0.2):
    """
    Generates a more realistic OHLC dataset for a given number of days.
    """
    np.random.seed(42) # for reproducibility
    # Generate daily returns from a normal distribution
    daily_returns = np.random.normal(0, sigma / np.sqrt(252), n_days)

    prices = np.zeros((n_days, 4))
    last_close = s0

    for i in range(n_days):
        open_price = last_close
        close_price = open_price * (1 + daily_returns[i])

        # Generate high and low prices with a more constrained random spread
        high = max(open_price, close_price) * (1 + np.random.uniform(0, 0.02))
        low = min(open_price, close_price) * (1 - np.random.uniform(0, 0.02))

        # Ensure low is always positive
        if low <= 0:
            low = min(open_price, close_price) * 0.98

        prices[i, 0] = open_price
        prices[i, 1] = high
        prices[i, 2] = low
        prices[i, 3] = close_price

        last_close = close_price

    return prices.astype(np.float64)

def run_demo():
    """
    Demonstrates the usage of the kinyu_historical library to calculate
    various historical volatility estimators.
    """
    # Generate a more realistic dataset with 250 days of OHLC data
    prices = generate_ohlc_data(n_days=250)

    print("--- Historical Volatility Estimation Demo ---")
    print(f"Using {prices.shape[0]} periods of OHLC data.")

    # Create an instance of the HistoricalVolatility calculator
    hv = HistoricalVolatility(prices)

    # Trading days in a year for annualization
    trading_days_per_year = 252
    annualization_factor = np.sqrt(trading_days_per_year)

    # Calculate and print each volatility estimator
    estimators = {
        "Close-to-Close": hv.close_to_close,
        "Parkinson": hv.parkinson,
        "Garman-Klass": hv.garman_klass,
        "Rogers-Satchell": hv.rogers_satchell,
        "Yang-Zhang": hv.yang_zhang,
    }

    print("\nResults (Annualized):")
    print("-" * 25)
    for name, estimator_fn in estimators.items():
        daily_vol = estimator_fn()
        annualized_vol = daily_vol * annualization_factor
        print(f"{name:<20}: {annualized_vol:.4f}")
    print("-" * 25)

if __name__ == "__main__":
    run_demo()