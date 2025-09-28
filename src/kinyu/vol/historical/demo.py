import numpy as np
from kinyu_historical import HistoricalVolatility

def run_demo():
    """
    Demonstrates the usage of the kinyu_historical library to calculate
    various historical volatility estimators.
    """
    # Sample OHLC data: Open, High, Low, Close
    # Data is arranged in rows, each row representing a time period (e.g., a day)
    prices = np.array([
        [100.0, 102.0, 99.0, 101.0],
        [101.0, 103.0, 100.5, 102.5],
        [102.5, 104.0, 102.0, 103.0],
        [103.0, 103.5, 101.5, 102.0],
        [102.0, 104.0, 101.0, 103.5],
    ], dtype=np.float64)

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