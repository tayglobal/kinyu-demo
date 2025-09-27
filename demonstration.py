import corr
import pprint
import random
from datetime import date, timedelta

def generate_time_series_data(num_series=5, num_days=365):
    """
    Generates a dictionary of time series data with some correlation patterns.
    """
    start_date = date(2022, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(num_days)]

    series_data = {}

    # Generate a base random walk for ts1
    ts1_values = [100 + random.uniform(-1, 1) for _ in range(num_days)]
    for i in range(1, num_days):
        ts1_values[i] = ts1_values[i-1] + random.uniform(-1, 1)
    series_data["ts1"] = {d.strftime("%Y-%m-%d"): v for d, v in zip(dates, ts1_values)}

    # ts2 is highly correlated with ts1
    ts2_values = [v + random.uniform(-0.2, 0.2) for v in ts1_values]
    series_data["ts2"] = {d.strftime("%Y-%m-%d"): v for d, v in zip(dates, ts2_values)}

    # ts3 is negatively correlated with ts1
    ts3_values = [-v + random.uniform(-1, 1) for v in ts1_values]
    series_data["ts3"] = {d.strftime("%Y-%m-%d"): v for d, v in zip(dates, ts3_values)}

    # ts4 is an independent random walk
    ts4_values = [100 + random.uniform(-5, 5) for _ in range(num_days)]
    for i in range(1, num_days):
        ts4_values[i] = ts4_values[i-1] + random.uniform(-5, 5)
    series_data["ts4"] = {d.strftime("%Y-%m-%d"): v for d, v in zip(dates, ts4_values)}

    # ts5 is somewhat correlated with ts2
    ts5_values = [v * 0.5 + random.uniform(-3, 3) for v in ts2_values]
    series_data["ts5"] = {d.strftime("%Y-%m-%d"): v for d, v in zip(dates, ts5_values)}

    return series_data

def demonstrate_correlation():
    """
    Demonstrates the use of the `corr` library to calculate a correlation matrix
    and to make a non-positive definite matrix positive definite.
    """
    # 1. Generate 5 time series with 1 year of data
    print("--- Generating 5 time series with 1 year of data ---")
    series_data = generate_time_series_data()
    print("Generated 5 time series.")
    # print("Sample of generated data for ts1:")
    # pprint.pprint({k: series_data['ts1'][k] for k in list(series_data['ts1'])[:5]})


    print("\n--- Calculating Correlation Matrix for 5 Time Series ---")
    correlation_matrix = corr.calculate_correlation_matrix(series_data)
    print("Calculated Correlation Matrix:")
    pprint.pprint(correlation_matrix)

    # 2. Demonstrate making a matrix positive definite
    print("\n--- Correcting a Non-Positive Definite Matrix ---")
    non_pd_matrix = [
        [1.0, 1.2, 0.3],
        [1.2, 1.0, 0.5],
        [0.3, 0.5, 1.0]
    ]
    print("\nNon-Positive Definite Matrix:")
    pprint.pprint(non_pd_matrix)

    corrected_matrix = corr.py_make_positive_definite(non_pd_matrix, 0.01)
    print("\nCorrected Positive Definite Matrix:")
    pprint.pprint(corrected_matrix)

if __name__ == "__main__":
    demonstrate_correlation()