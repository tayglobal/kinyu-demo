import sys
import os
from pprint import pprint

# Add the source directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../')))

from kinyu.corr import calculate_correlation_matrix, make_positive_definite

def generate_sample_data():
    """Generates sample time series data for demonstration."""
    return {
        "AssetA": {
            "2023-01-01": 100.0,
            "2023-01-02": 101.0,
            "2023-01-03": 102.0,
            "2023-01-04": 103.0,
        },
        "AssetB": {
            "2023-01-01": 200.0,
            "2023-01-02": 202.0,
            "2023-01-03": 204.0,
            "2023-01-04": 206.0,
        },
        "AssetC": {
            "2023-01-01": 50.0,
            "2023-01-02": 49.0,
            "2023-01-03": 48.0,
            "2023-01-04": 47.0,
        },
    }

def main():
    """Main function to demonstrate the correlation library."""
    print("--- Correlation Library Demonstration ---")

    # 1. Calculate Correlation Matrix
    print("\n1. Calculating correlation matrix from time series data...")
    time_series_data = generate_sample_data()
    print("\nSample Time Series Data:")
    pprint(time_series_data)

    corr_matrix = calculate_correlation_matrix(time_series_data)
    print("\nCalculated Correlation Matrix:")
    pprint(corr_matrix)

    # 2. Make a Non-Positive Definite Matrix Positive Definite
    print("\n2. Correcting a non-positive definite matrix...")
    non_pd_matrix = [
        [1.0, 0.9, 0.2],
        [0.9, 1.0, 0.9],
        [0.2, 0.9, 1.0],
    ]
    print("\nOriginal Non-Positive Definite Matrix:")
    pprint(non_pd_matrix)

    # This matrix is likely not positive definite because the high correlation
    # between B and C (0.9) and A and B (0.9) is inconsistent with the low
    # correlation between A and C (0.2).

    min_eigenvalue = 0.01
    pd_matrix = make_positive_definite(non_pd_matrix, min_eigenvalue)
    print(f"\nCorrected Positive Definite Matrix (min eigenvalue = {min_eigenvalue}):")
    pprint(pd_matrix)

    print("\n--- End of Demonstration ---")

if __name__ == "__main__":
    main()