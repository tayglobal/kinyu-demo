
import numpy as np
from kinyu.credit.implied.credit_curve import (
    build_survival_curve,
    survival_to_default_probabilities,
    credit_spread_curve,
)

def main():
    """An example script to demonstrate the credit curve library."""
    # Input data
    dates = [365, 730, 1095, 1460, 1825]  # in days
    discount_factors = [0.95, 0.9, 0.85, 0.8, 0.75]
    cds_spreads = [0.01, 0.012, 0.014, 0.016, 0.018]  # in decimal form
    recovery_rate = 0.4

    print("Input Data:")
    print(f"  Dates: {dates}")
    print(f"  Discount Factors: {discount_factors}")
    print(f"  CDS Spreads: {cds_spreads}")
    print(f"  Recovery Rate: {recovery_rate}")
    print("\n")

    # Build the survival curve
    survival_probabilities = build_survival_curve(
        dates, discount_factors, cds_spreads, recovery_rate
    )
    print("Survival Probabilities:")
    for i, date in enumerate(dates):
        print(f"  Year {date/365:.0f}: {survival_probabilities[i+1]:.4f}")
    print("\n")

    # Convert to default probabilities
    default_probabilities = survival_to_default_probabilities(survival_probabilities)
    print("Default Probabilities:")
    for i, date in enumerate(dates):
        print(f"  Year {date/365:.0f}: {default_probabilities[i+1]:.4f}")
    print("\n")

    # Recalculate the credit spread curve
    recalculated_spreads = credit_spread_curve(
        dates, survival_probabilities, discount_factors, recovery_rate
    )
    print("Recalculated Credit Spreads (for verification):")
    for i, date in enumerate(dates):
        print(f"  Year {date/365:.0f}: {recalculated_spreads[i]:.4f}")

if __name__ == "__main__":
    main()
