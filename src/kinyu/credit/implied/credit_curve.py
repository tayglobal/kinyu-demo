
import numpy as np

def build_survival_curve(dates, discount_factors, cds_spreads, recovery_rate):
    """
    Builds a survival curve from a series of CDS spreads.
    Args:
        dates (list): List of dates for the CDS maturities.
        discount_factors (list): List of discount factors for the corresponding dates.
        cds_spreads (list): List of CDS spreads for the corresponding maturities.
        recovery_rate (float): The recovery rate on default.
    Returns:
        list: A list of survival probabilities at each date.
    """
    survival_probabilities = [1.0]
    loss_rate = 1.0 - recovery_rate

    for i in range(len(dates)):
        spread = cds_spreads[i]
        
        # Calculate the known parts of the premium and default legs
        premium_leg_known = 0
        default_leg_known = 0
        for j in range(i):
            dt_j = (dates[j] - (dates[j-1] if j > 0 else 0)) / 365.0
            df_j = discount_factors[j]
            sp_prev_j = survival_probabilities[j]
            sp_curr_j = survival_probabilities[j+1]
            
            premium_leg_known += dt_j * df_j * (sp_prev_j + sp_curr_j) / 2.0
            default_leg_known += df_j * (sp_prev_j - sp_curr_j)

        # Now solve for the current survival probability
        dt_i = (dates[i] - (dates[i-1] if i > 0 else 0)) / 365.0
        df_i = discount_factors[i]
        sp_prev_i = survival_probabilities[i]

        numerator = loss_rate * default_leg_known + loss_rate * df_i * sp_prev_i - spread * premium_leg_known - spread * dt_i * df_i * sp_prev_i / 2.0
        denominator = spread * dt_i * df_i / 2.0 + loss_rate * df_i

        if denominator == 0:
            sp_i = 0.0
        else:
            sp_i = numerator / denominator
            
        survival_probabilities.append(sp_i)

    return survival_probabilities

def survival_to_default_probabilities(survival_probabilities):
    """
    Converts a survival curve to a list of default probabilities.

    Args:
        survival_probabilities (list): A list of survival probabilities.

    Returns:
        list: A list of default probabilities.
    """
    default_probabilities = [0.0]
    for i in range(1, len(survival_probabilities)):
        default_probabilities.append(survival_probabilities[i-1] - survival_probabilities[i])
    return default_probabilities

def credit_spread_curve(dates, survival_probabilities, discount_factors, recovery_rate):
    """
    Calculates the credit spread curve from a survival curve.

    Args:
        dates (list): List of dates.
        survival_probabilities (list): List of survival probabilities.
        discount_factors (list): List of discount factors.
        recovery_rate (float): The recovery rate on default.

    Returns:
        list: A list of credit spreads.
    """
    spreads = []
    loss_rate = 1.0 - recovery_rate
    for i in range(len(dates)):
        numerator = 0
        denominator = 0
        for j in range(i + 1):
            prev_date = dates[j-1] if j > 0 else 0
            dt = (dates[j] - prev_date) / 365.0
            dp = survival_probabilities[j] - survival_probabilities[j+1]
            numerator += loss_rate * dp * discount_factors[j]
            denominator += dt * discount_factors[j] * (survival_probabilities[j+1] + survival_probabilities[j]) / 2
        spreads.append(numerator / denominator if denominator != 0 else 0)
    return spreads
