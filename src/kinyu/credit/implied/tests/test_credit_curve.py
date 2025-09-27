
import unittest
import numpy as np
from kinyu.credit.implied.credit_curve import (
    build_survival_curve,
    survival_to_default_probabilities,
    credit_spread_curve,
)

class TestCreditCurve(unittest.TestCase):

    def test_build_survival_curve(self):
        dates = [365, 730, 1095, 1460, 1825]
        discount_factors = [0.95, 0.9, 0.85, 0.8, 0.75]
        cds_spreads = [0.01, 0.012, 0.014, 0.016, 0.018]
        recovery_rate = 0.4

        survival_probabilities = build_survival_curve(
            dates, discount_factors, cds_spreads, recovery_rate
        )

        # The first survival probability is always 1.0
        self.assertAlmostEqual(survival_probabilities[0], 1.0)

        # The survival probabilities should be decreasing
        for i in range(len(survival_probabilities) - 1):
            self.assertGreaterEqual(survival_probabilities[i], survival_probabilities[i+1])

        # A simple check for a single period
        dt = dates[0] / 365.0
        spread = cds_spreads[0]
        df = discount_factors[0]
        loss_rate = 1.0 - recovery_rate
        # premium_leg = spread * dt * df * (q1 + 1)/2
        # default_leg = loss_rate * df * (1 - q1)
        # premium_leg = default_leg
        # spread * dt * df * (q1 + 1)/2 = loss_rate * df * (1 - q1)
        # spread * dt * (q1 + 1)/2 = loss_rate * (1 - q1)
        # spread * dt * q1 + spread * dt = 2 * loss_rate - 2 * loss_rate * q1
        # q1 * (spread * dt + 2 * loss_rate) = 2 * loss_rate - spread * dt
        # q1 = (2 * loss_rate - spread * dt) / (spread * dt + 2 * loss_rate)
        q1 = (2 * loss_rate - spread * dt) / (spread * dt + 2 * loss_rate)
        self.assertAlmostEqual(survival_probabilities[1], q1, places=6)


    def test_survival_to_default_probabilities(self):
        survival_probabilities = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
        default_probabilities = survival_to_default_probabilities(survival_probabilities)

        self.assertEqual(len(default_probabilities), len(survival_probabilities))
        self.assertAlmostEqual(default_probabilities[0], 0.0)
        for i in range(1, len(survival_probabilities)):
            self.assertAlmostEqual(
                default_probabilities[i],
                survival_probabilities[i-1] - survival_probabilities[i]
            )

    def test_credit_spread_curve(self):
        dates = [365, 730, 1095, 1460, 1825]
        discount_factors = [0.95, 0.9, 0.85, 0.8, 0.75]
        survival_probabilities = [1.0, 0.98, 0.95, 0.92, 0.88, 0.83]
        recovery_rate = 0.4

        spreads = credit_spread_curve(dates, survival_probabilities, discount_factors, recovery_rate)

        # Test that the calculated spreads can be used to rebuild the survival curve
        rebuilt_survival_probabilities = build_survival_curve(
            dates, discount_factors, spreads, recovery_rate
        )

        for i in range(len(survival_probabilities)):
            self.assertAlmostEqual(
                survival_probabilities[i], rebuilt_survival_probabilities[i], places=2
            )

if __name__ == '__main__':
    unittest.main()
