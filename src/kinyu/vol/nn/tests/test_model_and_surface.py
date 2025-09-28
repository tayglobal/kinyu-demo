import unittest
import torch
from kinyu.vol.nn.model import VolatilityModel
from kinyu.vol.nn.surface import VolatilitySurface
from kinyu.vol.nn.train import train_model

class TestModelAndSurface(unittest.TestCase):

    def test_model_creation(self):
        model = VolatilityModel()
        self.assertIsInstance(model, VolatilityModel)

        # Test a forward pass
        input_tensor = torch.tensor([[0.1, 0.5]], dtype=torch.float32)
        output = model(input_tensor)
        self.assertEqual(output.shape, (1, 1))
        self.assertTrue(output.item() >= 0)

    def test_surface_lookup(self):
        model = VolatilityModel()
        spot_price = 100
        surface = VolatilitySurface(model, spot_price)

        vol = surface.get_volatility(110, 0.5)
        self.assertIsInstance(vol, float)
        self.assertTrue(vol >= 0)

    def test_no_butterfly_arbitrage(self):
        """
        Tests for static arbitrage by checking the butterfly spread.
        A non-negative butterfly spread is a condition for no arbitrage.
        """
        # Train a model with the arbitrage penalty enabled
        spot_price = 100.0
        options = [
            (90, 1.0, 0.3, 'P'), (100, 1.0, 0.2, 'P'), (110, 1.0, 0.3, 'C'),
            (95, 0.5, 0.28, 'P'), (100, 0.5, 0.22, 'C'), (105, 0.5, 0.28, 'C'),
        ]
        # Use a lower learning rate and more epochs for stable convergence with the penalty
        model = train_model(options, spot_price, epochs=100, lr=0.005, penalty_weight=2.0)
        surface = VolatilitySurface(model, spot_price)

        # Check the butterfly spread at a specific TTE
        tte = 1.0
        strike_center = 100.0
        strike_wing = 5.0

        k1 = strike_center - strike_wing
        k2 = strike_center
        k3 = strike_center + strike_wing

        # We check the convexity of volatility itself, as a proxy for price convexity
        v1 = surface.get_volatility(k1, tte)
        v2 = surface.get_volatility(k2, tte)
        v3 = surface.get_volatility(k3, tte)

        butterfly_spread = v1 - 2 * v2 + v3
        self.assertTrue(butterfly_spread >= -1e-5, f"Negative butterfly spread found: {butterfly_spread}")

if __name__ == '__main__':
    unittest.main()