import unittest
import torch
import numpy as np
from kinyu.vol.nn.data import calculate_moneyness, OptionDataset

class TestData(unittest.TestCase):

    def test_calculate_moneyness(self):
        self.assertAlmostEqual(calculate_moneyness(100, 110), np.log(1.1))
        self.assertAlmostEqual(calculate_moneyness(100, 100), 0.0)
        self.assertAlmostEqual(calculate_moneyness(100, 90), np.log(0.9))

    def test_option_dataset(self):
        options = [
            (110, 0.5, 0.2, 'C'),
            (100, 0.5, 0.15, 'C'),
            (90, 0.5, 0.25, 'P'),
        ]
        spot_price = 100
        dataset = OptionDataset(options, spot_price)

        self.assertEqual(len(dataset), 3)

        inputs, target = dataset[0]
        self.assertTrue(torch.is_tensor(inputs))
        self.assertTrue(torch.is_tensor(target))

        expected_moneyness = calculate_moneyness(spot_price, 110)
        self.assertAlmostEqual(inputs[0].item(), expected_moneyness)
        self.assertAlmostEqual(inputs[1].item(), 0.5)
        self.assertAlmostEqual(target[0].item(), 0.2)

if __name__ == '__main__':
    unittest.main()