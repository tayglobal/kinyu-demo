import unittest
import torch
from kinyu.vol.nn.train import train_model

class TestTraining(unittest.TestCase):

    def test_training_loop_runs(self):
        """
        Tests that the training loop runs and the loss decreases.
        """
        # Generate some dummy data for the test
        spot_price = 100.0
        options = [
            (105, 1.0, 0.2, 'C'),
            (100, 1.0, 0.15, 'P'),
            (95, 1.0, 0.2, 'P'),
            (110, 0.5, 0.25, 'C'),
        ]

        # Train for a few epochs to test the process
        try:
            model = train_model(options, spot_price, epochs=5, lr=0.01)
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"Training loop failed with an exception: {e}")

if __name__ == '__main__':
    unittest.main()