import torch
from torch.utils.data import Dataset
import numpy as np

def calculate_moneyness(spot_price, strike_price):
    """
    Calculates the log-moneyness of an option.

    Args:
        spot_price (float): The current price of the underlying asset.
        strike_price (float): The strike price of the option.

    Returns:
        float: The log-moneyness value.
    """
    return np.log(strike_price / spot_price)

class OptionDataset(Dataset):
    """
    A PyTorch Dataset for option data with pruning and weighting.
    """
    def __init__(self, options, spot_price, moneyness_range=(-0.5, 0.5), atm_weight=2.0):
        """
        Initializes the dataset.

        Args:
            options (list): A list of tuples, where each tuple represents an
                            option and contains (strike, tte, vol, put_call_flag).
            spot_price (float): The current spot price of the underlying asset.
            moneyness_range (tuple): The range of moneyness to keep. Options outside
                                     this range are pruned.
            atm_weight (float): The weight to apply to at-the-money options.
        """
        self.inputs = []
        self.targets = []
        self.weights = []

        atm_threshold = 0.05  # Defines the moneyness region for ATM options

        for strike, tte, vol, flag in options:
            moneyness = calculate_moneyness(spot_price, strike)

            # Pruning: Skip options outside the desired moneyness range
            if not (moneyness_range[0] <= moneyness <= moneyness_range[1]):
                continue

            self.inputs.append([moneyness, tte])
            self.targets.append([vol])

            # Weighting: Give more weight to ATM options
            if abs(moneyness) < atm_threshold:
                self.weights.append(atm_weight)
            else:
                self.weights.append(1.0)

        if not self.inputs:
            raise ValueError("No options data left after pruning. Check moneyness_range.")

        self.inputs = torch.tensor(self.inputs, dtype=torch.float32)
        self.targets = torch.tensor(self.targets, dtype=torch.float32)
        self.weights = torch.tensor(self.weights, dtype=torch.float32)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]