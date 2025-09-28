import torch
from .model import VolatilityModel
from .data import calculate_moneyness

class VolatilitySurface:
    """
    Represents the calibrated volatility surface.
    """
    def __init__(self, model, spot_price):
        """
        Initializes the volatility surface.

        Args:
            model (VolatilityModel): The trained volatility model.
            spot_price (float): The spot price used during training.
        """
        self.model = model
        self.spot_price = spot_price
        self.model.eval()  # Set the model to evaluation mode

    def get_volatility(self, strike_price, time_to_expiration):
        """
        Gets the implied volatility for a given strike and expiration.

        Args:
            strike_price (float): The strike price.
            time_to_expiration (float): The time to expiration.

        Returns:
            float: The implied volatility.
        """
        moneyness = calculate_moneyness(self.spot_price, strike_price)

        with torch.no_grad():
            input_tensor = torch.tensor([[moneyness, time_to_expiration]], dtype=torch.float32)
            volatility = self.model(input_tensor).item()

        return volatility