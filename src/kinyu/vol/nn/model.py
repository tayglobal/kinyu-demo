import torch
import torch.nn as nn

class VolatilityModel(nn.Module):
    """
    A neural network model for the volatility surface.
    """
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=1):
        """
        Initializes the model.

        Args:
            input_dim (int): The input dimension (e.g., moneyness, time to expiration).
            hidden_dim (int): The dimension of the hidden layers.
            output_dim (int): The output dimension (implied volatility).
        """
        super(VolatilityModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus()  # Use Softplus for a smooth, non-negative output
        )

    def forward(self, x):
        """
        Performs the forward pass.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: The output tensor of shape (batch_size, output_dim).
        """
        return self.network(x)