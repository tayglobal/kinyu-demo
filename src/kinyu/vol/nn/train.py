import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from .model import VolatilityModel
from .data import OptionDataset

def arbitrage_penalty(model, inputs):
    """
    Calculates a penalty for arbitrage opportunities by penalizing the
    second derivative of the volatility with respect to moneyness.
    A positive second derivative helps prevent butterfly arbitrage.
    """
    inputs.requires_grad_(True)
    vols = model(inputs)

    # First derivative with respect to moneyness (k)
    first_grad_outputs = torch.ones_like(vols)
    dv_dk = torch.autograd.grad(vols, inputs, grad_outputs=first_grad_outputs, create_graph=True)[0][:, 0]

    # Second derivative with respect to moneyness (k)
    second_grad_outputs = torch.ones_like(dv_dk)
    d2v_dk2 = torch.autograd.grad(dv_dk, inputs, grad_outputs=second_grad_outputs, create_graph=True)[0][:, 0]

    # Penalize non-convexity (negative second derivative)
    penalty = torch.mean(torch.relu(-d2v_dk2))
    return penalty

def train_model(options, spot_price, epochs=100, lr=0.001, penalty_weight=0.01):
    """
    Trains the volatility model using a weighted sampler.

    Args:
        options (list): A list of option data.
        spot_price (float): The current spot price.
        epochs (int): The number of training epochs.
        lr (float): The learning rate.
        penalty_weight (float): The weight for the arbitrage penalty.

    Returns:
        VolatilityModel: The trained model.
    """
    dataset = OptionDataset(options, spot_price)

    # Use a weighted sampler to focus on ATM options
    sampler = WeightedRandomSampler(dataset.weights, len(dataset.weights))
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)

    model = VolatilityModel()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    for epoch in range(epochs):
        for inputs, targets in dataloader:
            optimizer.zero_grad()

            predictions = model(inputs)
            loss = loss_fn(predictions, targets)

            # Add arbitrage penalty
            penalty = arbitrage_penalty(model, inputs)
            total_loss = loss + penalty_weight * penalty

            total_loss.backward()
            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss.item():.4f}, Penalty: {penalty.item():.4f}')

    return model