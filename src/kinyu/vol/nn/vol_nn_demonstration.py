import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from kinyu.vol.nn.data import OptionDataset
from kinyu.vol.nn.train import train_model
from kinyu.vol.nn.surface import VolatilitySurface

def generate_dummy_data(num_samples=1000):
    """
    Generates a dummy dataset of option prices.
    The volatility surface is modeled with a simple parabolic smile.
    """
    np.random.seed(42)
    spot_price = 100.0

    strikes = np.random.uniform(80, 120, num_samples)
    ttes = np.random.uniform(0.1, 2.0, num_samples)

    # Simple vol smile model: vol = a(k-b)^2 + c
    moneyness = np.log(strikes / spot_price)
    a = 0.2
    b = 0.0
    c = 0.1
    vols = a * (moneyness - b)**2 + c + np.random.normal(0, 0.02, num_samples)

    # Add dummy put/call flags ('C' for call, 'P' for put)
    flags = ['C' if strike > spot_price else 'P' for strike in strikes]

    options = list(zip(strikes, ttes, vols, flags))
    return options, spot_price

def plot_surfaces(options, spot_price, surface):
    """
    Plots the original and calibrated volatility surfaces.
    """
    fig = plt.figure(figsize=(14, 7))

    # Original data
    ax1 = fig.add_subplot(121, projection='3d')
    strikes = [opt[0] for opt in options]
    ttes = [opt[1] for opt in options]
    vols = [opt[2] for opt in options]
    ax1.scatter(strikes, ttes, vols, c='b', marker='.', label='Original Data')
    ax1.set_title('Original Volatility Surface')
    ax1.set_xlabel('Strike Price')
    ax1.set_ylabel('Time to Expiration')
    ax1.set_zlabel('Implied Volatility')
    ax1.legend()

    # Calibrated surface
    ax2 = fig.add_subplot(122, projection='3d')
    strike_grid = np.linspace(80, 120, 20)
    tte_grid = np.linspace(0.1, 2.0, 20)
    X, Y = np.meshgrid(strike_grid, tte_grid)
    Z = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = surface.get_volatility(X[i, j], Y[i, j])

    ax2.plot_surface(X, Y, Z, cmap='viridis', label='Calibrated Surface')
    ax2.set_title('Calibrated Volatility Surface (NN)')
    ax2.set_xlabel('Strike Price')
    ax2.set_ylabel('Time to Expiration')
    ax2.set_zlabel('Implied Volatility')

    plt.tight_layout()
    # Save the plot in the same directory as the script
    import os
    script_dir = os.path.dirname(__file__)
    output_path = os.path.join(script_dir, 'vol_surface_comparison.png')
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.show()


if __name__ == '__main__':
    # 1. Generate Data
    print("Generating dummy option data...")
    options, spot_price = generate_dummy_data(num_samples=2000)

    # 2. Train Model
    print("Training volatility model...")
    # Using more epochs for a better fit
    trained_model = train_model(options, spot_price, epochs=200, lr=0.001, penalty_weight=0.05)

    # 3. Create Surface
    print("Creating volatility surface...")
    vol_surface = VolatilitySurface(trained_model, spot_price)

    # 4. Test a lookup
    strike = 105.0
    tte = 1.0
    vol = vol_surface.get_volatility(strike, tte)
    print(f"Volatility for Strike={strike}, TTE={tte}: {vol:.4f}")

    # 5. Plot results
    print("Plotting results...")
    plot_surfaces(options, spot_price, vol_surface)