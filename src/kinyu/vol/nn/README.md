# Neural Network Volatility Surface Calibration

This library provides a PyTorch-based solution for calibrating a volatility surface from a set of option prices. It uses a neural network to learn the complex relationship between moneyness, time to expiration, and implied volatility, while also ensuring the resulting surface is smooth and free of static arbitrage.

## Core Concepts

### 1. Neural Network Model

The core of the library is a simple feed-forward neural network (`VolatilityModel`) that learns the volatility surface. The model takes two inputs:
- **Log-Moneyness:** `log(strike / spot)`
- **Time to Expiration (TTE)**

The network consists of several hidden layers with ReLU activations and a `Softplus` activation on the final output layer. The `Softplus` function ensures that the predicted volatility is always non-negative.

### 2. Data Handling

The `OptionDataset` class handles the preprocessing of option data. It includes two key features:
- **Pruning:** Options that are deep in-the-money or out-of-the-money are pruned based on a configurable `moneyness_range`. This helps focus the model on the most liquid and relevant options.
- **Weighting:** At-the-money (ATM) options are given a higher weight during training. This is achieved using a `WeightedRandomSampler` in PyTorch, which ensures that the model pays closer attention to the most important part of the volatility smile.

### 3. No-Arbitrage Constraint

A key feature of this library is the inclusion of a no-arbitrage penalty in the loss function. Static arbitrage opportunities can arise if the volatility surface is not smooth. Specifically, a non-convex volatility smile can lead to butterfly arbitrage.

To prevent this, we add a penalty term to the loss function that is proportional to the negative part of the second derivative of the volatility with respect to moneyness (`d²v/dk²`). By penalizing non-convexity, the training process is guided towards producing a smooth, arbitrage-free surface.

## Installation

To use this library, you need to install the required Python packages. Ensure you have `torch` installed, which is the core dependency.

1.  **Install requirements:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Ensure `PYTHONPATH` is set correctly:**
    Because the library is part of a larger project structure, you need to set the `PYTHONPATH` to the `src` directory to ensure the modules can be imported correctly.

## Running the Demonstration

A demonstration script, `vol_nn_demonstration.py`, is included to showcase the library's functionality.

To run the demo, execute the following command from the root of the project:
```bash
PYTHONPATH=src python src/kinyu/vol/nn/vol_nn_demonstration.py
```

This script will:
1.  Generate a dummy dataset of option prices with a known volatility smile.
2.  Train the neural network model on this data.
3.  Use the trained model to generate a calibrated volatility surface.
4.  Plot the original data and the calibrated surface for comparison.

## Demonstration Output

The following plot shows the result of the demonstration. On the left is the noisy input data, and on the right is the smooth, arbitrage-free volatility surface learned by the neural network.

![Calibrated Volatility Surface](vol_surface_comparison.png)