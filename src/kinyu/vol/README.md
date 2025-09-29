# Volatility Libraries

This module contains a collection of libraries for volatility modeling, including historical volatility estimation, implied volatility calibration, and neural network-based volatility surfaces.

## Core Libraries

Below is a summary of the core libraries available in this module. For more detailed information, please refer to the README file in each subdirectory.

### 1. [Historical Volatility](./historical/)

The `historical` library provides a set of high-performance historical volatility estimators implemented in Rust with Python bindings. It includes common estimators such as:
- Close-to-Close
- Parkinson (High-Low)
- Garman-Klass
- Rogers-Satchell
- Yang-Zhang

These tools are essential for analyzing past price movements to forecast future volatility.

### 2. [Implied Volatility](./implied/)

The `implied` library offers tools for calibrating a Stochastic Volatility Inspired (SVI) volatility surface from market option prices. Key features include:
- Calibration of the SVI model parameters (`a`, `b`, `rho`, `m`, `sigma`).
- A robust calibration process using the Nelder-Mead optimization algorithm.
- Functionality to query the calibrated surface for implied volatility at any moneyness and expiration.

This library is designed for pricing and risk-managing derivatives.

### 3. [Neural Network Volatility](./nn/)

The `nn` library provides a modern approach to volatility surface calibration using a PyTorch-based neural network. This approach offers several advantages:
- **Flexibility:** The neural network can capture complex patterns in the volatility surface without being tied to a specific parametric form.
- **No-Arbitrage Constraint:** A penalty term is included in the loss function to ensure the resulting surface is smooth and free of static arbitrage opportunities.
- **Weighted Training:** The model gives more importance to at-the-money (ATM) options, which are typically the most liquid and important for calibration.

This library is suitable for creating highly accurate and smooth volatility surfaces for advanced modeling.