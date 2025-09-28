# SVI Implied Volatility Library

This library provides a Rust implementation for calibrating a Stochastic Volatility Inspired (SVI) volatility surface from a set of option prices. It includes Python bindings for easy integration into Python-based financial analysis workflows.

## Features

- Calibrate a full SVI volatility surface across multiple expiries.
- Uses the Nelder-Mead optimization algorithm for robust parameter fitting.
- Includes a Black-Scholes pricer and an implied volatility calculator.
- Provides a simple Python API for calibration and querying the volatility surface.

## Installation

This library is built as a Python extension module using `maturin`. To install it, you first need to build the wheel from the project's root directory:

```bash
# From the /app directory
maturin build --release --manifest-path /app/src/kinyu/vol/implied/Cargo.toml -o /app/wheels
```

Then, you can install the generated wheel using `pip`:

```bash
pip install /app/wheels/implied-*.whl
```

## Usage and Demonstration

For a comprehensive guide on how to use the library, please see the demonstration Jupyter Notebook:

[**SVI Calibration Demo.ipynb**](/notebooks/SVI_Calibration_Demo.ipynb)

This notebook covers:

- Generating realistic option data.
- Calibrating the `SVIVolatilitySurface`.
- Querying the surface for volatility values.
- Visualizing the calibrated surface against the input market data.