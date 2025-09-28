# Correlation Matrix Library

This library provides tools for calculating correlation matrices from time series data and for correcting non-positive definite correlation matrices. The core logic is implemented in Rust for performance, with Python bindings for ease of use.

## Features

- **Correlation Matrix Calculation**: Computes the correlation matrix for a set of time series. The time series are automatically aligned by their common dates.
- **Positive Definite Correction**: Includes a function to adjust a non-positive definite matrix to be positive definite by cleaning its eigenvalues.

## How It Works

The library consists of two main parts:

1.  **Rust Core**: The heavy lifting is done in Rust using the `nalgebra` library for efficient matrix operations. The core functions handle time series alignment, correlation calculation, and eigenvalue adjustments.
2.  **Python Wrapper**: The Rust functions are exposed to Python using `pyo3`. A simple Python wrapper in `src/kinyu/corr/__init__.py` makes the functions easily accessible as a standard Python package.

## Installation

To use this library, you need to have Rust and Python installed.

1.  **Install Dependencies**:
    First, install the required Python packages from the project root:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Build the Rust Extension**:
    Next, build and install the Rust extension in editable mode. This command compiles the Rust code and links it to the Python package.
    ```bash
    pip install -e .
    ```

## Running the Demonstration

A demonstration script is included at `src/kinyu/corr/demo.py`. To run it, execute the following command from the project root:

```bash
python src/kinyu/corr/demo.py
```

### Demonstration Output

Here is the expected output from running the demonstration script:

```
--- Correlation Library Demonstration ---

1. Calculating correlation matrix from time series data...

Sample Time Series Data:
{'AssetA': {'2023-01-01': 100.0,
            '2023-01-02': 101.0,
            '2023-01-03': 102.0,
            '2023-01-04': 103.0},
 'AssetB': {'2023-01-01': 200.0,
            '2023-01-02': 202.0,
            '2023-01-03': 204.0,
            '2023-01-04': 206.0},
 'AssetC': {'2023-01-01': 50.0,
            '2023-01-02': 49.0,
            '2023-01-03': 48.0,
            '2023-01-04': 47.0}}

Calculated Correlation Matrix:
[[1.0000000000000002, 1.0000000000000002, -1.0000000000000002],
 [1.0000000000000002, 1.0000000000000002, -1.0000000000000002],
 [-1.0000000000000002, -1.0000000000000002, 1.0000000000000002]]

2. Correcting a non-positive definite matrix...

Original Non-Positive Definite Matrix:
[[1.0, 0.9, 0.2], [0.9, 1.0, 0.9], [0.2, 0.9, 1.0]]

Corrected Positive Definite Matrix (min eigenvalue = 0.01):
[[1.043022480632264, 0.8341892507190934, 0.24302248063226456],
 [0.8341892507190936, 1.1006695722158428, 0.8341892507190936],
 [0.24302248063226461, 0.8341892507190934, 1.0430224806322637]]

--- End of Demonstration ---
```