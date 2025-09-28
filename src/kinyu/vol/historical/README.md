# Kinyu Historical Volatility Library

This library provides a set of tools for calculating historical volatility using several common estimators. It is written in Rust for performance and includes Python bindings for ease of use.

## Installation

### Prerequisites
- Rust toolchain (install via [rustup](https://rustup.rs/))
- Python 3.8+
- `numpy`, `setuptools`, `setuptools-rust`

### Building the Library
The library is part of a larger Python project and should be installed from the project's root directory.

1.  Navigate to the root of the `kinyu` project:
    ```bash
    cd /app
    ```
2.  Install the package in editable mode. This will build the Rust extension and make it available in your Python environment.
    ```bash
    pip install -e .
    ```
    Alternatively, to build a wheel and install it:
    ```bash
    python setup.py bdist_wheel
    pip install --force-reinstall dist/kinyu-*.whl
    ```

## Usage

### Running the Rust Tests
To verify the correctness of the core Rust implementation, you can run the unit tests:
```bash
cd /app/src/kinyu/vol/historical
cargo test
```

### Running the Python Demo
A Python demo script is included to demonstrate the library's functionality. To run it, execute the following command from the project root:
```bash
python /app/src/kinyu/vol/historical/demo.py
```

## Historical Volatility Estimators

This library implements the following historical volatility estimators:

### 1. Close-to-Close
The most basic estimator. It is calculated as the standard deviation of the natural logarithm of the closing price returns. While simple, it is often inefficient as it ignores intraday price action.

### 2. Parkinson (1980)
The Parkinson estimator uses the high and low prices of the day to estimate volatility. It is more efficient than the close-to-close method, but it assumes a lognormal price process with no drift and no opening price jumps.

### 3. Garman-Klass (1980)
This estimator extends the Parkinson model by incorporating the opening and closing prices, making it even more efficient. It is also based on the assumption of a no-drift lognormal process. Under certain conditions (e.g., low volatility with opening/closing gaps), this estimator can produce a negative variance, in which case this implementation returns a volatility of `0.0`.

### 4. Rogers-Satchell (1991)
The Rogers-Satchell estimator is notable because it is independent of the asset's drift, making it more robust when prices exhibit a trend. It incorporates open, high, low, and close prices.

### 5. Yang-Zhang (2000)
Considered a benchmark estimator, the Yang-Zhang model is a weighted average of the Rogers-Satchell estimator, the close-to-open volatility (overnight gap), and the open-to-close volatility. It has a minimum estimation error and is independent of drift.

## Python Demo Output

Here is the sample output from running the `demo.py` script:

```
--- Historical Volatility Estimation Demo ---
Using 5 periods of OHLC data.

Results (Annualized):
-------------------------
Close-to-Close      : 0.3865
Parkinson           : 0.1600
Garman-Klass        : 0.0000
Rogers-Satchell     : 0.1703
Yang-Zhang          : 0.4018
-------------------------
```