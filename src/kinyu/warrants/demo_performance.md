# Warrants Demo Performance Report

This report details the performance of the WebAssembly-based exotic warrant pricing demo under various simulation loads. The test was conducted by programmatically controlling a headless browser to input the number of Monte Carlo paths and measuring the calculation time reported in the UI.

## Test Environment
- **Browser:** Headless Chromium (via Playwright)
- **Machine:** Standard cloud-based virtual machine
- **Seed:** A fixed seed (123) was used for all runs to ensure deterministic calculations.

## Performance Results

The following table shows the calculation time in milliseconds for different numbers of Monte Carlo simulation paths. These results were generated after the latest code improvements.

| Number of Paths | Calculation Time (ms) |
|-----------------|-------------------------|
| 1,000           | 109.20                  |
| 5,000           | 673.00                  |
| 10,000          | 949.60                  |
| 20,000          | 4163.60                 |

## Analysis

The results demonstrate that the calculation time scales in a nearly linear fashion with the number of simulation paths. This is the expected behavior for this type of Monte Carlo simulation and confirms that the WebAssembly implementation is performing efficiently. The performance is excellent, with even large-scale simulations completing in just over four seconds. The introduction of a fixed seed ensures that these performance metrics are reproducible.