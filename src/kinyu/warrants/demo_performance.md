# Warrants Demo Performance Report

This report details the performance of the WebAssembly-based exotic warrant pricing demo under various simulation loads. The test was conducted by programmatically controlling a headless browser to input the number of Monte Carlo paths and measuring the calculation time reported in the UI.

## Test Environment
- **Browser:** Headless Chromium (via Playwright)
- **Machine:** Standard cloud-based virtual machine

## Performance Results

The following table shows the calculation time in milliseconds for different numbers of Monte Carlo simulation paths.

| Number of Paths | Calculation Time (ms) |
|-----------------|-------------------------|
| 1,000           | 113.10                  |
| 5,000           | 611.10                  |
| 10,000          | 1212.50                 |
| 20,000          | 2098.70                 |

## Analysis

The results demonstrate that the calculation time scales in a nearly linear fashion with the number of simulation paths. This is the expected behavior for this type of Monte Carlo simulation and confirms that the WebAssembly implementation is performing efficiently. The performance is excellent, with even large-scale simulations completing in just over two seconds.