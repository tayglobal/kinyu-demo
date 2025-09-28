# Exotic Warrant Pricing - WebAssembly Demo

This demo provides an interactive web interface for pricing exotic warrants using a WebAssembly-compiled version of the Rust pricing engine.

## Features

- **Interactive Parameter Input**: Adjust all warrant pricing parameters through a user-friendly web interface
- **Real-time Calculation**: Fast Monte Carlo pricing using WebAssembly for near-native performance
- **Multiple Presets**: Quick access to different risk scenarios (low risk, high risk, zero correlation, high buyback)
- **Dynamic Curve Management**: Add/remove points from forward rate and credit spread curves
- **Detailed Results**: View calculated price with timing and parameter information

## Quick Start

### Option 1: Using Python HTTP Server (Recommended)

1. **Start the server**:
   ```bash
   cd src/kinyu/warrants
   python3 serve_demo.py
   ```

2. **Open your browser** and navigate to:
   ```
   http://localhost:8000/warrants_demo.html
   ```

### Option 2: Using any HTTP Server

If you have another HTTP server available (like `python -m http.server`, Node.js `http-server`, etc.), you can serve the files from the `src/kinyu/warrants` directory.

**Important**: WebAssembly modules require proper MIME types and CORS headers, so make sure your server is configured correctly.

## Building from Source

If you need to rebuild the WebAssembly module:

```bash
cd src/kinyu/warrants
./build_wasm.sh
```

This will:
1. Check for and install `wasm-pack` if needed
2. Compile the Rust code to WebAssembly
3. Generate JavaScript bindings
4. Create the `pkg/` directory with all necessary files

## Understanding the Parameters

### Basic Parameters
- **Initial Stock Price (S₀)**: Current stock price
- **Strike Discount Factor**: Multiplier for weekly strike resets (e.g., 0.9 = 90% of spot)
- **Buyback Price**: Price at which issuer can call the warrant
- **Time to Maturity**: Years until expiration
- **Volatility (σ)**: Annualized stock price volatility
- **Equity-Credit Correlation**: Correlation between stock and credit shocks
- **Recovery Rate**: Payoff if issuer defaults

### Monte Carlo Parameters
- **Number of Paths**: More paths = more accurate but slower
- **Number of Time Steps**: More steps = more accurate but slower
- **Polynomial Degree**: Degree for Longstaff-Schwartz regression
- **Random Seed**: For reproducible results

### Curves
- **Forward Rate Curve**: Risk-neutral interest rates over time
- **Credit Spread Curve**: Hazard rates (default probabilities) over time

## Preset Scenarios

### Low Risk
- Low credit spreads (1% annual)
- Standard correlation (-0.5)
- Normal buyback price ($15)

### High Risk
- High credit spreads (20% annual)
- Same correlation and other parameters
- Shows impact of credit risk on warrant value

### Zero Correlation
- Same as low risk but with zero equity-credit correlation
- Demonstrates correlation effect

### High Buyback
- Very high buyback price ($1000)
- Makes call feature unlikely to be exercised
- Shows maximum warrant value

## Technical Details

### WebAssembly Implementation
- Simplified Rust implementation without external dependencies
- Custom random number generator for WebAssembly compatibility
- Box-Muller transform for normal distribution generation
- Gaussian elimination for polynomial regression

### Performance
- Typical calculation time: 100-500ms for 5000 paths
- Memory usage: ~10-50MB depending on parameters
- Browser compatibility: Modern browsers with WebAssembly support

### Limitations
- Maximum paths: ~50,000 (browser memory limits)
- Maximum time steps: ~1000 (performance considerations)
- Polynomial degree: 1-5 (numerical stability)

## Troubleshooting

### "WebAssembly module not loading"
- Ensure you're using an HTTP server (not opening file directly)
- Check browser console for CORS errors
- Verify `pkg/` directory exists with all WebAssembly files

### "Calculation taking too long"
- Reduce number of paths (try 1000-2000)
- Reduce number of time steps (try 100-200)
- Lower polynomial degree (try 1-2)

### "Invalid input parameters"
- Check that all numeric fields have valid values
- Ensure curves have at least the minimum required points
- Verify ranges (e.g., correlation between -1 and 1)

## File Structure

```
src/kinyu/warrants/
├── warrants_demo.html          # Main demo interface
├── serve_demo.py              # Python HTTP server
├── build_wasm.sh              # WebAssembly build script
├── pkg/                       # Generated WebAssembly files
│   ├── warrants_wasm.js       # JavaScript bindings
│   ├── warrants_wasm_bg.wasm  # WebAssembly binary
│   └── ...
├── wasm-build/                # WebAssembly build directory
│   ├── Cargo.toml            # WebAssembly-specific Cargo config
│   └── ...
└── src/
    ├── lib.rs                # Original Python/Rust implementation
    └── wasm_lib.rs           # WebAssembly-specific implementation
```

## Browser Compatibility

- **Chrome/Chromium**: Full support
- **Firefox**: Full support  
- **Safari**: Full support (macOS 11+, iOS 14+)
- **Edge**: Full support

## Performance Tips

1. **Start with presets** to understand the tool
2. **Use fewer paths** for quick experimentation (1000-2000)
3. **Increase paths** for final calculations (5000-10000)
4. **Adjust time steps** based on maturity (252 for 1 year is good)
5. **Use polynomial degree 2** for most cases (good balance of accuracy/speed)

## Next Steps

- Try different parameter combinations to understand their impact
- Compare results across different risk scenarios
- Experiment with curve shapes and their effects
- Use the tool for educational purposes or preliminary analysis

For production use, consider the full Python implementation with more sophisticated dependencies and higher precision.
