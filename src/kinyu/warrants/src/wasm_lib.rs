// WebAssembly-specific implementation with simplified dependencies
use wasm_bindgen::prelude::*;
use js_sys;

// Simple random number generator for WebAssembly
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        // Ensure the seed is non-zero to avoid issues with the LCG
        let state = if seed == 0 { 1 } else { seed };
        Self { state }
    }
    
    fn new_random() -> Self {
        // Use current time and additional randomness for better seed generation
        let time_seed = js_sys::Date::now() as u64;
        // Add some additional randomness by using the time in different ways
        let additional_random = (time_seed ^ (time_seed >> 32)) ^ (time_seed << 16);
        // Add more entropy by using performance counter if available
        let perf_seed = js_sys::Reflect::get(&js_sys::global(), &"performance".into())
            .ok()
            .and_then(|perf| js_sys::Reflect::get(&perf, &"now".into()).ok())
            .and_then(|now| now.as_f64())
            .unwrap_or(0.0) as u64;
        let final_seed = additional_random ^ (perf_seed << 8) ^ (perf_seed >> 8);
        Self { state: final_seed }
    }
    
    fn next(&mut self) -> f64 {
        // Linear congruential generator with better constants
        self.state = self.state.wrapping_mul(6364136223846793005u64).wrapping_add(1442695040888963407u64);
        let result = (self.state >> 32) as f64 / 4294967296.0;
        // Ensure we never return exactly 0 or 1
        if result <= 0.0 { 1e-10 } else if result >= 1.0 { 1.0 - 1e-10 } else { result }
    }
    
    fn normal(&mut self) -> f64 {
        // Box-Muller transform for normal distribution
        let u1 = self.next();
        let u2 = self.next();
        
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// Helper function to interpolate the credit curve linearly
fn interpolate(curve: &[(f64, f64)], time: f64) -> f64 {
    if curve.is_empty() {
        return 0.0;
    }
    if time <= curve[0].0 {
        return curve[0].1;
    }
    if let Some(last_point) = curve.last() {
        if time >= last_point.0 {
            return last_point.1;
        }
    }

    for i in 0..curve.len() - 1 {
        let p1 = curve[i];
        let p2 = curve[i + 1];
        if time >= p1.0 && time <= p2.0 {
            let t1 = p1.0;
            let v1 = p1.1;
            let t2 = p2.0;
            let v2 = p2.1;
            return v1 + (v2 - v1) * (time - t1) / (t2 - t1);
        }
    }
    curve.last().unwrap().1
}

// Simplified Monte Carlo simulation for WebAssembly
fn simulate_paths(
    s0: f64,
    forward_curve: &[(f64, f64)],
    sigma: f64,
    t: f64,
    n_steps: usize,
    n_paths: usize,
    credit_spreads: &[(f64, f64)],
    equity_credit_corr: f64,
    seed: Option<u64>,
) -> (Vec<Vec<f64>>, Vec<f64>) {
    let dt = t / n_steps as f64;
    let mut rng = match seed {
        Some(s) => SimpleRng::new(s),
        None => SimpleRng::new_random(),
    };

    let mut paths = vec![vec![0.0; n_paths]; n_steps + 1];
    let mut default_times = vec![t + 1.0; n_paths];

    for j in 0..n_paths {
        paths[0][j] = s0;
        for i in 1..=n_steps {
            let current_time = i as f64 * dt;
            let z1 = rng.normal();
            let z2 = rng.normal();
            let e_stock = z1;
            let e_credit = equity_credit_corr * z1 + (1.0 - equity_credit_corr * equity_credit_corr).sqrt() * z2;

            let r = interpolate(forward_curve, current_time);
            paths[i][j] = paths[i - 1][j] * ((r - 0.5 * sigma * sigma) * dt + sigma * e_stock * dt.sqrt()).exp();

            if default_times[j] > t {
                let hazard_rate = interpolate(credit_spreads, current_time);
                let prob_default = 1.0 - (-hazard_rate * dt).exp();
                let u_credit = 0.5 * (1.0 + erf(e_credit / 2.0_f64.sqrt()));
                if u_credit < prob_default {
                    default_times[j] = current_time;
                }
            }
        }
    }
    (paths, default_times)
}

// Simple error function approximation
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

// Polynomial regression for LSMC
fn poly_regression(x: &[f64], y: &[f64], degree: usize) -> Option<Vec<f64>> {
    if x.is_empty() {
        return None;
    }
    
    let n = x.len();
    let m = degree + 1;
    
    // Build Vandermonde matrix
    let mut a = vec![vec![0.0; m]; m];
    let mut b = vec![0.0; m];
    
    for i in 0..m {
        for j in 0..m {
            for k in 0..n {
                a[i][j] += x[k].powi((i + j) as i32);
            }
        }
        for k in 0..n {
            b[i] += y[k] * x[k].powi(i as i32);
        }
    }
    
    // Solve using Gaussian elimination
    gaussian_elimination(&mut a, &mut b)
}

fn gaussian_elimination(a: &mut [Vec<f64>], b: &mut [f64]) -> Option<Vec<f64>> {
    let n = a.len();
    
    // Forward elimination
    for i in 0..n {
        // Find pivot
        let mut max_row = i;
        for k in i + 1..n {
            if a[k][i].abs() > a[max_row][i].abs() {
                max_row = k;
            }
        }
        
        if a[max_row][i].abs() < 1e-10 {
            return None; // Singular matrix
        }
        
        // Swap rows
        if max_row != i {
            a.swap(i, max_row);
            b.swap(i, max_row);
        }
        
        // Eliminate
        for k in i + 1..n {
            let factor = a[k][i] / a[i][i];
            for j in i..n {
                a[k][j] -= factor * a[i][j];
            }
            b[k] -= factor * b[i];
        }
    }
    
    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = b[i];
        for j in i + 1..n {
            x[i] -= a[i][j] * x[j];
        }
        x[i] /= a[i][i];
    }
    
    Some(x)
}

// Core pricing function for WebAssembly
fn price_exotic_warrant_core(
    s0: f64,
    strike_discount: f64,
    buyback_price: f64,
    t: f64,
    forward_curve: &[(f64, f64)],
    sigma: f64,
    credit_spreads: &[(f64, f64)],
    equity_credit_corr: f64,
    recovery_rate: f64,
    monthly_exercise_limit: f64,
    n_paths: usize,
    n_steps: usize,
    poly_degree: usize,
    seed: Option<u64>,
) -> f64 {
    // Add some basic validation
    if s0 <= 0.0 || strike_discount <= 0.0 || t <= 0.0 || sigma <= 0.0 || n_paths == 0 || n_steps == 0 {
        return 0.0;
    }
    let dt = t / n_steps as f64;
    let (stock_paths, default_times) = simulate_paths(
        s0, forward_curve, sigma, t, n_steps, n_paths, credit_spreads, equity_credit_corr, seed
    );

    let mut strikes = vec![vec![0.0; n_paths]; n_steps + 1];
    for j in 0..n_paths {
        let mut current_strike = s0 * strike_discount;
        strikes[0][j] = current_strike;
        let mut last_week = -1;
        for i in 1..=n_steps {
            let current_time = i as f64 * dt;
            let current_week = (current_time * 52.0 - 1e-9).floor() as i32;
            if current_week > last_week {
                current_strike = stock_paths[i - 1][j] * strike_discount;
                last_week = current_week;
            }
            strikes[i][j] = current_strike;
        }
    }

    let mut warrant_values = vec![0.0; n_paths];
    for j in 0..n_paths {
        if default_times[j] <= t {
            warrant_values[j] = recovery_rate;
        } else {
            let final_price = stock_paths[n_steps][j];
            let final_strike = strikes[n_steps][j];
            warrant_values[j] = (final_price - final_strike).max(0.0);
        }
    }

    for i in (0..n_steps).rev() {
        let current_time = i as f64 * dt;
        let r = interpolate(forward_curve, current_time);

        let mut x_regression = Vec::new();
        let mut y_regression = Vec::new();

        for j in 0..n_paths {
            if default_times[j] > current_time {
                let stock_price = stock_paths[i][j];
                let strike = strikes[i][j];
                if (stock_price - strike).max(0.0) > 0.0 {
                    x_regression.push(stock_price);
                    y_regression.push(warrant_values[j] * (-r * dt).exp());
                }
            }
        }

        let beta = if !x_regression.is_empty() {
            poly_regression(&x_regression, &y_regression, poly_degree)
        } else {
            None
        };

        for j in 0..n_paths {
            let r = interpolate(forward_curve, current_time);
            let discounted_value = warrant_values[j] * (-r * dt).exp();

            if default_times[j] > current_time {
                let stock_price = stock_paths[i][j];
                let strike = strikes[i][j];

                if (stock_price - strike).max(0.0) > 0.0 {
                    if let Some(ref b) = beta {
                        let mut continuation_value = 0.0;
                        for d in 0..b.len() {
                            continuation_value += b[d] * stock_price.powi(d as i32);
                        }

                        if continuation_value > buyback_price {
                            warrant_values[j] = buyback_price;
                        } else {
                            warrant_values[j] = discounted_value;
                        }
                    } else {
                        warrant_values[j] = discounted_value;
                    }
                } else {
                    warrant_values[j] = discounted_value;
                }
            } else {
                warrant_values[j] = recovery_rate;
            }
        }
    }

    warrant_values.iter().sum::<f64>() / n_paths as f64
}

// WebAssembly wrapper function
#[wasm_bindgen]
pub fn price_exotic_warrant_wasm(
    s0: f64,
    strike_discount: f64,
    buyback_price: f64,
    t: f64,
    forward_curve: &[f64], // Flattened array: [t1, r1, t2, r2, ...]
    sigma: f64,
    credit_spreads: &[f64], // Flattened array: [t1, s1, t2, s2, ...]
    equity_credit_corr: f64,
    recovery_rate: f64,
    monthly_exercise_limit: f64,
    n_paths: usize,
    n_steps: usize,
    poly_degree: usize,
    seed: Option<u64>,
) -> f64 {
    // Convert flattened arrays to Vec<(f64, f64)>
    let forward_curve_vec: Vec<(f64, f64)> = forward_curve
        .chunks(2)
        .map(|chunk| (chunk[0], chunk[1]))
        .collect();
    
    let credit_spreads_vec: Vec<(f64, f64)> = credit_spreads
        .chunks(2)
        .map(|chunk| (chunk[0], chunk[1]))
        .collect();
    
    // Debug: log the seed value
    let seed_value = match seed {
        Some(s) => s,
        None => 0,
    };
    
    price_exotic_warrant_core(
        s0, strike_discount, buyback_price, t, &forward_curve_vec, sigma,
        &credit_spreads_vec, equity_credit_corr, recovery_rate, monthly_exercise_limit,
        n_paths, n_steps, poly_degree, seed
    )
}

// Simple test function to verify WebAssembly is working
#[wasm_bindgen]
pub fn test_wasm() -> f64 {
    42.0
}

// Simple test calculation
#[wasm_bindgen]
pub fn simple_test_calculation() -> f64 {
    let s0 = 100.0;
    let strike_discount = 0.9;
    let buyback_price = 15.0;
    let t = 1.0;
    let forward_curve = vec![(0.0, 0.05), (1.0, 0.05)];
    let sigma = 0.2;
    let credit_spreads = vec![(0.0, 0.01), (1.0, 0.01)]; // Start at time 0
    let equity_credit_corr = -0.5;
    let recovery_rate = 0.4;
    let monthly_exercise_limit = 1.0;
    let n_paths = 1000; // Smaller for testing
    let n_steps = 100;  // Smaller for testing
    let poly_degree = 2;
    
    price_exotic_warrant_core(
        s0, strike_discount, buyback_price, t, &forward_curve, sigma,
        &credit_spreads, equity_credit_corr, recovery_rate, monthly_exercise_limit,
        n_paths, n_steps, poly_degree, None
    )
}

// Debug function to test individual components
#[wasm_bindgen]
pub fn debug_calculation() -> f64 {
    let s0 = 100.0;
    let strike_discount = 0.9;
    let buyback_price = 15.0;
    let t = 1.0;
    let forward_curve = vec![(0.0, 0.05), (1.0, 0.05)];
    let sigma = 0.2;
    let credit_spreads = vec![(0.0, 0.01), (1.0, 0.01)];
    let equity_credit_corr = -0.5;
    let recovery_rate = 0.4;
    let n_paths = 10; // Very small for debugging
    let n_steps = 10; // Very small for debugging
    let poly_degree = 2;
    
    // Test interpolation
    let r0 = interpolate(&forward_curve, 0.0);
    let r1 = interpolate(&forward_curve, 0.5);
    let r2 = interpolate(&forward_curve, 1.0);
    
    // Test credit spread interpolation
    let s0_val = interpolate(&credit_spreads, 0.0);
    let s1_val = interpolate(&credit_spreads, 0.5);
    let s2_val = interpolate(&credit_spreads, 1.0);
    
    // Test random generation
    let mut rng = SimpleRng::new_random();
    let rand1 = rng.normal();
    let rand2 = rng.normal();
    
    // Return a combination of test values to see if they're reasonable
    r0 + r1 + r2 + s0_val + s1_val + s2_val + rand1 + rand2
}

// Very simple test - just return a basic calculation
#[wasm_bindgen]
pub fn simple_option_test() -> f64 {
    let s0: f64 = 100.0;
    let strike: f64 = 90.0; // strike_discount * s0
    let payoff = if s0 > strike { s0 - strike } else { 0.0 };
    payoff
}

// Test the core calculation with minimal parameters
#[wasm_bindgen]
pub fn minimal_test() -> f64 {
    // Just test if the core function can run without crashing
    let s0 = 100.0;
    let strike_discount = 0.9;
    let buyback_price = 15.0;
    let t = 0.1; // Very short time
    let forward_curve = vec![(0.0, 0.05), (0.1, 0.05)];
    let sigma = 0.2;
    let credit_spreads = vec![(0.0, 0.001), (0.1, 0.001)]; // Very low credit risk
    let equity_credit_corr = 0.0; // No correlation
    let recovery_rate = 0.4;
    let monthly_exercise_limit = 1.0;
    let n_paths = 10; // Very small
    let n_steps = 2;  // Very small
    let poly_degree = 1; // Simple polynomial
    
    price_exotic_warrant_core(
        s0, strike_discount, buyback_price, t, &forward_curve, sigma,
        &credit_spreads, equity_credit_corr, recovery_rate, monthly_exercise_limit,
        n_paths, n_steps, poly_degree, None
    )
}

// Debug function to test random number generation
#[wasm_bindgen]
pub fn test_random_generation() -> f64 {
    let mut rng = SimpleRng::new_random();
    let mut sum = 0.0;
    for _ in 0..1000 {
        sum += rng.normal();
    }
    sum / 1000.0 // Should be close to 0
}

// Test function to verify random seed generation produces different results
#[wasm_bindgen]
pub fn test_random_seed_variation() -> f64 {
    let mut rng1 = SimpleRng::new_random();
    let mut rng2 = SimpleRng::new_random();
    
    // Generate a few random numbers from each RNG
    let val1 = rng1.normal();
    let val2 = rng2.normal();
    
    // Return the difference to show they're different
    (val1 - val2).abs()
}

// Test function to demonstrate different prices with different seeds
#[wasm_bindgen]
pub fn test_price_variation_with_seeds() -> f64 {
    let s0 = 100.0;
    let strike_discount = 0.9;
    let buyback_price = 15.0;
    let t = 1.0;
    let forward_curve = vec![(0.0, 0.05), (1.0, 0.05)];
    let sigma = 0.2;
    let credit_spreads = vec![(0.0, 0.01), (1.0, 0.01)];
    let equity_credit_corr = -0.5;
    let recovery_rate = 0.4;
    let monthly_exercise_limit = 1.0;
    let n_paths = 1000;
    let n_steps = 100;
    let poly_degree = 2;
    
    // Calculate price with seed 12345
    let price1 = price_exotic_warrant_core(
        s0, strike_discount, buyback_price, t, &forward_curve, sigma,
        &credit_spreads, equity_credit_corr, recovery_rate, monthly_exercise_limit,
        n_paths, n_steps, poly_degree, Some(12345)
    );
    
    // Calculate price with seed 54321
    let price2 = price_exotic_warrant_core(
        s0, strike_discount, buyback_price, t, &forward_curve, sigma,
        &credit_spreads, equity_credit_corr, recovery_rate, monthly_exercise_limit,
        n_paths, n_steps, poly_degree, Some(54321)
    );
    
    // Return the difference to show they're different
    (price1 - price2).abs()
}

// Test function to verify seed is working by testing random number generation directly
#[wasm_bindgen]
pub fn test_seed_functionality() -> f64 {
    // Test with two different seeds
    let mut rng1 = SimpleRng::new(12345);
    let mut rng2 = SimpleRng::new(54321);
    
    // Generate a few random numbers from each
    let val1 = rng1.normal();
    let val2 = rng2.normal();
    
    // Return the difference - should be non-zero if seeds work
    (val1 - val2).abs()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolate_basic() {
        let curve = vec![(0.0, 0.01), (1.0, 0.02), (2.0, 0.03)];
        
        // Test exact matches
        assert!((interpolate(&curve, 0.0) - 0.01).abs() < 1e-9);
        assert!((interpolate(&curve, 1.0) - 0.02).abs() < 1e-9);
        assert!((interpolate(&curve, 2.0) - 0.03).abs() < 1e-9);
        
        // Test interpolation
        assert!((interpolate(&curve, 0.5) - 0.015).abs() < 1e-9);
        assert!((interpolate(&curve, 1.5) - 0.025).abs() < 1e-9);
        
        // Test extrapolation
        assert!((interpolate(&curve, -1.0) - 0.01).abs() < 1e-9);
        assert!((interpolate(&curve, 3.0) - 0.03).abs() < 1e-9);
    }

    #[test]
    fn test_interpolate_empty_curve() {
        let curve = vec![];
        assert!((interpolate(&curve, 0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_interpolate_single_point() {
        let curve = vec![(1.0, 0.05)];
        assert!((interpolate(&curve, 0.0) - 0.05).abs() < 1e-9);
        assert!((interpolate(&curve, 1.0) - 0.05).abs() < 1e-9);
        assert!((interpolate(&curve, 2.0) - 0.05).abs() < 1e-9);
    }

    #[test]
    fn test_simple_rng_seed() {
        let mut rng1 = SimpleRng::new(12345);
        let mut rng2 = SimpleRng::new(12345);
        
        // Same seed should produce same sequence
        let val1_1 = rng1.next();
        let val2_1 = rng2.next();
        assert!((val1_1 - val2_1).abs() < 1e-9);
        
        let val1_2 = rng1.next();
        let val2_2 = rng2.next();
        assert!((val1_2 - val2_2).abs() < 1e-9);
    }

    #[test]
    fn test_simple_rng_different_seeds() {
        let mut rng1 = SimpleRng::new(12345);
        let mut rng2 = SimpleRng::new(54321);
        
        // Different seeds should produce different values
        let val1 = rng1.next();
        let val2 = rng2.next();
        assert!((val1 - val2).abs() > 1e-9);
    }

    #[test]
    fn test_simple_rng_range() {
        let mut rng = SimpleRng::new(42);
        
        // Generate many values and check they're in [0, 1)
        for _ in 0..1000 {
            let val = rng.next();
            assert!(val >= 0.0);
            assert!(val < 1.0);
        }
    }

    #[test]
    fn test_simple_rng_normal_distribution() {
        let mut rng = SimpleRng::new(42);
        let mut sum = 0.0;
        let n = 10000;
        
        // Generate many normal values and check mean is close to 0
        for _ in 0..n {
            sum += rng.normal();
        }
        
        let mean = sum / n as f64;
        assert!(mean.abs() < 0.1); // Should be close to 0
    }

    #[test]
    fn test_erf_function() {
        // Test some known values of the error function
        assert!((erf(0.0) - 0.0).abs() < 1e-9);
        assert!(erf(1.0) > 0.8);
        assert!(erf(-1.0) < -0.8);
        assert!(erf(2.0) > 0.99);
        assert!(erf(-2.0) < -0.99);
    }

    #[test]
    fn test_poly_regression_simple() {
        // Test with a simple quadratic: y = 2 + 3x + 4x^2
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![9.0, 24.0, 47.0, 78.0, 117.0]; // 2 + 3*1 + 4*1^2 = 9, etc.
        
        let result = poly_regression(&x, &y, 2);
        assert!(result.is_some());
        
        let beta = result.unwrap();
        assert_eq!(beta.len(), 3);
        
        // Check coefficients are close to expected values
        assert!((beta[0] - 2.0).abs() < 0.1);
        assert!((beta[1] - 3.0).abs() < 0.1);
        assert!((beta[2] - 4.0).abs() < 0.1);
    }

    #[test]
    fn test_poly_regression_empty() {
        let x = vec![];
        let y = vec![];
        
        let result = poly_regression(&x, &y, 2);
        assert!(result.is_none());
    }

    #[test]
    fn test_poly_regression_linear() {
        // Test with a simple linear: y = 1 + 2x
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![3.0, 5.0, 7.0];
        
        let result = poly_regression(&x, &y, 1);
        assert!(result.is_some());
        
        let beta = result.unwrap();
        assert_eq!(beta.len(), 2);
        assert!((beta[0] - 1.0).abs() < 0.1);
        assert!((beta[1] - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_price_exotic_warrant_core_basic() {
        let s0 = 100.0;
        let strike_discount = 0.9;
        let buyback_price = 15.0;
        let t = 0.1; // Short time for fast test
        let forward_curve = vec![(0.0, 0.05), (0.1, 0.05)];
        let sigma = 0.2;
        let credit_spreads = vec![(0.0, 0.001), (0.1, 0.001)];
        let equity_credit_corr = 0.0;
        let recovery_rate = 0.4;
        let monthly_exercise_limit = 1.0;
        let n_paths = 100; // Small for fast test
        let n_steps = 10;
        let poly_degree = 1;
        
        let price = price_exotic_warrant_core(
            s0, strike_discount, buyback_price, t, &forward_curve, sigma,
            &credit_spreads, equity_credit_corr, recovery_rate, monthly_exercise_limit,
            n_paths, n_steps, poly_degree, Some(42)
        );
        
        // Basic sanity checks
        assert!(price >= 0.0, "Price should be non-negative");
        assert!(price < 100.0, "Price should be reasonable");
    }

    #[test]
    fn test_price_exotic_warrant_core_invalid_inputs() {
        let forward_curve = vec![(0.0, 0.05), (1.0, 0.05)];
        let credit_spreads = vec![(0.0, 0.01), (1.0, 0.01)];
        
        // Test with invalid inputs - should return 0
        let price1 = price_exotic_warrant_core(
            0.0, 0.9, 15.0, 1.0, &forward_curve, 0.2, &credit_spreads, 0.0, 0.4, 1.0, 100, 10, 2, None
        );
        assert_eq!(price1, 0.0);
        
        let price2 = price_exotic_warrant_core(
            100.0, 0.0, 15.0, 1.0, &forward_curve, 0.2, &credit_spreads, 0.0, 0.4, 1.0, 100, 10, 2, None
        );
        assert_eq!(price2, 0.0);
        
        let price3 = price_exotic_warrant_core(
            100.0, 0.9, 15.0, 0.0, &forward_curve, 0.2, &credit_spreads, 0.0, 0.4, 1.0, 100, 10, 2, None
        );
        assert_eq!(price3, 0.0);
        
        let price4 = price_exotic_warrant_core(
            100.0, 0.9, 15.0, 1.0, &forward_curve, 0.0, &credit_spreads, 0.0, 0.4, 1.0, 100, 10, 2, None
        );
        assert_eq!(price4, 0.0);
        
        let price5 = price_exotic_warrant_core(
            100.0, 0.9, 15.0, 1.0, &forward_curve, 0.2, &credit_spreads, 0.0, 0.4, 1.0, 0, 10, 2, None
        );
        assert_eq!(price5, 0.0);
        
        let price6 = price_exotic_warrant_core(
            100.0, 0.9, 15.0, 1.0, &forward_curve, 0.2, &credit_spreads, 0.0, 0.4, 1.0, 100, 0, 2, None
        );
        assert_eq!(price6, 0.0);
    }

    #[test]
    fn test_price_exotic_warrant_core_deterministic() {
        let s0 = 100.0;
        let strike_discount = 0.9;
        let buyback_price = 15.0;
        let t = 0.1;
        let forward_curve = vec![(0.0, 0.05), (0.1, 0.05)];
        let sigma = 0.2;
        let credit_spreads = vec![(0.0, 0.001), (0.1, 0.001)];
        let equity_credit_corr = 0.0;
        let recovery_rate = 0.4;
        let monthly_exercise_limit = 1.0;
        let n_paths = 100;
        let n_steps = 10;
        let poly_degree = 1;
        let seed = Some(12345);
        
        // Run the same calculation twice with the same seed
        let price1 = price_exotic_warrant_core(
            s0, strike_discount, buyback_price, t, &forward_curve, sigma,
            &credit_spreads, equity_credit_corr, recovery_rate, monthly_exercise_limit,
            n_paths, n_steps, poly_degree, seed
        );
        
        let price2 = price_exotic_warrant_core(
            s0, strike_discount, buyback_price, t, &forward_curve, sigma,
            &credit_spreads, equity_credit_corr, recovery_rate, monthly_exercise_limit,
            n_paths, n_steps, poly_degree, seed
        );
        
        // Should be identical with the same seed
        assert!((price1 - price2).abs() < 1e-9);
    }

    #[test]
    fn test_wasm_bindgen_functions() {
        // Test simple option calculation (doesn't use js-sys)
        let option_price = simple_option_test();
        assert_eq!(option_price, 10.0); // 100 - 90 = 10
        
        // Test seed functionality (doesn't use js-sys)
        let seed_func_test = test_seed_functionality();
        assert!(seed_func_test > 0.0); // Should be different with different seeds
    }

    #[test]
    fn test_price_variation_with_seeds() {
        // Test that different seeds produce different results by calling the function directly
        let s0 = 100.0;
        let strike_discount = 0.9;
        let buyback_price = 15.0;
        let t = 0.1;
        let forward_curve = vec![(0.0, 0.05), (0.1, 0.05)];
        let sigma = 0.2;
        let credit_spreads = vec![(0.0, 0.001), (0.1, 0.001)];
        let equity_credit_corr = 0.0;
        let recovery_rate = 0.4;
        let monthly_exercise_limit = 1.0;
        let n_paths = 100;
        let n_steps = 10;
        let poly_degree = 1;
        
        let price1 = price_exotic_warrant_core(
            s0, strike_discount, buyback_price, t, &forward_curve, sigma,
            &credit_spreads, equity_credit_corr, recovery_rate, monthly_exercise_limit,
            n_paths, n_steps, poly_degree, Some(12345)
        );
        
        let price2 = price_exotic_warrant_core(
            s0, strike_discount, buyback_price, t, &forward_curve, sigma,
            &credit_spreads, equity_credit_corr, recovery_rate, monthly_exercise_limit,
            n_paths, n_steps, poly_degree, Some(54321)
        );
        
        let price_diff = (price1 - price2).abs();
        assert!(price_diff > 0.0);
    }

    #[test]
    fn test_minimal_calculation() {
        // Test minimal calculation without calling WASM functions that use js-sys
        let s0 = 100.0;
        let strike_discount = 0.9;
        let buyback_price = 15.0;
        let t = 0.1;
        let forward_curve = vec![(0.0, 0.05), (0.1, 0.05)];
        let sigma = 0.2;
        let credit_spreads = vec![(0.0, 0.001), (0.1, 0.001)];
        let equity_credit_corr = 0.0;
        let recovery_rate = 0.4;
        let monthly_exercise_limit = 1.0;
        let n_paths = 10;
        let n_steps = 2;
        let poly_degree = 1;
        
        let price = price_exotic_warrant_core(
            s0, strike_discount, buyback_price, t, &forward_curve, sigma,
            &credit_spreads, equity_credit_corr, recovery_rate, monthly_exercise_limit,
            n_paths, n_steps, poly_degree, Some(42)
        );
        
        assert!(price >= 0.0);
        assert!(price.is_finite());
    }

    #[test]
    fn test_simple_test_calculation() {
        // Test simple calculation without calling WASM functions that use js-sys
        let s0 = 100.0;
        let strike_discount = 0.9;
        let buyback_price = 15.0;
        let t = 1.0;
        let forward_curve = vec![(0.0, 0.05), (1.0, 0.05)];
        let sigma = 0.2;
        let credit_spreads = vec![(0.0, 0.01), (1.0, 0.01)];
        let equity_credit_corr = -0.5;
        let recovery_rate = 0.4;
        let monthly_exercise_limit = 1.0;
        let n_paths = 1000;
        let n_steps = 100;
        let poly_degree = 2;
        
        let price = price_exotic_warrant_core(
            s0, strike_discount, buyback_price, t, &forward_curve, sigma,
            &credit_spreads, equity_credit_corr, recovery_rate, monthly_exercise_limit,
            n_paths, n_steps, poly_degree, Some(123)
        );
        
        assert!(price >= 0.0);
        assert!(price.is_finite());
        assert!(price < 100.0); // Should be reasonable
    }
}
