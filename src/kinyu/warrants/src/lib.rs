use nalgebra::{DMatrix, DVector};
use rand::distributions::Distribution;
use statrs::distribution::Normal;
use rand::rngs::StdRng;
use rand::SeedableRng;
use pyo3::prelude::*;

// Function to simulate stock price paths using Geometric Brownian Motion
fn simulate_gbm(
    s0: f64,
    mu: f64,
    sigma: f64,
    t: f64,
    n_steps: usize,
    n_paths: usize,
    seed: u64,
) -> DMatrix<f64> {
    let dt = t / n_steps as f64;
    let mut rng = StdRng::seed_from_u64(seed);
    // Corrected Normal distribution for GBM
    let normal = Normal::new(0.0, 1.0).unwrap();

    let mut paths = DMatrix::from_element(n_steps + 1, n_paths, 0.0);

    for j in 0..n_paths {
        paths[(0, j)] = s0;
        for i in 1..=n_steps {
            let z = normal.sample(&mut rng);
            paths[(i, j)] = paths[(i - 1, j)] * ((mu - 0.5 * sigma.powi(2)) * dt + sigma * z * dt.sqrt()).exp();
        }
    }

    paths
}

// Polynomial regression for LSMC
fn poly_regression(x: &DVector<f64>, y: &DVector<f64>, degree: usize) -> Option<DVector<f64>> {
    let mut x_matrix = DMatrix::from_element(x.len(), degree + 1, 1.0);
    for i in 1..=degree {
        for j in 0..x.len() {
            x_matrix[(j, i)] = x[j].powi(i as i32);
        }
    }
    let xt = x_matrix.transpose();
    let xtx = xt.clone() * x_matrix;

    if let Some(inv_xtx) = xtx.try_inverse() {
        let beta = inv_xtx * xt * y;
        Some(beta)
    } else {
        None
    }
}

#[pyfunction]
fn price_exotic_warrant(
    s0: f64,
    strike_discount: f64,
    buyback_price: f64,
    t: f64,
    r: f64,
    sigma: f64,
    n_paths: usize,
    n_steps: usize,
    poly_degree: usize,
    seed: u64,
) -> PyResult<f64> {
    let dt = t / n_steps as f64;
    let stock_paths = simulate_gbm(s0, r, sigma, t, n_steps, n_paths, seed);

    let mut strikes = DMatrix::from_element(n_steps + 1, n_paths, 0.0);
    for j in 0..n_paths {
        let mut current_strike = s0 * strike_discount;
        strikes[(0, j)] = current_strike;
        let mut last_week = -1;
        for i in 1..=n_steps {
            let current_time = i as f64 * dt;
            let current_week = (current_time * 52.0 - 1e-9).floor() as i32;
            if current_week > last_week {
                current_strike = stock_paths[(i - 1, j)] * strike_discount;
                last_week = current_week;
            }
            strikes[(i, j)] = current_strike;
        }
    }

    let mut warrant_values = DVector::from_fn(n_paths, |j, _| {
        let final_price = stock_paths[(n_steps, j)];
        let final_strike = strikes[(n_steps, j)];
        (final_price - final_strike).max(0.0)
    });

    for i in (1..n_steps).rev() {
        warrant_values = warrant_values.map(|v| v * (-r * dt).exp());

        let x_vec = stock_paths.row(i).transpose().clone_owned();

        let beta = match poly_regression(&x_vec, &warrant_values, poly_degree) {
            Some(b) => b,
            None => continue, // Skip if regression fails
        };

        for j in 0..n_paths {
            let mut continuation_value = 0.0;
            let stock_price = x_vec[j];
            for d in 0..=poly_degree {
                continuation_value += beta[d] * stock_price.powi(d as i32);
            }

            if continuation_value > buyback_price {
                warrant_values[j] = buyback_price;
            }
        }
    }

    let price = warrant_values.mean() * (-r * dt).exp();
    Ok(price)
}

#[pymodule]
fn warrants(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(price_exotic_warrant, m)?)?;
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;

    #[test]
    fn test_gbm_simulation() {
        let s0 = 100.0;
        let mu = 0.05;
        let sigma = 0.2;
        let t = 1.0;
        let n_steps = 100;
        let n_paths = 10000;
        let seed = 123;

        let paths = simulate_gbm(s0, mu, sigma, t, n_steps, n_paths, seed);
        let final_prices = paths.row(n_steps);
        let mean_final_price = final_prices.mean();
        let expected_final_price = s0 * (mu * t).exp();

        // Check if the mean of simulated prices is close to the expected price
        assert!((mean_final_price - expected_final_price).abs() < 2.0);
    }

    #[test]
    fn test_poly_regression() {
        // y = 2 + 3x + 4x^2
        let x = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = DVector::from_vec(vec![9.0, 24.0, 47.0, 78.0, 117.0]);
        let degree = 2;

        let beta = poly_regression(&x, &y, degree).unwrap();

        assert!(beta.len() == 3);
        assert!((beta[0] - 2.0).abs() < 1e-9); // intercept
        assert!((beta[1] - 3.0).abs() < 1e-9); // coeff for x
        assert!((beta[2] - 4.0).abs() < 1e-9); // coeff for x^2
    }

    #[test]
    fn test_warrant_pricing_low_buyback() {
        // If buyback_price is very low, issuer should call immediately.
        // Price should be close to buyback_price, discounted.
        let price = price_exotic_warrant(
            100.0, 0.9, 0.1, 1.0, 0.05, 0.2, 1000, 100, 3, 42
        ).unwrap();
        let dt = 1.0 / 100.0;
        let expected = 0.1 * (-0.05f64 * dt).exp();
        assert!((price - expected).abs() < 0.1);
    }

    #[test]
    fn test_warrant_pricing_high_buyback() {
        // If buyback_price is very high, it should never be called.
        // Price should be greater than zero (for a typical case).
        let price = price_exotic_warrant(
            100.0, 0.9, 1000.0, 1.0, 0.05, 0.2, 5000, 200, 3, 42
        ).unwrap();
        assert!(price > 0.0);
    }

    #[test]
    fn test_warrant_pricing_zero_sigma() {
        // If sigma is 0, price path is deterministic.
        let s0 = 100.0;
        let r = 0.05;
        let t = 1.0;
        let n_steps = 52;
        let dt = t / n_steps as f64;
        let strike_discount = 0.9;

        let mut s_t = s0;
        let mut strike = s0 * strike_discount;
        let mut last_week = -1;

        for i in 1..=n_steps {
            s_t *= (r * dt).exp();
            let current_time = i as f64 * dt;
            let current_week = (current_time * 52.0 - 1e-9).floor() as i32;
            if current_week > last_week {
                strike = s0 * (r * (i-1) as f64 * dt).exp() * strike_discount;
                last_week = current_week;
            }
        }

        let expected_payoff = (s_t - strike).max(0.0);
        let expected_price = expected_payoff * (-r * t).exp();

        let price = price_exotic_warrant(
            s0, strike_discount, 1000.0, t, r, 0.0, 1, n_steps, 1, 42
        ).unwrap();

        // With n_paths=1 and sigma=0, there's no randomness.
        assert!((price - expected_price).abs() < 1e-9);
    }
}