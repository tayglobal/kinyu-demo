use nalgebra::{DMatrix, DVector, Cholesky, Matrix2};
use rand::distributions::Distribution;
use statrs::distribution::{Normal, ContinuousCDF};
use rand::rngs::StdRng;
use rand::SeedableRng;
use pyo3::prelude::*;

// Helper function to interpolate the credit curve linearly
fn interpolate(curve: &Vec<[f64; 2]>, time: f64) -> f64 {
    if curve.is_empty() {
        return 0.0;
    }
    if time <= curve[0][0] {
        return curve[0][1];
    }
    if let Some(last_point) = curve.last() {
        if time >= last_point[0] {
            return last_point[1];
        }
    }

    for i in 0..curve.len() - 1 {
        let p1 = curve[i];
        let p2 = curve[i + 1];
        if time >= p1[0] && time <= p2[0] {
            let t1 = p1[0];
            let v1 = p1[1];
            let t2 = p2[0];
            let v2 = p2[1];
            return v1 + (v2 - v1) * (time - t1) / (t2 - t1);
        }
    }
    curve.last().unwrap()[1]
}

// Function to simulate correlated stock price paths and default times
fn simulate_correlated_paths(
    s0: f64,
    forward_curve: &Vec<[f64; 2]>,
    sigma: f64,
    t: f64,
    n_steps: usize,
    n_paths: usize,
    seed: u64,
    credit_spreads: &Vec<[f64; 2]>,
    equity_credit_corr: f64,
) -> (DMatrix<f64>, DVector<f64>) {
    let dt = t / n_steps as f64;
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0, 1.0).unwrap();

    let corr_matrix = Matrix2::new(1.0, equity_credit_corr, equity_credit_corr, 1.0);
    let cholesky = Cholesky::new(corr_matrix).unwrap();
    let l_factor = cholesky.l();

    let mut paths = DMatrix::from_element(n_steps + 1, n_paths, 0.0);
    let mut default_times = DVector::from_element(n_paths, t + 1.0);

    for j in 0..n_paths {
        paths[(0, j)] = s0;
        for i in 1..=n_steps {
            let current_time = i as f64 * dt;
            let z1 = normal.sample(&mut rng);
            let z2 = normal.sample(&mut rng);
            let e_stock = l_factor[(0, 0)] * z1;
            let e_credit = l_factor[(1, 0)] * z1 + l_factor[(1, 1)] * z2;

            let r = interpolate(forward_curve, current_time);
            paths[(i, j)] = paths[(i - 1, j)] * ((r - 0.5 * sigma.powi(2)) * dt + sigma * e_stock * dt.sqrt()).exp();

            if default_times[j] > t {
                let hazard_rate = interpolate(credit_spreads, current_time);
                let prob_default = 1.0 - (-hazard_rate * dt).exp();
                let u_credit = normal.cdf(e_credit);
                if u_credit < prob_default {
                    default_times[j] = current_time;
                }
            }
        }
    }
    (paths, default_times)
}


// Polynomial regression for LSMC
fn poly_regression(x: &DVector<f64>, y: &DVector<f64>, degree: usize) -> Option<DVector<f64>> {
    if x.len() == 0 {
        return None;
    }
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
    forward_curve: Vec<[f64; 2]>,
    sigma: f64,
    credit_spreads: Vec<[f64; 2]>,
    equity_credit_corr: f64,
    recovery_rate: f64,
    n_paths: usize,
    n_steps: usize,
    poly_degree: usize,
    seed: u64,
) -> PyResult<f64> {
    let dt = t / n_steps as f64;
    let (stock_paths, default_times) = simulate_correlated_paths(
        s0, &forward_curve, sigma, t, n_steps, n_paths, seed, &credit_spreads, equity_credit_corr
    );

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
        if default_times[j] <= t {
            return recovery_rate;
        }
        let final_price = stock_paths[(n_steps, j)];
        let final_strike = strikes[(n_steps, j)];
        (final_price - final_strike).max(0.0)
    });

    for i in (0..n_steps).rev() {
        let current_time = i as f64 * dt;
        let r = interpolate(&forward_curve, current_time);

        let mut x_regression = Vec::new();
        let mut y_regression = Vec::new();

        for j in 0..n_paths {
            if default_times[j] > current_time {
                let stock_price = stock_paths[(i, j)];
                let strike = strikes[(i, j)];
                if (stock_price - strike).max(0.0) > 0.0 {
                    x_regression.push(stock_price);
                    y_regression.push(warrant_values[j] * (-r * dt).exp());
                }
            }
        }

        let beta = if !x_regression.is_empty() {
            let x_vec = DVector::from_vec(x_regression);
            let y_vec = DVector::from_vec(y_regression);
            poly_regression(&x_vec, &y_vec, poly_degree)
        } else {
            None
        };

        for j in 0..n_paths {
            let r = interpolate(&forward_curve, current_time);
            let discounted_value = warrant_values[j] * (-r * dt).exp();

            if default_times[j] > current_time {
                let stock_price = stock_paths[(i, j)];
                let strike = strikes[(i, j)];

                if (stock_price - strike).max(0.0) > 0.0 {
                     if let Some(ref b) = beta {
                        let mut continuation_value = 0.0;
                        for d in 0..=b.len() - 1 {
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

    let price = warrant_values.mean();
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
    fn test_interpolate() {
        let curve = vec![[0.0, 0.01], [1.0, 0.02], [2.0, 0.03]];
        assert!((interpolate(&curve, 0.5) - 0.015).abs() < 1e-9);
        assert!((interpolate(&curve, -1.0) - 0.01).abs() < 1e-9);
        assert!((interpolate(&curve, 3.0) - 0.03).abs() < 1e-9);
    }

    #[test]
    fn test_poly_regression() {
        let x = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = DVector::from_vec(vec![9.0, 24.0, 47.0, 78.0, 117.0]);
        let degree = 2;

        let beta = poly_regression(&x, &y, degree).unwrap();

        assert!(beta.len() == 3);
        assert!((beta[0] - 2.0).abs() < 1e-9);
        assert!((beta[1] - 3.0).abs() < 1e-9);
        assert!((beta[2] - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_warrant_pricing_no_default_risk() {
        // Test with zero credit spread, should be close to previous results
        let credit_spreads = vec![[1.0, 0.0]];
        let forward_curve = vec![[0.0, 0.05], [1.0, 0.05]];
        let price = price_exotic_warrant(
            100.0, 0.9, 1000.0, 1.0, forward_curve, 0.2, credit_spreads, 0.0, 0.4, 5000, 200, 2, 42
        ).unwrap();
        assert!(price > 5.0 && price < 15.0); // A reasonable range
    }

    #[test]
    fn test_warrant_pricing_high_default_risk() {
        // High spread should lower the price
        let credit_spreads_low = vec![[1.0, 0.001]];
        let forward_curve = vec![[0.0, 0.05], [1.0, 0.05]];
        let price_low_risk = price_exotic_warrant(
            100.0, 0.9, 1000.0, 1.0, forward_curve.clone(), 0.2, credit_spreads_low, 0.0, 0.4, 5000, 200, 2, 42
        ).unwrap();

        let credit_spreads_high = vec![[1.0, 0.20]]; // 20% spread
        let price_high_risk = price_exotic_warrant(
            100.0, 0.9, 1000.0, 1.0, forward_curve, 0.2, credit_spreads_high, 0.0, 0.4, 5000, 200, 2, 42
        ).unwrap();

        assert!(price_high_risk < price_low_risk);
    }
}