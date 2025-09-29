use nalgebra::{DMatrix, DVector, Cholesky, Matrix2};
use rand::distributions::Distribution;
use statrs::distribution::{Normal, ContinuousCDF};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::{HashMap, HashSet};
use std::cmp::Ordering;

#[cfg(feature = "pyo3")]
use pyo3::prelude::*;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

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

// Function to simulate correlated stock price paths and default times.
// This function models the underlying stock price and the possibility of the issuer defaulting.
// It uses a Cholesky decomposition to introduce correlation between the stock price process
// and the credit default process.
fn simulate_correlated_paths(
    s0: f64,
    forward_curve: &Vec<[f64; 2]>,
    sigma: f64,
    t: f64,
    n_steps: usize,
    n_paths: usize,
    credit_spreads: &Option<Vec<[f64; 2]>>,
    equity_credit_corr: f64,
) -> (DMatrix<f64>, DVector<f64>) {
    let dt = t / n_steps as f64;
    let mut rng = StdRng::from_entropy();
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

            // If a credit curve is provided, simulate the default event.
            if let Some(spreads) = credit_spreads {
                // A default occurs if a uniform random variable (derived from the correlated normal)
                // falls below the cumulative default probability for the time step.
                if default_times[j] > t { // Only check if not already defaulted.
                    let hazard_rate = interpolate(spreads, current_time);
                    let prob_default = 1.0 - (-hazard_rate * dt).exp();
                    let u_credit = normal.cdf(e_credit); // Transform to uniform for probability comparison.
                    if u_credit < prob_default {
                        default_times[j] = current_time;
                    }
                }
            }
            // If credit_spreads is None, we skip the default simulation entirely.
        }
    }
    (paths, default_times)
}


// This function performs a polynomial regression, which is a core component of the
// Longstaff-Schwartz Monte Carlo (LSMC) method. It estimates the "continuation value"
// of the warrant—the expected value of holding it rather than exercising it.
// It solves the standard regression equation: beta = (X'X)^-1 * X'y
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
    // Calculate the coefficients (beta) using the normal equation: (X'X)^-1 * X'y
    let xt = x_matrix.transpose();
    let xtx = xt.clone() * x_matrix;

    if let Some(inv_xtx) = xtx.try_inverse() {
        let beta = inv_xtx * xt * y;
        Some(beta)
    } else {
        None
    }
}

// This is the core pricing function for the exotic warrant, using a Long-Schwartz Monte Carlo method.
// It handles complex features like issuer credit risk, correlation, and monthly exercise limits.
fn price_exotic_warrant_core(
    s0: f64,
    strike_discount: f64,
    buyback_price: f64,
    t: f64,
    forward_curve: Vec<[f64; 2]>,
    sigma: f64,
    credit_spreads: Option<Vec<[f64; 2]>>,
    equity_credit_corr: f64,
    recovery_rate: f64,
    monthly_exercise_limit: f64,
    n_paths: usize,
    n_steps: usize,
    poly_degree: usize,
) -> f64 {
    let dt = t / n_steps as f64;
    // Step 1: Simulate the correlated stock price paths and issuer default times.
    let (stock_paths, default_times) = simulate_correlated_paths(
        s0, &forward_curve, sigma, t, n_steps, n_paths, &credit_spreads, equity_credit_corr
    );

    // Step 2: Calculate the evolving strike price for each path.
    // The strike is reset weekly based on the previous week's stock price.
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

    // Step 3: Initialize the warrant values at the final time step (maturity).
    // If the issuer has defaulted, the value is the recovery rate. Otherwise, it's the standard
    // call option payoff: max(S - K, 0).
    let mut warrant_values = DVector::from_fn(n_paths, |j, _| {
        if default_times[j] <= t {
            return recovery_rate;
        }
        let final_price = stock_paths[(n_steps, j)];
        let final_strike = strikes[(n_steps, j)];
        (final_price - final_strike).max(0.0)
    });

    // This map will track the number of warrants exercised in each calendar month
    // to enforce the monthly exercise limit.
    let mut monthly_exercise_counts: HashMap<usize, usize> = HashMap::new();

    // Step 4: Perform the backward induction (Long-Schwartz algorithm).
    // We step back from maturity, deciding at each time step whether to exercise or hold.
    for i in (0..n_steps).rev() {
        let current_time = i as f64 * dt;
        let r = interpolate(&forward_curve, current_time);

        // Step 4a: Identify in-the-money paths to use for regression.
        // We only regress on paths where immediate exercise has a positive value, as these
        // are the only paths where the exercise decision is relevant.
        let mut x_regression = Vec::new();
        let mut y_regression = Vec::new();

        for j in 0..n_paths {
            if default_times[j] > current_time { // Path has not defaulted yet.
                let stock_price = stock_paths[(i, j)];
                let strike = strikes[(i, j)];
                if (stock_price - strike).max(0.0) > 0.0 {
                    x_regression.push(stock_price);
                    // The 'y' values are the discounted future values from the next time step.
                    y_regression.push(warrant_values[j] * (-r * dt).exp());
                }
            }
        }

        // Step 4b: Run the regression to get the coefficients of the continuation value polynomial.
        let beta = if !x_regression.is_empty() {
            let x_vec = DVector::from_vec(x_regression);
            let y_vec = DVector::from_vec(y_regression);
            poly_regression(&x_vec, &y_vec, poly_degree)
        } else {
            None
        };

        // Step 4c: Identify all paths where immediate exercise is optimal.
        // This is where the core LSMC decision logic and the new monthly limit are applied.
        let mut potential_exercisers: Vec<(usize, f64, f64)> = Vec::new(); // (path_idx, exercise_value, profitability)

        for j in 0..n_paths {
            if default_times[j] > current_time {
                let stock_price = stock_paths[(i, j)];
                let strike = strikes[(i, j)];
                let exercise_value = (stock_price - strike).max(0.0);

                if exercise_value > 0.0 {
                    let r = interpolate(&forward_curve, current_time);
                    let discounted_value = warrant_values[j] * (-r * dt).exp();

                    // Estimate the continuation value for this specific path using the regression coefficients.
                    let continuation_value = if let Some(ref b) = beta {
                        let mut cv = 0.0;
                        for d in 0..=b.len() - 1 {
                            cv += b[d] * stock_price.powi(d as i32);
                        }
                        cv
                    } else {
                        // If regression fails (e.g., no in-the-money paths), assume hold is optimal.
                        discounted_value
                    };

                    let effective_exercise_value = exercise_value.min(buyback_price);

                    // If exercising now is more valuable than holding, add it to the list of candidates.
                    if effective_exercise_value > continuation_value {
                        let profitability = effective_exercise_value - continuation_value;
                        potential_exercisers.push((j, effective_exercise_value, profitability));
                    }
                }
            }
        }

        // Step 4d: Enforce the monthly exercise limit.
        // First, sort the candidates by profitability (how much better exercise is than holding).
        // This assumes the most "in-the-money" holders will exercise first.
        potential_exercisers.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(Ordering::Equal));

        // Determine the limit for the current calendar month.
        let current_month = (current_time * 12.0).floor() as usize;
        let max_monthly_exercises = (n_paths as f64 * monthly_exercise_limit).floor() as usize;
        let exercised_count_this_month = monthly_exercise_counts.entry(current_month).or_insert(0);
        let mut allowed_exercises = max_monthly_exercises.saturating_sub(*exercised_count_this_month);

        // Step 4e: Update warrant values based on the (limited) exercise decisions.
        // We iterate through the profitable exercisers and "fill" the monthly quota.
        let mut exercised_paths = HashSet::new();
        for (path_idx, exercise_value, _) in potential_exercisers.into_iter() {
            if allowed_exercises > 0 {
                // This path exercises: its value for this time step is the exercise value.
                warrant_values[path_idx] = exercise_value;
                exercised_paths.insert(path_idx);
                allowed_exercises -= 1;
            } else {
                // The monthly limit has been reached. No more exercises are allowed.
                // Paths that were optimal to exercise but were blocked are now forced to "hold".
                break;
            }
        }

        // Update the total count of exercises for the month.
        if !exercised_paths.is_empty() {
            *exercised_count_this_month += exercised_paths.len();
        }

        // Step 4f: Update the values for all paths that did not exercise in this step.
        for j in 0..n_paths {
            if default_times[j] <= current_time {
                // If the path defaulted previously, its value is the fixed recovery rate.
                warrant_values[j] = recovery_rate;
            } else if !exercised_paths.contains(&j) {
                // If the path did not exercise (either by choice or due to the limit),
                // its value is the discounted value from the next time step (the continuation value).
                let r = interpolate(&forward_curve, current_time);
                warrant_values[j] *= (-r * dt).exp();
            }
            // If the path was in exercised_paths, its value was already set and is not changed here.
        }
    }

    // Step 5: The final warrant price is the average of all path values at time 0.
    let price = warrant_values.mean();
    price
}

// Python wrapper function
#[cfg(feature = "pyo3")]
#[pyfunction(
    signature = (
        s0,
        strike_discount,
        buyback_price,
        t,
        forward_curve,
        sigma,
        credit_spreads,
        equity_credit_corr,
        recovery_rate,
        monthly_exercise_limit,
        n_paths,
        n_steps,
        poly_degree
    )
)]
fn price_exotic_warrant(
    s0: f64,
    strike_discount: f64,
    buyback_price: f64,
    t: f64,
    forward_curve: Vec<[f64; 2]>,
    sigma: f64,
    credit_spreads: Option<Vec<[f64; 2]>>,
    equity_credit_corr: f64,
    recovery_rate: f64,
    monthly_exercise_limit: f64,
    n_paths: usize,
    n_steps: usize,
    poly_degree: usize,
) -> PyResult<f64> {
    Ok(price_exotic_warrant_core(
        s0, strike_discount, buyback_price, t, forward_curve, sigma,
        credit_spreads, equity_credit_corr, recovery_rate,
        monthly_exercise_limit, n_paths, n_steps, poly_degree
    ))
}

#[cfg(feature = "pyo3")]
#[pymodule]
fn warrants(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(price_exotic_warrant, m)?)?;
    Ok(())
}

// WebAssembly wrapper function
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn price_exotic_warrant_wasm(
    s0: f64,
    strike_discount: f64,
    buyback_price: f64,
    t: f64,
    forward_curve: &[f64], // Flattened array: [t1, r1, t2, r2, ...]
    sigma: f64,
    credit_spreads: &[f64], // Flattened array: [t1, s1, t2, s2, ...]. Pass empty slice for no credit risk.
    equity_credit_corr: f64,
    recovery_rate: f64,
    monthly_exercise_limit: f64,
    n_paths: usize,
    n_steps: usize,
    poly_degree: usize,
) -> f64 {
    // Convert flattened arrays to Vec<[f64; 2]>
    let forward_curve_vec: Vec<[f64; 2]> = forward_curve
        .chunks(2)
        .map(|chunk| [chunk[0], chunk[1]])
        .collect();
    
    // If the credit_spreads slice is empty, treat it as None. Otherwise, parse it.
    // This allows the caller (e.g., JavaScript) to omit the credit curve easily.
    let credit_spreads_opt = if credit_spreads.is_empty() {
        None
    } else {
        Some(
            credit_spreads
                .chunks(2)
                .map(|chunk| [chunk[0], chunk[1]])
                .collect(),
        )
    };
    
    price_exotic_warrant_core(
        s0, strike_discount, buyback_price, t, forward_curve_vec, sigma,
        credit_spreads_opt, equity_credit_corr, recovery_rate,
        monthly_exercise_limit, n_paths, n_steps, poly_degree
    )
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
        let price = price_exotic_warrant_core(
            100.0, 0.9, 1000.0, 1.0, forward_curve, 0.2, Some(credit_spreads), 0.0, 0.4, 1.0, 5000, 200, 2
        );
        assert!(price > 5.0 && price < 15.0); // A reasonable range
    }

    #[test]
    fn test_warrant_pricing_high_default_risk() {
        // High spread should lower the price
        let credit_spreads_low = vec![[1.0, 0.001]];
        let forward_curve = vec![[0.0, 0.05], [1.0, 0.05]];
        let price_low_risk = price_exotic_warrant_core(
            100.0, 0.9, 1000.0, 1.0, forward_curve.clone(), 0.2, Some(credit_spreads_low), 0.0, 0.4, 1.0, 5000, 200, 2
        );

        let credit_spreads_high = vec![[1.0, 0.20]]; // 20% spread
        let price_high_risk = price_exotic_warrant_core(
            100.0, 0.9, 1000.0, 1.0, forward_curve, 0.2, Some(credit_spreads_high), 0.0, 0.4, 1.0, 5000, 200, 2
        );

        assert!(price_high_risk < price_low_risk);
    }

    #[test]
    fn test_warrant_pricing_with_exercise_limit() {
        // A scenario where the warrant is likely to be exercised
        let forward_curve = vec![[0.0, 0.05], [1.0, 0.05]];
        let credit_spreads = vec![[1.0, 0.0]]; // No default risk
        let s0 = 100.0;
        let strike_discount = 0.9;
        let t = 1.0;
        let sigma = 0.3; // Higher vol to increase exercise probability

        // Price with no exercise limit
        let price_no_limit = price_exotic_warrant_core(
            s0, strike_discount, 1000.0, t, forward_curve.clone(), sigma, Some(credit_spreads.clone()), 0.0, 0.4, 1.0, 5000, 200, 2
        );

        // Price with a strict exercise limit (e.g., 10% per month)
        let price_with_limit = price_exotic_warrant_core(
            s0, strike_discount, 1000.0, t, forward_curve.clone(), sigma, Some(credit_spreads.clone()), 0.0, 0.4, 0.1, 5000, 200, 2
        );

        // The constrained warrant should be less valuable
        assert!(price_with_limit < price_no_limit);
    }

    #[test]
    fn test_warrant_pricing_no_credit_risk() {
        // Test that pricing with `None` for credit spreads is equivalent to pricing with zero default risk.
        let forward_curve = vec![[0.0, 0.05], [1.0, 0.05]];
        let s0 = 100.0;
        let strike_discount = 0.9;
        let t = 1.0;
        let sigma = 0.2;

        // Price with no credit curve provided.
        let price_no_credit_risk = price_exotic_warrant_core(
            s0, strike_discount, 1000.0, t, forward_curve.clone(), sigma, None, 0.0, 0.4, 1.0, 5000, 200, 2
        );

        // Price with a zero credit spread (which should also model no default risk).
        let credit_spreads_zero = vec![[1.0, 0.0]];
        let price_zero_spread = price_exotic_warrant_core(
            s0, strike_discount, 1000.0, t, forward_curve.clone(), sigma, Some(credit_spreads_zero), 0.0, 0.4, 1.0, 5000, 200, 2
        );

        // The prices should be very close (allowing for small Monte Carlo variation).
        assert!((price_no_credit_risk - price_zero_spread).abs() < 0.5);
        // Also check against a reasonable range.
        assert!(price_no_credit_risk > 5.0 && price_no_credit_risk < 15.0);
    }
}