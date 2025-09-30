use pyo3::prelude::*;
use nalgebra::{DMatrix, DVector};
use std::collections::HashSet;
use chrono::{Duration, NaiveDate, Utc};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::{Normal, Distribution};

#[pyclass]
#[derive(Clone, Debug)]
pub struct WarrantParams {
    #[pyo3(get, set)]
    pub spot: f64,
    #[pyo3(get, set)]
    pub vol: f64,
    #[pyo3(get, set)]
    pub risk_free_rate: f64,
    #[pyo3(get, set)]
    pub maturity_date: NaiveDate,
    #[pyo3(get, set)]
    pub strike_reset_period_days: i64,
    #[pyo3(get, set)]
    pub strike_discount: f64,
    #[pyo3(get, set)]
    pub buyback_price: f64,
    #[pyo3(get, set)]
    pub exercise_limit_percentage: f64,
    #[pyo3(get, set)]
    pub exercise_limit_period_days: i64,
    #[pyo3(get, set)]
    pub next_exercise_reset_date: NaiveDate,
    #[pyo3(get, set)]
    pub exercised_this_period_percentage: f64,
    #[pyo3(get, set)]
    pub num_paths: usize,
    #[pyo3(get, set)]
    pub num_steps: usize,
    #[pyo3(get, set)]
    pub seed: Option<u64>,
}

#[pymethods]
impl WarrantParams {
    #[new]
    #[pyo3(signature = (
        spot, vol, risk_free_rate, maturity_date, strike_reset_period_days, strike_discount,
        buyback_price, exercise_limit_percentage, exercise_limit_period_days, next_exercise_reset_date,
        exercised_this_period_percentage, num_paths, num_steps, seed
    ))]
    fn new(
        spot: f64, vol: f64, risk_free_rate: f64, maturity_date: NaiveDate,
        strike_reset_period_days: i64, strike_discount: f64, buyback_price: f64,
        exercise_limit_percentage: f64, exercise_limit_period_days: i64,
        next_exercise_reset_date: NaiveDate, exercised_this_period_percentage: f64,
        num_paths: usize, num_steps: usize, seed: Option<u64>
    ) -> Self {
        WarrantParams {
            spot, vol, risk_free_rate, maturity_date, strike_reset_period_days, strike_discount,
            buyback_price, exercise_limit_percentage, exercise_limit_period_days,
            next_exercise_reset_date, exercised_this_period_percentage, num_paths, num_steps, seed
        }
    }
}

#[derive(Debug, Clone)]
struct PathState {
    stock_price: f64,
    strike_price: f64,
    max_quota_for_period: f64,
    time_to_maturity: f64,
}

fn generate_paths(params: &WarrantParams) -> (Vec<Vec<PathState>>, f64) {
    let today = Utc::now().naive_utc().date();
    let time_to_maturity_days = (params.maturity_date - today).num_days();
    if time_to_maturity_days <= 0 { return (Vec::new(), 0.0); }
    let time_to_maturity_years = time_to_maturity_days as f64 / 365.25;
    let dt = time_to_maturity_years / params.num_steps as f64;
    let sqrt_dt = dt.sqrt();

    let mut rng = match params.seed { Some(seed) => StdRng::seed_from_u64(seed), None => StdRng::from_entropy() };
    let normal_dist = Normal::new(0.0, 1.0).unwrap();
    let mut all_paths: Vec<Vec<PathState>> = Vec::with_capacity(params.num_paths);

    for _ in 0..params.num_paths {
        let mut path: Vec<PathState> = Vec::with_capacity(params.num_steps + 1);
        let mut current_s = params.spot;
        let mut current_k = params.spot * (1.0 - params.strike_discount);
        let mut current_date = today;
        let mut next_strike_reset_date = today + Duration::days(params.strike_reset_period_days);
        let mut next_quota_reset_date = params.next_exercise_reset_date;
        let initial_quota = params.exercise_limit_percentage - params.exercised_this_period_percentage;

        path.push(PathState {
            stock_price: current_s, strike_price: current_k, max_quota_for_period: initial_quota,
            time_to_maturity: time_to_maturity_years,
        });

        for i in 1..=params.num_steps {
            current_date = today + Duration::days((i as f64 * dt * 365.25).round() as i64);
            let z: f64 = normal_dist.sample(&mut rng);
            current_s *= ((params.risk_free_rate - 0.5 * params.vol.powi(2)) * dt + params.vol * sqrt_dt * z).exp();

            if current_date >= next_strike_reset_date {
                current_k = current_s * (1.0 - params.strike_discount);
                next_strike_reset_date += Duration::days(params.strike_reset_period_days);
            }
            let mut current_max_quota = path.last().unwrap().max_quota_for_period;
            if current_date >= next_quota_reset_date {
                current_max_quota = params.exercise_limit_percentage;
                next_quota_reset_date += Duration::days(params.exercise_limit_period_days);
            }
            path.push(PathState {
                stock_price: current_s, strike_price: current_k, max_quota_for_period: current_max_quota,
                time_to_maturity: time_to_maturity_years - i as f64 * dt,
            });
        }
        all_paths.push(path);
    }
    (all_paths, dt)
}

fn build_basis_matrix(states: &[(&PathState, f64)]) -> DMatrix<f64> {
    let n_samples = states.len();
    // New basis: {q, q^2, m*q}. All terms are zero if q=0, which forces
    // the continuation value C(S, q) to be zero if the quota is zero.
    let n_features = 3;
    let mut x = DMatrix::zeros(n_samples, n_features);
    for (i, (state, quota)) in states.iter().enumerate() {
        let m = state.stock_price / state.strike_price;
        let q = *quota;
        x[(i, 0)] = q;
        x[(i, 1)] = q * q;
        x[(i, 2)] = m * q;
    }
    x
}

#[pyfunction]
fn price_warrant(params: WarrantParams) -> PyResult<f64> {
    let (paths, dt) = generate_paths(&params);
    if paths.is_empty() { return Ok(0.0); }

    let num_paths = params.num_paths;
    let num_steps = params.num_steps;
    let discount_factor = (-(params.risk_free_rate * dt)).exp();

    // V now stores the TOTAL value of the warrant for each path
    let mut V: DVector<f64> = DVector::from_fn(num_paths, |i, _| {
        let final_state = &paths[i][num_steps];
        let per_unit_intrinsic = (final_state.stock_price - final_state.strike_price).max(0.0);
        let per_unit_value = per_unit_intrinsic.min(params.buyback_price);
        per_unit_value * final_state.max_quota_for_period
    });

    let mut remaining_quota: DVector<f64> = DVector::from_fn(num_paths, |i, _| paths[i][num_steps].max_quota_for_period);

    for t in (0..num_steps).rev() {
        // Update remaining quota for paths that cross a reset date
        for i in 0..num_paths {
            if paths[i][t + 1].max_quota_for_period != paths[i][t].max_quota_for_period {
                remaining_quota[i] = paths[i][t + 1].max_quota_for_period;
            }
        }

        let continuation_v = &V * discount_factor;

        let itm_path_indices: Vec<usize> = (0..num_paths)
            .filter(|&i| paths[i][t].stock_price > paths[i][t].strike_price && remaining_quota[i] > 1e-9)
            .collect();

        let itm_set: HashSet<usize> = itm_path_indices.iter().cloned().collect();
        let mut beta_opt: Option<DVector<f64>> = None;

        if !itm_path_indices.is_empty() {
            let itm_states_with_quota: Vec<(&PathState, f64)> = itm_path_indices.iter().map(|&i| (&paths[i][t], remaining_quota[i])).collect();
            let X = build_basis_matrix(&itm_states_with_quota);
            let Y = DVector::from_iterator(itm_path_indices.len(), itm_path_indices.iter().map(|&i| continuation_v[i]));

            if let Ok(beta) = X.svd(true, true).solve(&Y, 1e-10) {
                beta_opt = Some(beta);
            }
        }

        let mut next_V = V.clone();
        for i in 0..num_paths {
            let hold_value = continuation_v[i];

            if !itm_set.contains(&i) {
                next_V[i] = hold_value;
                continue;
            }

            let state = &paths[i][t];
            let per_unit_exercise_value = (state.stock_price - state.strike_price).max(0.0).min(params.buyback_price);
            let q_i = remaining_quota[i];

            if let Some(beta) = &beta_opt {
                // Find optimal exercise amount `e` to maximize:
                // f(e) = e * P + C(S, q - e)
                // where P is per_unit_exercise_value and C is the continuation value from regression.
                // C(S, q) = beta[0]*q + beta[1]*q^2 + beta[2]*m*q
                let m = state.stock_price / state.strike_price;
                let b0 = beta[0]; // Coeff for q
                let b1 = beta[1]; // Coeff for q^2
                let b2 = beta[2]; // Coeff for m*q

                // f(e) is a quadratic in e: A*e^2 + B*e + C_val
                let A = b1;
                // dC/dq at q=q_i
                let dC_dq = b0 + 2.0 * b1 * q_i + b2 * m;
                let B = per_unit_exercise_value - dC_dq;

                let optimal_e = if A >= -1e-9 { // Convex or linear: max is at a boundary
                    if B > 0.0 { q_i } else { 0.0 }
                } else { // Concave: max is at vertex, clamped to [0, q_i]
                    let e_star = -B / (2.0 * A);
                    e_star.max(0.0).min(q_i)
                };

                if optimal_e > 1e-9 {
                    let exercised_value = optimal_e * per_unit_exercise_value;

                    let q_after_exercise = q_i - optimal_e;
                    let features_after = DVector::from_vec(vec![q_after_exercise, q_after_exercise * q_after_exercise, m * q_after_exercise]);
                    let continuation_after_exercise = beta.dot(&features_after);

                    let total_value = exercised_value + continuation_after_exercise;
                    next_V[i] = total_value;
                    remaining_quota[i] = q_after_exercise;
                } else {
                    next_V[i] = hold_value;
                }
            } else { // Fallback if regression fails
                let exercise_all_value = per_unit_exercise_value * q_i;
                if exercise_all_value > hold_value {
                    next_V[i] = exercise_all_value;
                    remaining_quota[i] = 0.0;
                } else {
                    next_V[i] = hold_value;
                }
            }
        }
        V = next_V;
    }

    // The final price is the mean of the risk-neutral expectation of the warrant's total value at t=0
    Ok(V.mean())
}

#[pymodule]
fn _floating_strike_warrant(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(price_warrant, m)?)?;
    m.add_class::<WarrantParams>()?;
    Ok(())
}