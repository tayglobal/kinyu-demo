use pyo3::prelude::*;
use nalgebra::{DMatrix, DVector};
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
    let n_features = 6;
    let mut x = DMatrix::zeros(n_samples, n_features);
    for (i, (state, quota)) in states.iter().enumerate() {
        let m = state.stock_price / state.strike_price;
        let q = *quota;
        x[(i, 0)] = 1.0;
        x[(i, 1)] = m;
        x[(i, 2)] = m * m;
        x[(i, 3)] = q;
        x[(i, 4)] = q * q;
        x[(i, 5)] = m * q;
    }
    x
}

#[pyfunction]
fn price_warrant(params: WarrantParams) -> PyResult<f64> {
    let (paths, dt) = generate_paths(&params);
    if paths.is_empty() { return Ok(0.0); }

    let num_paths = params.num_paths;
    let num_steps = params.num_steps;

    let mut cash_flows = DMatrix::zeros(num_paths, num_steps + 1);
    let mut remaining_quota: DVector<f64> = DVector::from_fn(num_paths, |i, _| paths[i][num_steps].max_quota_for_period);

    let mut V: DVector<f64> = DVector::from_fn(num_paths, |i, _| {
        let final_state = &paths[i][num_steps];
        let intrinsic_value = (final_state.stock_price - final_state.strike_price).max(0.0);
        let capped_value = intrinsic_value.min(params.buyback_price);
        cash_flows[(i, num_steps)] = capped_value * remaining_quota[i];
        capped_value
    });

    for t in (0..num_steps).rev() {
        for i in 0..num_paths {
            if paths[i][t+1].max_quota_for_period != paths[i][t].max_quota_for_period {
                remaining_quota[i] = paths[i][t].max_quota_for_period;
            }
        }

        let continuation_v = &V * (-(params.risk_free_rate * dt)).exp();
        let mut estimated_continuation_v = continuation_v.clone();

        let itm_path_indices: Vec<usize> = (0..num_paths)
            .filter(|&i| paths[i][t].stock_price > paths[i][t].strike_price)
            .collect();

        if !itm_path_indices.is_empty() {
            let itm_states_with_quota: Vec<(&PathState, f64)> = itm_path_indices.iter().map(|&i| (&paths[i][t], remaining_quota[i])).collect();
            let X = build_basis_matrix(&itm_states_with_quota);
            let Y = DVector::from_iterator(itm_path_indices.len(), itm_path_indices.iter().map(|&i| continuation_v[i]));

            if let Ok(beta) = X.clone().svd(true, true).solve(&Y, 1e-10) {
                let regression_fitted_values = &X * &beta;
                for (idx, &i) in itm_path_indices.iter().enumerate() {
                    estimated_continuation_v[i] = regression_fitted_values[idx].max(0.0);
                }
            }
        }

        for i in 0..num_paths {
            let state = &paths[i][t];
            let intrinsic_value = (state.stock_price - state.strike_price).max(0.0);
            let hold_value = estimated_continuation_v[i];

            let exercise_is_optimal = intrinsic_value > hold_value;

            let holder_optimal_value = if exercise_is_optimal { intrinsic_value } else { hold_value };
            let final_per_unit_value = holder_optimal_value.min(params.buyback_price);

            V[i] = final_per_unit_value;

            if exercise_is_optimal && remaining_quota[i] > 0.0 {
                cash_flows[(i, t)] = V[i] * remaining_quota[i];
                remaining_quota[i] = 0.0;
            } else {
                cash_flows[(i, t)] = 0.0;
            }
        }
    }

    let mut total_pv = 0.0;
    for i in 0..num_paths {
        let mut path_pv = 0.0;
        for t in 0..=num_steps {
            let discount = (-(t as f64 * dt) * params.risk_free_rate).exp();
            path_pv += cash_flows[(i, t)] * discount;
        }
        total_pv += path_pv;
    }

    Ok(total_pv / num_paths as f64)
}

#[pymodule]
fn _floating_strike_warrant(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(price_warrant, m)?)?;
    m.add_class::<WarrantParams>()?;
    Ok(())
}