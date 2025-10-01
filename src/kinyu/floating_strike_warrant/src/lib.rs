use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::StandardNormal;

#[cfg(feature = "python")]
use pyo3::{prelude::*, wrap_pyfunction};

#[cfg(test)]
use std::cell::{Cell, RefCell};

/// Parameters that fully describe the floating-strike warrant simulation.
#[derive(Debug, Clone)]
pub struct FloatingStrikeWarrantParams {
    /// Current underlying stock price.
    pub initial_price: f64,
    /// Continuously compounded risk-free rate.
    pub risk_free_rate: f64,
    /// Annualised volatility of the underlying (in decimal, e.g. 0.2 for 20%).
    pub volatility: f64,
    /// Time to maturity in years.
    pub maturity: f64,
    /// Number of time steps per year used in the simulation grid.
    pub steps_per_year: usize,
    /// Strike reset level expressed as a fraction of the prevailing stock price (e.g. 0.9 for a 10% discount).
    pub strike_discount: f64,
    /// Number of simulation steps between successive strike resets.
    pub strike_reset_steps: usize,
    /// Issuer buy-back price that caps the warrant value.
    pub buyback_price: f64,
    /// Underlying price level that enables the holder put right.
    pub holder_put_trigger_price: f64,
    /// Price paid to the holder when the put right is exercised.
    pub holder_put_price: f64,
    /// Maximum fraction of the position that can be exercised within a quota period (e.g. 0.1 for 10%).
    pub exercise_limit_fraction: f64,
    /// Length of the exercise quota period measured in simulation steps (e.g. trading days per month).
    pub exercise_limit_period_steps: usize,
    /// Simulation step index (starting at 0) at which the current exercise quota resets.
    pub next_limit_reset_step: usize,
    /// Fraction of the position that has already been exercised within the current quota period.
    pub exercised_fraction_current_period: f64,
    /// Number of Monte Carlo sample paths.
    pub num_paths: usize,
    /// Optional RNG seed to make the simulation reproducible.
    pub seed: Option<u64>,
}

/// Price a floating-strike warrant with strike resets, issuer call right, holder put right and exercise quota limits using LSMC.
///
/// The implementation follows a quota-aware Longstaff-Schwartz Monte Carlo scheme with a polynomial basis.
/// SVD is used to stabilise the regression at each backward induction step.
pub fn price_warrant(params: &FloatingStrikeWarrantParams) -> f64 {
    validate(params);

    let grid = simulate_state_grid(params);
    lsmc_price_internal(params, &grid, None)
}

/// Per-step expectations from the Monte Carlo valuation.
#[derive(Debug, Clone)]
pub struct WarrantTimeseriesPoint {
    /// Simulation step index (0 corresponds to valuation time).
    pub step: usize,
    /// Elapsed time in years at the start of the step.
    pub time: f64,
    /// Expected warrant value at the start of the step.
    pub expected_price: f64,
    /// Expected immediate exercise value subject to quota limits.
    pub exercise_upper_bound: f64,
    /// Expected payoff from invoking the holder put protection.
    pub put_lower_bound: f64,
    /// Issuer buyback price that caps the warrant value.
    pub buyback_upper_bound: f64,
}

/// Price the warrant and capture the simulated time series of bounds and values.
pub fn price_warrant_with_timeseries(
    params: &FloatingStrikeWarrantParams,
) -> (f64, Vec<WarrantTimeseriesPoint>) {
    validate(params);

    let grid = simulate_state_grid(params);
    let mut series = Vec::with_capacity(grid.total_steps + 1);
    let price = lsmc_price_internal(params, &grid, Some(&mut series));
    series.reverse();
    (price, series)
}

fn simulate_state_grid(params: &FloatingStrikeWarrantParams) -> SimulationGrid {
    let total_steps = (params.maturity * params.steps_per_year as f64).ceil() as usize;
    let total_steps = total_steps.max(1);
    let dt = if total_steps == 0 {
        0.0
    } else {
        if params.maturity == 0.0 {
            0.0
        } else {
            params.maturity / total_steps as f64
        }
    };
    let discount_factor = (-params.risk_free_rate * dt).exp();
    let drift = (params.risk_free_rate - 0.5 * params.volatility * params.volatility) * dt;
    let vol_term = params.volatility * dt.sqrt();

    let mut rng: StdRng = match params.seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_entropy(),
    };

    let mut states = vec![vec![State::default(); params.num_paths]; total_steps + 1];

    for path in 0..params.num_paths {
        let mut spot = params.initial_price;
        let mut strike = params.strike_discount * spot;
        let mut quota = (params.exercise_limit_fraction - params.exercised_fraction_current_period)
            .max(0.0)
            .min(params.exercise_limit_fraction);
        let remaining = (1.0 - params.exercised_fraction_current_period).max(0.0);
        let mut next_quota_reset = params
            .next_limit_reset_step
            .min(total_steps + params.exercise_limit_period_steps);
        let mut quota_period_index = 0usize;

        for step in 0..=total_steps {
            if step == next_quota_reset {
                quota = params.exercise_limit_fraction.min(remaining);
                next_quota_reset = next_quota_reset
                    .saturating_add(params.exercise_limit_period_steps)
                    .min(total_steps + params.exercise_limit_period_steps);
                quota_period_index = quota_period_index.saturating_add(1);
            }

            let steps_to_quota_reset = next_quota_reset.saturating_sub(step);
            let steps_to_strike_reset = if params.strike_reset_steps == 0 {
                0
            } else {
                let modulo = step % params.strike_reset_steps;
                let until_next_reset = if modulo == 0 {
                    params.strike_reset_steps
                } else {
                    params.strike_reset_steps - modulo
                };
                until_next_reset.min(total_steps.saturating_sub(step))
            };

            states[step][path] = State {
                spot,
                strike,
                time_to_maturity: (total_steps - step) as f64 * dt,
                steps_to_strike_reset,
                steps_to_quota_reset,
                quota_base: quota,
                quota_period_index,
            };

            if step == total_steps {
                break;
            }

            let normal: f64 = rng.sample(StandardNormal);
            spot *= (drift + vol_term * normal).exp();

            if params.strike_reset_steps > 0 && (step + 1) % params.strike_reset_steps == 0 {
                strike = params.strike_discount * spot;
            }
        }
    }

    SimulationGrid {
        states,
        total_steps,
        dt,
        discount_factor,
    }
}

struct SimulationGrid {
    states: Vec<Vec<State>>,
    total_steps: usize,
    dt: f64,
    discount_factor: f64,
}

#[cfg(test)]
#[derive(Clone, Debug)]
#[allow(dead_code)]
struct ExerciseRecord {
    step: usize,
    path: usize,
    amount: f64,
}

#[cfg(test)]
thread_local! {
    static LOG_ENABLED: Cell<bool> = Cell::new(false);
    static EXERCISE_LOG: RefCell<Vec<ExerciseRecord>> = RefCell::new(Vec::new());
}

#[cfg(test)]
fn enable_exercise_logging() {
    LOG_ENABLED.with(|flag| flag.set(true));
    EXERCISE_LOG.with(|log| log.borrow_mut().clear());
}

#[cfg(test)]
fn take_exercise_log() -> Vec<ExerciseRecord> {
    LOG_ENABLED.with(|flag| flag.set(false));
    EXERCISE_LOG.with(|log| log.borrow_mut().drain(..).collect())
}

#[cfg(test)]
fn log_exercise(step: usize, path: usize, amount: f64) {
    if amount <= 0.0 {
        return;
    }

    LOG_ENABLED.with(|flag| {
        if flag.get() {
            EXERCISE_LOG.with(|log| {
                log.borrow_mut().push(ExerciseRecord { step, path, amount });
            });
        }
    });
}

fn lsmc_price_internal(
    params: &FloatingStrikeWarrantParams,
    grid: &SimulationGrid,
    mut series: Option<&mut Vec<WarrantTimeseriesPoint>>,
) -> f64 {
    let total_steps = grid.total_steps;
    let dt = grid.dt;
    let discount_factor = grid.discount_factor;
    let states = &grid.states;

    let initial_remaining = (1.0 - params.exercised_fraction_current_period)
        .max(0.0)
        .min(1.0);
    let mut remaining_fraction = vec![initial_remaining; params.num_paths];
    let mut period_quota_remaining = vec![0.0; params.num_paths];
    let mut current_period_index = vec![usize::MAX; params.num_paths];
    let mut intrinsic_now = vec![0.0; params.num_paths];
    let mut available_now = vec![0.0; params.num_paths];
    let mut values = vec![0.0; params.num_paths];
    let mut discounted_future = vec![0.0; params.num_paths];

    for step in (0..=total_steps).rev() {
        let mut exercise_sum = 0.0;
        let mut put_sum = 0.0;

        for path in 0..params.num_paths {
            let state = states[step][path];

            if current_period_index[path] != state.quota_period_index {
                current_period_index[path] = state.quota_period_index;
                let base_quota = state
                    .quota_base
                    .max(0.0)
                    .min(params.exercise_limit_fraction)
                    .min(remaining_fraction[path]);
                period_quota_remaining[path] = base_quota;
            }

            intrinsic_now[path] = (state.spot - state.strike).max(0.0);
            available_now[path] = period_quota_remaining[path]
                .min(remaining_fraction[path])
                .max(0.0);
            exercise_sum += intrinsic_now[path] * available_now[path];
            if state.spot <= params.holder_put_trigger_price {
                put_sum += params.holder_put_price * remaining_fraction[path];
            }
        }

        if step == total_steps {
            for path in 0..params.num_paths {
                let state = states[step][path];
                let intrinsic = intrinsic_now[path];
                let available = available_now[path];
                let exercise_amount = if intrinsic > 0.0 { available } else { 0.0 };
                let call_payoff = intrinsic * exercise_amount;
                let put_payoff = if state.spot <= params.holder_put_trigger_price {
                    params.holder_put_price * remaining_fraction[path]
                } else {
                    0.0
                };

                let mut value = call_payoff;
                if put_payoff > value {
                    period_quota_remaining[path] = 0.0;
                    remaining_fraction[path] = 0.0;
                    value = put_payoff;
                } else {
                    #[cfg(test)]
                    if exercise_amount > 0.0 {
                        log_exercise(step, path, exercise_amount);
                    }
                    period_quota_remaining[path] =
                        (period_quota_remaining[path] - exercise_amount).max(0.0);
                    remaining_fraction[path] =
                        (remaining_fraction[path] - exercise_amount).max(0.0);
                }

                values[path] = value.min(params.buyback_price);
            }
        } else {
            for path in 0..params.num_paths {
                discounted_future[path] = values[path] * discount_factor;
            }

            let mut design_matrix = Vec::new();
            let mut responses = Vec::new();
            let mut itm_paths = Vec::new();

            for path in 0..params.num_paths {
                let intrinsic = intrinsic_now[path];
                let available = available_now[path];

                if intrinsic > 0.0 && available > 0.0 {
                    let basis = basis(&states[step][path], dt, available, remaining_fraction[path]);
                    design_matrix.extend_from_slice(&basis);
                    responses.push(discounted_future[path]);
                    itm_paths.push(path);
                }
            }

            let mut continuation_estimates: Vec<Option<f64>> = vec![None; params.num_paths];
            let mut beta_opt: Option<Vec<f64>> = None;

            if !itm_paths.is_empty() {
                let rows = itm_paths.len();
                let x = DMatrix::from_row_slice(rows, BASIS_DIM, &design_matrix);
                let y = DVector::from_vec(responses);
                let svd = x.svd(true, true);
                let beta_matrix = svd.solve(&y, 1e-12).expect("SVD regression solve failed");
                let beta_vec: Vec<f64> = beta_matrix.column(0).iter().copied().collect();
                beta_opt = Some(beta_vec.clone());

                for (row_idx, path_idx) in itm_paths.iter().enumerate() {
                    let slice = &design_matrix[row_idx * BASIS_DIM..(row_idx + 1) * BASIS_DIM];
                    let cont = dot(slice, &beta_vec);
                    continuation_estimates[*path_idx] = Some(cont);
                }
            }

            for path in 0..params.num_paths {
                let state = states[step][path];
                let intrinsic = intrinsic_now[path];
                let available = available_now[path];
                let continuation = continuation_estimates[path].unwrap_or(discounted_future[path]);

                let immediate = intrinsic * available;

                let should_exercise = if intrinsic > 0.0 && available > 1e-8 {
                    if let Some(beta) = &beta_opt {
                        let dq = available.min(0.01);
                        if dq > 0.0 {
                            let cont_current = continuation_estimates[path].unwrap_or_else(|| {
                                let phi = basis(
                                    &states[step][path],
                                    dt,
                                    available,
                                    remaining_fraction[path],
                                );
                                dot(&phi, beta)
                            });
                            let reduced_available = (available - dq).max(0.0);
                            let reduced_remaining = (remaining_fraction[path] - dq).max(0.0);
                            let phi_minus = basis(
                                &states[step][path],
                                dt,
                                reduced_available,
                                reduced_remaining,
                            );
                            let cont_minus = dot(&phi_minus, beta);
                            let lambda = (cont_current - cont_minus) / dq;
                            intrinsic > lambda
                        } else {
                            false
                        }
                    } else {
                        immediate > continuation
                    }
                } else {
                    false
                };

                let mut call_value = continuation;
                let mut call_exercised = false;
                let mut exercise_amount = 0.0;
                if should_exercise && immediate > 0.0 {
                    call_value = immediate;
                    call_exercised = true;
                    exercise_amount = available;
                }

                let put_payoff = if state.spot <= params.holder_put_trigger_price {
                    params.holder_put_price * remaining_fraction[path]
                } else {
                    0.0
                };

                let mut use_put = false;
                let mut value = call_value;
                if put_payoff > value {
                    use_put = true;
                    value = put_payoff;
                }

                if use_put {
                    period_quota_remaining[path] = 0.0;
                    remaining_fraction[path] = 0.0;
                } else if call_exercised {
                    #[cfg(test)]
                    log_exercise(step, path, exercise_amount);
                    period_quota_remaining[path] =
                        (period_quota_remaining[path] - exercise_amount).max(0.0);
                    remaining_fraction[path] =
                        (remaining_fraction[path] - exercise_amount).max(0.0);
                }

                value = value.min(params.buyback_price);
                values[path] = value;
            }
        }

        if let Some(series_vec) = series.as_mut() {
            let expected_price = values.iter().sum::<f64>() / params.num_paths as f64;
            (**series_vec).push(WarrantTimeseriesPoint {
                step,
                time: step as f64 * dt,
                expected_price,
                exercise_upper_bound: exercise_sum / params.num_paths as f64,
                put_lower_bound: put_sum / params.num_paths as f64,
                buyback_upper_bound: params.buyback_price,
            });
        }
    }

    let sum: f64 = values.iter().sum();
    sum / params.num_paths as f64
}

fn validate(params: &FloatingStrikeWarrantParams) {
    assert!(
        params.initial_price.is_finite() && params.initial_price > 0.0,
        "Initial price must be positive"
    );
    assert!(
        params.volatility.is_finite() && params.volatility >= 0.0,
        "Volatility must be non-negative"
    );
    assert!(
        params.maturity.is_finite() && params.maturity >= 0.0,
        "Maturity must be non-negative"
    );
    assert!(params.steps_per_year > 0, "Steps per year must be positive");
    assert!(
        params.strike_discount.is_finite() && params.strike_discount > 0.0,
        "Strike discount must be positive"
    );
    assert!(
        params.strike_reset_steps > 0,
        "Strike reset steps must be positive"
    );
    assert!(
        params.buyback_price.is_finite() && params.buyback_price >= 0.0,
        "Buyback price must be non-negative"
    );
    assert!(
        params.holder_put_trigger_price.is_finite() && params.holder_put_trigger_price >= 0.0,
        "Holder put trigger price must be non-negative"
    );
    assert!(
        params.holder_put_price.is_finite() && params.holder_put_price >= 0.0,
        "Holder put price must be non-negative"
    );
    assert!(
        params.exercise_limit_fraction.is_finite() && params.exercise_limit_fraction >= 0.0,
        "Exercise limit must be non-negative"
    );
    assert!(
        params.exercise_limit_period_steps > 0,
        "Exercise limit period steps must be positive"
    );
    assert!(params.num_paths > 0, "Number of paths must be positive");
    assert!(
        params.exercised_fraction_current_period >= 0.0,
        "Already exercised fraction must be non-negative"
    );
}

#[derive(Debug, Clone, Copy)]
struct State {
    spot: f64,
    strike: f64,
    time_to_maturity: f64,
    steps_to_strike_reset: usize,
    steps_to_quota_reset: usize,
    quota_base: f64,
    quota_period_index: usize,
}

impl Default for State {
    fn default() -> Self {
        Self {
            spot: 0.0,
            strike: 0.0,
            time_to_maturity: 0.0,
            steps_to_strike_reset: 0,
            steps_to_quota_reset: 0,
            quota_base: 0.0,
            quota_period_index: 0,
        }
    }
}

const BASIS_DIM: usize = 14;

fn basis(
    state: &State,
    dt: f64,
    available_quota: f64,
    remaining_fraction: f64,
) -> [f64; BASIS_DIM] {
    let strike = if state.strike.abs() < 1e-12 {
        state.spot.max(1e-12)
    } else {
        state.strike
    };
    let m = state.spot / strike - 1.0;
    let quota = available_quota.max(0.0);
    let remaining = remaining_fraction.max(0.0);
    let tau = state.time_to_maturity.max(0.0);
    let to_strike_reset = state.steps_to_strike_reset as f64 * dt;
    let to_quota_reset = state.steps_to_quota_reset as f64 * dt;

    [
        1.0,
        m,
        m * m,
        quota,
        quota * quota,
        remaining,
        remaining * remaining,
        tau,
        tau * tau,
        m * quota,
        m * remaining,
        quota * remaining,
        to_strike_reset,
        to_quota_reset,
    ]
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(feature = "python")]
#[allow(unsafe_op_in_unsafe_fn)]
#[pyfunction]
#[pyo3(signature = (
    initial_price,
    risk_free_rate,
    volatility,
    maturity,
    steps_per_year,
    strike_discount,
    strike_reset_steps,
    buyback_price,
    holder_put_trigger_price,
    holder_put_price,
    exercise_limit_fraction,
    exercise_limit_period_steps,
    next_limit_reset_step,
    exercised_fraction_current_period,
    num_paths,
    seed=None,
))]
fn price_warrant_py(
    initial_price: f64,
    risk_free_rate: f64,
    volatility: f64,
    maturity: f64,
    steps_per_year: usize,
    strike_discount: f64,
    strike_reset_steps: usize,
    buyback_price: f64,
    holder_put_trigger_price: f64,
    holder_put_price: f64,
    exercise_limit_fraction: f64,
    exercise_limit_period_steps: usize,
    next_limit_reset_step: usize,
    exercised_fraction_current_period: f64,
    num_paths: usize,
    seed: Option<u64>,
) -> PyResult<f64> {
    let params = FloatingStrikeWarrantParams {
        initial_price,
        risk_free_rate,
        volatility,
        maturity,
        steps_per_year,
        strike_discount,
        strike_reset_steps,
        buyback_price,
        holder_put_trigger_price,
        holder_put_price,
        exercise_limit_fraction,
        exercise_limit_period_steps,
        next_limit_reset_step,
        exercised_fraction_current_period,
        num_paths,
        seed,
    };

    Ok(price_warrant(&params))
}

#[cfg(feature = "python")]
#[allow(unsafe_op_in_unsafe_fn)]
#[pyfunction]
#[pyo3(signature = (
    initial_price,
    risk_free_rate,
    volatility,
    maturity,
    steps_per_year,
    strike_discount,
    strike_reset_steps,
    buyback_price,
    holder_put_trigger_price,
    holder_put_price,
    exercise_limit_fraction,
    exercise_limit_period_steps,
    next_limit_reset_step,
    exercised_fraction_current_period,
    num_paths,
    seed=None,
))]
fn price_warrant_timeseries_py(
    initial_price: f64,
    risk_free_rate: f64,
    volatility: f64,
    maturity: f64,
    steps_per_year: usize,
    strike_discount: f64,
    strike_reset_steps: usize,
    buyback_price: f64,
    holder_put_trigger_price: f64,
    holder_put_price: f64,
    exercise_limit_fraction: f64,
    exercise_limit_period_steps: usize,
    next_limit_reset_step: usize,
    exercised_fraction_current_period: f64,
    num_paths: usize,
    seed: Option<u64>,
) -> PyResult<(f64, Vec<(usize, f64, f64, f64, f64, f64)>)> {
    let params = FloatingStrikeWarrantParams {
        initial_price,
        risk_free_rate,
        volatility,
        maturity,
        steps_per_year,
        strike_discount,
        strike_reset_steps,
        buyback_price,
        holder_put_trigger_price,
        holder_put_price,
        exercise_limit_fraction,
        exercise_limit_period_steps,
        next_limit_reset_step,
        exercised_fraction_current_period,
        num_paths,
        seed,
    };

    let (price, series) = price_warrant_with_timeseries(&params);
    let records = series
        .into_iter()
        .map(|point: WarrantTimeseriesPoint| {
            (
                point.step,
                point.time,
                point.expected_price,
                point.exercise_upper_bound,
                point.put_lower_bound,
                point.buyback_upper_bound,
            )
        })
        .collect();

    Ok((price, records))
}

#[cfg(feature = "python")]
#[allow(deprecated)]
#[pymodule]
fn floating_strike_warrant(_py: Python<'_>, module: &PyModule) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(price_warrant_py, module)?)?;
    module.add_function(wrap_pyfunction!(price_warrant_timeseries_py, module)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn baseline_params() -> FloatingStrikeWarrantParams {
        FloatingStrikeWarrantParams {
            initial_price: 100.0,
            risk_free_rate: 0.01,
            volatility: 0.2,
            maturity: 1.0,
            steps_per_year: 52,
            strike_discount: 0.9,
            strike_reset_steps: 4,
            buyback_price: 100.0,
            holder_put_trigger_price: 0.0,
            holder_put_price: 0.0,
            exercise_limit_fraction: 0.2,
            exercise_limit_period_steps: 21,
            next_limit_reset_step: 10,
            exercised_fraction_current_period: 0.0,
            num_paths: 2048,
            seed: Some(42),
        }
    }

    fn approx_eq(a: f64, b: f64, tol: f64) {
        let diff = (a - b).abs();
        assert!(diff <= tol, "|{a} - {b}| = {diff} > {tol}");
    }

    #[test]
    fn zero_volatility_produces_small_value_when_atm() {
        let params = FloatingStrikeWarrantParams {
            initial_price: 100.0,
            risk_free_rate: 0.0,
            volatility: 0.0,
            maturity: 0.5,
            steps_per_year: 52,
            strike_discount: 1.0,
            strike_reset_steps: 1,
            buyback_price: 10.0,
            holder_put_trigger_price: 0.0,
            holder_put_price: 0.0,
            exercise_limit_fraction: 0.1,
            exercise_limit_period_steps: 21,
            next_limit_reset_step: 10,
            exercised_fraction_current_period: 0.0,
            num_paths: 1024,
            seed: Some(42),
        };

        let price = price_warrant(&params);
        assert!(price >= 0.0);
        assert!(price < 1.0);
    }

    #[test]
    fn positive_value_for_in_the_money_setup() {
        let params = FloatingStrikeWarrantParams {
            initial_price: 120.0,
            risk_free_rate: 0.01,
            volatility: 0.25,
            maturity: 1.0,
            steps_per_year: 52,
            strike_discount: 0.9,
            strike_reset_steps: 4,
            buyback_price: 40.0,
            holder_put_trigger_price: 0.0,
            holder_put_price: 0.0,
            exercise_limit_fraction: 0.2,
            exercise_limit_period_steps: 21,
            next_limit_reset_step: 5,
            exercised_fraction_current_period: 0.05,
            num_paths: 2048,
            seed: Some(7),
        };

        let price = price_warrant(&params);
        assert!(price.is_finite());
        assert!(price > 0.0);
    }

    #[test]
    fn price_is_zero_when_quota_fully_exercised() {
        let mut params = baseline_params();
        params.exercised_fraction_current_period = params.exercise_limit_fraction;
        params.next_limit_reset_step = params.steps_per_year * 2;
        params.maturity = 0.25;
        let price = price_warrant(&params);
        assert!(price <= 1e-10, "price = {price}");
    }

    #[test]
    fn price_is_zero_with_zero_buyback_price() {
        let mut params = baseline_params();
        params.buyback_price = 0.0;
        let price = price_warrant(&params);
        assert!(price.abs() <= 1e-10);
    }

    #[test]
    fn high_buyback_price_does_not_cap_value() {
        let mut capped = baseline_params();
        capped.buyback_price = 5.0;
        let capped_price = price_warrant(&capped);

        let mut uncapped = capped.clone();
        uncapped.buyback_price = 1_000.0;
        let uncapped_price = price_warrant(&uncapped);

        assert!(uncapped_price >= capped_price);
        assert!(uncapped_price > 0.0);
    }

    #[test]
    fn immediate_maturity_equals_intrinsic_times_quota() {
        let params = FloatingStrikeWarrantParams {
            maturity: 0.0,
            strike_reset_steps: 10,
            exercise_limit_fraction: 0.2,
            exercised_fraction_current_period: 0.05,
            ..baseline_params()
        };

        let expected_quota = (params.exercise_limit_fraction
            - params.exercised_fraction_current_period)
            .max(0.0)
            .min(params.exercise_limit_fraction);
        let intrinsic = params.initial_price * (1.0 - params.strike_discount);
        let expected = intrinsic * expected_quota;

        let price = price_warrant(&params);
        approx_eq(price, expected.min(params.buyback_price), 1e-9);
    }

    #[test]
    fn no_quota_limit_matches_unconstrained_behaviour() {
        let mut constrained = baseline_params();
        constrained.exercise_limit_fraction = 0.1;
        constrained.exercise_limit_period_steps = 21;
        let constrained_price = price_warrant(&constrained);

        let mut unconstrained = constrained.clone();
        unconstrained.exercise_limit_fraction = 1.0;
        unconstrained.exercise_limit_period_steps = 1;
        let unconstrained_price = price_warrant(&unconstrained);

        assert!(unconstrained_price >= constrained_price);
    }

    #[test]
    fn no_strike_reset_behaves_like_fixed_strike() {
        let mut params = baseline_params();
        params.strike_reset_steps = 10_000;
        let grid = super::simulate_state_grid(&params);
        for path in 0..params.num_paths {
            for step in 0..=grid.total_steps {
                let state = grid.states[step][path];
                approx_eq(
                    state.strike,
                    params.strike_discount * params.initial_price,
                    1e-9,
                );
            }
        }
    }

    #[test]
    fn reproducible_with_same_seed() {
        let params = baseline_params();
        let price1 = price_warrant(&params);
        let price2 = price_warrant(&params);
        approx_eq(price1, price2, 1e-12);
    }

    #[test]
    fn price_increases_with_volatility() {
        let mut params = baseline_params();
        params.exercise_limit_fraction = 1.0;
        params.exercise_limit_period_steps = 1;
        params.buyback_price = 1_000.0;
        params.strike_discount = 1.0;
        params.num_paths = 8192;
        params.volatility = 0.1;
        let low_vol = price_warrant(&params);
        params.volatility = 0.5;
        let high_vol = price_warrant(&params);
        assert!(high_vol > low_vol);
    }

    #[test]
    fn price_increases_with_higher_strike_discount() {
        let mut params = baseline_params();
        params.strike_discount = 0.9;
        let higher_discount = price_warrant(&params);
        params.strike_discount = 0.8;
        let deeper_discount = price_warrant(&params);
        assert!(deeper_discount > higher_discount);
    }

    #[test]
    fn strike_and_quota_reset_behaviour_matches_schedule() {
        let mut params = baseline_params();
        params.steps_per_year = 12;
        params.maturity = 0.5;
        params.strike_reset_steps = 2;
        params.exercise_limit_period_steps = 3;
        params.next_limit_reset_step = 2;
        params.exercise_limit_fraction = 0.2;
        params.num_paths = 1;
        params.volatility = 0.0;
        let grid = super::simulate_state_grid(&params);

        let states = &grid.states;
        // Strike should reset every 2 steps to discounted spot
        for step in 0..=grid.total_steps {
            let state = states[step][0];
            if step > 0 && step % params.strike_reset_steps == 0 {
                approx_eq(state.strike, state.spot * params.strike_discount, 1e-9);
            }

            let expected_steps_to_reset = if step == grid.total_steps {
                0
            } else {
                let modulo = step % params.strike_reset_steps;
                let until_next = if modulo == 0 {
                    params.strike_reset_steps
                } else {
                    params.strike_reset_steps - modulo
                };
                until_next.min(grid.total_steps - step)
            };
            assert_eq!(state.steps_to_strike_reset, expected_steps_to_reset);
        }

        // Quota should reset at the configured limit reset step
        let before_reset = states[params.next_limit_reset_step.saturating_sub(1)][0];
        let at_reset = states[params.next_limit_reset_step][0];
        assert!(at_reset.quota_base >= before_reset.quota_base);
        approx_eq(at_reset.quota_base, params.exercise_limit_fraction, 1e-9);
    }

    #[test]
    fn quota_refills_after_reset() {
        let mut params = baseline_params();
        params.exercise_limit_fraction = 0.15;
        params.exercised_fraction_current_period = 0.1;
        params.next_limit_reset_step = 1;
        params.num_paths = 1;
        params.volatility = 0.0;
        let grid = super::simulate_state_grid(&params);

        let before_reset = grid.states[0][0].quota_base;
        let after_reset = grid.states[params.next_limit_reset_step][0].quota_base;
        assert!(after_reset >= before_reset);
        approx_eq(after_reset, params.exercise_limit_fraction, 1e-9);
    }

    #[test]
    fn partial_exercise_occurs_under_quota_cap() {
        let mut params = baseline_params();
        params.num_paths = 1;
        params.volatility = 0.0;
        params.exercise_limit_fraction = 0.05;
        params.exercise_limit_period_steps = 10;
        params.buyback_price = 1_000.0;
        params.exercised_fraction_current_period = 0.98;
        params.next_limit_reset_step = 1;

        super::enable_exercise_logging();
        let price = price_warrant(&params);
        assert!(price > 0.0);
        let log = super::take_exercise_log();
        assert!(!log.is_empty());

        let mut partial_found = false;
        for record in log {
            if record.amount > 0.0 && record.amount < params.exercise_limit_fraction {
                partial_found = true;
                break;
            }
        }
        assert!(
            partial_found,
            "expected at least one partial exercise event"
        );
    }

    #[test]
    fn holder_put_triggers_when_spot_below_threshold() {
        let mut params = baseline_params();
        params.initial_price = 50.0;
        params.risk_free_rate = 0.0;
        params.strike_discount = 1.0;
        params.volatility = 0.0;
        params.maturity = 0.25;
        params.steps_per_year = 12;
        params.num_paths = 1;
        params.exercise_limit_fraction = 1.0;
        params.exercise_limit_period_steps = 1;
        params.holder_put_trigger_price = 60.0;
        params.holder_put_price = 7.5;
        params.buyback_price = 100.0;

        let price = price_warrant(&params);
        approx_eq(price, params.holder_put_price, 1e-9);
    }

    #[test]
    fn holder_put_respects_buyback_cap() {
        let mut params = baseline_params();
        params.initial_price = 30.0;
        params.risk_free_rate = 0.0;
        params.strike_discount = 1.0;
        params.volatility = 0.0;
        params.maturity = 0.25;
        params.steps_per_year = 12;
        params.num_paths = 1;
        params.exercise_limit_fraction = 1.0;
        params.exercise_limit_period_steps = 1;
        params.holder_put_trigger_price = 40.0;
        params.holder_put_price = 25.0;
        params.buyback_price = 5.0;

        let price = price_warrant(&params);
        approx_eq(price, params.buyback_price, 1e-9);
    }

    #[test]
    fn handles_large_number_of_paths() {
        let mut params = baseline_params();
        params.num_paths = 100_000;
        params.steps_per_year = 26;
        params.maturity = 0.5;
        let price = price_warrant(&params);
        assert!(price.is_finite());
        assert!(price >= 0.0);
    }
}
