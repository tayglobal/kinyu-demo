use nalgebra::{DMatrix, DVector};
use rand::Rng;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand_distr::StandardNormal;

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

/// Price a floating-strike warrant with strike resets, issuer call right and exercise quota limits using LSMC.
///
/// The implementation follows a quota-aware Longstaff-Schwartz Monte Carlo scheme with a polynomial basis.
/// SVD is used to stabilise the regression at each backward induction step.
pub fn price_warrant(params: &FloatingStrikeWarrantParams) -> f64 {
    validate(params);

    let total_steps = (params.maturity * params.steps_per_year as f64).ceil() as usize;
    let total_steps = total_steps.max(1);
    let dt = params.maturity / total_steps as f64;
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

        for step in 0..=total_steps {
            if step == next_quota_reset {
                quota = params.exercise_limit_fraction.min(remaining);
                next_quota_reset = next_quota_reset
                    .saturating_add(params.exercise_limit_period_steps)
                    .min(total_steps + params.exercise_limit_period_steps);
            }

            let steps_to_quota_reset = next_quota_reset.saturating_sub(step);
            let steps_to_strike_reset = if params.strike_reset_steps == 0 {
                0
            } else {
                let remainder = params.strike_reset_steps - (step % params.strike_reset_steps);
                if remainder == params.strike_reset_steps {
                    0
                } else {
                    remainder
                }
            };

            states[step][path] = State {
                spot,
                strike,
                quota,
                remaining,
                time_to_maturity: (total_steps - step) as f64 * dt,
                steps_to_strike_reset,
                steps_to_quota_reset,
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

    let mut values = vec![0.0; params.num_paths];
    for path in 0..params.num_paths {
        let state = states[total_steps][path];
        let intrinsic = (state.spot - state.strike).max(0.0);
        let available = state.quota.min(state.remaining).max(0.0);
        let value = intrinsic * available;
        values[path] = value.min(params.buyback_price);
    }

    let mut discounted_future = vec![0.0; params.num_paths];

    for step in (0..total_steps).rev() {
        let mut design_matrix = Vec::new();
        let mut responses = Vec::new();
        let mut itm_paths = Vec::new();

        for path in 0..params.num_paths {
            discounted_future[path] = values[path] * discount_factor;
            let state = states[step][path];
            let intrinsic = (state.spot - state.strike).max(0.0);
            let available = state.quota.min(state.remaining).max(0.0);

            if intrinsic > 0.0 && available > 0.0 {
                let basis = basis(&state, dt);
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
            let intrinsic = (state.spot - state.strike).max(0.0);
            let available = state.quota.min(state.remaining).max(0.0);
            let continuation = continuation_estimates[path].unwrap_or(discounted_future[path]);

            let immediate = intrinsic * available;

            let should_exercise = if intrinsic > 0.0 && available > 1e-8 {
                if let Some(beta) = &beta_opt {
                    let dq = available.min(0.01);
                    if dq > 0.0 {
                        let cont_current = continuation_estimates[path].unwrap_or_else(|| {
                            let phi = basis(&state, dt);
                            dot(&phi, beta)
                        });
                        let mut reduced_state = state;
                        reduced_state.quota = (state.quota - dq).max(0.0);
                        let phi_minus = basis(&reduced_state, dt);
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

            let mut value = if should_exercise && immediate > 0.0 {
                immediate
            } else {
                continuation
            };

            value = value.min(params.buyback_price);
            values[path] = value;
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
        params.maturity.is_finite() && params.maturity > 0.0,
        "Maturity must be positive"
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
    quota: f64,
    remaining: f64,
    time_to_maturity: f64,
    steps_to_strike_reset: usize,
    steps_to_quota_reset: usize,
}

impl Default for State {
    fn default() -> Self {
        Self {
            spot: 0.0,
            strike: 0.0,
            quota: 0.0,
            remaining: 0.0,
            time_to_maturity: 0.0,
            steps_to_strike_reset: 0,
            steps_to_quota_reset: 0,
        }
    }
}

const BASIS_DIM: usize = 14;

fn basis(state: &State, dt: f64) -> [f64; BASIS_DIM] {
    let strike = if state.strike.abs() < 1e-12 {
        state.spot.max(1e-12)
    } else {
        state.strike
    };
    let m = state.spot / strike - 1.0;
    let quota = state.quota.max(0.0);
    let remaining = state.remaining.max(0.0);
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

#[cfg(test)]
mod tests {
    use super::*;

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
}
