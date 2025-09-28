use argmin::core::{CostFunction, Error as ArgminError, Executor, State};
use argmin::solver::neldermead::NelderMead;
use nalgebra::DVector;
use pyo3::create_exception;
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use thiserror::Error;

pub mod pricing {
    use super::{OptionData, OptionType};
    use roots::{find_root_brent, SimpleConvergency};
    use statrs::distribution::{ContinuousCDF, Normal};

    /// Calculates the Black-Scholes price of a European option.
    pub fn black_scholes_price(
        spot: f64,
        strike: f64,
        risk_free_rate: f64,
        time_to_expiry: f64,
        volatility: f64,
        option_type: OptionType,
    ) -> f64 {
        if time_to_expiry <= 0.0 || volatility <= 0.0 {
            return (spot - strike).max(0.0);
        }

        let d1 = ((spot / strike).ln()
            + (risk_free_rate + 0.5 * volatility.powi(2)) * time_to_expiry)
            / (volatility * time_to_expiry.sqrt());
        let d2 = d1 - volatility * time_to_expiry.sqrt();

        let normal = Normal::new(0.0, 1.0).unwrap();

        match option_type {
            OptionType::Call => {
                spot * normal.cdf(d1)
                    - strike * (-risk_free_rate * time_to_expiry).exp() * normal.cdf(d2)
            }
            OptionType::Put => {
                strike * (-risk_free_rate * time_to_expiry).exp() * normal.cdf(-d2)
                    - spot * normal.cdf(-d1)
            }
        }
    }

    /// Calculates the implied volatility for a given option price using Brent's method.
    pub fn implied_volatility(
        option_data: &OptionData,
        market_price: f64,
    ) -> Result<f64, String> {
        let mut convergency = SimpleConvergency {
            eps: 1e-9,
            max_iter: 100,
        };
        let f = |vol: f64| -> f64 {
            black_scholes_price(
                option_data.spot,
                option_data.strike,
                option_data.risk_free_rate,
                option_data.expiry,
                vol,
                option_data.option_type,
            ) - market_price
        };

        find_root_brent(0.0001, 1.0, &f, &mut convergency)
            .map_err(|e| format!("Root finding failed: {}", e))
    }
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OptionType {
    Call,
    Put,
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct OptionData {
    pub strike: f64,
    pub spot: f64,
    pub expiry: f64,
    pub price: f64,
    pub risk_free_rate: f64,
    pub option_type: OptionType,
    pub weight: f64,
}

#[pymethods]
impl OptionData {
    #[new]
    #[pyo3(signature = (strike, spot, expiry, price, risk_free_rate, option_type, weight=1.0))]
    fn new(
        strike: f64,
        spot: f64,
        expiry: f64,
        price: f64,
        risk_free_rate: f64,
        option_type: OptionType,
        weight: f64,
    ) -> Self {
        Self {
            strike,
            spot,
            expiry,
            price,
            risk_free_rate,
            option_type,
            weight,
        }
    }

    /// Calculates the log-forward moneyness `k = ln(K/F)`.
    /// The forward price F is calculated as F = S * exp(rT).
    pub fn moneyness(&self) -> f64 {
        self.strike.ln() - self.spot.ln() - self.risk_free_rate * self.expiry
    }
}

#[pyclass(get_all, set_all)]
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SVIParameters {
    pub a: f64,
    pub b: f64,
    pub rho: f64,
    pub m: f64,
    pub sigma: f64,
}

#[pymethods]
impl SVIParameters {
    #[new]
    fn new(a: f64, b: f64, rho: f64, m: f64, sigma: f64) -> Self {
        Self {
            a,
            b,
            rho,
            m,
            sigma,
        }
    }
}

#[derive(Error, Debug)]
pub enum CalibrationError {
    #[error("Not enough data points to calibrate the model")]
    NotEnoughData,
    #[error("Optimization failed to converge: {0}")]
    OptimizationFailed(String),
    #[error("Invalid input parameters: {0}")]
    InvalidInput(String),
}

create_exception!(kinyu_implied_vol, PyCalibrationError, pyo3::exceptions::PyException);

impl From<CalibrationError> for PyErr {
    fn from(err: CalibrationError) -> PyErr {
        PyCalibrationError::new_err(err.to_string())
    }
}

struct SVICost<'a> {
    options: &'a [OptionData],
}

impl CostFunction for SVICost<'_> {
    type Param = DVector<f64>;
    type Output = f64;

    fn cost(&self, params: &Self::Param) -> Result<Self::Output, ArgminError> {
        let svi_params = SVIParameters {
            a: params[0],
            b: params[1],
            rho: params[2],
            m: params[3],
            sigma: params[4],
        };

        if svi_params.b < 0.0 || svi_params.sigma <= 0.0 || svi_params.rho.abs() >= 1.0 {
            return Ok(f64::INFINITY);
        }
        if svi_params.a + svi_params.b * svi_params.sigma * (1.0 - svi_params.rho.powi(2)).sqrt()
            < 0.0
        {
            return Ok(f64::INFINITY);
        }

        let errors: Vec<f64> = self
            .options
            .iter()
            .map(|option| {
                let implied_vol = match pricing::implied_volatility(option, option.price) {
                    Ok(vol) => vol,
                    Err(_) => return f64::NAN, // Signal failure for this point
                };

                let market_total_variance = implied_vol.powi(2) * option.expiry;

                let k = option.moneyness();
                let model_total_variance = svi_params.a
                    + svi_params.b
                        * (svi_params.rho * (k - svi_params.m)
                            + ((k - svi_params.m).powi(2) + svi_params.sigma.powi(2)).sqrt());

                let error = model_total_variance - market_total_variance;
                option.weight * error.powi(2)
            })
            .collect();

        if errors.iter().any(|&e| e.is_nan()) {
            return Ok(f64::INFINITY);
        }

        let total_error: f64 = errors.into_iter().sum();
        Ok(total_error)
    }
}

#[pyclass]
pub struct SVIVolatilitySurface {
    // We use a BTreeMap to keep the expiries sorted.
    // The key is u64 representation of f64 expiry to ensure Ord trait is implemented.
    slices: BTreeMap<u64, SVIVolatilitySlice>,
}

#[pymethods]
impl SVIVolatilitySurface {
    #[staticmethod]
    #[pyo3(signature = (options, initial_params_map=None))]
    pub fn calibrate(
        options: Vec<OptionData>,
        initial_params_map: Option<BTreeMap<u64, SVIParameters>>,
    ) -> PyResult<Self> {
        let mut options_by_expiry: BTreeMap<u64, Vec<OptionData>> = BTreeMap::new();
        for option in &options {
            options_by_expiry
                .entry(option.expiry.to_bits())
                .or_default()
                .push(*option);
        }

        let mut calibrated_slices = BTreeMap::new();
        for (expiry_bits, opts) in options_by_expiry {
            let initial_params = initial_params_map
                .as_ref()
                .and_then(|m| m.get(&expiry_bits).copied());
            let slice = SVIVolatilitySlice::calibrate(&opts, initial_params)?;
            calibrated_slices.insert(expiry_bits, slice);
        }

        Ok(SVIVolatilitySurface {
            slices: calibrated_slices,
        })
    }

    pub fn volatility(&self, moneyness: f64, expiry: f64) -> Option<f64> {
        if self.slices.is_empty() {
            return None;
        }

        let expiry_bits = expiry.to_bits();
        // Check for an exact match first
        if let Some(slice) = self.slices.get(&expiry_bits) {
            return Some(slice.volatility(moneyness));
        }

        // If no exact match, find the closest slice by expiry
        let next_slice = self.slices.range(expiry_bits..).next();
        let prev_slice = self.slices.range(..expiry_bits).next_back();

        match (prev_slice, next_slice) {
            (Some((_, prev)), Some((_, next))) => {
                if (expiry - prev.expiry).abs() < (expiry - next.expiry).abs() {
                    Some(prev.volatility(moneyness))
                } else {
                    Some(next.volatility(moneyness))
                }
            }
            (Some((_, prev)), None) => Some(prev.volatility(moneyness)),
            (None, Some((_, next))) => Some(next.volatility(moneyness)),
            (None, None) => None, // Should be unreachable if slices is not empty
        }
    }
}

struct SVIVolatilitySlice {
    params: SVIParameters,
    expiry: f64,
}

impl SVIVolatilitySlice {
    fn calibrate(
        options: &[OptionData],
        initial_params: Option<SVIParameters>,
    ) -> Result<Self, CalibrationError> {
        if options.len() < 5 {
            return Err(CalibrationError::NotEnoughData);
        }

        let first_expiry = options[0].expiry;
        if options
            .iter()
            .any(|opt| (opt.expiry - first_expiry).abs() > 1e-9)
        {
            return Err(CalibrationError::InvalidInput(
                "All options must have the same expiry for a slice calibration".to_string(),
            ));
        }
        if first_expiry <= 0.0 {
            return Err(CalibrationError::InvalidInput(
                "Expiry must be positive".to_string(),
            ));
        }

        let cost_function = SVICost { options };

        let p = initial_params.unwrap_or_else(|| {
            let mid_moneyness =
                options.iter().map(|o| o.moneyness()).sum::<f64>() / options.len() as f64;
            SVIParameters {
                a: 0.1,
                b: 0.1,
                rho: -0.5,
                m: mid_moneyness,
                sigma: 0.1,
            }
        });

        let initial_guess = DVector::from_vec(vec![p.a, p.b, p.rho, p.m, p.sigma]);

        let mut initial_simplex = vec![initial_guess.clone()];
        for i in 0..initial_guess.len() {
            let mut point = initial_guess.clone();
            point[i] *= 1.2;
            initial_simplex.push(point);
        }

        let solver = NelderMead::new(initial_simplex);

        let res = Executor::new(cost_function, solver)
            .configure(|state| state.max_iters(20000))
            .run()
            .map_err(|e| CalibrationError::OptimizationFailed(e.to_string()))?;

        let best_params = res.state().get_best_param().ok_or_else(|| {
            CalibrationError::OptimizationFailed(
                "Optimizer did not return a best parameter set.".to_string(),
            )
        })?;

        Ok(SVIVolatilitySlice {
            params: SVIParameters {
                a: best_params[0],
                b: best_params[1],
                rho: best_params[2],
                m: best_params[3],
                sigma: best_params[4],
            },
            expiry: first_expiry,
        })
    }

    fn volatility(&self, moneyness: f64) -> f64 {
        let k = moneyness;
        let p = &self.params;
        let total_variance = p.a
            + p.b
                * (p.rho * (k - p.m)
                    + ((k - p.m).powi(2) + p.sigma.powi(2)).sqrt());

        if total_variance < 0.0 {
            return 0.0;
        }

        (total_variance / self.expiry).sqrt()
    }
}


#[pyfunction]
fn black_scholes_price_py(
    spot: f64,
    strike: f64,
    risk_free_rate: f64,
    time_to_expiry: f64,
    volatility: f64,
    option_type: OptionType,
) -> f64 {
    pricing::black_scholes_price(spot, strike, risk_free_rate, time_to_expiry, volatility, option_type)
}

#[pyfunction]
fn implied_volatility_py(option_data: &OptionData, market_price: f64) -> PyResult<f64> {
    pricing::implied_volatility(option_data, market_price).map_err(|e| PyCalibrationError::new_err(e))
}

#[pymodule]
fn implied(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<OptionType>()?;
    m.add_class::<OptionData>()?;
    m.add_class::<SVIParameters>()?;
    m.add_class::<SVIVolatilitySurface>()?;
    m.add("CalibrationError", _py.get_type::<PyCalibrationError>())?;
    m.add_function(wrap_pyfunction!(black_scholes_price_py, m)?)?;
    m.add_function(wrap_pyfunction!(implied_volatility_py, m)?)?;
    Ok(())
}


#[cfg(test)]
mod pricing_tests {
    use super::pricing::*;
    use super::{OptionData, OptionType};

    #[test]
    fn test_black_scholes_pricing() {
        let spot = 100.0;
        let strike = 105.0;
        let risk_free_rate = 0.05;
        let time_to_expiry = 1.0;
        let volatility = 0.2;

        let call_price = black_scholes_price(
            spot,
            strike,
            risk_free_rate,
            time_to_expiry,
            volatility,
            OptionType::Call,
        );
        let put_price = black_scholes_price(
            spot,
            strike,
            risk_free_rate,
            time_to_expiry,
            volatility,
            OptionType::Put,
        );

        // Expected values calculated from an external source
        assert!((call_price - 8.021).abs() < 0.001);
        assert!((put_price - 7.900).abs() < 0.001);
    }

    #[test]
    fn test_implied_volatility_calculation() {
        let option = OptionData {
            strike: 105.0,
            spot: 100.0,
            expiry: 1.0,
            price: 8.021, // Market price
            risk_free_rate: 0.05,
            option_type: OptionType::Call,
            weight: 1.0,
        };

        let implied_vol = implied_volatility(&option, option.price).unwrap();
        assert!((implied_vol - 0.2).abs() < 1e-4);
    }
}

#[cfg(test)]
mod surface_tests {
    use super::*;
    use rand::Rng;

    // Generates realistic option prices based on a known "true" SVI smile.
    fn generate_realistic_options(
        num_options: usize,
        expiry: f64,
        true_svi: &SVIParameters,
    ) -> Vec<OptionData> {
        let mut rng = rand::thread_rng();
        let spot = 100.0;
        let risk_free_rate = 0.05;

        (0..num_options)
            .map(|_| {
                let strike = rng.gen_range(80.0..120.0);
                let mut option = OptionData {
                    strike,
                    spot,
                    expiry,
                    price: 0.0, // Placeholder
                    risk_free_rate,
                    option_type: OptionType::Call,
                    weight: 1.0,
                };

                let k = option.moneyness();
                let total_variance = true_svi.a
                    + true_svi.b
                        * (true_svi.rho * (k - true_svi.m)
                            + ((k - true_svi.m).powi(2) + true_svi.sigma.powi(2)).sqrt());
                if total_variance < 0.0 {
                    return None; // Avoid generating options with negative variance
                }
                let vol = (total_variance / expiry).sqrt();

                option.price = pricing::black_scholes_price(
                    spot,
                    strike,
                    risk_free_rate,
                    expiry,
                    vol,
                    option.option_type,
                );
                Some(option)
            })
            .filter_map(|x| x) // Filter out None values
            .collect()
    }

    #[test]
    fn test_surface_calibration_and_querying() {
        let true_svi_1y = SVIParameters {
            a: 0.04,
            b: 0.4,
            rho: -0.7,
            m: 0.1,
            sigma: 0.2,
        };
        let true_svi_2y = SVIParameters {
            a: 0.035,
            b: 0.35,
            rho: -0.65,
            m: 0.12,
            sigma: 0.22,
        };

        let mut options = generate_realistic_options(50, 1.0, &true_svi_1y);
        options.extend(generate_realistic_options(50, 2.0, &true_svi_2y));

        let surface = SVIVolatilitySurface::calibrate(options, None).unwrap();

        // 1. Check that two slices were created
        assert_eq!(surface.slices.len(), 2);

        // 2. Test exact expiry match
        let vol_1y = surface.volatility(0.0, 1.0).unwrap();
        let expected_vol_1y = surface.slices.get(&1.0f64.to_bits()).unwrap().volatility(0.0);
        assert_eq!(vol_1y, expected_vol_1y);

        // 3. Test closest expiry match
        let vol_1_1y = surface.volatility(0.0, 1.1).unwrap(); // 1.1 is closer to 1.0
        assert_eq!(vol_1_1y, expected_vol_1y);

        let vol_1_8y = surface.volatility(0.0, 1.8).unwrap(); // 1.8 is closer to 2.0
        let expected_vol_2y = surface.slices.get(&2.0f64.to_bits()).unwrap().volatility(0.0);
        assert_eq!(vol_1_8y, expected_vol_2y);

        // 4. Test out of bounds query (should return None as extrapolation is not implemented)
        // Note: My current implementation finds the *closest*, so it will never return None if slices exist.
        // This test is to confirm that behavior.
        let vol_far_past = surface.volatility(0.0, 0.1).unwrap();
        assert_eq!(vol_far_past, expected_vol_1y);

        let vol_far_future = surface.volatility(0.0, 5.0).unwrap();
        assert_eq!(vol_far_future, expected_vol_2y);
    }
}