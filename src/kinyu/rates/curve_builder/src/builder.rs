use crate::curve::DiscountCurve;
use crate::instruments::Instrument;
use crate::pricing;
use argmin::core::{CostFunction, Error, Executor, State};
use argmin::solver::neldermead::NelderMead;
use chrono::NaiveDate;
use nalgebra::DVector;
use std::vec;

/// Builds a discount curve by finding the set of zero rates that best fits the market instruments.
#[derive(Clone)]
pub struct CurveBuilder {
    pub base_date: NaiveDate,
    pub instruments: Vec<Instrument>,
    pub pillars: Vec<NaiveDate>,
}

impl CurveBuilder {
    pub fn new(base_date: NaiveDate, instruments: Vec<Instrument>) -> Self {
        // Determine pillar dates from the instruments' end dates.
        let mut pillars: Vec<NaiveDate> = instruments.iter().map(|i| i.end_date()).collect();
        pillars.sort();
        pillars.dedup();

        CurveBuilder {
            base_date,
            instruments,
            pillars,
        }
    }

    /// Run the optimization to build the curve.
    pub fn build(&self) -> DiscountCurve {
        // The Nelder-Mead solver requires an initial simplex (a set of starting points).
        // We'll create one around an initial guess for the zero rates (e.g., all 1%).
        let initial_guess = DVector::from_vec(vec![0.01; self.pillars.len()]);
        let mut initial_simplex = Vec::new();
        initial_simplex.push(initial_guess.clone());
        for i in 0..self.pillars.len() {
            let mut next_guess = initial_guess.clone();
            next_guess[i] += 0.005; // Create points around the initial guess
            initial_simplex.push(next_guess);
        }

        // Set up the Nelder-Mead solver.
        let solver = NelderMead::new(initial_simplex);

        // Run the optimization
        let res = Executor::new(self.clone(), solver)
            .configure(|state| state.max_iters(1000))
            .run()
            .expect("Optimization failed");

        // The best parameters are the optimized zero rates.
        let best_rates = res.state.get_best_param().unwrap();

        // Create the final curve with the optimized rates.
        DiscountCurve::new(self.base_date, self.pillars.clone(), best_rates.data.as_vec().to_vec())
    }
}

/// Implement the CostFunction trait for CurveBuilder to be used with argmin.
impl CostFunction for CurveBuilder {
    type Param = DVector<f64>;
    type Output = f64;

    /// The cost function calculates the sum of squared errors between the
    /// market rates and the rates implied by the curve.
    fn cost(&self, param: &Self::Param) -> Result<Self::Output, Error> {
        let rates = param.data.as_vec();
        let curve = DiscountCurve::new(self.base_date, self.pillars.clone(), rates.to_vec());

        let mut total_error = 0.0;

        for instrument in &self.instruments {
            let market_rate = instrument.rate();
            let priced_rate = match instrument {
                Instrument::Ois(ois) => pricing::price_ois_rate(ois, &curve),
                Instrument::Fra(fra) => pricing::price_fra_rate(fra, &curve),
                Instrument::Future(future) => pricing::price_future_rate(future, &curve),
                Instrument::Swap(swap) => pricing::price_swap_rate(swap, &curve),
            };
            total_error += (market_rate - priced_rate).powi(2);
        }
        Ok(total_error)
    }
}