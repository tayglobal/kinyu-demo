use chrono::NaiveDate;
use std::collections::BTreeMap;

// A simple day count convention
pub fn year_fraction(start: NaiveDate, end: NaiveDate) -> f64 {
    (end - start).num_days() as f64 / 365.0
}

/// Represents an interest rate curve, defined by zero rates at a set of pillar dates.
#[derive(Debug, Clone)]
pub struct DiscountCurve {
    pub base_date: NaiveDate,
    // Pillar dates and their corresponding zero rates.
    pub pillars: Vec<NaiveDate>,
    pub zero_rates: BTreeMap<NaiveDate, f64>,
}

impl DiscountCurve {
    /// Creates a new curve from a set of pillar dates and zero rates.
    pub fn new(base_date: NaiveDate, pillars: Vec<NaiveDate>, rates: Vec<f64>) -> Self {
        let mut zero_rates = BTreeMap::new();
        // The rate at the base date is assumed to be the first rate.
        zero_rates.insert(base_date, rates[0]);
        for (i, &pillar) in pillars.iter().enumerate() {
            // The first rate is for the period from base_date to pillars[0], and so on.
            // So rates[i] corresponds to pillars[i].
            zero_rates.insert(pillar, rates[i]);
        }

        DiscountCurve {
            base_date,
            pillars,
            zero_rates,
        }
    }

    /// Updates the zero rates of the curve. Used by the optimizer.
    pub fn update_rates(&mut self, rates: &[f64]) {
        for (i, pillar) in self.pillars.iter().enumerate() {
            self.zero_rates.insert(*pillar, rates[i]);
        }
    }

    /// Get the discount factor for a given date.
    /// Performs linear interpolation on zero rates.
    pub fn df(&self, date: NaiveDate) -> f64 {
        let zero_rate = self.zero_rate(date);
        let t = year_fraction(self.base_date, date);
        (-zero_rate * t).exp()
    }

    /// Get the zero rate for a given date, using linear interpolation.
    pub fn zero_rate(&self, date: NaiveDate) -> f64 {
        if date <= self.base_date {
            return 0.0;
        }

        // Check if we have the exact date
        if let Some(&rate) = self.zero_rates.get(&date) {
            return rate;
        }

        // Find bracketing dates for interpolation
        let (prev_date, prev_rate) = self.zero_rates.range(..date).next_back().unwrap_or((&self.base_date, &0.0));
        let (next_date, next_rate) = self.zero_rates.range(date..).next().unwrap_or((prev_date, prev_rate));

        if prev_date == next_date {
            return *prev_rate;
        }

        let t_prev = year_fraction(self.base_date, *prev_date);
        let t_next = year_fraction(self.base_date, *next_date);
        let t = year_fraction(self.base_date, date);

        // Linear interpolation of zero rates
        let r = prev_rate + (next_rate - prev_rate) * (t - t_prev) / (t_next - t_prev);

        r
    }
}