use crate::curve::{DiscountCurve, year_fraction};
use crate::instruments::{Swap, Frequency, OIS, FRA, IRFuture};
use chrono::{NaiveDate, Duration};

/// Prices an interest rate swap's NPV for a given fixed rate.
pub fn price_swap_npv(swap: &Swap, curve: &DiscountCurve) -> f64 {
    let pv_fixed = calculate_fixed_leg_pv(swap, curve);
    let pv_floating = calculate_floating_leg_pv(swap, curve);
    pv_floating - pv_fixed
}

/// Calculates the par rate for a swap (the rate that makes NPV=0).
pub fn price_swap_rate(swap: &Swap, curve: &DiscountCurve) -> f64 {
    let floating_leg_pv = calculate_floating_leg_pv(swap, curve);
    let annuity = calculate_annuity(swap, curve);
    if annuity == 0.0 { 0.0 } else { floating_leg_pv / annuity }
}

/// Calculates the rate for an OIS instrument from the curve.
pub fn price_ois_rate(ois: &OIS, curve: &DiscountCurve) -> f64 {
    let t = year_fraction(curve.base_date, ois.end_date);
    let df = curve.df(ois.end_date);
    (1.0 / df - 1.0) / t
}

/// Calculates the forward rate for a FRA instrument from the curve.
pub fn price_fra_rate(fra: &FRA, curve: &DiscountCurve) -> f64 {
    let df_start = curve.df(fra.start_date);
    let df_end = curve.df(fra.end_date);
    let t = year_fraction(fra.start_date, fra.end_date);
    (df_start / df_end - 1.0) / t
}

/// Calculates the implied rate for an IR Future from the curve.
/// This is treated identically to a FRA for pricing purposes.
pub fn price_future_rate(future: &IRFuture, curve: &DiscountCurve) -> f64 {
    let df_start = curve.df(future.start_date);
    let df_end = curve.df(future.end_date);
    let t = year_fraction(future.start_date, future.end_date);
    (df_start / df_end - 1.0) / t
}

/// Calculates the Present Value of the fixed leg of a swap.
fn calculate_fixed_leg_pv(swap: &Swap, curve: &DiscountCurve) -> f64 {
    swap.rate * calculate_annuity(swap, curve)
}

/// Calculates the annuity of the fixed leg (sum of year fractions * discount factors).
fn calculate_annuity(swap: &Swap, curve: &DiscountCurve) -> f64 {
    let payment_dates = generate_payment_dates(swap.start_date, swap.end_date, swap.frequency);
    let mut annuity = 0.0;
    let mut prev_date = swap.start_date;

    for date in payment_dates {
        let tau = year_fraction(prev_date, date);
        let df = curve.df(date);
        annuity += tau * df;
        prev_date = date;
    }
    annuity
}

/// Calculates the Present Value of the floating leg of a swap.
fn calculate_floating_leg_pv(swap: &Swap, curve: &DiscountCurve) -> f64 {
    // PV of floating leg is DF(start) - DF(end)
    curve.df(swap.start_date) - curve.df(swap.end_date)
}

// This helper function should be in a shared utility module.
fn generate_payment_dates(start: NaiveDate, end: NaiveDate, freq: Frequency) -> Vec<NaiveDate> {
    let mut dates = Vec::new();
    let mut current = start;

    let period_days = match freq {
        Frequency::Quarterly => 91,
        Frequency::SemiAnnual => 182,
        Frequency::Annual => 365,
    };

    loop {
        current = current + Duration::days(period_days);
        if current >= end {
            dates.push(end);
            break;
        }
        dates.push(current);
    }
    dates
}