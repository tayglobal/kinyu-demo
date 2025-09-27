use chrono::NaiveDate;
use curve_builder::builder::CurveBuilder;
use curve_builder::instruments::{Instrument, OIS, FRA, Swap, Frequency, IRFuture};
use curve_builder::pricing::price_swap_npv;

fn main() {
    // 1. Set up the market data
    let base_date = NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();

    let instruments = vec![
        // OIS for the short end
        Instrument::Ois(OIS { end_date: NaiveDate::from_ymd_opt(2023, 2, 1).unwrap(), rate: 0.005 }), // 1M
        Instrument::Ois(OIS { end_date: NaiveDate::from_ymd_opt(2023, 4, 1).unwrap(), rate: 0.006 }), // 3M

        // FRAs/Futures for the middle part
        Instrument::Fra(FRA { start_date: NaiveDate::from_ymd_opt(2023, 4, 1).unwrap(), end_date: NaiveDate::from_ymd_opt(2023, 7, 1).unwrap(), rate: 0.0075 }),
        Instrument::Future(IRFuture { start_date: NaiveDate::from_ymd_opt(2023, 7, 1).unwrap(), end_date: NaiveDate::from_ymd_opt(2023, 10, 1).unwrap(), price: 99.20 }), // rate = 0.8%

        // Swaps for the long end
        Instrument::Swap(Swap { start_date: base_date, end_date: NaiveDate::from_ymd_opt(2024, 1, 1).unwrap(), rate: 0.01, frequency: Frequency::SemiAnnual }), // 1Y
        Instrument::Swap(Swap { start_date: base_date, end_date: NaiveDate::from_ymd_opt(2025, 1, 1).unwrap(), rate: 0.015, frequency: Frequency::SemiAnnual }), // 2Y
        Instrument::Swap(Swap { start_date: base_date, end_date: NaiveDate::from_ymd_opt(2028, 1, 1).unwrap(), rate: 0.02, frequency: Frequency::SemiAnnual }), // 5Y
    ];

    // 2. Build the curve using the global optimization builder
    println!("--- Building Curve with Global Optimization ---");
    let builder = CurveBuilder::new(base_date, instruments);
    let curve = builder.build();

    println!("\n--- Globally Optimized Discount Curve ---");
    println!("Base Date: {}", curve.base_date);
    println!("Pillar Dates: {:?}", curve.pillars);
    println!("Optimized Zero Rates:");
    for (date, rate) in &curve.zero_rates {
        println!("  Date: {}, Zero Rate: {:.6}%", date, rate * 100.0);
    }
    println!("----------------------------------------\n");

    // 3. Define a new swap to price
    let swap_to_price = Swap {
        start_date: base_date,
        end_date: NaiveDate::from_ymd_opt(2026, 1, 1).unwrap(), // 3Y
        rate: 0.021, // 2.1% fixed rate
        frequency: Frequency::SemiAnnual,
    };

    println!("--- Pricing a 3-Year Interest Rate Swap ---");
    println!("Start Date: {}", swap_to_price.start_date);
    println!("End Date: {}", swap_to_price.end_date);
    println!("Fixed Rate: {}%", swap_to_price.rate * 100.0);
    println!("-------------------------------------------\n");

    // 4. Price the swap using the new curve
    let npv = price_swap_npv(&swap_to_price, &curve);

    println!("--- Swap Pricing Result ---");
    println!("The NPV of the swap is: {:.6}", npv);
    if npv > 0.0 {
        println!("This is favorable to the floating-rate payer / fixed-rate receiver.");
    } else {
        println!("This is favorable to the fixed-rate payer / floating-rate receiver.");
    }
    println!("---------------------------\n");
}