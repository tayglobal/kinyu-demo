use chrono::NaiveDate;

#[derive(Debug, Clone, Copy)]
pub enum Frequency {
    Quarterly,
    SemiAnnual,
    Annual,
}

#[derive(Debug, Clone, Copy)]
pub struct OIS {
    pub end_date: NaiveDate,
    pub rate: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct FRA {
    pub start_date: NaiveDate,
    pub end_date: NaiveDate,
    pub rate: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct IRFuture {
    pub start_date: NaiveDate,
    pub end_date: NaiveDate,
    pub price: f64,
}

impl IRFuture {
    /// The rate is derived from the price, typically as 100 - price.
    /// We'll return it as a decimal.
    pub fn rate(&self) -> f64 {
        (100.0 - self.price) / 100.0
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Swap {
    pub start_date: NaiveDate,
    pub end_date: NaiveDate,
    pub rate: f64,
    pub frequency: Frequency,
}

#[derive(Debug, Clone, Copy)]
pub enum Instrument {
    Ois(OIS),
    Fra(FRA),
    Future(IRFuture),
    Swap(Swap),
}

impl Instrument {
    pub fn end_date(&self) -> NaiveDate {
        match self {
            Instrument::Ois(i) => i.end_date,
            Instrument::Fra(i) => i.end_date,
            Instrument::Future(i) => i.end_date,
            Instrument::Swap(i) => i.end_date,
        }
    }

    pub fn rate(&self) -> f64 {
        match self {
            Instrument::Ois(i) => i.rate,
            Instrument::Fra(i) => i.rate,
            Instrument::Future(i) => i.rate(),
            Instrument::Swap(i) => i.rate,
        }
    }
}