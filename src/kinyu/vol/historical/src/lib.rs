use nalgebra::DMatrix;
use pyo3::prelude::*;
use numpy::PyReadonlyArray2;

/// Historical Volatility Estimators
///
/// This struct holds OHLC price data and provides methods to calculate historical volatility
/// using various estimators.
/// The input matrix should have columns in the order: Open, High, Low, Close.
pub struct HistoricalVolatility {
    // open, high, low, close
    prices: DMatrix<f64>,
}

impl HistoricalVolatility {
    pub fn new(prices: DMatrix<f64>) -> Self {
        if prices.ncols() != 4 {
            panic!("Input matrix must have 4 columns for Open, High, Low, Close.");
        }
        HistoricalVolatility { prices }
    }

    /// Simple close-to-close historical volatility estimator.
    pub fn close_to_close(&self) -> f64 {
        let close_prices = self.prices.column(3);
        let mut log_returns = DMatrix::from_element(close_prices.nrows() - 1, 1, 0.0);
        for i in 1..close_prices.nrows() {
            log_returns[i - 1] = (close_prices[i] / close_prices[i - 1]).ln();
        }

        let n = log_returns.nrows() as f64;
        let mean = log_returns.mean();
        let variance = log_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
        variance.sqrt()
    }

    /// Parkinson (1980) estimator.
    /// Uses high and low prices.
    pub fn parkinson(&self) -> f64 {
        let n = self.prices.nrows() as f64;
        let sum_of_squares = self.prices.row_iter().map(|row| {
            let high = row[1];
            let low = row[2];
            (high / low).ln().powi(2)
        }).sum::<f64>();

        (sum_of_squares / (n * 4.0 * (2.0f64).ln())).sqrt()
    }

    /// Garman-Klass (1980) estimator.
    /// Uses open, high, low, and close prices.
    pub fn garman_klass(&self) -> f64 {
        let n = self.prices.nrows() as f64;
        let sum = self.prices.row_iter().map(|row| {
            let open = row[0];
            let high = row[1];
            let low = row[2];
            let close = row[3];
            let hl_ratio = (high / low).ln().powi(2);
            let co_ratio = (close / open).ln().powi(2);
            0.5 * hl_ratio - (2.0 * (2.0f64).ln() - 1.0) * co_ratio
        }).sum::<f64>();

        if sum < 0.0 {
            return 0.0;
        }

        (sum / n).sqrt()
    }

    /// Rogers-Satchell (1991) estimator.
    /// Uses open, high, low, and close prices.
    pub fn rogers_satchell(&self) -> f64 {
        let n = self.prices.nrows() as f64;
        let sum = self.prices.row_iter().map(|row| {
            let o = row[0];
            let h = row[1];
            let l = row[2];
            let c = row[3];
            (h / c).ln() * (h / o).ln() + (l / c).ln() * (l / o).ln()
        }).sum::<f64>();

        (sum / n).sqrt()
    }

    /// Yang-Zhang (2000) estimator.
    /// This estimator is a combination of overnight volatility and intraday volatility.
    pub fn yang_zhang(&self) -> f64 {
        let n_int = self.prices.nrows();
        let n = n_int as f64;

        // Overnight volatility (close-to-open)
        let overnight_returns: Vec<f64> = (0..n_int - 1).map(|i| {
            let open_i1 = self.prices[(i + 1, 0)];
            let close_i = self.prices[(i, 3)];
            (open_i1 / close_i).ln()
        }).collect();
        let overnight_mean = overnight_returns.iter().sum::<f64>() / (n - 1.0);
        let overnight_var = overnight_returns.iter().map(|r| (r - overnight_mean).powi(2)).sum::<f64>() / (n - 2.0);

        // Open-to-close volatility
        let open_close_returns: Vec<f64> = self.prices.row_iter().map(|row| {
            let open = row[0];
            let close = row[3];
            (close / open).ln()
        }).collect();
        let open_close_mean = open_close_returns.iter().sum::<f64>() / n;
        let open_close_var = open_close_returns.iter().map(|r| (r - open_close_mean).powi(2)).sum::<f64>() / (n - 1.0);

        let rogers_satchell_vol_sq = self.rogers_satchell().powi(2);

        let k = 0.34 / (1.34 + (n + 1.0) / (n - 1.0));

        (overnight_var + k * open_close_var + (1.0 - k) * rogers_satchell_vol_sq).sqrt()
    }
}


#[pyclass(name = "HistoricalVolatility")]
struct PyHistoricalVolatility {
    hv: HistoricalVolatility,
}

#[pymethods]
impl PyHistoricalVolatility {
    #[new]
    fn new(prices: PyReadonlyArray2<f64>) -> Self {
        let prices_slice = prices.as_slice().expect("Input array must be C-contiguous");
        let prices_mat = DMatrix::from_row_slice(
            prices.shape()[0],
            prices.shape()[1],
            prices_slice,
        );
        PyHistoricalVolatility {
            hv: HistoricalVolatility::new(prices_mat),
        }
    }

    fn close_to_close(&self) -> f64 {
        self.hv.close_to_close()
    }

    fn parkinson(&self) -> f64 {
        self.hv.parkinson()
    }

    fn garman_klass(&self) -> f64 {
        self.hv.garman_klass()
    }

    fn rogers_satchell(&self) -> f64 {
        self.hv.rogers_satchell()
    }

    fn yang_zhang(&self) -> f64 {
        self.hv.yang_zhang()
    }
}

#[pymodule]
fn kinyu_historical(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyHistoricalVolatility>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;
    use approx::assert_relative_eq;

    fn get_sample_data() -> DMatrix<f64> {
        // Sample data from https://www.ivolatility.com/help/3.2.1.h_volatility.html
        // For simplicity, we use a small dataset.
        // O, H, L, C
        DMatrix::from_rows(&[
            nalgebra::RowDVector::from_vec(vec![100.0, 102.0, 99.0, 101.0]),
            nalgebra::RowDVector::from_vec(vec![101.0, 103.0, 100.5, 102.5]),
            nalgebra::RowDVector::from_vec(vec![102.5, 104.0, 102.0, 103.0]),
            nalgebra::RowDVector::from_vec(vec![103.0, 103.5, 101.5, 102.0]),
        ])
    }

    #[test]
    fn test_close_to_close() {
        let prices = get_sample_data();
        let hv = HistoricalVolatility::new(prices);
        let vol = hv.close_to_close();
        // Manual check:
        // C: 101.0, 102.5, 103.0, 102.0
        // Ret: ln(102.5/101.0)=0.0147, ln(103/102.5)=0.0048, ln(102/103)=-0.0097
        // Mean: 0.0032
        // Std Dev: 0.0124 (daily)
        assert_relative_eq!(vol, 0.0124, epsilon = 1e-4);
    }

    #[test]
    fn test_parkinson() {
        let prices = get_sample_data();
        let hv = HistoricalVolatility::new(prices);
        let vol = hv.parkinson();
        assert_relative_eq!(vol, 0.01425, epsilon = 1e-5);
    }

    #[test]
    fn test_garman_klass() {
        let prices = get_sample_data();
        let hv = HistoricalVolatility::new(prices);
        let vol = hv.garman_klass();
        assert_relative_eq!(vol, 0.01548, epsilon = 1e-5);
    }

    #[test]
    fn test_rogers_satchell() {
        let prices = get_sample_data();
        let hv = HistoricalVolatility::new(prices);
        let vol = hv.rogers_satchell();
        assert_relative_eq!(vol, 0.01517, epsilon = 1e-5);
    }

    #[test]
    fn test_yang_zhang() {
        let prices = get_sample_data();
        let hv = HistoricalVolatility::new(prices);
        let vol = hv.yang_zhang();
        assert_relative_eq!(vol, 0.01472, epsilon = 1e-5);
    }
}