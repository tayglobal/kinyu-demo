use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyList};
use chrono::NaiveDate;
use nalgebra::{DMatrix, DVector};
use std::collections::{BTreeMap, BTreeSet};

/// A time series is represented as a map from date to value.
pub type TimeSeries = BTreeMap<NaiveDate, f64>;

/// Aligns multiple time series to a common set of dates.
///
/// # Arguments
///
/// * `series` - A slice of `TimeSeries`.
///
/// # Returns
///
/// A `DMatrix<f64>` where each column represents a time series and each row
/// corresponds to a date present in all time series.
pub fn align_time_series(series: &[TimeSeries]) -> DMatrix<f64> {
    if series.is_empty() {
        return DMatrix::zeros(0, 0);
    }

    // Find the intersection of all dates.
    let mut dates_iter = series.iter().map(|s| s.keys().cloned().collect::<BTreeSet<_>>());
    let common_dates = dates_iter.next().map_or_else(BTreeSet::new, |first| {
        dates_iter.fold(first, |acc, dates| {
            acc.intersection(&dates).cloned().collect()
        })
    });

    let n_dates = common_dates.len();
    let n_series = series.len();
    let mut matrix = DMatrix::zeros(n_dates, n_series);

    for (j, s) in series.iter().enumerate() {
        for (i, date) in common_dates.iter().enumerate() {
            if let Some(value) = s.get(date) {
                matrix[(i, j)] = *value;
            }
        }
    }

    matrix
}

/// Calculates the correlation matrix for a set of time series.
///
/// # Arguments
///
/// * `series` - A slice of `TimeSeries`.
///
/// # Returns
///
/// A `DMatrix<f64>` representing the correlation matrix.
pub fn correlation_matrix(series: &[TimeSeries]) -> DMatrix<f64> {
    let aligned_data = align_time_series(series);
    let (rows, cols) = aligned_data.shape();

    if rows < 2 {
        return DMatrix::identity(cols, cols);
    }

    let means = DVector::from_iterator(
        cols,
        (0..cols).map(|j| aligned_data.column(j).mean()),
    );

    let mut centered_data = DMatrix::zeros(rows, cols);
    for j in 0..cols {
        let mean = means[j];
        for i in 0..rows {
            centered_data[(i, j)] = aligned_data[(i, j)] - mean;
        }
    }
    let mut cov = DMatrix::zeros(cols, cols);
    for i in 0..cols {
        for j in i..cols {
            let col_i = centered_data.column(i);
            let col_j = centered_data.column(j);
            let covariance = col_i.dot(&col_j) / (rows as f64 - 1.0);
            cov[(i, j)] = covariance;
            cov[(j, i)] = covariance;
        }
    }

    let std_devs = DVector::from_iterator(
        cols,
        (0..cols).map(|i| cov[(i, i)].sqrt()),
    );

    let mut corr = DMatrix::zeros(cols, cols);
    for i in 0..cols {
        for j in i..cols {
            if std_devs[i] > 0.0 && std_devs[j] > 0.0 {
                let correlation = cov[(i, j)] / (std_devs[i] * std_devs[j]);
                corr[(i, j)] = correlation;
                corr[(j, i)] = correlation;
            } else {
                corr[(i, j)] = if i == j { 1.0 } else { 0.0 };
                corr[(j, i)] = if i == j { 1.0 } else { 0.0 };
            }
        }
    }
    corr
}

/// Checks if a matrix is positive definite.
///
/// # Arguments
///
/// * `matrix` - The matrix to check.
///
/// # Returns
///
/// `true` if the matrix is positive definite, `false` otherwise.
pub fn is_positive_definite(matrix: &DMatrix<f64>) -> bool {
    if matrix.is_empty() || !matrix.is_square() {
        return false;
    }
    match matrix.clone().symmetric_eigen() {
        e => e.eigenvalues.iter().all(|&val| val > 0.0),
    }
}

/// Corrects a non-positive definite matrix to be positive definite using eigenvalue cleaning.
///
/// # Arguments
///
/// * `matrix` - The matrix to correct.
/// * `min_eigenvalue` - The minimum allowed eigenvalue.
///
/// # Returns
///
/// A new positive definite matrix.
pub fn make_positive_definite(matrix: &DMatrix<f64>, min_eigenvalue: f64) -> DMatrix<f64> {
    if matrix.is_empty() || !matrix.is_square() {
        return matrix.clone();
    }

    let eigen = matrix.clone().symmetric_eigen();
    let mut eigenvalues = eigen.eigenvalues;
    let eigenvectors = eigen.eigenvectors;

    for val in eigenvalues.iter_mut() {
        if *val < min_eigenvalue {
            *val = min_eigenvalue;
        }
    }

    &eigenvectors * DMatrix::from_diagonal(&eigenvalues) * eigenvectors.transpose()
}

/// Converts a Python dictionary of time series to a Vec<TimeSeries>.
fn py_dict_to_time_series(py: Python, dict: &PyDict) -> PyResult<Vec<TimeSeries>> {
    let mut series_vec = Vec::new();
    for (_key, value) in dict.iter() {
        let ts_dict: &PyDict = value.downcast()?;
        let mut ts = TimeSeries::new();
        for (date_str, val) in ts_dict.iter() {
            let date = NaiveDate::parse_from_str(&date_str.to_string(), "%Y-%m-%d")
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;
            let value: &PyFloat = val.downcast()?;
            ts.insert(date, value.value());
        }
        series_vec.push(ts);
    }
    Ok(series_vec)
}

/// Converts a DMatrix to a Python list of lists.
fn dmatrix_to_py_list(py: Python, matrix: &DMatrix<f64>) -> PyObject {
    let mut outer_list = Vec::new();
    for i in 0..matrix.nrows() {
        let mut inner_list = Vec::new();
        for j in 0..matrix.ncols() {
            inner_list.push(matrix[(i, j)]);
        }
        outer_list.push(PyList::new(py, inner_list));
    }
    PyList::new(py, outer_list).to_object(py)
}

#[pyfunction]
fn calculate_correlation_matrix(py: Python, series_dict: &PyDict) -> PyResult<PyObject> {
    let series = py_dict_to_time_series(py, series_dict)?;
    let corr_matrix = correlation_matrix(&series);
    Ok(dmatrix_to_py_list(py, &corr_matrix))
}

#[pyfunction]
fn py_make_positive_definite(py: Python, matrix_list: &PyList, min_eigenvalue: f64) -> PyResult<PyObject> {
    let nrows = matrix_list.len();
    if nrows == 0 {
        return Ok(PyList::new(py, Vec::<PyObject>::new()).to_object(py));
    }
    let ncols = matrix_list.get_item(0)?.downcast::<PyList>()?.len();

    let mut data = Vec::with_capacity(nrows * ncols);
    for i in 0..nrows {
        let row: &PyList = matrix_list.get_item(i)?.downcast()?;
        for j in 0..ncols {
            data.push(row.get_item(j)?.extract::<f64>()?);
        }
    }

    let matrix = DMatrix::from_row_slice(nrows, ncols, &data);
    let pd_matrix = make_positive_definite(&matrix, min_eigenvalue);
    Ok(dmatrix_to_py_list(py, &pd_matrix))
}

#[pymodule]
fn corr(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calculate_correlation_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(py_make_positive_definite, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn create_date(year: i32, month: u32, day: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(year, month, day).unwrap()
    }

    #[test]
    fn test_align_time_series() {
        let date1 = create_date(2023, 1, 1);
        let date2 = create_date(2023, 1, 2);
        let date3 = create_date(2023, 1, 3);

        let mut ts1 = TimeSeries::new();
        ts1.insert(date1, 1.0);
        ts1.insert(date2, 2.0);

        let mut ts2 = TimeSeries::new();
        ts2.insert(date2, 3.0);
        ts2.insert(date3, 4.0);

        let series = vec![ts1, ts2];
        let aligned = align_time_series(&series);

        let expected = DMatrix::from_row_slice(1, 2, &[2.0, 3.0]);
        assert_eq!(aligned, expected);
    }

    #[test]
    fn test_correlation_matrix_perfect_correlation() {
        let date1 = create_date(2023, 1, 1);
        let date2 = create_date(2023, 1, 2);

        let mut ts1 = TimeSeries::new();
        ts1.insert(date1, 1.0);
        ts1.insert(date2, 2.0);

        let mut ts2 = TimeSeries::new();
        ts2.insert(date1, 2.0);
        ts2.insert(date2, 4.0);

        let series = vec![ts1, ts2];
        let corr = correlation_matrix(&series);

        assert!((corr[(0, 0)] - 1.0).abs() < 1e-9);
        assert!((corr[(1, 1)] - 1.0).abs() < 1e-9);
        assert!((corr[(0, 1)] - 1.0).abs() < 1e-9);
        assert!((corr[(1, 0)] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_correlation_matrix_no_correlation() {
        let date1 = create_date(2023, 1, 1);
        let date2 = create_date(2023, 1, 2);

        let mut ts1 = TimeSeries::new();
        ts1.insert(date1, 1.0);
        ts1.insert(date2, 2.0);

        let mut ts2 = TimeSeries::new();
        ts2.insert(date1, 2.0);
        ts2.insert(date2, 1.0);

        let series = vec![ts1, ts2];
        let corr = correlation_matrix(&series);

        assert!((corr[(0, 0)] - 1.0).abs() < 1e-9);
        assert!((corr[(1, 1)] - 1.0).abs() < 1e-9);
        assert!((corr[(0, 1)] - (-1.0)).abs() < 1e-9);
        assert!((corr[(1, 0)] - (-1.0)).abs() < 1e-9);
    }

    #[test]
    fn test_positive_definite() {
        let mut matrix = DMatrix::from_row_slice(2, 2, &[
            1.0, 0.8,
            0.8, 1.0,
        ]);
        assert!(is_positive_definite(&matrix));

        matrix[(0, 1)] = 1.2;
        matrix[(1, 0)] = 1.2;
        assert!(!is_positive_definite(&matrix));
    }

    #[test]
    fn test_make_positive_definite() {
        let mut matrix = DMatrix::from_row_slice(2, 2, &[
            1.0, 1.2,
            1.2, 1.0,
        ]);
        assert!(!is_positive_definite(&matrix));

        let corrected_matrix = make_positive_definite(&matrix, 0.01);
        assert!(is_positive_definite(&corrected_matrix));
    }
}