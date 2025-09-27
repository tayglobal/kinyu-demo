use chrono::NaiveDate;
use nalgebra::DMatrix;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

/// A Python module implemented in Rust.
#[pymodule]
fn corr(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<TimeSeries>()?;
    m.add_function(wrap_pyfunction!(calculate_correlation_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(get_eigen_decomposition, m)?)?;
    Ok(())
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct TimeSeries {
    #[pyo3(get, set)]
    pub dates: Vec<String>,
    #[pyo3(get, set)]
    pub values: Vec<f64>,
}

#[pymethods]
impl TimeSeries {
    #[new]
    fn new(dates: Vec<String>, values: Vec<f64>) -> Self {
        TimeSeries { dates, values }
    }
}

#[pyfunction]
fn calculate_correlation_matrix(series: Vec<TimeSeries>, make_positive_definite: bool) -> PyResult<Vec<Vec<f64>>> {
    let (matrix, _) = align_time_series(series)?;
    let mut correlation_matrix = calculate_correlation(&matrix);

    if make_positive_definite && !is_positive_definite(&correlation_matrix) {
        correlation_matrix = adjust_to_positive_definite(&correlation_matrix);
    }

    Ok(correlation_matrix.row_iter().map(|r| r.iter().cloned().collect()).collect())
}

fn is_positive_definite(matrix: &DMatrix<f64>) -> bool {
    if matrix.is_empty() {
        return true;
    }
    let eigen = nalgebra::linalg::SymmetricEigen::new(matrix.clone());
    eigen.eigenvalues.iter().all(|&e| e > 0.0)
}

fn adjust_to_positive_definite(matrix: &DMatrix<f64>) -> DMatrix<f64> {
    let eigen = nalgebra::linalg::SymmetricEigen::new(matrix.clone());
    let mut eigenvalues = eigen.eigenvalues.clone();
    for val in eigenvalues.iter_mut() {
        if *val <= 0.0 {
            *val = 1e-8; // replace with small positive value
        }
    }
    let new_eigenvalues = DMatrix::from_diagonal(&eigenvalues);
    let eigenvectors = eigen.eigenvectors.clone();
    let new_matrix = eigenvectors.clone() * new_eigenvalues * eigenvectors.transpose();

    // Rescale to make it a correlation matrix again
    let mut correlation_matrix = new_matrix.clone();
    for i in 0..new_matrix.nrows() {
        for j in 0..new_matrix.ncols() {
            let d_i = new_matrix[(i,i)].sqrt();
            let d_j = new_matrix[(j,j)].sqrt();
            if d_i > 0.0 && d_j > 0.0 {
                correlation_matrix[(i,j)] = new_matrix[(i,j)] / (d_i * d_j);
            }
        }
    }
    correlation_matrix
}

/// Performs eigendecomposition on a correlation matrix.
/// This is a common first step in Principal Component Analysis (PCA) and can be used
/// as a basis for a factor model, but is not a full statistical factor model itself.
#[pyfunction]
fn get_eigen_decomposition(correlation_matrix: Vec<Vec<f64>>) -> PyResult<(Vec<f64>, Vec<Vec<f64>>)> {
    let rows = correlation_matrix.len();
    if rows == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    let cols = correlation_matrix[0].len();
    let matrix = DMatrix::from_row_slice(rows, cols, &correlation_matrix.concat());

    let eigen = nalgebra::linalg::SymmetricEigen::new(matrix);
    let eigenvalues = eigen.eigenvalues.iter().cloned().collect();
    let eigenvectors = eigen.eigenvectors.row_iter().map(|r| r.iter().cloned().collect()).collect();

    Ok((eigenvalues, eigenvectors))
}


fn align_time_series(series: Vec<TimeSeries>) -> PyResult<(DMatrix<f64>, Vec<NaiveDate>)> {
    let mut date_map = HashMap::new();
    for (i, ts) in series.iter().enumerate() {
        for (j, date_str) in ts.dates.iter().enumerate() {
            let date = NaiveDate::parse_from_str(date_str, "%Y-%m-%d")
                .map_err(|e| PyValueError::new_err(format!("Invalid date format for date '{}': {}", date_str, e)))?;
            date_map.entry(date).or_insert_with(Vec::new).push((i, ts.values[j]));
        }
    }

    let mut sorted_dates: Vec<NaiveDate> = date_map.keys().cloned().collect();
    sorted_dates.sort();

    let mut matrix = DMatrix::from_element(sorted_dates.len(), series.len(), f64::NAN);

    for (date_idx, date) in sorted_dates.iter().enumerate() {
        if let Some(values) = date_map.get(date) {
            for &(series_idx, value) in values {
                matrix[(date_idx, series_idx)] = value;
            }
        }
    }

    Ok((matrix, sorted_dates))
}


fn calculate_correlation(matrix: &DMatrix<f64>) -> DMatrix<f64> {
    let (rows, cols) = matrix.shape();
    let mut centered_matrix = DMatrix::from_element(rows, cols, 0.0);

    for c in 0..cols {
        let col_vec: Vec<f64> = matrix.column(c).iter().cloned().filter(|x| !x.is_nan()).collect();
        if col_vec.is_empty() { continue; }
        let mean = col_vec.iter().sum::<f64>() / col_vec.len() as f64;
        for r in 0..rows {
            if !matrix[(r, c)].is_nan() {
                centered_matrix[(r, c)] = matrix[(r, c)] - mean;
            }
        }
    }

    let mut cov_matrix = DMatrix::from_element(cols, cols, 0.0);
    for i in 0..cols {
        for j in 0..cols {
            let mut valid_count = 0;
            let mut cov = 0.0;
            for r in 0..rows {
                if !matrix[(r,i)].is_nan() && !matrix[(r,j)].is_nan() {
                    cov += centered_matrix[(r,i)] * centered_matrix[(r,j)];
                    valid_count += 1;
                }
            }
            if valid_count > 1 {
                cov_matrix[(i,j)] = cov / (valid_count -1) as f64;
            }
        }
    }

    let mut corr_matrix = DMatrix::from_element(cols, cols, 0.0);
    for i in 0..cols {
        for j in 0..cols {
            let std_i = cov_matrix[(i,i)].sqrt();
            let std_j = cov_matrix[(j,j)].sqrt();
            if std_i > 0.0 && std_j > 0.0 {
                corr_matrix[(i,j)] = cov_matrix[(i,j)] / (std_i * std_j)
            }
        }
    }

    corr_matrix
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DMatrix;

    fn create_test_series() -> Vec<TimeSeries> {
        let ts1 = TimeSeries {
            dates: vec!["2023-01-01".to_string(), "2023-01-02".to_string(), "2023-01-03".to_string()],
            values: vec![1.0, 2.0, 3.0],
        };
        let ts2 = TimeSeries {
            dates: vec!["2023-01-01".to_string(), "2023-01-02".to_string(), "2023-01-03".to_string()],
            values: vec![2.0, 3.0, 4.0],
        };
        let ts3 = TimeSeries {
            dates: vec!["2023-01-01".to_string(), "2023-01-02".to_string(), "2023-01-03".to_string()],
            values: vec![1.0, 1.0, 1.0],
        };
        vec![ts1, ts2, ts3]
    }

    #[test]
    fn test_align_time_series_invalid_date() {
        let ts1 = TimeSeries {
            dates: vec!["2023-01-01".to_string(), "not-a-date".to_string()],
            values: vec![1.0, 2.0],
        };
        let series = vec![ts1];
        let result = align_time_series(series);
        assert!(result.is_err());
    }

    #[test]
    fn test_align_time_series_matching_dates() {
        let series = create_test_series();
        let (matrix, dates) = align_time_series(series).unwrap();

        assert_eq!(dates.len(), 3);
        assert_eq!(matrix.nrows(), 3);
        assert_eq!(matrix.ncols(), 3);
        assert_eq!(matrix[(0, 0)], 1.0);
        assert_eq!(matrix[(1, 1)], 3.0);
    }

    #[test]
    fn test_align_time_series_mismatched_dates() {
        let ts1 = TimeSeries {
            dates: vec!["2023-01-01".to_string(), "2023-01-03".to_string()],
            values: vec![1.0, 3.0],
        };
        let ts2 = TimeSeries {
            dates: vec!["2023-01-02".to_string(), "2023-01-03".to_string()],
            values: vec![2.0, 4.0],
        };
        let series = vec![ts1, ts2];
        let (matrix, dates) = align_time_series(series).unwrap();

        assert_eq!(dates.len(), 3);
        assert_eq!(matrix.nrows(), 3);
        assert_eq!(matrix.ncols(), 2);

        // Check values and NaNs
        assert_eq!(matrix[(0, 0)], 1.0);
        assert!(matrix[(0, 1)].is_nan());
        assert!(matrix[(1, 0)].is_nan());
        assert_eq!(matrix[(1, 1)], 2.0);
        assert_eq!(matrix[(2, 0)], 3.0);
        assert_eq!(matrix[(2, 1)], 4.0);
    }

    #[test]
    fn test_calculate_correlation_simple() {
        let matrix = DMatrix::from_row_slice(3, 2, &[
            1.0, 2.0,
            2.0, 3.0,
            3.0, 4.0,
        ]);
        let corr = calculate_correlation(&matrix);
        assert!((corr[(0, 1)] - 1.0).abs() < 1e-9);
        assert!((corr[(1, 0)] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_positive_definite() {
        let pos_def_matrix = DMatrix::from_row_slice(2, 2, &[
            1.0, 0.5,
            0.5, 1.0,
        ]);
        assert!(is_positive_definite(&pos_def_matrix));

        let non_pos_def_matrix = DMatrix::from_row_slice(2, 2, &[
            1.0, 1.0,
            1.0, 1.0,
        ]);
        assert!(!is_positive_definite(&non_pos_def_matrix));
    }

    #[test]
    fn test_adjust_to_positive_definite() {
        let _non_pos_def_matrix = DMatrix::from_row_slice(3, 3, &[
            1.0, 0.9, 0.9,
            0.9, 1.0, 0.9,
            0.9, 0.9, 1.0,
        ]);
        // This matrix is often non-positive definite in practice
        // let's make a singular one
         let non_pos_def_matrix = DMatrix::from_row_slice(2, 2, &[
            1.0, 1.0,
            1.0, 1.0,
        ]);
        assert!(!is_positive_definite(&non_pos_def_matrix));
        let adjusted_matrix = adjust_to_positive_definite(&non_pos_def_matrix);
        assert!(is_positive_definite(&adjusted_matrix));
    }

    #[test]
    fn test_stress_large_dataset() {
        let num_series = 50;
        let num_dates = 365;
        let mut series = Vec::new();
        let start_date = NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();

        for i in 0..num_series {
            let mut dates = Vec::new();
            let mut values = Vec::new();
            for j in 0..num_dates {
                let current_date = start_date.checked_add_signed(chrono::Duration::days(j as i64)).unwrap();
                dates.push(current_date.format("%Y-%m-%d").to_string());
                values.push(i as f64 + j as f64 + (i * j) as f64 % 100.0);
            }
            series.push(TimeSeries::new(dates, values));
        }

        let (matrix, _) = align_time_series(series).unwrap();
        assert_eq!(matrix.nrows(), num_dates);
        assert_eq!(matrix.ncols(), num_series);

        let corr_matrix = calculate_correlation(&matrix);
        assert_eq!(corr_matrix.nrows(), num_series);
        assert_eq!(corr_matrix.ncols(), num_series);
    }
}