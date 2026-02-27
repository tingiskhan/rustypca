use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use super::ppca::{PPCA, PPCAConfig};
use nalgebra::DMatrix;
use numpy::{PyArray2};

/// Probabilistic PCA model with PyO3 bindings
#[pyclass]
pub struct PPCARust {
    inner: PPCA,
}

#[pymethods]
impl PPCARust {
    #[new]
    #[pyo3(signature = (n_components=2, max_iterations=100, tol=1e-4, loading_signs=None, random_state=None))]
    fn new(n_components: usize, max_iterations: usize, tol: f64, loading_signs: Option<Vec<i8>>, random_state: Option<u64>) -> Self {
        let config = PPCAConfig {
            n_components,
            max_iterations,
            tol,
            random_state,
            loading_signs,
        };
        PPCARust {
            inner: PPCA::with_config(config),
        }
    }

    /// Fit the model to data with missing value mask
    fn fit(
        &mut self,
        _py: Python,
        X: &PyArray2<f64>,
        mask: Option<&PyArray2<bool>>,
    ) -> PyResult<()> {
        let X_matrix = convert_py_array_to_matrix(X)?;
        
        let mask_matrix = if let Some(m) = mask {
            convert_py_bool_array_to_matrix(m)?
        } else {
            DMatrix::from_element(X_matrix.nrows(), X_matrix.ncols(), false)
        };

        self.inner.fit(&X_matrix, &mask_matrix)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(())
    }

    /// Transform data using the fitted model
    fn transform(&self, py: Python, X: &PyArray2<f64>) -> PyResult<Py<PyArray2<f64>>> {
        let X_matrix = convert_py_array_to_matrix(X)?;
        let Y_matrix = self.inner.transform(&X_matrix)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        // Convert to Vec<Vec<f64>> for from_vec2
        let nrows = Y_matrix.nrows();
        let mut result: Vec<Vec<f64>> = Vec::with_capacity(nrows);
        for i in 0..nrows {
            let row: Vec<f64> = Y_matrix.row(i).iter().copied().collect();
            result.push(row);
        }
        
        let arr = PyArray2::from_vec2(py, &result)
            .map_err(|e| PyValueError::new_err(format!("Failed to create array: {}", e)))?;
        Ok(arr.to_owned())
    }

    /// Reconstruct data from latent representation
    fn inverse_transform(&self, py: Python, Y: &PyArray2<f64>) -> PyResult<Py<PyArray2<f64>>> {
        let Y_matrix = convert_py_array_to_matrix(Y)?;
        let X_matrix = self.inner.inverse_transform(&Y_matrix)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        let nrows = X_matrix.nrows();
        let mut result: Vec<Vec<f64>> = Vec::with_capacity(nrows);
        for i in 0..nrows {
            let row: Vec<f64> = X_matrix.row(i).iter().copied().collect();
            result.push(row);
        }
        
        let arr = PyArray2::from_vec2(py, &result)
            .map_err(|e| PyValueError::new_err(format!("Failed to create array: {}", e)))?;
        Ok(arr.to_owned())
    }

    /// Fit model and transform data
    fn fit_transform(
        &mut self,
        py: Python,
        X: &PyArray2<f64>,
        mask: Option<&PyArray2<bool>>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        self.fit(py, X, mask)?;
        self.transform(py, X)
    }

    /// Get the explained variance ratio from the fitted model
    fn explained_variance_ratio(&self) -> PyResult<Vec<f64>> {
        let evr = self.inner.explained_variance_ratio()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(evr.iter().copied().collect())
    }

    /// Get the noise variance from the fitted model
    fn noise_variance(&self) -> PyResult<f64> {
        self.inner.noise_variance()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Compute reconstruction error
    fn reconstruction_error(
        &self,
        X: &PyArray2<f64>,
        mask: Option<&PyArray2<bool>>,
    ) -> PyResult<f64> {
        let X_matrix = convert_py_array_to_matrix(X)?;
        
        let mask_matrix = if let Some(m) = mask {
            convert_py_bool_array_to_matrix(m)?
        } else {
            DMatrix::from_element(X_matrix.nrows(), X_matrix.ncols(), false)
        };

        self.inner.reconstruction_error(&X_matrix, &mask_matrix)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

fn convert_py_array_to_matrix(arr: &PyArray2<f64>) -> PyResult<DMatrix<f64>> {
    let arr_readonly = arr.readonly();
    let shape = arr_readonly.dims();
    let slice = arr_readonly.as_slice().map_err(|_| PyValueError::new_err("Failed to get array data"))?;
    
    let matrix = DMatrix::from_row_slice(shape[0], shape[1], slice);
    Ok(matrix)
}

fn convert_py_bool_array_to_matrix(arr: &PyArray2<bool>) -> PyResult<DMatrix<bool>> {
    let arr_readonly = arr.readonly();
    let shape = arr_readonly.dims();
    let slice = arr_readonly.as_slice().map_err(|_| PyValueError::new_err("Failed to get array data"))?;
    
    let matrix = DMatrix::from_row_slice(shape[0], shape[1], slice);
    Ok(matrix)
}

#[pymodule]
fn ppca_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PPCARust>()?;
    Ok(())
}
