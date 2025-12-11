use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use ppca_core::{PPCA, PPCAConfig, PPCAError};
use nalgebra::DMatrix;
use numpy::{PyArray2, ToPyArray};

/// Probabilistic PCA model with PyO3 bindings
#[pyclass]
pub struct PPCARust {
    inner: PPCA,
}

#[pymethods]
impl PPCARust {
    #[new]
    #[pyo3(signature = (n_components=2, max_iterations=100, tol=1e-4))]
    fn new(n_components: usize, max_iterations: usize, tol: f64) -> Self {
        let config = PPCAConfig {
            n_components,
            max_iterations,
            tol,
            random_state: None,
        };
        PPCARust {
            inner: PPCA::with_config(config),
        }
    }

    /// Fit the model to data with missing value mask
    fn fit(
        &mut self,
        py: Python,
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
        
        let result = Y_matrix.to_pyarray(py).to_owned();
        Ok(result)
    }

    /// Reconstruct data from latent representation
    fn inverse_transform(&self, py: Python, Y: &PyArray2<f64>) -> PyResult<Py<PyArray2<f64>>> {
        let Y_matrix = convert_py_array_to_matrix(Y)?;
        let X_matrix = self.inner.inverse_transform(&Y_matrix)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        let result = X_matrix.to_pyarray(py).to_owned();
        Ok(result)
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
