use nalgebra::DMatrix;
use numpy::PyArray2;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use super::ppca::{NoiseType, PPCAConfig, PPCA};

/// Probabilistic PCA model with PyO3 bindings.
#[pyclass]
pub struct PPCARust {
    inner: PPCA,
}

#[pymethods]
impl PPCARust {
    #[new]
    #[pyo3(signature = (n_components=2, max_iterations=100, tol=1e-4, random_state=None, noise_type="isotropic", l2_penalty=0.0))]
    fn new(
        n_components: usize,
        max_iterations: usize,
        tol: f64,
        random_state: Option<u64>,
        noise_type: &str,
        l2_penalty: f64,
    ) -> PyResult<Self> {
        let noise_type_enum = match noise_type {
            "isotropic" => NoiseType::Isotropic,
            "diagonal" => NoiseType::Diagonal,
            other => {
                return Err(PyValueError::new_err(format!(
                    "noise_type must be 'isotropic' or 'diagonal', got '{}'",
                    other
                )))
            }
        };
        let config = PPCAConfig {
            n_components,
            max_iterations,
            tol,
            random_state,
            noise_type: noise_type_enum,
            l2_penalty,
        };
        Ok(PPCARust {
            inner: PPCA::with_config(config),
        })
    }

    fn fit(&mut self, _py: Python, x: &PyArray2<f64>, mask: Option<&PyArray2<bool>>) -> PyResult<()> {
        let x_mat = py_to_matrix(x)?;
        let mask_mat = match mask {
            Some(m) => py_bool_to_matrix(m)?,
            None => DMatrix::from_element(x_mat.nrows(), x_mat.ncols(), false),
        };
        self.inner
            .fit(&x_mat, &mask_mat)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn transform(&self, py: Python, x: &PyArray2<f64>) -> PyResult<Py<PyArray2<f64>>> {
        let x_mat = py_to_matrix(x)?;
        let y = self
            .inner
            .transform(&x_mat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        matrix_to_py(py, &y)
    }

    fn inverse_transform(&self, py: Python, y: &PyArray2<f64>) -> PyResult<Py<PyArray2<f64>>> {
        let y_mat = py_to_matrix(y)?;
        let x = self
            .inner
            .inverse_transform(&y_mat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        matrix_to_py(py, &x)
    }

    fn fit_transform(
        &mut self,
        py: Python,
        x: &PyArray2<f64>,
        mask: Option<&PyArray2<bool>>,
    ) -> PyResult<Py<PyArray2<f64>>> {
        self.fit(py, x, mask)?;
        self.transform(py, x)
    }

    fn explained_variance_ratio(&self) -> PyResult<Vec<f64>> {
        self.inner
            .explained_variance_ratio()
            .map(|v| v.iter().copied().collect())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn loadings(&self, py: Python) -> PyResult<Py<PyArray2<f64>>> {
        let r = self
            .inner
            .result()
            .ok_or_else(|| PyValueError::new_err("Model not fitted"))?;
        matrix_to_py(py, &r.loadings)
    }

    fn noise_variance(&self) -> PyResult<f64> {
        self.inner
            .noise_variance()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn noise_variances(&self) -> PyResult<Vec<f64>> {
        self.inner
            .noise_variances()
            .map(|v| v.iter().copied().collect())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn n_iter(&self) -> PyResult<usize> {
        self.inner
            .n_iter()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn log_likelihoods(&self) -> PyResult<Vec<f64>> {
        self.inner
            .log_likelihoods()
            .map(|v| v.clone())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn reconstruction_error(
        &self,
        x: &PyArray2<f64>,
        mask: Option<&PyArray2<bool>>,
    ) -> PyResult<f64> {
        let x_mat = py_to_matrix(x)?;
        let mask_mat = match mask {
            Some(m) => py_bool_to_matrix(m)?,
            None => DMatrix::from_element(x_mat.nrows(), x_mat.ncols(), false),
        };
        self.inner
            .reconstruction_error(&x_mat, &mask_mat)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

// ── Conversion helpers ──────────────────────────────────────────────────────

fn py_to_matrix(arr: &PyArray2<f64>) -> PyResult<DMatrix<f64>> {
    let ro = arr.readonly();
    let shape = ro.dims();
    let slice = ro
        .as_slice()
        .map_err(|_| PyValueError::new_err("Failed to read array data"))?;
    Ok(DMatrix::from_row_slice(shape[0], shape[1], slice))
}

fn py_bool_to_matrix(arr: &PyArray2<bool>) -> PyResult<DMatrix<bool>> {
    let ro = arr.readonly();
    let shape = ro.dims();
    let slice = ro
        .as_slice()
        .map_err(|_| PyValueError::new_err("Failed to read array data"))?;
    Ok(DMatrix::from_row_slice(shape[0], shape[1], slice))
}

fn matrix_to_py(py: Python, m: &DMatrix<f64>) -> PyResult<Py<PyArray2<f64>>> {
    let rows: Vec<Vec<f64>> = (0..m.nrows())
        .map(|i| m.row(i).iter().copied().collect())
        .collect();
    let arr = PyArray2::from_vec2(py, &rows)
        .map_err(|e| PyValueError::new_err(format!("Failed to create array: {}", e)))?;
    Ok(arr.to_owned())
}

#[pymodule]
fn ppca_rs(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PPCARust>()?;
    Ok(())
}
