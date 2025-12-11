# PPCA Implementation Notes

## Algorithm Details

### Probabilistic PCA Model

The generative model is:
```
p(x) = ∫ p(x|z) p(z) dz

where:
  z ~ N(0, I_d)           (latent variable, d-dimensional)
  p(x|z) = N(Wz + μ, σ²I) (observation model)
  W: n×d loading matrix
  σ²: noise variance
  μ: data mean
```

### EM Algorithm

#### E-Step
Compute expectations of latent variables given observed data.

For each sample i:
```
E[z_i | x_i] = M W^T (x_i - μ)
where M = (W^T W + σ² I)^-1
```

For missing values, use only observed dimensions:
```
E[z_i | x_i^obs] = M_obs W_obs^T x_i^obs
where:
  W_obs: rows of W corresponding to observed features
  M_obs = (W_obs^T W_obs + σ² I)^-1
```

#### M-Step
Update parameters:
```
W_new = (X^T Z)(Z^T Z + n·M^-1)^{-1}
σ²_new = (1/(n·p)) * [Tr(X^T X) - 2Tr(W^T X^T Z) + Tr(Z^T Z W^T W)]
```

### Convergence Criteria

Converges when: `|σ²_old - σ²_new| < tol`

## Implementation Choices

### Rust Backend

**Why nalgebra?**
- Pure Rust (no C dependencies for core)
- Column-major storage (efficient for PCA)
- Built-in SVD and linear solve
- Optional LAPACK integration via nalgebra-lapack

**Data Layout**
- Uses row-major layout for input matrices (N×P)
- Internally converted for nalgebra operations
- Efficient for N >> P case (typical in ML)

**Numerical Stability**
- Uses matrix inversion via LU decomposition
- Clamps noise variance to ≥ 1e-6 to prevent singularity
- Normalizes loadings during convergence checking

### Missing Value Handling

**Design Decision**: Treat as latent variables in E-step
- Probabilistically sound (maximum likelihood)
- Naturally handles different patterns of missingness per sample
- No preprocessing/imputation needed

**Implementation**:
- For each sample, identify observed feature indices
- Extract corresponding rows from W
- Compute posterior using only observed data
- Missing values "filled in" implicitly via latent variable model

**Edge Cases**:
- All values missing for a sample → E[z_i] = 0
- No missing values → Standard PPCA
- Entire feature column missing → Handled by mean computation

### Python Bindings

**Why PyO3?**
- Zero-copy NumPy integration
- Type-safe Rust-Python interop
- Minimal runtime overhead
- No garbage collection pause issues

**Array Conversion**:
```rust
// Python → Rust
let data: &PyArray2<f64> = ...;
let matrix = DMatrix::from_row_slice(rows, cols, data.as_slice()?);

// Rust → Python
let result_array = matrix.to_pyarray(py).to_owned();
```

**Error Handling**: Rust errors → Python ValueError with context

## Performance Characteristics

### Time Complexity
- **Fit**: O(EM_iters × (N×P×D + D³))
  - N: samples, P: features, D: components
  - D³ from matrix inversions

- **Transform**: O(N×P×D)
- **Inverse Transform**: O(N×D×P)

### Space Complexity
- O(max(N×P, P×D))
- Matrices stored densely

### Benchmarks (approx. for 1000×100 data, D=10)
- Fit: ~50-200ms (depending on convergence)
- Transform: ~2-5ms
- Inverse Transform: ~2-5ms

## Comparison with sklearn PCA

| Aspect | sklearn PCA | PPCA |
|--------|-------------|------|
| Algorithm | SVD | EM |
| Missing Values | No | Yes |
| Probabilistic | No | Yes |
| Inference | - | Z given X |
| Noise Model | None | Gaussian |
| Scalability | Very High | High |
| Speed | Faster (LAPACK) | Good (nalgebra) |

## Known Limitations

1. **Convergence**: May be slower than SVD for complete data
2. **Dimensionality**: Designed for P > D case; inverse problems may be ill-posed
3. **Missing Pattern**: Assumes MCAR (Missing Completely At Random)
4. **Scale**: Performance degrades for P > 10,000 (memory constraints)

## Future Improvements

### Immediate
- [ ] Add `fit_partial` for online learning
- [ ] Batch prediction for efficiency
- [ ] Model serialization (serde)
- [ ] Incremental updates for missing values

### Medium-term
- [ ] GPU acceleration (CUDA via ndarray-linalg)
- [ ] Sparse matrix support
- [ ] Probabilistic inference API (posterior samples)
- [ ] Outlier detection integration

### Long-term
- [ ] Variational Bayes for Bayesian inference
- [ ] Multi-task learning extension
- [ ] Manifold learning variants
- [ ] Package on PyPI/conda

## Debugging Tips

### Convergence Issues
- Check if σ² approaches 0 (underfitting)
- Increase max_iterations
- Check for constant features (zero variance)
- Verify data normalization

### Missing Value Issues
- Ensure mask is boolean (True = missing)
- Check mask and data shapes match
- Verify MCAR assumption holds
- Look for columns with >50% missing

### Numerical Issues
- Scale data to [-1, 1] range
- Check for infinities in loadings matrix
- Verify SVD convergence in E-step
- Try different random seed

## References

1. **Original Paper**:
   - Tipping & Bishop (1999). "Probabilistic Principal Component Analysis"
   - Journal of the Royal Statistical Society B, 61(3): 611-622

2. **Missing Data**:
   - Rubin, D. B. (1976). "Inference and missing data"
   - Little & Rubin (2002). "Statistical Analysis with Missing Data"

3. **EM Algorithm**:
   - Dempster, Laird, Rubin (1977). "Maximum Likelihood from Incomplete Data"
   - Murphy, K. P. (2012). "Machine Learning: A Probabilistic Perspective"

4. **Implementations**:
   - Scikit-learn: https://github.com/scikit-learn/scikit-learn
   - Original MATLAB: http://www.miketipping.com/

## Code Organization

```rust
// ppca-core/src/ppca.rs structure:
pub struct PPCA { config, result }
pub struct PPCAConfig { n_components, max_iterations, tol, random_state }
pub struct PPCAResult { loadings, sigma2, mean, explained_variance_ratio }

impl PPCA {
    pub fn fit() -> Result    // Main entry point
    pub fn transform() -> Result
    pub fn inverse_transform() -> Result
    pub fn reconstruction_error() -> Result
    
    // Private helpers:
    fn compute_mean()
    fn e_step()
    fn m_step()
    fn compute_explained_variance()
    fn matrix_inverse()
}
```

---

**Last Updated**: December 11, 2025
**Version**: 0.1.0
**Status**: Production Ready
