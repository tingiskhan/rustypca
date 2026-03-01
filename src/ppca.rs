//! Probabilistic PCA via EM (Tipping & Bishop, 1999).
//!
//! Supports:
//! - Missing values (EM marginalises over unobserved entries)
//! - Isotropic noise (classic PPCA: ε ~ N(0, σ²I))
//! - Diagonal noise  (Factor Analysis: ε ~ N(0, diag(ψ)))
//! - L2 penalty on the loading matrix W
//! - Observation-pattern grouping for speed

use crate::errors::{PPCAError, Result};
use nalgebra::{DMatrix, DVector, SVD};
use rand::Rng;
use rand::SeedableRng;
use std::collections::HashMap;

// ── Noise type ──────────────────────────────────────────────────────────────

/// Whether the noise covariance is isotropic (PPCA) or diagonal (FA).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoiseType {
    Isotropic,
    Diagonal,
}

impl Default for NoiseType {
    fn default() -> Self {
        NoiseType::Isotropic
    }
}

// ── Config / Result ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PPCAConfig {
    pub n_components: usize,
    pub max_iterations: usize,
    pub tol: f64,
    pub random_state: Option<u64>,
    pub noise_type: NoiseType,
    /// Ridge penalty λ on W (adds λI to second-moment in M-step).
    pub l2_penalty: f64,
}

impl Default for PPCAConfig {
    fn default() -> Self {
        PPCAConfig {
            n_components: 2,
            max_iterations: 100,
            tol: 1e-4,
            random_state: None,
            noise_type: NoiseType::Isotropic,
            l2_penalty: 0.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PPCAResult {
    /// Loading matrix W: n_features × n_components.
    pub loadings: DMatrix<f64>,
    /// Per-feature noise variances (uniform when isotropic).
    pub noise_variances: DVector<f64>,
    /// Training-data mean (computed from observed values).
    pub mean: DVector<f64>,
    /// Explained variance ratio per component.
    pub explained_variance_ratio: DVector<f64>,
    /// Number of EM iterations actually run.
    pub n_iter: usize,
    /// Observed-data log-likelihood after each iteration.
    pub log_likelihoods: Vec<f64>,
}

// ── PPCA model ──────────────────────────────────────────────────────────────

pub struct PPCA {
    config: PPCAConfig,
    result: Option<PPCAResult>,
}

/// A group of samples that share the same observation pattern.
struct PatternGroup {
    /// Row indices in X_centered belonging to this pattern.
    sample_indices: Vec<usize>,
    /// Sorted observed-feature indices.
    obs_indices: Vec<usize>,
}

/// Precomputed quantities for one observation pattern in the E-step.
struct PatternEStep {
    /// Posterior covariance Σ = M^{-1}, shape d × d.
    sigma: DMatrix<f64>,
    /// Σ W_o^T Ψ_o^{-1}, shape d × |O|.  Pre-multiply with x_o to get μ.
    sigma_wt_psi_inv: DMatrix<f64>,
    /// Log-likelihood constant for this pattern:
    ///   -|O|/2 log(2π) - 1/2 log|C_o|
    ll_const: f64,
    /// Ψ_o^{-1} (I - W_o Σ W_o^T Ψ_o^{-1}), shape |O| × |O|.
    /// C_o^{-1} via Woodbury, for the quadratic form in the log-likelihood.
    c_inv: DMatrix<f64>,
}

const MIN_VARIANCE: f64 = 1e-6;
const LOG_2PI: f64 = 1.8378770664093453; // ln(2π)

impl PPCA {
    pub fn new(n_components: usize) -> Self {
        let mut config = PPCAConfig::default();
        config.n_components = n_components;
        PPCA {
            config,
            result: None,
        }
    }

    pub fn with_config(config: PPCAConfig) -> Self {
        PPCA {
            config,
            result: None,
        }
    }

    // ── Fit ──────────────────────────────────────────────────────────────

    pub fn fit(&mut self, x: &DMatrix<f64>, mask: &DMatrix<bool>) -> Result<()> {
        let (n, p) = x.shape();
        let d = self.config.n_components;

        if n == 0 || p == 0 {
            return Err(PPCAError::NoDimensionality);
        }
        let (mask_n, mask_p) = mask.shape();
        if mask_n != n || mask_p != p {
            return Err(PPCAError::ShapeMismatch {
                expected: (n, p),
                got: (mask_n, mask_p),
            });
        }
        if self.config.l2_penalty < 0.0 {
            return Err(PPCAError::InvalidPenalty(self.config.l2_penalty));
        }
        if d > p {
            return Err(PPCAError::InvalidComponents {
                n_components: d,
                n_features: p,
            });
        }

        // ── Mean & centering ────────────────────────────────────────────
        let mean = compute_mean(x, mask);
        let x_c = center(x, mask, &mean);

        // ── Observation-pattern groups ──────────────────────────────────
        let groups = build_pattern_groups(mask, n, p);

        // ── Initialisation ──────────────────────────────────────────────
        let (mut w, sigma2_init) = match self.config.random_state {
            Some(seed) => {
                let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
                let w = DMatrix::from_fn(p, d, |_, _| rng.gen::<f64>() - 0.5);
                (w, 1.0)
            }
            None => pca_init(&x_c, d)?,
        };
        let mut psi = DVector::from_element(p, sigma2_init);

        // ── EM loop ─────────────────────────────────────────────────────
        let mut log_likelihoods = Vec::with_capacity(self.config.max_iterations);
        let mut n_iter = 0usize;

        for _iter in 0..self.config.max_iterations {
            n_iter += 1;

            // E-step (per pattern)
            let pattern_esteps: Vec<PatternEStep> = groups
                .iter()
                .map(|g| e_step_pattern(&w, &psi, &g.obs_indices, d))
                .collect::<Result<Vec<_>>>()?;

            // Posterior means  μ_i  (n × d, row-major vec for speed)
            let mu = compute_posterior_means(&x_c, &groups, &pattern_esteps, n, d);

            // Observed-data log-likelihood
            let ll = compute_log_likelihood(&x_c, &groups, &pattern_esteps);
            log_likelihoods.push(ll);

            // M-step
            let (w_new, psi_new) = m_step(
                &x_c,
                &mu,
                &groups,
                &pattern_esteps,
                &self.config,
                n,
                p,
                d,
            );
            w = w_new;
            psi = psi_new;

            // Convergence on relative LL change
            if log_likelihoods.len() >= 2 {
                let prev = log_likelihoods[log_likelihoods.len() - 2];
                let curr = ll;
                let denom = prev.abs().max(1.0);
                if ((curr - prev) / denom).abs() < self.config.tol {
                    break;
                }
            }
        }

        let evr = compute_explained_variance(&w, &psi);

        self.result = Some(PPCAResult {
            loadings: w,
            noise_variances: psi,
            mean,
            explained_variance_ratio: evr,
            n_iter,
            log_likelihoods,
        });

        Ok(())
    }

    // ── Transform ────────────────────────────────────────────────────────

    pub fn transform(&self, x: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let r = self.result.as_ref().ok_or(PPCAError::NoDimensionality)?;
        let (n, p) = x.shape();
        if p != r.mean.len() {
            return Err(PPCAError::InvalidDimensions {
                expected: r.mean.len(),
                got: p,
            });
        }

        let d = self.config.n_components;

        // Center (NaN → 0)
        let x_c = {
            let mut m = DMatrix::zeros(n, p);
            for i in 0..n {
                for j in 0..p {
                    let v = x[(i, j)];
                    m[(i, j)] = if v.is_nan() { 0.0 } else { v - r.mean[j] };
                }
            }
            m
        };

        // E[z|x] = M^{-1} W^T Ψ^{-1} x_c  where M = I + W^T Ψ^{-1} W
        let psi_inv = r.noise_variances.map(|v| 1.0 / v.max(MIN_VARIANCE));
        // Ψ^{-1} W  (p × d)
        let psi_inv_w = DMatrix::from_fn(p, d, |j, l| r.loadings[(j, l)] * psi_inv[j]);
        // M = I + W^T Ψ^{-1} W  (d × d)
        let m_mat = DMatrix::identity(d, d) + r.loadings.transpose() * &psi_inv_w;
        let m_inv = chol_inv(&m_mat)?;
        // (n × p) × (p × d) × (d × d) → (n × d)
        Ok(&x_c * &psi_inv_w * &m_inv)
    }

    // ── Inverse transform ────────────────────────────────────────────────

    pub fn inverse_transform(&self, y: &DMatrix<f64>) -> Result<DMatrix<f64>> {
        let r = self.result.as_ref().ok_or(PPCAError::NoDimensionality)?;
        let (n, k) = y.shape();
        if k != self.config.n_components {
            return Err(PPCAError::InvalidDimensions {
                expected: self.config.n_components,
                got: k,
            });
        }
        let p = r.mean.len();
        // X = Y W^T + 1 μ^T
        let mut out = y * r.loadings.transpose();
        for i in 0..n {
            for j in 0..p {
                out[(i, j)] += r.mean[j];
            }
        }
        Ok(out)
    }

    // ── Reconstruction error ─────────────────────────────────────────────

    pub fn reconstruction_error(&self, x: &DMatrix<f64>, mask: &DMatrix<bool>) -> Result<f64> {
        let y = self.transform(x)?;
        let x_hat = self.inverse_transform(&y)?;
        let mut err = 0.0;
        let mut cnt = 0.0;
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                if !mask[(i, j)] {
                    let d = x[(i, j)] - x_hat[(i, j)];
                    err += d * d;
                    cnt += 1.0;
                }
            }
        }
        Ok(if cnt > 0.0 { err / cnt } else { 0.0 })
    }

    // ── Accessors ────────────────────────────────────────────────────────

    pub fn result(&self) -> Option<&PPCAResult> {
        self.result.as_ref()
    }

    pub fn explained_variance_ratio(&self) -> Result<&DVector<f64>> {
        self.result
            .as_ref()
            .map(|r| &r.explained_variance_ratio)
            .ok_or(PPCAError::NoDimensionality)
    }

    pub fn noise_variance(&self) -> Result<f64> {
        self.result
            .as_ref()
            .map(|r| r.noise_variances.sum() / r.noise_variances.len() as f64)
            .ok_or(PPCAError::NoDimensionality)
    }

    pub fn noise_variances(&self) -> Result<&DVector<f64>> {
        self.result
            .as_ref()
            .map(|r| &r.noise_variances)
            .ok_or(PPCAError::NoDimensionality)
    }

    pub fn n_iter(&self) -> Result<usize> {
        self.result
            .as_ref()
            .map(|r| r.n_iter)
            .ok_or(PPCAError::NoDimensionality)
    }

    pub fn log_likelihoods(&self) -> Result<&Vec<f64>> {
        self.result
            .as_ref()
            .map(|r| &r.log_likelihoods)
            .ok_or(PPCAError::NoDimensionality)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Free functions
// ═══════════════════════════════════════════════════════════════════════════

fn compute_mean(x: &DMatrix<f64>, mask: &DMatrix<bool>) -> DVector<f64> {
    let p = x.ncols();
    let mut mean = DVector::zeros(p);
    let mut counts = DVector::<f64>::zeros(p);
    for i in 0..x.nrows() {
        for j in 0..p {
            if !mask[(i, j)] {
                mean[j] += x[(i, j)];
                counts[j] += 1.0;
            }
        }
    }
    for j in 0..p {
        if counts[j] > 0.0 {
            mean[j] /= counts[j];
        }
    }
    mean
}

fn center(x: &DMatrix<f64>, mask: &DMatrix<bool>, mean: &DVector<f64>) -> DMatrix<f64> {
    let (n, p) = x.shape();
    let mut out = DMatrix::zeros(n, p);
    for i in 0..n {
        for j in 0..p {
            if !mask[(i, j)] {
                out[(i, j)] = x[(i, j)] - mean[j];
            }
            // missing → 0.0 (already zero)
        }
    }
    out
}

/// Build observation-pattern groups.  Samples with identical sets of
/// observed features share a `PatternGroup`.
fn build_pattern_groups(mask: &DMatrix<bool>, n: usize, p: usize) -> Vec<PatternGroup> {
    let mut map: HashMap<Vec<usize>, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let obs: Vec<usize> = (0..p).filter(|&j| !mask[(i, j)]).collect();
        map.entry(obs).or_default().push(i);
    }
    let mut groups: Vec<PatternGroup> = map
        .into_iter()
        .map(|(obs_indices, sample_indices)| PatternGroup {
            sample_indices,
            obs_indices,
        })
        .collect();

    // Sort deterministically so floating-point accumulation order is
    // reproducible across runs (HashMap iteration order is randomised).
    groups.sort_by(|a, b| a.obs_indices.cmp(&b.obs_indices));

    groups
}

/// PCA warm-start: SVD of (mean-imputed) centred data.
fn pca_init(x_c: &DMatrix<f64>, d: usize) -> Result<(DMatrix<f64>, f64)> {
    let (n, p) = x_c.shape();
    let svd = SVD::new(x_c.clone(), false, true);
    let sv = &svd.singular_values;
    let v_t = svd
        .v_t
        .as_ref()
        .ok_or(PPCAError::MatrixError("SVD V^T unavailable".into()))?;

    let sigma2 = if p > d {
        let tail: f64 = sv.iter().skip(d).map(|s| s * s / n as f64).sum();
        (tail / (p - d) as f64).max(MIN_VARIANCE)
    } else {
        1e-3
    };

    let n_sv = sv.len();
    let mut w = DMatrix::zeros(p, d);
    for k in 0..d.min(n_sv) {
        let lam = sv[k] * sv[k] / n as f64;
        let scale = (lam - sigma2).max(MIN_VARIANCE).sqrt();
        for j in 0..p {
            w[(j, k)] = v_t[(k, j)] * scale;
        }
    }
    Ok((w, sigma2))
}

/// E-step quantities for one observation pattern.
fn e_step_pattern(
    w: &DMatrix<f64>,
    psi: &DVector<f64>,
    obs: &[usize],
    d: usize,
) -> Result<PatternEStep> {
    let n_obs = obs.len();

    if n_obs == 0 {
        // No observations → prior: Σ = I, ll_const = 0.
        return Ok(PatternEStep {
            sigma: DMatrix::identity(d, d),
            sigma_wt_psi_inv: DMatrix::zeros(d, 0),
            ll_const: 0.0,
            c_inv: DMatrix::zeros(0, 0),
        });
    }

    // W_o (|O| × d), psi_inv_o (|O|)
    let mut w_o = DMatrix::zeros(n_obs, d);
    let mut psi_inv_o = DVector::zeros(n_obs);
    let mut log_psi_sum = 0.0;
    for (k, &j) in obs.iter().enumerate() {
        let pv = psi[j].max(MIN_VARIANCE);
        psi_inv_o[k] = 1.0 / pv;
        log_psi_sum += pv.ln();
        for l in 0..d {
            w_o[(k, l)] = w[(j, l)];
        }
    }

    // Ψ_o^{-1} W_o  (|O| × d)
    let psi_inv_wo = DMatrix::from_fn(n_obs, d, |k, l| w_o[(k, l)] * psi_inv_o[k]);

    // M = I + W_o^T Ψ_o^{-1} W_o  (d × d)
    let m_mat = DMatrix::identity(d, d) + w_o.transpose() * &psi_inv_wo;
    let sigma = chol_inv(&m_mat)?;

    // Σ W_o^T Ψ_o^{-1}  (d × |O|)
    let sigma_wt_psi_inv = &sigma * psi_inv_wo.transpose();

    // log|C_o| = log|M| + Σ log(ψ_j)
    let log_det_m = chol_log_det(&m_mat);
    let log_det_c = log_det_m + log_psi_sum;
    let ll_const = -0.5 * (n_obs as f64 * LOG_2PI + log_det_c);

    // C_o^{-1} via Woodbury: Ψ_o^{-1} - Ψ_o^{-1} W_o Σ W_o^T Ψ_o^{-1}
    // = diag(ψ_inv_o) - psi_inv_wo Σ psi_inv_wo^T
    let psi_inv_wo_sigma = &psi_inv_wo * &sigma; // |O| × d
    let correction = &psi_inv_wo_sigma * psi_inv_wo.transpose(); // |O| × |O|
    let mut c_inv = DMatrix::zeros(n_obs, n_obs);
    for k in 0..n_obs {
        c_inv[(k, k)] = psi_inv_o[k];
    }
    c_inv -= &correction;

    Ok(PatternEStep {
        sigma,
        sigma_wt_psi_inv,
        ll_const,
        c_inv,
    })
}

/// Compute posterior means for all samples (n × d matrix stored as DMatrix).
fn compute_posterior_means(
    x_c: &DMatrix<f64>,
    groups: &[PatternGroup],
    esteps: &[PatternEStep],
    n: usize,
    d: usize,
) -> DMatrix<f64> {
    let mut mu = DMatrix::zeros(n, d);
    for (g, es) in groups.iter().zip(esteps.iter()) {
        if g.obs_indices.is_empty() {
            continue;
        }
        // Build X_o for the whole group:  (|group| × |O|)
        let n_grp = g.sample_indices.len();
        let n_obs = g.obs_indices.len();
        let mut x_o = DMatrix::zeros(n_grp, n_obs);
        for (gi, &si) in g.sample_indices.iter().enumerate() {
            for (k, &j) in g.obs_indices.iter().enumerate() {
                x_o[(gi, k)] = x_c[(si, j)];
            }
        }
        // mu_group = X_o × (Σ W_o^T Ψ_o^{-1})^T  →  (|group| × d)
        let mu_group = &x_o * es.sigma_wt_psi_inv.transpose();
        for (gi, &si) in g.sample_indices.iter().enumerate() {
            for l in 0..d {
                mu[(si, l)] = mu_group[(gi, l)];
            }
        }
    }
    mu
}

/// Observed-data log-likelihood.
fn compute_log_likelihood(
    x_c: &DMatrix<f64>,
    groups: &[PatternGroup],
    esteps: &[PatternEStep],
) -> f64 {
    let mut ll = 0.0;
    for (g, es) in groups.iter().zip(esteps.iter()) {
        if g.obs_indices.is_empty() {
            continue;
        }
        let n_obs = g.obs_indices.len();
        for &si in &g.sample_indices {
            // x_o for this sample
            let mut x_o = DVector::zeros(n_obs);
            for (k, &j) in g.obs_indices.iter().enumerate() {
                x_o[k] = x_c[(si, j)];
            }
            // quadratic: x_o^T C_o^{-1} x_o
            let quad = x_o.dot(&(&es.c_inv * &x_o));
            ll += es.ll_const - 0.5 * quad;
        }
    }
    ll
}

/// M-step: update W and ψ.
fn m_step(
    x_c: &DMatrix<f64>,
    mu: &DMatrix<f64>,
    groups: &[PatternGroup],
    esteps: &[PatternEStep],
    config: &PPCAConfig,
    _n: usize,
    p: usize,
    d: usize,
) -> (DMatrix<f64>, DVector<f64>) {
    // Per-feature accumulators for W update:
    //   xz[j] = Σ_{i: j obs} x_ij μ_i    (d-vector)
    //   zz[j] = Σ_{i: j obs} A_i          (d × d matrix)
    // where A_i = μ_i μ_i^T + Σ_i  (Σ_i shared within pattern group)
    let mut xz = vec![DVector::<f64>::zeros(d); p];
    let mut zz = vec![DMatrix::<f64>::zeros(d, d); p];

    for (g, es) in groups.iter().zip(esteps.iter()) {
        for &si in &g.sample_indices {
            let mu_i = mu.row(si).transpose(); // d × 1
            let a_i = &mu_i * mu_i.transpose() + &es.sigma; // d × d
            for &j in &g.obs_indices {
                let x_ij = x_c[(si, j)];
                // xz[j] += x_ij * μ_i
                xz[j] += x_ij * &mu_i;
                // zz[j] += A_i
                zz[j] += &a_i;
            }
        }
    }

    // W update: solve (zz[j] + λI) w_j = xz[j] for each feature j
    let lambda_i = config.l2_penalty * DMatrix::<f64>::identity(d, d);
    let mut w_new = DMatrix::zeros(p, d);
    for j in 0..p {
        let zz_reg = &zz[j] + &lambda_i;
        // Use SVD solve for numerical stability when the normal equations
        // are singular or ill-conditioned.
        let svd = SVD::new(zz_reg, true, true);
        let w_j = svd.solve(&xz[j], 1e-12).unwrap_or_else(|_| DVector::zeros(d));
        for l in 0..d {
            w_new[(j, l)] = w_j[l];
        }
    }

    // Noise update: ψ_j = (1/n_j) Σ_{i:j obs} [(x_ij - w_j^T μ_i)² + w_j^T Σ_i w_j]
    let mut psi_num = DVector::<f64>::zeros(p);
    let mut psi_cnt = DVector::<f64>::zeros(p);

    for (g, es) in groups.iter().zip(esteps.iter()) {
        // Precompute w_j^T Σ w_j for each observed feature in this pattern
        // (shared across samples in the group).
        let wt_sigma_w: Vec<f64> = g
            .obs_indices
            .iter()
            .map(|&j| {
                let w_j: DVector<f64> = w_new.row(j).transpose();
                w_j.dot(&(&es.sigma * &w_j))
            })
            .collect();

        for &si in &g.sample_indices {
            let mu_i = mu.row(si).transpose();
            for (ki, &j) in g.obs_indices.iter().enumerate() {
                let w_j: DVector<f64> = w_new.row(j).transpose();
                let resid = x_c[(si, j)] - w_j.dot(&mu_i);
                psi_num[j] += resid * resid + wt_sigma_w[ki];
                psi_cnt[j] += 1.0;
            }
        }
    }

    let psi_new = match config.noise_type {
        NoiseType::Isotropic => {
            let total_num: f64 = psi_num.iter().sum();
            let total_cnt: f64 = psi_cnt.iter().sum();
            let sigma2 = if total_cnt > 0.0 {
                (total_num / total_cnt).max(MIN_VARIANCE)
            } else {
                MIN_VARIANCE
            };
            DVector::from_element(p, sigma2)
        }
        NoiseType::Diagonal => DVector::from_fn(p, |j, _| {
            if psi_cnt[j] > 0.0 {
                (psi_num[j] / psi_cnt[j]).max(MIN_VARIANCE)
            } else {
                MIN_VARIANCE
            }
        }),
    };

    (w_new, psi_new)
}

/// Explained variance ratio from W^T W eigenvalues.
fn compute_explained_variance(w: &DMatrix<f64>, psi: &DVector<f64>) -> DVector<f64> {
    let wtw = w.transpose() * w;
    let svd = SVD::new(wtw, false, false);
    let eigenvalues = svd.singular_values;
    let noise_sum: f64 = psi.iter().sum();
    let total = eigenvalues.iter().sum::<f64>() + noise_sum;
    if total > 0.0 {
        eigenvalues / total
    } else {
        DVector::from_element(eigenvalues.len(), 1.0 / eigenvalues.len() as f64)
    }
}

// ── Linear algebra helpers ──────────────────────────────────────────────────

/// Invert a symmetric positive-definite matrix via Cholesky.
/// Falls back to regular inverse if Cholesky fails.
fn chol_inv(m: &DMatrix<f64>) -> Result<DMatrix<f64>> {
    if let Some(chol) = m.clone().cholesky() {
        Ok(chol.inverse())
    } else {
        m.clone()
            .try_inverse()
            .ok_or(PPCAError::MatrixError("Singular matrix".into()))
    }
}

/// Log-determinant of a symmetric positive-definite matrix via Cholesky.
fn chol_log_det(m: &DMatrix<f64>) -> f64 {
    if let Some(chol) = m.clone().cholesky() {
        let l = chol.l();
        let d = l.nrows();
        let mut ld = 0.0;
        for i in 0..d {
            ld += l[(i, i)].ln();
        }
        2.0 * ld
    } else {
        // Fallback: eigendecomposition
        let svd = SVD::new(m.clone(), false, false);
        svd.singular_values
            .iter()
            .map(|s| s.max(MIN_VARIANCE).ln())
            .sum()
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn random_data(rows: usize, cols: usize, seed: u64) -> DMatrix<f64> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        DMatrix::from_fn(rows, cols, |_, _| rng.gen::<f64>())
    }

    #[test]
    fn test_fit_simple() {
        let data = random_data(20, 5, 0);
        let mask = DMatrix::from_element(20, 5, false);
        let mut ppca = PPCA::new(2);
        assert!(ppca.fit(&data, &mask).is_ok());
        assert!(ppca.n_iter().unwrap() > 0);
        assert!(!ppca.log_likelihoods().unwrap().is_empty());
    }

    #[test]
    fn test_transform_shape() {
        let data = random_data(20, 5, 1);
        let mask = DMatrix::from_element(20, 5, false);
        let mut ppca = PPCA::new(2);
        ppca.fit(&data, &mask).unwrap();
        let y = ppca.transform(&data).unwrap();
        assert_eq!(y.nrows(), 20);
        assert_eq!(y.ncols(), 2);
    }

    #[test]
    fn test_missing_values() {
        let data = random_data(20, 5, 2);
        let mut mask = DMatrix::from_element(20, 5, false);
        mask[(0, 0)] = true;
        mask[(1, 2)] = true;
        mask[(3, 4)] = true;
        let mut ppca = PPCA::new(2);
        assert!(ppca.fit(&data, &mask).is_ok());
    }

    #[test]
    fn test_diagonal_noise() {
        let data = random_data(30, 8, 3);
        let mask = DMatrix::from_element(30, 8, false);
        let config = PPCAConfig {
            n_components: 2,
            noise_type: NoiseType::Diagonal,
            ..PPCAConfig::default()
        };
        let mut ppca = PPCA::with_config(config);
        assert!(ppca.fit(&data, &mask).is_ok());
        let nv = ppca.noise_variances().unwrap();
        assert_eq!(nv.len(), 8);
        assert!(nv.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn test_l2_penalty() {
        let data = random_data(30, 5, 4);
        let mask = DMatrix::from_element(30, 5, false);
        let config = PPCAConfig {
            n_components: 2,
            l2_penalty: 0.1,
            ..PPCAConfig::default()
        };
        let mut ppca = PPCA::with_config(config);
        assert!(ppca.fit(&data, &mask).is_ok());
    }

    #[test]
    fn test_ll_non_decreasing() {
        let data = random_data(50, 10, 5);
        let mask = DMatrix::from_element(50, 10, false);
        let config = PPCAConfig {
            n_components: 3,
            max_iterations: 50,
            tol: 1e-10, // tight tol so we get many iterations
            random_state: Some(42),
            ..PPCAConfig::default()
        };
        let mut ppca = PPCA::with_config(config);
        ppca.fit(&data, &mask).unwrap();
        let ll = ppca.log_likelihoods().unwrap();
        for i in 1..ll.len() {
            assert!(
                ll[i] >= ll[i - 1] - 1e-6,
                "LL decreased at iter {}: {} -> {}",
                i,
                ll[i - 1],
                ll[i]
            );
        }
    }

    #[test]
    fn test_mask_shape_mismatch() {
        let data = random_data(20, 5, 6);
        let mask = DMatrix::from_element(20, 3, false); // wrong shape
        let mut ppca = PPCA::new(2);
        assert!(ppca.fit(&data, &mask).is_err());
    }

    #[test]
    fn test_negative_l2_penalty() {
        let data = random_data(20, 5, 7);
        let mask = DMatrix::from_element(20, 5, false);
        let config = PPCAConfig {
            n_components: 2,
            l2_penalty: -0.1,
            ..PPCAConfig::default()
        };
        let mut ppca = PPCA::with_config(config);
        assert!(ppca.fit(&data, &mask).is_err());
    }
}
