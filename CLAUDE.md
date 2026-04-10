# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
make develop   # Build Rust extension in-place (required after any Rust changes)
make test      # Run pytest with 95% coverage requirement
make format    # Format Rust (cargo fmt) and Python (ruff)
make check     # Check formatting/linting without modifying files
```

To run a single test:
```bash
pytest rustypca/tests/test_ppca.py::TestMissingValues::test_fit_with_missing -v
```

## Architecture

rustypca is a hybrid Python+Rust project. The Python layer exposes a scikit-learn-compatible API; the Rust core handles all numerical computation.

**Data flow:**
1. `PPCA.fit(X)` in `rustypca/_rustypca.py` validates input via `check_array`, computes the NaN mask, then delegates to `PPCARust.fit(X, mask)` via PyO3.
2. The Rust EM algorithm in `src/ppca.rs` groups observations by missing-data pattern (`PatternGroup`) to reuse per-pattern E-step computations (`PatternEStep` caches posterior covariance and `Σ W_o^T Ψ_o^{-1}`).
3. Fitted results (`PPCAResult`) are surfaced back to Python attributes (`components_`, `noise_variances_`, etc.).

**Key files:**
- `rustypca/_rustypca.py` — Python `PPCA` class (sklearn `BaseEstimator + TransformerMixin`)
- `src/ppca.rs` — Core EM algorithm (~700 lines); all linear algebra via `nalgebra`
- `src/bindings.rs` — PyO3 bindings that bridge Python ↔ Rust
- `src/errors.rs` — `PPCAError` enum

**Noise types:** `"isotropic"` (uniform noise, strict PPCA) or `"diagonal"` (per-feature noise, Factor Analysis).

**Missing values in `transform`:** NaN entries are replaced with the training mean before projection — this is intentional and lives in `_rustypca.py`, not the Rust core.

**Convergence criterion:** relative log-likelihood change `|LL[i] - LL[i-1]| / max(|LL[i-1]|, 1.0) < tol`. Minimum noise variance is clamped to `1e-6` to prevent numerical collapse.

**Build toolchain:** `maturin` (build backend), `pyo3 0.23`, `nalgebra 0.33`. Any Rust change requires `make develop` before tests will reflect it.
