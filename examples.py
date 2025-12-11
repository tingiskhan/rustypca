"""
Example usage of the PPCA library.

This script demonstrates:
1. Basic PPCA usage
2. PPCA with missing values
3. Comparison with scikit-learn PCA
4. Reconstruction error computation
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA as SklearnPCA

# Import PPCA (requires building the Rust extension first)
try:
    from ppca import PPCA
except ImportError:
    print("ERROR: PPCA not installed. Run 'maturin develop' in ppca-py directory first.")
    exit(1)


def example_basic_usage():
    """Basic PPCA usage example."""
    print("=" * 60)
    print("Example 1: Basic PPCA Usage")
    print("=" * 60)
    
    # Generate random data
    np.random.seed(42)
    X = np.random.randn(100, 10)
    
    # Create and fit PPCA
    ppca = PPCA(n_components=2)
    X_transformed = ppca.fit_transform(X)
    
    print(f"Original shape: {X.shape}")
    print(f"Transformed shape: {X_transformed.shape}")
    
    # Reconstruct
    X_reconstructed = ppca.inverse_transform(X_transformed)
    print(f"Reconstructed shape: {X_reconstructed.shape}")
    
    # Reconstruction error
    error = ppca.reconstruction_error(X)
    print(f"Reconstruction error: {error:.6f}")
    print()


def example_with_missing_values():
    """PPCA with missing values example."""
    print("=" * 60)
    print("Example 2: PPCA with Missing Values")
    print("=" * 60)
    
    # Generate data with missing values
    np.random.seed(42)
    X = np.random.randn(100, 10)
    missing_mask = np.random.rand(100, 10) < 0.15  # 15% missing
    
    print(f"Number of missing values: {missing_mask.sum()} / {missing_mask.size}")
    print(f"Missing percentage: {100 * missing_mask.mean():.1f}%")
    
    # Fit PPCA with missing values
    ppca = PPCA(n_components=3, max_iterations=100)
    ppca.fit(X, missing_mask=missing_mask)
    
    # Transform
    X_transformed = ppca.transform(X)
    print(f"Transformed shape: {X_transformed.shape}")
    
    # Compute error only on observed values
    error = ppca.reconstruction_error(X, missing_mask=missing_mask)
    print(f"Reconstruction error (observed values only): {error:.6f}")
    print()


def example_comparison_with_sklearn():
    """Compare PPCA with scikit-learn PCA."""
    print("=" * 60)
    print("Example 3: PPCA vs Scikit-learn PCA")
    print("=" * 60)
    
    # Load iris dataset
    iris = load_iris()
    X = iris.data
    
    n_components = 2
    
    # PPCA
    ppca = PPCA(n_components=n_components)
    X_ppca = ppca.fit_transform(X)
    ppca_error = ppca.reconstruction_error(X)
    
    # Scikit-learn PCA
    sklearn_pca = SklearnPCA(n_components=n_components)
    X_sklearn = sklearn_pca.fit_transform(X)
    X_sklearn_reconstructed = sklearn_pca.inverse_transform(X_sklearn)
    sklearn_error = np.mean((X - X_sklearn_reconstructed) ** 2)
    
    print(f"Data shape: {X.shape}")
    print(f"Number of components: {n_components}")
    print("\nPPCA:")
    print(f"  Transformed shape: {X_ppca.shape}")
    print(f"  Reconstruction error: {ppca_error:.6f}")
    
    print("\nScikit-learn PCA:")
    print(f"  Transformed shape: {X_sklearn.shape}")
    print(f"  Reconstruction error: {sklearn_error:.6f}")
    
    # Compare explained variance
    if hasattr(ppca, 'explained_variance_ratio_'):
        print(f"Explained variance ratio (PPCA): {ppca.explained_variance_ratio_}")
    if hasattr(sklearn_pca, 'explained_variance_ratio_'):
        print(f"Explained variance ratio (Sklearn): {sklearn_pca.explained_variance_ratio_}")
    print()


def example_convergence_with_components():
    """Show how reconstruction error improves with components."""
    print("=" * 60)
    print("Example 4: Reconstruction Error vs. Number of Components")
    print("=" * 60)
    
    np.random.seed(42)
    X = np.random.randn(200, 15)
    
    print("PPCA components | Reconstruction Error")
    print("-" * 40)
    for n_comp in [1, 2, 3, 5, 7, 10, 15]:
        ppca = PPCA(n_components=n_comp)
        ppca.fit(X)
        error = ppca.reconstruction_error(X)
        print(f"        {n_comp:2d}       |      {error:.6f}")
    print()


def example_parameters():
    """Show PPCA parameter interface."""
    print("=" * 60)
    print("Example 5: PPCA Parameters")
    print("=" * 60)
    
    ppca = PPCA(n_components=3, max_iterations=150, tol=1e-5)
    
    params = ppca.get_params()
    print("Current parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Update parameters
    ppca.set_params(n_components=5, max_iterations=200)
    print("\nAfter set_params(n_components=5, max_iterations=200):")
    params = ppca.get_params()
    for key, value in params.items():
        print(f"  {key}: {value}")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("PPCA Library Examples")
    print("=" * 60 + "\n")
    
    try:
        example_basic_usage()
        example_with_missing_values()
        example_comparison_with_sklearn()
        example_convergence_with_components()
        example_parameters()
        
        print("=" * 60)
        print("All examples completed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
