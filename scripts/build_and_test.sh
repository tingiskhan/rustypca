#!/bin/bash
# Build and test script for PPCA

set -e

echo "Building PPCA..."
cd ppca-py

# Install build dependencies if needed
python -m pip install -q maturin numpy scikit-learn pytest 2>/dev/null || true

# Build the Rust extension
echo "Building Rust extension..."
maturin develop

# Run tests
echo "Running tests..."
python -m pytest tests/test_ppca.py -v

echo "All tests passed!"
