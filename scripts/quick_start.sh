#!/usr/bin/env bash
# Quick Start Guide for PPCA

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  PPCA - Probabilistic PCA with Missing Value Support       ║"
echo "║  Quick Start Guide                                         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

echo "✓ Python3 found: $(python3 --version)"

# Check if Rust is available
if ! command -v cargo &> /dev/null; then
    echo "ℹ️  Rust not found. Installing..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source $HOME/.cargo/env
fi

echo "✓ Rust found: $(rustc --version)"

# Install Python build dependencies
echo ""
echo "📦 Installing Python dependencies..."
pip install -q maturin numpy scikit-learn pytest 2>/dev/null || pip3 install -q maturin numpy scikit-learn pytest

echo "✓ Dependencies installed"

# Build the extension
echo ""
echo "🔨 Building PPCA Rust extension..."
cd ppca-py

if ! maturin develop; then
    echo "❌ Build failed"
    exit 1
fi

echo "✓ Build successful"

# Run tests
echo ""
echo "🧪 Running tests..."
if python3 -m pytest tests/test_ppca.py -v --tb=short; then
    echo ""
    echo "✓ All tests passed!"
else
    echo ""
    echo "⚠️  Some tests failed"
fi

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  ✅ PPCA is ready to use!                                  ║"
echo "╠════════════════════════════════════════════════════════════╣"
echo "║  Next steps:                                               ║"
echo "║  1. Check examples.py for usage examples                   ║"
echo "║  2. Read README.md for API documentation                   ║"
echo "║  3. Review DEVELOPMENT.md for contribution guidelines      ║"
echo "║  4. Try: python3 ../examples.py                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
