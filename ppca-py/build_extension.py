"""Configuration and build tools for the PPCA package."""

import os
import subprocess
import sys
from pathlib import Path


def build_rust_extension():
    """Build the Rust extension using maturin."""
    ppca_py_dir = Path(__file__).parent
    try:
        subprocess.run(
            ["maturin", "develop"],
            cwd=str(ppca_py_dir),
            check=True,
            capture_output=False
        )
    except FileNotFoundError:
        print("maturin not found. Please install it with: pip install maturin")
        return False
    except subprocess.CalledProcessError as e:
        print(f"Failed to build Rust extension: {e}")
        return False
    return True


if __name__ == "__main__":
    build_rust_extension()
