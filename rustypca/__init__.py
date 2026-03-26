"""Probabilistic Principal Component Analysis with missing value support."""

from importlib.metadata import PackageNotFoundError, version

from ._rustypca import PPCA

__all__ = ["PPCA"]

try:
    __version__ = version("rustypca")
except PackageNotFoundError:
    __version__ = "unknown"
