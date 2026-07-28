"""pyfastmm -- Python bindings for FaSTMM2 (Fast Superposition T-Matrix Method)."""

from pyfastmm._fastmm import FaSTMM2, eps_from_index, get_tmatrix_size

__all__ = ["FaSTMM2", "eps_from_index", "get_tmatrix_size"]
__version__ = "1.0.0"
