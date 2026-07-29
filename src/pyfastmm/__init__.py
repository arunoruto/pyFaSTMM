"""pyfastmm -- Python bindings for FaSTMM2 (Fast Superposition T-Matrix Method)."""

from pyfastmm._fastmm import (
    FaSTMM2,
    eps_from_index,
    eps_relative_to_medium,
    get_tmatrix_size,
    k_in_medium,
    medium_flux_norm,
)

__all__ = [
    "FaSTMM2",
    "eps_from_index",
    "eps_relative_to_medium",
    "get_tmatrix_size",
    "k_in_medium",
    "medium_flux_norm",
]
__version__ = "1.0.0"
