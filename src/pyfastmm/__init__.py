"""pyfastmm -- Python bindings for FaSTMM2 (Fast Superposition T-Matrix Method)."""

from pyfastmm._cluster import FaSTMM, radiative_transfer_phase_function, rotate_for_incidence
from pyfastmm._fastmm import (
    FaSTMM2,
    eps_from_index,
    eps_relative_to_medium,
    get_tmatrix_size,
    k_in_medium,
    medium_flux_norm,
)

__all__ = [
    "FaSTMM",
    "FaSTMM2",
    "eps_from_index",
    "eps_relative_to_medium",
    "get_tmatrix_size",
    "k_in_medium",
    "medium_flux_norm",
    "radiative_transfer_phase_function",
    "rotate_for_incidence",
]
__version__ = "1.0.0"
