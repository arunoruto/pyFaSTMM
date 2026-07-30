"""pyfastmm -- Python bindings for FaSTMM2 (Fast Superposition T-Matrix Method)."""

from pyfastmm._cluster import FaSTMM, radiative_transfer_phase_function, rotate_for_incidence
from pyfastmm._dask import run_resolved_dask
from pyfastmm._results import SolveResult, TMatrixResult
from pyfastmm._fastmm import (
    FaSTMM2,
    eps_from_index,
    eps_relative_to_medium,
    get_tmatrix_size,
    k_in_medium,
    medium_flux_norm,
    resolve_n_threads,
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
    "resolve_n_threads",
    "rotate_for_incidence",
    "run_resolved_dask",
    "SolveResult",
    "TMatrixResult",
]
__version__ = "1.0.0"
