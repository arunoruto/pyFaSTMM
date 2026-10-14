"""pyfastmm -- Python bindings for FaSTMM2 (Fast Superposition T-Matrix Method)."""

from importlib.metadata import version as _version

from pyfastmm._cluster import FaSTMM, radiative_transfer_phase_function, rotate_for_incidence
from pyfastmm._config import SweepConfig, load_config
from pyfastmm._dask import run_resolved_dask
from pyfastmm._results import SolveResult, TMatrixResult
from pyfastmm._fastmm import (
    FaSTMM2,
    eps_from_index,
    eps_relative_to_medium,
    get_tmatrix_size,
    k_in_medium,
    resolve_n_threads,
)

__all__ = [
    "FaSTMM",
    "FaSTMM2",
    "eps_from_index",
    "eps_relative_to_medium",
    "get_tmatrix_size",
    "k_in_medium",
    "load_config",
    "radiative_transfer_phase_function",
    "resolve_n_threads",
    "rotate_for_incidence",
    "run_resolved_dask",
    "SolveResult",
    "SweepConfig",
    "TMatrixResult",
]
__version__ = _version("pyfastmm")
