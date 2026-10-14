"""Tests for pyfastmm._results -- typed results stored on FaSTMM.results."""

import numpy as np
import pytest

from pyfastmm import FaSTMM, SolveResult, TMatrixResult

_COORDS = [[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]]
_RADII = [1.0, 1.0]
_N = 1.5 + 0.01j
_WL = 5.0  # same length unit as coords/radii


def _cluster():
    return FaSTMM(_COORDS, _RADII, _N)


def test_results_none_until_first_call():
    assert _cluster().results is None


def test_solve_stores_validated_results_and_returns_matching_dict():
    """The two access styles -- returned dict and stored model -- must be
    the same data: model attributes == dict values, to_dict() closes the
    loop."""
    f = _cluster()
    returned = f.solve(_WL, N_theta=11, N_phi=1, acc=1)

    res = f.results
    assert isinstance(res, SolveResult)
    assert res.c_ext == returned["c_ext"]
    assert res.wavelength_vacuum == _WL
    np.testing.assert_array_equal(res.mueller, returned["mueller"])
    assert res.to_dict().keys() == returned.keys()


def test_orientation_averaged_results_have_no_jones():
    f = _cluster()
    returned = f.solve(_WL, N_theta=11, N_phi=1, N_ave=2, acc=1)
    res = f.results
    assert isinstance(res, SolveResult)
    assert res.jones is None
    assert "jones" not in returned  # to_dict() drops the absent block


def test_compute_tmatrix_stores_tmatrix_results():
    f = _cluster()
    returned = f.compute_tmatrix(_WL, t_order=4, acc=1)
    res = f.results
    assert isinstance(res, TMatrixResult)
    np.testing.assert_array_equal(res.Taa, returned["Taa"])
    assert res.wavelength_vacuum == _WL


def test_results_hold_last_call_only():
    """Documented semantics: results is the *latest* outcome -- a
    compute_tmatrix after a solve replaces the SolveResult."""
    f = _cluster()
    f.solve(_WL, N_theta=11, N_phi=1, acc=1)
    assert isinstance(f.results, SolveResult)
    f.compute_tmatrix(_WL, t_order=4, acc=1)
    assert isinstance(f.results, TMatrixResult)


def test_solve_result_rejects_non_finite_mueller():
    """The load-bearing validation: an all-NaN Mueller (the historical
    RR-bug signature for absorbing media) must fail loudly at
    construction, not flow silently into downstream analysis."""
    good = _cluster()
    good.solve(_WL, N_theta=11, N_phi=1, acc=1)
    assert good.results is not None
    fields = good.results.to_dict()
    fields["mueller"] = np.full_like(fields["mueller"], np.nan)
    with pytest.raises(ValueError, match="non-finite"):
        SolveResult(**fields)


def test_tmatrix_result_rejects_non_finite_blocks():
    good = _cluster()
    good.compute_tmatrix(_WL, t_order=4, acc=1)
    assert good.results is not None
    fields = good.results.to_dict()
    fields["Tbb"] = np.full_like(fields["Tbb"], np.nan + 0j)
    with pytest.raises(ValueError, match="non-finite"):
        TMatrixResult(**fields)
