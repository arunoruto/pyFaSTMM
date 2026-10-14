"""Smoke test for the f2py extension itself (not the Python-facing API).

Skipped if the extension cannot be imported.
"""

import numpy as np
import pytest


def _import_ext():
    return pytest.importorskip("pyfastmm._fastmm_ext")


def test_fastmm2_solve_fixed_single_sphere():
    ext = _import_ext()
    coords = np.asfortranarray([[0.0], [0.0], [0.0]])
    radii = np.array([1.0])
    eps_r = np.array([3.0])
    eps_i = np.array([0.1])

    mueller, jones, crs = ext.fastmm2_f2py_bindings.fastmm2_solve_fixed(
        coords, radii, eps_r, eps_i, 1.0, 19, 1, 2, 2, 1e-4, 5, 50, 0, 1
    )
    assert mueller.shape == (19, 18)
    assert jones.shape == (19, 6)
    assert crs.shape == (5,)
    assert crs[0] > 0  # Cext


def test_fastmm2_solve_averaged_single_sphere():
    ext = _import_ext()
    coords = np.asfortranarray([[0.0], [0.0], [0.0]])
    radii = np.array([1.0])
    eps_r = np.array([3.0])
    eps_i = np.array([0.1])

    mueller, crs = ext.fastmm2_f2py_bindings.fastmm2_solve_averaged(
        coords, radii, eps_r, eps_i, 1.0, 19, 4, 2, 0, 2, 2, 1e-4, 5, 50, 0, 1
    )
    assert mueller.shape == (19, 17)
    assert crs.shape == (5,)


def test_fastmm2_compute_tmatrix_single_sphere():
    ext = _import_ext()
    coords = np.asfortranarray([[0.0], [0.0], [0.0]])
    radii = np.array([1.0])
    eps_r = np.array([3.0])
    eps_i = np.array([0.1])
    t_order = 4
    nm = (t_order + 1) ** 2 - 1

    Taa, Tab, Tba, Tbb = ext.fastmm2_f2py_bindings.fastmm2_compute_tmatrix(
        coords, radii, eps_r, eps_i, 1.0, t_order, nm, 2, 2, 1e-4, 5, 50, 0, 1
    )
    for m in (Taa, Tab, Tba, Tbb):
        assert m.shape == (nm, nm)
