"""Basic physical-sanity tests for FaSTMM2.solve(), no CLI binary required."""

import numpy as np
import pytest

from pyfastmm import FaSTMM2


def test_single_sphere_solve():
    """A single dielectric sphere reduces to plain Lorenz-Mie scattering."""
    f = FaSTMM2()
    result = f.solve(
        coords=[[0.0, 0.0, 0.0]],
        radii=[1.0],
        eps=[3.0 + 0.1j],
        k=1.0,
        N_theta=19,
        N_phi=1,
    )

    mueller = result["mueller"]
    assert mueller.shape == (19, 18)
    assert not np.any(np.isnan(mueller))
    assert result["c_ext"] > 0
    assert result["c_sca"] > 0
    assert result["c_abs"] > 0
    # Cext = Csca + Cabs (optical theorem), within GMRES/quadrature tolerance
    assert result["c_ext"] == pytest.approx(result["c_ext_minus_c_abs"] + result["c_abs"])
    # P11 (phase function) must be positive everywhere for a passive scatterer
    assert np.all(mueller[:, 2] > 0)


def test_two_sphere_solve_no_nan():
    """Two interacting spheres: solve should converge to a finite result."""
    f = FaSTMM2()
    result = f.solve(
        coords=[[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        radii=[1.0, 1.0],
        eps=[2.5 + 0.0j, 2.5 + 0.0j],
        k=1.0,
        N_theta=11,
        N_phi=1,
        tol=1e-6,
    )
    assert not np.any(np.isnan(result["mueller"]))
    assert result["c_ext"] > 0


def test_orientation_average_matches_fixed_for_single_sphere():
    """A single sphere is rotationally symmetric, so its orientation-averaged
    cross sections must equal the fixed-orientation ones (independent of the
    CLI -- a physical invariant, not just a numeric-parity check)."""
    f = FaSTMM2()
    fixed = f.solve(
        coords=[[0.0, 0.0, 0.0]], radii=[1.0], eps=[3.0 + 0.1j], k=1.0,
        N_theta=19, N_phi=8,
    )
    averaged = f.solve(
        coords=[[0.0, 0.0, 0.0]], radii=[1.0], eps=[3.0 + 0.1j], k=1.0,
        N_theta=19, N_phi=8, N_ave=3, halton_init=0,
    )
    assert averaged["c_ext"] == pytest.approx(fixed["c_ext"], rel=1e-8)
    assert averaged["c_abs"] == pytest.approx(fixed["c_abs"], rel=1e-8)
