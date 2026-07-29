"""Basic physical-sanity tests for FaSTMM2.solve(), no CLI binary required."""

import numpy as np
import pytest

from pyfastmm import FaSTMM2, eps_from_index


def test_eps_from_index_squares_refractive_index():
    assert eps_from_index(1.5 + 0.01j) == pytest.approx((1.5 + 0.01j) ** 2)
    np.testing.assert_allclose(
        eps_from_index([1.5 + 0.01j, 2.0 + 0.0j]),
        [(1.5 + 0.01j) ** 2, (2.0 + 0.0j) ** 2],
    )


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


def test_complex_k_absorbing_medium_no_nan():
    """Regression test for a real bug: mueller_matrix()/
    compute_cluster_fields_far() picked the far-field evaluation radius as
    RR = 1.0d6 * real(k) (external/fastmm2/src/mie.f90, dimensionally wrong
    -- RR is a distance in meters, real(k) has units of 1/length -- and
    numerically catastrophic: for optical k ~1e7 rad/m this put RR ~1e13 m,
    at which exp(i*k*RR) underflows to exact complex zero for *any*
    absorbing medium, i.e. any complex k with nonzero imaginary part --
    corrupting the entire Mueller/Jones matrix (division by exact zero)
    while cross sections stayed finite (they don't evaluate the field at
    RR). Fixed via src/pyfastmm/_fortran/patches/mie-far-field-distance-fix.patch
    (RR = 1.0d6 / real(k)). This is not an edge case for pyfastmm's own
    v1.1 medium support -- any nonzero medium.refractive_index absorption
    (k channel) triggers it, so a strongly absorbing composite (metal
    nanoparticle in a host medium, giving a negative-real-part effective
    permittivity) is exactly the regime that first surfaced this."""
    f = FaSTMM2()
    n_medium = 1.85 + 0.0011j  # a weakly-absorbing dielectric host
    eps = ((2.08 + 3.113j) / n_medium) ** 2  # metal-in-dielectric composite
    k = 2 * np.pi * n_medium / 0.52e-6

    result = f.solve(
        coords=[[0.0, 0.0, 0.0]], radii=[5e-9], eps=[eps], k=complex(k),
        N_theta=37, N_phi=4, formulation=0, acc=2, tol=1e-4,
    )
    mueller = result["mueller"]
    assert not np.any(np.isnan(mueller))
    assert np.all(mueller[:, 2] >= 0)  # P11 (phase function) positivity
    assert result["c_ext"] > 0
    assert result["c_abs"] > 0
    assert result["c_ext"] == pytest.approx(
        result["c_ext_minus_c_abs"] + result["c_abs"]
    )


def test_set_num_threads_does_not_change_results():
    """Thread count must be purely a performance knob -- results have to
    be identical (well within GMRES/quadrature tolerance) regardless of
    how many OpenMP threads the matvec/solver/far-field code use."""
    f = FaSTMM2()
    kwargs = dict(
        coords=[[-2.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        radii=[1.0, 1.0],
        eps=[2.5 + 0.1j, 2.5 + 0.1j],
        k=1.0,
        N_theta=11,
        N_phi=1,
        tol=1e-6,
    )
    results = [f.solve(**kwargs, n_threads=n) for n in (1, 2, 4)]
    for r in results[1:]:
        assert r["c_ext"] == pytest.approx(results[0]["c_ext"], rel=1e-10)
        assert r["c_abs"] == pytest.approx(results[0]["c_abs"], rel=1e-10)
        # atol matters here: some Mueller entries are near-zero by symmetry
        # for this cluster, where differing OpenMP reduction order across
        # thread counts leaves relative-only comparison noisy despite the
        # absolute difference being ~1e-18 (physically negligible).
        np.testing.assert_allclose(
            r["mueller"], results[0]["mueller"], rtol=1e-8, atol=1e-10
        )


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
