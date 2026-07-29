"""Tests for pyfastmm.Cluster -- the shared class-driven single-cluster API."""

import numpy as np
import pytest

from pyfastmm import Cluster, FaSTMM2, eps_from_index, radiative_transfer_phase_function


def test_vacuum_medium_matches_raw_fastmm2_call():
    """With the default (vacuum) medium, Cluster.solve() must reduce
    exactly to a raw FaSTMM2().solve() call -- no transform, no
    correction (medium_flux_norm(1+0j) == 1.0)."""
    coords = [[0.0, 0.0, 0.0]]
    radii = [1.0]
    n_particle = 3.0 + 0.1j

    cluster = Cluster(coords, radii, refractive_index=n_particle)

    raw = FaSTMM2().solve(coords, radii, [eps_from_index(n_particle)], 1.0, N_theta=19, N_phi=1)
    cluster_result = cluster.solve(
        wavelength_vacuum=2 * np.pi, N_theta=19, N_phi=1
    )

    assert cluster_result["c_ext"] == pytest.approx(raw["c_ext"])
    assert cluster_result["c_abs"] == pytest.approx(raw["c_abs"])
    np.testing.assert_allclose(cluster_result["mueller"], raw["mueller"])


def test_refractive_index_broadcasts_to_all_particles():
    coords = [[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]]
    radii = [1.0, 1.0]
    cluster = Cluster(coords, radii, refractive_index=1.5 + 0.01j)
    assert cluster.refractive_index.shape == (2,)
    np.testing.assert_array_equal(cluster.refractive_index, [1.5 + 0.01j, 1.5 + 0.01j])


def test_medium_flux_normalization_matches_raw_solve_times_re_n_medium():
    """Same regression this bug got fixed for in _config.py::SweepConfig.run()
    -- Cluster.solve() must apply the identical correction."""
    n_particle = 1.5 + 0.01j
    n_medium = 1.3 + 0.0j
    wavelength = 5.0

    cluster = Cluster(
        [[0.0, 0.0, 0.0]], [1.0], refractive_index=n_particle,
        medium_refractive_index=n_medium,
    )
    result = cluster.solve(wavelength, N_theta=19, N_phi=1)

    k_vacuum = 2 * np.pi / wavelength
    k_medium = k_vacuum * n_medium
    eps_eff = (n_particle / n_medium) ** 2
    raw = FaSTMM2().solve(
        [[0.0, 0.0, 0.0]], [1.0], [eps_eff], complex(k_medium), N_theta=19, N_phi=1
    )

    assert result["c_ext"] == pytest.approx(raw["c_ext"] * n_medium.real)
    assert result["c_abs"] == pytest.approx(raw["c_abs"] * n_medium.real)
    assert result["c_ext_minus_c_abs"] == pytest.approx(
        raw["c_ext_minus_c_abs"] * n_medium.real
    )
    # c_sca (far-field) is NOT corrected -- must equal the raw value
    # (modulo floating-point noise from e.g. OpenMP reduction order
    # differing between separately-run solves).
    assert result["c_sca"] == pytest.approx(raw["c_sca"])


def test_medium_correction_is_a_no_op_for_vacuum():
    cluster_vac = Cluster([[0, 0, 0]], [1.0], refractive_index=2.0 + 0.1j)
    cluster_explicit_vac = Cluster(
        [[0, 0, 0]], [1.0], refractive_index=2.0 + 0.1j,
        medium_refractive_index=1.0 + 0.0j,
    )
    r1 = cluster_vac.solve(2 * np.pi, N_theta=11, N_phi=1)
    r2 = cluster_explicit_vac.solve(2 * np.pi, N_theta=11, N_phi=1)
    assert r1["c_ext"] == pytest.approx(r2["c_ext"])


def test_solve_returns_k_medium_and_eps():
    cluster = Cluster(
        [[0, 0, 0]], [1.0], refractive_index=2.0 + 0.1j,
        medium_refractive_index=1.3 + 0.0j,
    )
    result = cluster.solve(5.0, N_theta=11, N_phi=1)
    assert result["k_medium"] == pytest.approx(2 * np.pi / 5.0 * 1.3)
    assert result["eps"] == pytest.approx(((2.0 + 0.1j) / 1.3) ** 2)


def test_single_sphere_orientation_invariant_to_incidence_angle():
    """A single sphere has no preferred axis -- rotating the incident
    direction must not change its cross sections, for any (polar,
    azimuthal) pair, regardless of rotation-order convention. A strong,
    convention-independent sanity check for _rotate_for_incidence()."""
    cluster = Cluster([[0.0, 0.0, 0.0]], [1.0], refractive_index=3.0 + 0.1j)
    baseline = cluster.solve(2 * np.pi, N_theta=11, N_phi=1)
    tilted = cluster.solve(
        2 * np.pi, incident_polar_deg=30.0, incident_azimuthal_deg=45.0,
        N_theta=11, N_phi=1,
    )
    assert tilted["c_ext"] == pytest.approx(baseline["c_ext"], rel=1e-8)
    assert tilted["c_abs"] == pytest.approx(baseline["c_abs"], rel=1e-8)


def test_compute_tmatrix_shape_and_medium_transform():
    from pyfastmm import get_tmatrix_size

    cluster = Cluster(
        [[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]], [1.0, 1.0],
        refractive_index=2.5 + 0.05j, medium_refractive_index=1.2 + 0.0j,
    )
    result = cluster.compute_tmatrix(5.0, t_order=4)
    nm = get_tmatrix_size(4)
    assert result["Taa"].shape == (nm, nm)
    assert result["k_medium"] == pytest.approx(2 * np.pi / 5.0 * 1.2)


def test_radiative_transfer_phase_function_normalizes_to_4pi():
    """Integrating the normalized S11 (with a matching quadrature) over
    the sphere must reproduce 4*pi -- the defining property of the
    radiative-transfer convention this function converts to."""
    cluster = Cluster([[0.0, 0.0, 0.0]], [1.0], refractive_index=1.5 + 0.01j)
    n_theta, n_phi = 91, 16
    result = cluster.solve(2 * np.pi, N_theta=n_theta, N_phi=n_phi)

    # Full (theta, phi) grid, not just the phi=0 cut, to integrate over
    # the whole sphere (a symmetric sphere's S11 doesn't depend on phi,
    # but reuse the actual grid rather than assuming that).
    mueller = result["mueller"]
    phase_norm = 4.0 * np.pi / (abs(result["k_medium"]) ** 2 * result["c_ext_minus_c_abs"])
    theta = mueller[:, 1]
    s11 = mueller[:, 2] * phase_norm
    dtheta = np.pi / (n_theta - 1)
    dphi = 2 * np.pi / n_phi
    integral = np.sum(s11 * np.sin(theta) * dtheta * dphi)
    assert integral == pytest.approx(4.0 * np.pi, rel=0.05)

    # The convenience function must match the same math for the phi=0 cut.
    pf = radiative_transfer_phase_function(result, n_theta=n_theta)
    np.testing.assert_allclose(pf[:, 1], s11[:n_theta])


def test_n_threads_defaults_to_one_and_does_not_change_results():
    """n_threads defaults to 1 (not "all cores") -- see solve()'s
    docstring for why -- and, like set_num_threads() itself, must be a
    pure performance knob: results identical regardless of value."""
    import inspect

    assert inspect.signature(Cluster.solve).parameters["n_threads"].default == 1
    assert inspect.signature(Cluster.compute_tmatrix).parameters["n_threads"].default == 1

    cluster = Cluster(
        [[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]], [1.0, 1.0],
        refractive_index=2.5 + 0.1j, medium_refractive_index=1.3 + 0.0j,
    )
    results = [
        cluster.solve(5.0, N_theta=11, N_phi=1, n_threads=n) for n in (1, 2, 4)
    ]
    for r in results[1:]:
        assert r["c_ext"] == pytest.approx(results[0]["c_ext"], rel=1e-10)
