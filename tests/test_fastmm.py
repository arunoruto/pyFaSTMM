"""Tests for pyfastmm.FaSTMM -- the shared class-driven single-cluster API."""

import numpy as np
import pytest

from pyfastmm import (
    FaSTMM,
    FaSTMM2,
    eps_from_index,
    radiative_transfer_phase_function,
)
from pyfastmm._fastmm import _mstm_medium_factor


def test_vacuum_medium_matches_raw_fastmm2_call():
    """With the default (vacuum) medium, FaSTMM.solve() must reduce
    exactly to a raw FaSTMM2().solve() call -- no transform, no
    correction (_mstm_medium_factor(1+0j) == 1.0)."""
    coords = [[0.0, 0.0, 0.0]]
    radii = [1.0]
    n_particle = 3.0 + 0.1j

    fastmm = FaSTMM(coords, radii, refractive_index=n_particle)

    raw = FaSTMM2().solve(coords, radii, [eps_from_index(n_particle)], 1.0, N_theta=19, N_phi=1)
    result = fastmm.solve(wavelength_vacuum=2 * np.pi, N_theta=19, N_phi=1)

    assert result["c_ext"] == pytest.approx(raw["c_ext"])
    assert result["c_abs"] == pytest.approx(raw["c_abs"])
    np.testing.assert_allclose(result["mueller"], raw["mueller"])


def test_refractive_index_broadcasts_to_all_particles():
    coords = [[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]]
    radii = [1.0, 1.0]
    fastmm = FaSTMM(coords, radii, refractive_index=1.5 + 0.01j)
    assert fastmm.refractive_index.shape == (2,)
    np.testing.assert_array_equal(fastmm.refractive_index, [1.5 + 0.01j, 1.5 + 0.01j])


def _host_medium_solve(n_medium, **kwargs):
    """A single absorbing sphere in a host, through FaSTMM and through a raw
    FaSTMM2 call with the same eps/k transform."""
    n_particle = 1.5 + 0.01j
    wavelength = 5.0
    fastmm = FaSTMM(
        [[0.0, 0.0, 0.0]], [1.0], refractive_index=n_particle,
        medium_refractive_index=n_medium,
    )
    result = fastmm.solve(wavelength, N_theta=19, N_phi=1, **kwargs)
    raw = FaSTMM2().solve(
        [[0.0, 0.0, 0.0]], [1.0], [(n_particle / n_medium) ** 2],
        complex(2 * np.pi / wavelength * n_medium), N_theta=19, N_phi=1,
    )
    return result, raw


_CROSS_SECTIONS = ("c_ext", "c_abs", "c_ext_minus_c_abs", "c_sca")


def test_host_medium_default_is_raw_fastmm2():
    """Default convention is physical: FaSTMM.solve() returns exactly what
    the raw FaSTMM2 call does (no Re(n_medium) factor)."""
    result, raw = _host_medium_solve(1.3 + 0.0j)
    for key in _CROSS_SECTIONS:
        assert result[key] == pytest.approx(raw[key])


def test_mstm_convention_is_re_n_medium_times_physical():
    n_medium = 1.3 + 0.002j
    physical, _ = _host_medium_solve(n_medium)
    mstm, _ = _host_medium_solve(n_medium, cross_section_convention="mstm")
    for key in _CROSS_SECTIONS:
        assert mstm[key] == pytest.approx(physical[key] * n_medium.real)
    assert _mstm_medium_factor(n_medium) == pytest.approx(n_medium.real)


def test_unknown_cross_section_convention_raises():
    fastmm = FaSTMM([[0.0, 0.0, 0.0]], [1.0], refractive_index=1.5 + 0.0j)
    with pytest.raises(ValueError, match="cross_section_convention"):
        fastmm.solve(1.0, N_theta=11, N_phi=1, cross_section_convention="vacuum")


# Single sphere in a non-absorbing host vs Mie theory with the relative
# index m / n_host and the size parameter in the host (Bohren & Huffman).
_WL = 1.0e-6
_N_FE = 2.868 + 4.192j  # iron (Querry) at 1 um, an absorbing test particle


def _mie_q(n_p, n_host, r, wl=_WL):
    miepython = pytest.importorskip("miepython")
    qe, qs, _, g = miepython.efficiencies_mx(n_p / n_host, 2 * np.pi * n_host * r / wl)
    return qe, qe - qs, g


@pytest.mark.parametrize("n_host", (1.0, 1.33, 1.8476))
@pytest.mark.parametrize("n_p,r", [(_N_FE, 5e-9), (_N_FE, 200e-9), (1.6 + 0j, 50e-9)])
def test_single_sphere_in_host_equals_mie(n_host, n_p, r):
    qe, qa, g = _mie_q(n_p, n_host, r)
    res = FaSTMM([[0, 0, 0]], [r], refractive_index=n_p, medium_refractive_index=n_host).solve(
        _WL, N_theta=91, N_phi=8, truncation_order=8,
    )
    area = np.pi * r * r
    assert res["c_ext"] / area == pytest.approx(qe, rel=1e-6)
    assert res["c_abs"] / area == pytest.approx(qa, rel=1e-6, abs=1e-12)
    assert res["asymmetry"] == pytest.approx(g, abs=2e-3)


@pytest.mark.parametrize("n_host", (1.0, 1.33, 1.8476))
def test_large_absorbing_sphere_extinguishes_twice_its_area(n_host):
    """Extinction paradox: physical Q_ext -> 2 for a large absorbing sphere
    in any non-absorbing host; the "mstm" convention would give 2 n_host.
    Checked on Mie (a 100 um sphere is out of reach for the solver), which
    the test above ties to the physical default."""
    qe, _, _ = _mie_q(_N_FE, n_host, 100e-6)
    assert qe == pytest.approx(2.0, rel=0.03)
    if n_host > 1.2:
        assert abs(qe - 2.0) < abs(qe - 2.0 * n_host) / 10


def test_truncation_order_validation():
    fastmm = FaSTMM([[0.0, 0.0, 0.0]], [1.0], refractive_index=1.5 + 0.0j)
    with pytest.raises(ValueError, match="truncation_order"):
        fastmm.solve(2 * np.pi, N_theta=11, N_phi=1, truncation_order=1)
    with pytest.raises(ValueError, match="truncation_formula"):
        fastmm.solve(2 * np.pi, N_theta=11, N_phi=1, truncation_formula=8)


def test_truncation_order_overrides_formula():
    fastmm = FaSTMM([[0.0, 0.0, 0.0]], [1.0], refractive_index=1.5 + 0.01j)
    kw = dict(N_theta=11, N_phi=1)
    a = fastmm.solve(2 * np.pi, truncation_order=12, **kw)
    b = fastmm.solve(2 * np.pi, truncation_formula=1, truncation_order=12, **kw)
    assert a["c_ext"] == b["c_ext"]


def test_medium_correction_is_a_no_op_for_vacuum():
    fastmm_vac = FaSTMM([[0, 0, 0]], [1.0], refractive_index=2.0 + 0.1j)
    fastmm_explicit_vac = FaSTMM(
        [[0, 0, 0]], [1.0], refractive_index=2.0 + 0.1j,
        medium_refractive_index=1.0 + 0.0j,
    )
    r1 = fastmm_vac.solve(2 * np.pi, N_theta=11, N_phi=1)
    r2 = fastmm_explicit_vac.solve(2 * np.pi, N_theta=11, N_phi=1)
    assert r1["c_ext"] == pytest.approx(r2["c_ext"])


def test_solve_returns_k_medium_and_eps():
    fastmm = FaSTMM(
        [[0, 0, 0]], [1.0], refractive_index=2.0 + 0.1j,
        medium_refractive_index=1.3 + 0.0j,
    )
    result = fastmm.solve(5.0, N_theta=11, N_phi=1)
    assert result["k_medium"] == pytest.approx(2 * np.pi / 5.0 * 1.3)
    assert result["eps"] == pytest.approx(((2.0 + 0.1j) / 1.3) ** 2)


def test_single_sphere_orientation_invariant_to_incidence_angle():
    """A single sphere has no preferred axis -- rotating the incident
    direction must not change its cross sections, for any (polar,
    azimuthal) pair, regardless of rotation-order convention. A strong,
    convention-independent sanity check for rotate_for_incidence()."""
    fastmm = FaSTMM([[0.0, 0.0, 0.0]], [1.0], refractive_index=3.0 + 0.1j)
    baseline = fastmm.solve(2 * np.pi, N_theta=11, N_phi=1)
    tilted = fastmm.solve(
        2 * np.pi, incident_polar_deg=30.0, incident_azimuthal_deg=45.0,
        N_theta=11, N_phi=1,
    )
    assert tilted["c_ext"] == pytest.approx(baseline["c_ext"], rel=1e-8)
    assert tilted["c_abs"] == pytest.approx(baseline["c_abs"], rel=1e-8)


def test_compute_tmatrix_shape_and_medium_transform():
    from pyfastmm import get_tmatrix_size

    fastmm = FaSTMM(
        [[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]], [1.0, 1.0],
        refractive_index=2.5 + 0.05j, medium_refractive_index=1.2 + 0.0j,
    )
    result = fastmm.compute_tmatrix(5.0, t_order=4)
    nm = get_tmatrix_size(4)
    assert result["Taa"].shape == (nm, nm)
    assert result["k_medium"] == pytest.approx(2 * np.pi / 5.0 * 1.2)


def test_radiative_transfer_phase_function_normalizes_to_4pi():
    """Integrating the normalized S11 (with a matching quadrature) over
    the sphere must reproduce 4*pi -- the defining property of the
    radiative-transfer convention this function converts to."""
    fastmm = FaSTMM([[0.0, 0.0, 0.0]], [1.0], refractive_index=1.5 + 0.01j)
    n_theta, n_phi = 91, 16
    result = fastmm.solve(2 * np.pi, N_theta=n_theta, N_phi=n_phi)

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
    docstring for why -- and must be a pure performance knob: results
    identical regardless of value."""
    import inspect

    assert inspect.signature(FaSTMM.solve).parameters["n_threads"].default == 1
    assert inspect.signature(FaSTMM.compute_tmatrix).parameters["n_threads"].default == 1

    fastmm = FaSTMM(
        [[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]], [1.0, 1.0],
        refractive_index=2.5 + 0.1j, medium_refractive_index=1.3 + 0.0j,
    )
    results = [
        fastmm.solve(5.0, N_theta=11, N_phi=1, n_threads=n) for n in (1, 2, 4)
    ]
    for r in results[1:]:
        assert r["c_ext"] == pytest.approx(results[0]["c_ext"], rel=1e-10)


# --- version/formulation dispatch -------------------------------------------


def test_default_version_is_2():
    fastmm = FaSTMM([[0.0, 0.0, 0.0]], [1.0], refractive_index=2.0 + 0.1j)
    assert fastmm.version == 2


def test_unsupported_version_raises_on_solve_not_construction():
    """Constructing with an unsupported version must succeed (it's just
    data) -- the error belongs on the call that actually needs to
    dispatch on it, matching how invalid formulation values are handled
    too (checked at call time, not eagerly)."""
    fastmm = FaSTMM(
        [[0.0, 0.0, 0.0]], [1.0], refractive_index=2.0 + 0.1j, version=3,
    )
    assert fastmm.version == 3
    with pytest.raises(ValueError, match="version"):
        fastmm.solve(2 * np.pi, N_theta=11, N_phi=1)
    with pytest.raises(ValueError, match="version"):
        fastmm.compute_tmatrix(2 * np.pi, t_order=4)


def test_unsupported_formulation_raises_for_version_2():
    fastmm = FaSTMM([[0.0, 0.0, 0.0]], [1.0], refractive_index=2.0 + 0.1j)
    with pytest.raises(ValueError, match="formulation"):
        fastmm.solve(2 * np.pi, N_theta=11, N_phi=1, formulation=99)
    with pytest.raises(ValueError, match="formulation"):
        fastmm.compute_tmatrix(2 * np.pi, t_order=4, formulation=99)


def test_all_v2_formulations_are_accepted():
    fastmm = FaSTMM([[0.0, 0.0, 0.0]], [1.0], refractive_index=2.0 + 0.1j)
    for formulation in (0, 1, 2):
        result = fastmm.solve(2 * np.pi, N_theta=11, N_phi=1, formulation=formulation)
        assert result["c_ext"] > 0


def test_absorbing_medium_c_sca_survives_propagator_attenuation():
    """Regression for the far-field attenuation bug:
    compute_cluster_fields_far integrated RR**2 * |E|**2 without dividing
    out the propagator's exp(-2*Im(k)*RR) decay, so an absorbing host
    medium suppressed c_sca by exp(-2e6*kappa/n) -- ~2e-7 for kappa=1e-5
    here, saturating at exp(-600) ~ 1e-260 once the RR cap engages.
    A weakly absorbing host must now give essentially the lossless-host
    c_sca; the tiny residual drift is the genuine physics of the medium."""
    coords = [[0.0, 0.0, 0.0]]
    radii = [1.0]
    n_particle = 1.5 + 0.01j

    def solve_c_sca(n_medium):
        fastmm = FaSTMM(
            coords, radii, refractive_index=n_particle,
            medium_refractive_index=n_medium,
        )
        return fastmm.solve(wavelength_vacuum=5.0, N_theta=19, N_phi=1)["c_sca"]

    lossless = solve_c_sca(1.3 + 0.0j)
    assert lossless > 0.0
    # kappa/n ~ 7.7e-6: below the RR-cap threshold (uncapped branch)
    assert solve_c_sca(1.3 + 1e-5j) == pytest.approx(lossless, rel=0.01)
    # kappa/n ~ 7.7e-4: RR cap engaged (pre-fix: c_sca ~ 1e-260 * true)
    assert solve_c_sca(1.3 + 1e-3j) == pytest.approx(lossless, rel=0.02)


def test_touching_pair_rotational_invariance_deep_rayleigh():
    """Regression for the deep-Rayleigh conditioning bug: the unscaled GMRES system for touching spheres at
    size parameter x << 1 is numerically singular in double precision
    (measured rcond ~ 1e-60 at Nmax=13, x ~ 0.04), so the "converged"
    solution -- and every cross section derived from it -- depended on the
    solver's iteration path: unpolarized c_ext for physically identical
    touching pairs oriented along x vs y differed by 3-12% across the
    wavelength grid. With the Jacobi-balanced solve
    (solver-deep-rayleigh-gmres-scaling.patch) results are
    orientation-independent; measured spread is ~1e-6 relative."""
    n_fe = 3.985 + 9.471j          # a metal (iron, Querry) at 2976nm
    n_med = 1.830993 + 0.001069j   # a weakly absorbing silicate host
    lam = 2.9762e-6
    r = 10e-9
    s = r / np.sqrt(2.0)
    res = {}
    for tag, coords in {
        "x": [[-r, 0.0, 0.0], [r, 0.0, 0.0]],
        "y": [[0.0, -r, 0.0], [0.0, r, 0.0]],
        "diag": [[-s, -s, 0.0], [s, s, 0.0]],
    }.items():
        fastmm = FaSTMM(coords, [r, r], refractive_index=n_fe,
                        medium_refractive_index=n_med)
        res[tag] = fastmm.solve(lam, N_theta=31, N_phi=8, formulation=0,
                                tol=1e-8, truncation_formula=1)["c_ext"]
    assert res["y"] == pytest.approx(res["x"], rel=1e-4)
    assert res["diag"] == pytest.approx(res["x"], rel=1e-4)


def test_c_sca_equals_c_ext_minus_c_abs_in_lossless_medium():
    """Energy-conservation anchor: with the attenuation fix, the
    far-field-integrated c_sca must match the optical-theorem c_ext - c_abs
    for an absorbing particle in a lossless medium (in either convention),
    limited only by angular quadrature (~5e-4 at this grid)."""
    fastmm = FaSTMM(
        [[0.0, 0.0, 0.0]], [1.0], refractive_index=1.5 + 0.01j,
        medium_refractive_index=1.3 + 0.0j,
    )
    result = fastmm.solve(wavelength_vacuum=5.0, N_theta=61, N_phi=8)
    assert result["c_sca"] == pytest.approx(result["c_ext_minus_c_abs"], rel=0.01)
