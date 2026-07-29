"""Tests for pyfastmm._config -- TOML sweep config parsing and execution."""

import os

import numpy as np
import pytest

pydantic = pytest.importorskip("pydantic")

from pyfastmm import FaSTMM2
from pyfastmm._config import (
    IncidentConfig,
    MediumConfig,
    ParticlesConfig,
    SolverConfig,
    SweepConfig,
    WavelengthsConfig,
    load_config,
    run_sweep,
)

_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_SWEEP_TOML = os.path.join(_DATA_DIR, "fractal_sweep.toml")


def test_load_config_shared_toml():
    cfg = load_config(_SWEEP_TOML)
    assert cfg.particles.positions_file == "fractal_N128_Df2.0.dat"
    assert cfg.particles.refractive_index == (1.5, 0.01)
    assert cfg.wavelengths.num == 5
    assert cfg.solver.formulation == 2


def test_particles_load_positions_and_eps():
    p = ParticlesConfig(
        positions_file="fractal_N128_Df2.0.dat", refractive_index=(1.5, 0.01)
    )
    positions = p.load_positions(_DATA_DIR)
    assert positions.shape == (128, 4)
    assert not np.any(np.isnan(positions))

    eps = p.eps()
    assert eps == pytest.approx((1.5 + 0.01j) ** 2)


def test_wavelengths_range_and_values():
    ranged = WavelengthsConfig(start=5.0, stop=8.0, num=4, scale=1.0)
    wl = ranged.get_wavelengths_m()
    assert wl.shape == (4,)
    np.testing.assert_allclose(wl, [5.0, 6.0, 7.0, 8.0])
    np.testing.assert_allclose(ranged.get_wavenumbers(), 2 * np.pi / wl)

    explicit = WavelengthsConfig(values=[1.0, 2.0], scale=2.0)
    np.testing.assert_allclose(explicit.get_wavelengths_m(), [2.0, 4.0])


def test_wavelengths_requires_values_or_range():
    with pytest.raises(pydantic.ValidationError):
        WavelengthsConfig()


def test_medium_applies_refractive_index():
    """A non-trivial medium refractive index is applied to compute k and eps
    (no longer ignored since v1.1)."""
    cfg = SweepConfig(
        particles=ParticlesConfig(
            positions_file="fractal_N128_Df2.0.dat", refractive_index=(1.5, 0.0)
        ),
        wavelengths=WavelengthsConfig(start=5.0, stop=6.0, num=1, scale=1.0),
        medium=MediumConfig(refractive_index=(1.3, 0.0)),
    )
    results = run_sweep(cfg, base_dir=_DATA_DIR)
    assert len(results) == 1
    assert results[0]["c_ext"] > 0


def test_medium_flux_normalization_matches_raw_solve_times_re_n_medium():
    """Regression test for a real cross-section bug: SweepConfig.run()
    used to report FaSTMM2's raw cross sections unchanged, but
    cross_sections()/sphere_absorbtion2() implicitly assume vacuum-like
    incident-flux normalization (Poynting flux ~ |E|^2) -- a real host
    medium needs an extra Re(n_medium), since flux ~ Re(n_medium)*|E|^2.
    Confirmed against MSTM (which does model this) on single sphere,
    touching pair, 4-chain, and a real 32-sphere fractal aggregate in an
    absorbing medium: ~46% off without this factor, <0.03% with it.

    This test doesn't have MSTM available to cross-check against, but
    pins the *mechanism*: SweepConfig.run()'s reported c_ext/c_abs must
    equal a raw FaSTMM2().solve() call (using the same k_medium/eps_eff
    SweepConfig.run() itself computes) times Re(n_medium) -- i.e. the
    correction is applied exactly once, not zero or twice.
    """
    n_particle = 1.5 + 0.01j
    n_medium = 1.3 + 0.0j
    wavelength = 5.0

    cfg = SweepConfig(
        particles=ParticlesConfig(
            positions_file="fractal_N128_Df2.0.dat", refractive_index=(1.5, 0.01)
        ),
        wavelengths=WavelengthsConfig(values=[wavelength], scale=1.0),
        medium=MediumConfig(refractive_index=(1.3, 0.0)),
    )
    results = cfg.run(base_dir=_DATA_DIR)

    positions = cfg.particles.load_positions(_DATA_DIR)
    coords, radii = positions[:, :3], positions[:, 3]
    eps_eff = (n_particle / n_medium) ** 2
    k_medium = 2.0 * np.pi * n_medium / wavelength

    raw = FaSTMM2().solve(
        coords, radii, np.full(len(radii), eps_eff), complex(k_medium)
    )

    assert results[0]["c_ext"] == pytest.approx(raw["c_ext"] * n_medium.real)
    assert results[0]["c_abs"] == pytest.approx(raw["c_abs"] * n_medium.real)
    assert results[0]["c_ext_minus_c_abs"] == pytest.approx(
        raw["c_ext_minus_c_abs"] * n_medium.real
    )


def test_incident_rotation_identity_at_zero_angle():
    inc = IncidentConfig(polar_angle_deg=0.0, azimuthal_angle_deg=0.0)
    positions = np.array([[1.0, 2.0, 3.0, 0.5]])
    np.testing.assert_array_equal(inc.rotate_positions(positions), positions)


def test_incident_rotation_preserves_geometry():
    # Rotation must preserve inter-particle distances (radii untouched).
    inc = IncidentConfig(polar_angle_deg=30.0, azimuthal_angle_deg=45.0)
    positions = np.array(
        [[0.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 0.5], [0.0, 2.0, 1.0, 0.5]]
    )
    rotated = inc.rotate_positions(positions)
    np.testing.assert_allclose(rotated[:, 3], positions[:, 3])  # radii unchanged
    d0 = np.linalg.norm(positions[0, :3] - positions[1, :3])
    d1 = np.linalg.norm(rotated[0, :3] - rotated[1, :3])
    assert d0 == pytest.approx(d1)


def test_incident_rotation_matches_validated_order_for_combined_angles():
    """Regression test for a real bug: with both polar and azimuthal
    nonzero simultaneously, rotation order matters (R = Ry @ Rz, Rz
    applied first) -- this function used to compose them the other way
    (R = Rz @ Ry), which t-bench's cross-tool validation against MSTM
    found disagrees by up to ~1.3% (matches to <0.0002% with the order
    pinned here). A single nonzero angle can't distinguish the two orders
    (trivial composition), so this needs both simultaneously nonzero."""
    theta = np.radians(30.0)
    phi = np.radians(45.0)
    cos_t, sin_t = np.cos(-theta), np.sin(-theta)
    Ry = np.array([[cos_t, 0, sin_t], [0, 1, 0], [-sin_t, 0, cos_t]])
    cos_p, sin_p = np.cos(-phi), np.sin(-phi)
    Rz = np.array([[cos_p, -sin_p, 0], [sin_p, cos_p, 0], [0, 0, 1]])
    expected_R = Ry @ Rz  # the validated order -- NOT Rz @ Ry

    inc = IncidentConfig(polar_angle_deg=30.0, azimuthal_angle_deg=45.0)
    positions = np.array([[1.0, 0.5, -0.3, 1.0]])
    rotated = inc.rotate_positions(positions)
    expected = positions.copy()
    expected[:, :3] = positions[:, :3] @ expected_R.T
    np.testing.assert_allclose(rotated, expected, atol=1e-12)


def test_solver_config_truncation_formula_default_and_override():
    assert SolverConfig().truncation_formula == 0
    assert SolverConfig(truncation_formula=1).truncation_formula == 1


def test_particles_and_medium_share_get_nk_behavior():
    # Both configs inherit get_nk() from the same mixin -- a fixed tuple
    # broadcasts identically regardless of which config class it's on.
    wavelengths = np.array([5.0, 6.0, 7.0])
    p = ParticlesConfig(refractive_index=(2.0, 0.5))
    m = MediumConfig(refractive_index=(2.0, 0.5))
    np.testing.assert_allclose(p.get_nk(wavelengths), m.get_nk(wavelengths))
    assert p.get_nk(wavelengths).shape == wavelengths.shape


def test_sweep_config_run_method_matches_run_sweep_free_function():
    # SweepConfig.run() is now the canonical implementation; run_sweep()
    # is a thin wrapper -- both must produce identical results.
    cfg = SweepConfig(
        particles=ParticlesConfig(refractive_index=(1.5, 0.01)),
        wavelengths=WavelengthsConfig(values=[5.0], scale=1.0),
        solver=dict(N_theta=11, N_phi=1, acc=1),
    )
    import pyfastmm._config as config_mod

    orig_load = config_mod.ParticlesConfig.load_positions
    config_mod.ParticlesConfig.load_positions = lambda *_a, **_kw: np.array(
        [[-1.5, 0.0, 0.0, 1.0], [1.5, 0.0, 0.0, 1.0]]
    )
    try:
        via_method = cfg.run()
        via_function = run_sweep(cfg)
    finally:
        config_mod.ParticlesConfig.load_positions = orig_load

    assert via_method[0]["c_ext"] == pytest.approx(via_function[0]["c_ext"])


def test_run_sweep_small_cluster_two_wavelengths():
    cfg = SweepConfig(
        particles=ParticlesConfig(refractive_index=(1.5, 0.01)),
        wavelengths=WavelengthsConfig(values=[5.0, 6.0], scale=1.0),
        solver=dict(N_theta=11, N_phi=1, acc=1),
    )
    # Bypass the file loader with an inline 2-sphere cluster.
    import pyfastmm._config as config_mod

    orig_load = config_mod.ParticlesConfig.load_positions
    config_mod.ParticlesConfig.load_positions = lambda *_a, **_kw: np.array(
        [[-1.5, 0.0, 0.0, 1.0], [1.5, 0.0, 0.0, 1.0]]
    )
    try:
        results = run_sweep(cfg)
    finally:
        config_mod.ParticlesConfig.load_positions = orig_load

    assert len(results) == 2
    assert results[0]["wavelength_m"] == pytest.approx(5.0)
    assert results[1]["wavelength_m"] == pytest.approx(6.0)
    for r in results:
        assert r["c_ext"] > 0
        assert not np.any(np.isnan(r["mueller"]))
