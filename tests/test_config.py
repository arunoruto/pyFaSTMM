"""Tests for pyfastmm._config -- TOML sweep config parsing and execution."""

import os

import numpy as np
import pytest

pydantic = pytest.importorskip("pydantic")

from pyfastmm._config import (
    IncidentConfig,
    MediumConfig,
    ParticlesConfig,
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


def test_medium_ignored_warns():
    cfg = SweepConfig(
        particles=ParticlesConfig(
            positions_file="fractal_N128_Df2.0.dat", refractive_index=(1.5, 0.0)
        ),
        wavelengths=WavelengthsConfig(start=5.0, stop=6.0, num=1, scale=1.0),
        medium=MediumConfig(refractive_index=(1.3, 0.0)),
    )
    with pytest.warns(UserWarning, match="no background-medium model"):
        run_sweep(cfg, base_dir=_DATA_DIR)


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
