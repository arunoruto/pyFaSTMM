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
    ResolvedSweepInputs,
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


def test_host_medium_cross_sections_are_physical_by_default():
    """SweepConfig.run() reports physical cross sections by default: exactly
    a raw FaSTMM2().solve() with the same k_medium/eps_eff, no Re(n_medium)
    factor (that is the opt-in "mstm" convention, see the next test)."""
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

    for key in ("c_ext", "c_abs", "c_ext_minus_c_abs"):
        assert results[0][key] == pytest.approx(raw[key])


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
    (R = Rz @ Ry), which cross-tool validation against MSTM
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
    assert SolverConfig().truncation_order is None
    assert SolverConfig(truncation_order=10).truncation_order == 10
    with pytest.raises(ValueError):
        SolverConfig(truncation_formula=8)  # explicit orders go in truncation_order
    with pytest.raises(ValueError):
        SolverConfig(truncation_order=1)


def test_solver_config_cross_section_convention():
    assert SolverConfig().cross_section_convention == "physical"
    assert SolverConfig(cross_section_convention="mstm").cross_section_convention == "mstm"
    with pytest.raises(ValueError):
        SolverConfig(cross_section_convention="vacuum")


def test_solver_config_n_threads_default_and_override():
    assert SolverConfig().n_threads == 1
    assert SolverConfig(n_threads=8).n_threads == 8


def test_solver_config_n_threads_accepts_negative_and_float():
    """SolverConfig.n_threads must accept the same negative-offset and
    float-fraction shapes resolve_n_threads() supports -- pydantic must not
    reject or silently coerce them (e.g. a float fraction truncated to an
    int would break the "fraction of cpu_count" meaning)."""
    assert SolverConfig(n_threads=-2).n_threads == -2
    assert SolverConfig(n_threads=0.5).n_threads == pytest.approx(0.5)


def test_sweep_config_run_passes_n_threads_through(monkeypatch):
    """SolverConfig.n_threads must reach FaSTMM.solve(), not silently
    fall back to FaSTMM's own default -- pin the actual value threaded
    through, not just that a solve happens to succeed."""
    import pyfastmm._config as config_mod

    seen = {}
    orig_solve = config_mod.FaSTMM.solve

    def spy_solve(self, *args, **kwargs):
        seen.update(kwargs)
        return orig_solve(self, *args, **kwargs)

    monkeypatch.setattr(config_mod.FaSTMM, "solve", spy_solve)

    cfg = SweepConfig(
        particles=ParticlesConfig(
            positions_file="fractal_N128_Df2.0.dat", refractive_index=(1.5, 0.0)
        ),
        wavelengths=WavelengthsConfig(start=5.0, stop=6.0, num=1, scale=1.0),
        solver=SolverConfig(
            n_threads=3, truncation_order=7, cross_section_convention="mstm",
            N_theta=11, N_phi=1,
        ),
    )
    cfg.run(base_dir=_DATA_DIR)
    assert seen["n_threads"] == 3
    assert seen["truncation_order"] == 7
    assert seen["cross_section_convention"] == "mstm"


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


def test_run_matches_resolve_then_run_resolved():
    """run() is now just run_resolved(resolve_inputs()) -- pin that the
    split didn't change behavior. This is the seam a distributed worker
    (e.g. a Dask task) is meant to use: resolve_inputs() on a machine with
    file access, run_resolved() anywhere, no filesystem needed there."""
    cfg = SweepConfig(
        particles=ParticlesConfig(refractive_index=(1.5, 0.01)),
        wavelengths=WavelengthsConfig(values=[5.0, 6.0], scale=1.0),
        solver=dict(N_theta=11, N_phi=1, acc=1),
    )
    import pyfastmm._config as config_mod

    orig_load = config_mod.ParticlesConfig.load_positions
    config_mod.ParticlesConfig.load_positions = lambda *_a, **_kw: np.array(
        [[-1.5, 0.0, 0.0, 1.0], [1.5, 0.0, 0.0, 1.0]]
    )
    try:
        via_run = cfg.run()
        resolved = cfg.resolve_inputs()
        via_split = cfg.run_resolved(resolved)
    finally:
        config_mod.ParticlesConfig.load_positions = orig_load

    assert isinstance(resolved, ResolvedSweepInputs)
    assert len(via_run) == len(via_split) == 2
    for a, b in zip(via_run, via_split):
        assert a["c_ext"] == pytest.approx(b["c_ext"])
        assert a["wavelength_m"] == pytest.approx(b["wavelength_m"])
        np.testing.assert_allclose(a["mueller"], b["mueller"])

    # The actual point of making run_resolved a staticmethod: callable via
    # the class with no SweepConfig instance at all -- what a Dask worker
    # does, since it only ever receives `resolved`, never a SweepConfig
    # (which would still carry particles.positions_file, a local file path
    # not meaningful on a remote machine).
    via_classmethod_call = SweepConfig.run_resolved(resolved)
    assert len(via_classmethod_call) == 2
    assert via_classmethod_call[0]["c_ext"] == pytest.approx(via_run[0]["c_ext"])


def _inline_two_sphere_cfg(n_wavelengths=4, **solver_kwargs):
    """A small config + monkeyed loader shared by the sweep_threads tests:
    real solves (cheap: 2 spheres, N_theta=11, N_phi=1), several
    wavelengths so concurrency actually happens."""
    return SweepConfig(
        particles=ParticlesConfig(refractive_index=(1.5, 0.01)),
        wavelengths=WavelengthsConfig(
            values=[5.0 + i for i in range(n_wavelengths)], scale=1.0
        ),
        solver=SolverConfig(N_theta=11, N_phi=1, acc=1, **solver_kwargs),
    )


def _resolve_with_inline_cluster(cfg):
    import pyfastmm._config as config_mod

    orig_load = config_mod.ParticlesConfig.load_positions
    config_mod.ParticlesConfig.load_positions = lambda *_a, **_kw: np.array(
        [[-1.5, 0.0, 0.0, 1.0], [1.5, 0.0, 0.0, 1.0]]
    )
    try:
        return cfg.resolve_inputs()
    finally:
        config_mod.ParticlesConfig.load_positions = orig_load


def test_solver_config_sweep_threads_default_and_shapes():
    """sweep_threads defaults to 1 (serial -- pre-existing behavior, and
    the safe value under a caller's own process-level parallelism) and
    must accept the same negative-offset/float-fraction shapes as
    n_threads without pydantic coercing them."""
    assert SolverConfig().sweep_threads == 1
    assert SolverConfig(sweep_threads=4).sweep_threads == 4
    assert SolverConfig(sweep_threads=-1).sweep_threads == -1
    assert SolverConfig(sweep_threads=0.5).sweep_threads == pytest.approx(0.5)


def test_sweep_threads_concurrent_matches_serial():
    """The whole point of sweep_threads: concurrent wavelengths must
    produce bit-comparable physics to the serial loop, in the same
    (wavelength) order regardless of completion order. A regression here
    means the thread pool scrambled results or the concurrent solves
    contaminated each other (cf. tests/test_concurrency.py, which pins
    the same property one level down at the FaSTMM2.solve() layer)."""
    resolved_serial = _resolve_with_inline_cluster(_inline_two_sphere_cfg())
    resolved_threaded = _resolve_with_inline_cluster(
        _inline_two_sphere_cfg(sweep_threads=4)
    )

    serial = SweepConfig.run_resolved(resolved_serial)
    threaded = SweepConfig.run_resolved(resolved_threaded)

    assert len(serial) == len(threaded) == 4
    for a, b in zip(serial, threaded):
        assert a["wavelength_m"] == pytest.approx(b["wavelength_m"])
        assert a["c_ext"] == pytest.approx(b["c_ext"])
        assert a["c_abs"] == pytest.approx(b["c_abs"])
        np.testing.assert_allclose(a["mueller"], b["mueller"])


def test_sweep_threads_results_stay_in_wavelength_order():
    resolved = _resolve_with_inline_cluster(_inline_two_sphere_cfg(sweep_threads=4))
    results = SweepConfig.run_resolved(resolved)
    wavelengths = [r["wavelength_m"] for r in results]
    assert wavelengths == sorted(wavelengths)
    np.testing.assert_allclose(wavelengths, [5.0, 6.0, 7.0, 8.0])


def test_sweep_threads_progress_callback_runs_on_calling_thread():
    """run_resolved()'s docstring promises progress_callback is only ever
    invoked from the calling thread (so UI-driving callbacks need no
    locking), with monotonically increasing fractions ending at 1.0."""
    import threading

    resolved = _resolve_with_inline_cluster(_inline_two_sphere_cfg(sweep_threads=4))
    caller = threading.get_ident()
    seen = []

    def cb(fraction, wavelength_m):
        assert threading.get_ident() == caller
        seen.append((fraction, wavelength_m))

    SweepConfig.run_resolved(resolved, progress_callback=cb)
    fractions = [f for f, _ in seen]
    assert fractions == sorted(fractions)
    assert fractions[-1] == pytest.approx(1.0)
    assert len(seen) == 4


def test_sweep_threads_oversubscription_warns():
    """sweep_threads * n_threads beyond cpu_count must warn (not raise,
    not pass silently) -- silent oversubscription is the exact failure
    mode both defaults being 1 exists to avoid."""
    cpu_count = os.cpu_count() or 1
    resolved = _resolve_with_inline_cluster(
        # 2 concurrent wavelengths, each claiming every CPU -> guaranteed
        # over budget on any machine.
        _inline_two_sphere_cfg(n_wavelengths=2, sweep_threads=2, n_threads=cpu_count)
    )
    with pytest.warns(UserWarning, match="oversubscribe"):
        SweepConfig.run_resolved(resolved)


def test_sweep_threads_propagates_solve_exceptions():
    """A failing solve must raise out of run_resolved(), not be swallowed
    by the pool."""
    import pyfastmm._config as config_mod

    resolved = _resolve_with_inline_cluster(_inline_two_sphere_cfg(sweep_threads=4))

    def boom(self, *args, **kwargs):
        raise RuntimeError("solver exploded")

    orig_solve = config_mod.FaSTMM.solve
    config_mod.FaSTMM.solve = boom
    try:
        with pytest.raises(RuntimeError, match="solver exploded"):
            SweepConfig.run_resolved(resolved)
    finally:
        config_mod.FaSTMM.solve = orig_solve


def test_resolved_sweep_inputs_pickles_roundtrip():
    """ResolvedSweepInputs must survive being pickled and reconstructed --
    the actual mechanism a Dask (or ProcessPoolExecutor) worker uses to
    receive it, since it's the thing shipped across that boundary instead
    of a file path."""
    import pickle

    cfg = SweepConfig(
        particles=ParticlesConfig(refractive_index=(1.5, 0.01)),
        wavelengths=WavelengthsConfig(values=[5.0, 6.0], scale=1.0),
        solver=dict(N_theta=11, N_phi=1, acc=1),
    )
    import pyfastmm._config as config_mod

    orig_load = config_mod.ParticlesConfig.load_positions
    config_mod.ParticlesConfig.load_positions = lambda *_a, **_kw: np.array(
        [[-1.5, 0.0, 0.0, 1.0], [1.5, 0.0, 0.0, 1.0]]
    )
    try:
        resolved = cfg.resolve_inputs()
    finally:
        config_mod.ParticlesConfig.load_positions = orig_load

    roundtripped = pickle.loads(pickle.dumps(resolved))
    np.testing.assert_array_equal(roundtripped.coords, resolved.coords)
    np.testing.assert_array_equal(roundtripped.radii, resolved.radii)
    np.testing.assert_array_equal(roundtripped.nk_particle, resolved.nk_particle)
    assert roundtripped.solver == resolved.solver

    # And the roundtripped object must actually be usable, not just equal.
    results = cfg.run_resolved(roundtripped)
    assert len(results) == 2
    assert results[0]["c_ext"] > 0
