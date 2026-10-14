"""Tests for pyfastmm._dask -- Dask-distributed sweep execution.

Skipped wholesale when `distributed` isn't installed: CI installs only the
`dev` extra (see pyproject.toml's comment on the `dask` extra), so these
run locally, where devenv's allExtras sync installs it. Two tiers:

- In-process ``LocalCluster`` tests: run anywhere the extra is installed,
  no external services -- threaded workers (``processes=False``), which
  still exercise real concurrency since the Fortran entry points release
  the GIL.
- External-scheduler tests: connect to the scheduler `devenv up` starts
  (see devenv.nix's ``processes``; override the address via
  ``PYFASTMM_DASK_SCHEDULER``) and *additionally* skip when nothing is
  listening there. These exercise the real out-of-process path --
  pickling of ResolvedSweepInputs and results across a socket, workers
  importing pyfastmm from the shared venv.
"""

import os
import threading

import numpy as np
import pytest

distributed = pytest.importorskip("distributed")
pytest.importorskip("pydantic")

from pyfastmm._config import (
    OutputConfig,
    ResolvedSweepInputs,
    SolverConfig,
    SweepConfig,
)
from pyfastmm._dask import run_resolved_dask

_SCHEDULER = os.environ.get("PYFASTMM_DASK_SCHEDULER", "tcp://127.0.0.1:8786")


def _resolved(n_wl: int = 3) -> ResolvedSweepInputs:
    """A small two-sphere sweep, built directly (no files needed -- that's
    the point of ResolvedSweepInputs)."""
    return ResolvedSweepInputs(
        coords=np.array([[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]]),
        radii=np.array([1.0, 1.0]),
        wavelengths_m=np.linspace(5.0, 7.0, n_wl),
        nk_particle=np.full(n_wl, 1.5 + 0.01j),
        nk_medium=np.full(n_wl, 1.0 + 0.0j),
        solver=SolverConfig(N_theta=11, N_phi=1, acc=1),
        output=OutputConfig(),
    )


@pytest.fixture(scope="module")
def local_client():
    with distributed.LocalCluster(
        n_workers=2,
        threads_per_worker=1,
        processes=False,  # threaded workers -- see module docstring
        dashboard_address=None,
    ) as cluster:
        with distributed.Client(cluster) as client:
            yield client


def test_run_resolved_dask_matches_serial(local_client):
    """The one property everything else rests on: the distributed sweep is
    the same physics as the serial one, in the same (wavelength) order."""
    serial = SweepConfig.run_resolved(_resolved())
    via_dask = run_resolved_dask(_resolved(), client=local_client)

    assert len(serial) == len(via_dask) == 3
    for a, b in zip(serial, via_dask):
        assert a["wavelength_m"] == pytest.approx(b["wavelength_m"])
        assert a["c_ext"] == pytest.approx(b["c_ext"])
        assert a["c_abs"] == pytest.approx(b["c_abs"])
        np.testing.assert_allclose(a["mueller"], b["mueller"])


def test_run_resolved_dask_caller_client_left_open(local_client):
    """A caller-provided client is never closed by run_resolved_dask --
    the caller owns it (the fixture reuses it across tests, so a close
    here would also break the other tests in this module)."""
    run_resolved_dask(_resolved(n_wl=1), client=local_client)
    assert local_client.status == "running"


def test_run_resolved_dask_progress_callback_contract(local_client):
    """Same contract as run_resolved(): called per completed solve with
    monotone fractions ending at 1.0, always from the calling thread."""
    caller = threading.get_ident()
    seen = []

    def cb(fraction, wavelength_m):
        assert threading.get_ident() == caller
        seen.append((fraction, wavelength_m))

    run_resolved_dask(_resolved(), progress_callback=cb, client=local_client)
    fractions = [f for f, _ in seen]
    assert fractions == sorted(fractions)
    assert fractions[-1] == pytest.approx(1.0)
    assert len(seen) == 3


def test_run_resolved_dask_empty_sweep_is_empty(local_client):
    assert run_resolved_dask(_resolved(n_wl=0), client=local_client) == []


def test_run_resolved_dask_propagates_solve_exceptions(local_client):
    """A failing solve must raise out of run_resolved_dask, not hang or
    return a partial result list. Provoked with real bad input rather
    than monkeypatching, since the failure happens on a worker where a
    monkeypatched module wouldn't exist. Deliberately a *Python-side*
    validation error (radii length mismatch -> _prepare_geometry raises
    ValueError before any Fortran runs): unvalidated garbage reaching
    the Fortran core (e.g. a negative radius) can hang it, which is a
    different, known limitation -- not what this test pins."""
    bad = _resolved()
    bad.radii = np.array([1.0])  # 2 spheres, 1 radius
    with pytest.raises(ValueError):
        run_resolved_dask(bad, client=local_client)


def _external_client():
    try:
        return distributed.Client(_SCHEDULER, timeout="2s")
    except OSError:
        pytest.skip(
            f"no Dask scheduler listening at {_SCHEDULER} -- start the local "
            "one with `devenv up` (see devenv.nix's processes)"
        )


def test_external_scheduler_run_dask_matches_serial():
    """The real out-of-process path: ResolvedSweepInputs and results
    pickle across a socket, workers import pyfastmm from the shared venv
    (see devenv.nix). Locally gated: skips unless `devenv up` (or an
    explicitly pointed-at scheduler) is running."""
    client = _external_client()
    try:
        serial = SweepConfig.run_resolved(_resolved())
        via_dask = run_resolved_dask(_resolved(), client=client)
    finally:
        client.close()

    for a, b in zip(serial, via_dask):
        assert a["c_ext"] == pytest.approx(b["c_ext"])
        np.testing.assert_allclose(a["mueller"], b["mueller"])


def test_external_scheduler_address_path_closes_its_own_client():
    """The scheduler_address= path (run_resolved_dask creates and owns the
    client) works end to end -- also covers SweepConfig.run_dask's usual
    calling convention without needing config files."""
    _external_client().close()  # only to check/skip on availability
    results = run_resolved_dask(_resolved(n_wl=2), scheduler_address=_SCHEDULER)
    assert len(results) == 2
    assert results[0]["c_ext"] > 0
