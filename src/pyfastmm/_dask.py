"""Dask-distributed sweep execution -- first-class, but strictly optional.

Requires the ``dask`` extra (``pip install pyfastmm[dask]``); nothing in
here is imported by ``pyfastmm``'s core solve path, and ``distributed``
itself is only imported inside the functions below, so importing this
module (or ``pyfastmm`` itself) works fine without it installed.

Builds directly on the seam ``SweepConfig`` already exposes for exactly
this purpose: :meth:`~pyfastmm._config.SweepConfig.resolve_inputs` does
all file I/O on the machine that has the files, producing a small,
picklable :class:`~pyfastmm._config.ResolvedSweepInputs`; this module
scatters that object to the workers **once** and submits one task per
wavelength against it (each task is a one-wavelength
``SweepConfig.run_resolved()`` call -- the exact same code path as a
local sweep, so there is no separate "distributed" physics to keep in
sync). Granularity note: per-wavelength tasks are the right size here --
a real solve runs seconds to minutes, dwarfing the ~ms task overhead,
and finer tasks would need Fortran-level splitting that doesn't exist.

Workers must be able to ``import pyfastmm`` (compiled extension included)
-- on a remote cluster that means shipping a wheel built for the
*worker's* platform/ABI (see ``[tool.cibuildwheel]`` and, for a worked
example, spcwth's ``dask_setup.py``, which installs the wheel on every
worker before submitting). The local ``devenv up`` scheduler/workers
(see devenv.nix's ``processes``) share this project's venv, so they can
already import it.

Threading interplay: each Dask task solves one wavelength, so
``solver.sweep_threads`` (the *within-process* concurrent-wavelength
knob) is inert here -- the cluster provides that axis instead. Keep
``solver.n_threads`` (OpenMP inside one solve) multiplied by the
worker's task slots within the worker's CPU budget, same rule as ever.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from pyfastmm._config import ResolvedSweepInputs


def _require_distributed():
    try:
        import distributed
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "pyfastmm's Dask support requires the 'distributed' package -- "
            "install the extra: pip install pyfastmm[dask] (or uv sync "
            "--extra dask)"
        ) from exc
    return distributed


def _slice_resolved(resolved: ResolvedSweepInputs, i: int) -> ResolvedSweepInputs:
    """A one-wavelength view of *resolved* (arrays sliced, geometry and
    configs shared) -- what a single Dask task actually solves."""
    return replace(
        resolved,
        wavelengths_m=resolved.wavelengths_m[i : i + 1],
        nk_particle=resolved.nk_particle[i : i + 1],
        nk_medium=resolved.nk_medium[i : i + 1],
    )


def _solve_index(resolved: ResolvedSweepInputs, i: int) -> dict:
    """Worker-side task: solve wavelength *i* of *resolved*. Module-level
    (not a closure) so it pickles by reference, and implemented as a
    one-wavelength ``run_resolved()`` so distributed and local sweeps
    share one code path."""
    from pyfastmm._config import SweepConfig

    return SweepConfig.run_resolved(_slice_resolved(resolved, i))[0]


def run_resolved_dask(
    resolved: ResolvedSweepInputs,
    progress_callback: Any = None,
    *,
    client: Any = None,
    scheduler_address: str | None = None,
) -> list[dict]:
    """Distributed equivalent of ``SweepConfig.run_resolved()``: one Dask
    task per wavelength, results returned in wavelength order.

    Exactly one way to reach a cluster, checked in this order:

    - *client*: an already-connected ``distributed.Client``. Used as-is
      and **never closed** by this function -- the caller owns it.
    - *scheduler_address*: e.g. ``"tcp://127.0.0.1:8786"`` (what
      ``devenv up``'s local scheduler/worker processes listen on). A
      Client is created and closed here.
    - neither: ``distributed.Client()``'s own default -- spins up a
      temporary in-process ``LocalCluster``, closed here afterwards.
      Handy for testing; for real local parallelism prefer
      ``solver.sweep_threads`` (no cluster round-trips at all).

    ``progress_callback`` keeps ``run_resolved()``'s exact contract:
    invoked as ``(fraction_done, wavelength_m)`` per *completed* solve
    (completion order, not sweep order), always from this calling thread.

    A failed solve raises out of this function immediately; a client
    created here is closed on the way out either way (``finally``).

    Returns
    -------
    list of dicts, identical shape/keys to ``run_resolved()``'s.
    """
    distributed = _require_distributed()

    n = len(resolved.wavelengths_m)
    if n == 0:
        return []

    owns_client = client is None
    if client is None:
        client = (
            distributed.Client(scheduler_address)
            if scheduler_address is not None
            else distributed.Client()
        )
    try:
        # One scatter, broadcast to every worker -- resolved is small (a
        # few short arrays + two pydantic configs), and every task needs
        # it; scattering once beats embedding it in all n task graphs.
        resolved_scattered = client.scatter(resolved, broadcast=True)
        futures = {
            client.submit(_solve_index, resolved_scattered, i): i for i in range(n)
        }
        slots: list[dict | None] = [None] * n
        done = 0
        for future in distributed.as_completed(futures):
            i = futures[future]
            slots[i] = future.result()
            done += 1
            if progress_callback is not None:
                progress_callback(done / n, float(resolved.wavelengths_m[i]))
        return cast("list[dict]", slots)  # every slot filled above
    finally:
        if owns_client:
            client.close()
