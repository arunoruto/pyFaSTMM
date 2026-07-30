#!/usr/bin/env python3
"""Benchmark harness for pyfastmm -- every performance claim gets a number.

Deterministic inputs: the first N spheres of the checked-in
tests/data/fractal_N128_Df2.0.dat aggregate (PyFracVAL lists monomers in
aggregation order, so a prefix is a connected sub-cluster), fixed eps/k.
Two axes, matching the two threading knobs:

- ``solve``: one solve, scaling over ``n_threads`` (OpenMP inside the
  solve -- expect the measured fall-off past physical-core saturation).
- ``sweep``: an 8-wavelength sweep, scaling over ``sweep_threads``
  (concurrent GIL-released solves -- expect near-linear until
  memory-bandwidth-bound).

Usage:
    uv run python benchmarks/bench_solve.py                # both suites
    uv run python benchmarks/bench_solve.py --suite solve --sizes 32 128
    uv run python benchmarks/bench_solve.py --json out.json --repeat 3

Timing convention: best-of-``--repeat`` (minimum wall time -- the least
noise-contaminated estimate of the true cost; means mix in scheduler
jitter). Not a pytest file on purpose: benchmarks are run deliberately,
not on every test invocation.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from pyfastmm import FaSTMM2

_PROJ_ROOT = Path(__file__).resolve().parent.parent
_CLUSTER_FILE = _PROJ_ROOT / "tests" / "data" / "fractal_N128_Df2.0.dat"

# Fixed physics, same regime as the test suite: moderate refractive
# contrast, slightly absorbing, ka ~ 1 per monomer.
_EPS = (1.5 + 0.01j) ** 2
_K = 1.2
_N_THETA, _N_PHI = 31, 8


def load_cluster(n: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(_CLUSTER_FILE)
    if n > len(data):
        raise SystemExit(f"cluster file has {len(data)} spheres, asked for {n}")
    return data[:n, :3], data[:n, 3]


def _best_of(fn, repeat: int) -> float:
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def bench_solve(sizes, threads, repeat):
    """One solve, scaling over n_threads (OpenMP inside the solve)."""
    f = FaSTMM2()
    rows = []
    for n in sizes:
        coords, radii = load_cluster(n)
        eps = np.full(n, _EPS)
        base = None
        for nt in threads:
            dt = _best_of(
                lambda: f.solve(
                    coords, radii, eps, _K,
                    N_theta=_N_THETA, N_phi=_N_PHI, acc=1, n_threads=nt,
                ),
                repeat,
            )
            base = base or dt
            rows.append({"N": n, "n_threads": nt, "s": dt, "speedup": base / dt})
            print(
                f"solve  N={n:4d}  n_threads={nt:2d}  {dt:8.3f} s"
                f"  ({base / dt:4.1f}x)"
            )
    return rows


def bench_sweep(sizes, threads, repeat, n_wavelengths=8):
    """An n_wavelengths sweep, scaling over sweep_threads (concurrent
    GIL-released solves; n_threads pinned to 1 inside each)."""
    from pyfastmm._config import (
        OutputConfig,
        ResolvedSweepInputs,
        SolverConfig,
        SweepConfig,
    )

    rows = []
    for n in sizes:
        coords, radii = load_cluster(n)
        base = None
        for st in threads:
            resolved = ResolvedSweepInputs(
                coords=coords,
                radii=radii,
                wavelengths_m=np.linspace(5.0, 8.0, n_wavelengths),
                nk_particle=np.full(n_wavelengths, 1.5 + 0.01j),
                nk_medium=np.full(n_wavelengths, 1.0 + 0.0j),
                solver=SolverConfig(
                    N_theta=_N_THETA, N_phi=_N_PHI, acc=1, sweep_threads=st
                ),
                output=OutputConfig(),
            )
            dt = _best_of(lambda: SweepConfig.run_resolved(resolved), repeat)
            base = base or dt
            rows.append({"N": n, "sweep_threads": st, "s": dt, "speedup": base / dt})
            print(
                f"sweep  N={n:4d}  sweep_threads={st:2d}  {dt:8.3f} s"
                f"  ({n_wavelengths} wl, {base / dt:4.1f}x)"
            )
    return rows


def _metadata():
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_PROJ_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        rev = "unknown"
    return {
        "git_rev": rev,
        "cpu_count": os.cpu_count(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Benchmark harness for pyfastmm solve/sweep scaling."
    )
    ap.add_argument("--suite", choices=["solve", "sweep", "all"], default="all")
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 32, 128])
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--json", type=Path, help="also dump results as JSON")
    args = ap.parse_args()

    meta = _metadata()
    results: dict[str, object] = {"meta": meta}
    print(f"pyfastmm bench  rev={meta['git_rev']}  cpus={meta['cpu_count']}")
    if args.suite in ("solve", "all"):
        results["solve"] = bench_solve(args.sizes, args.threads, args.repeat)
    if args.suite in ("sweep", "all"):
        # Sweep scaling is only interesting on a non-trivial cluster.
        sweep_sizes = [s for s in args.sizes if s >= 32] or args.sizes[-1:]
        results["sweep"] = bench_sweep(sweep_sizes, args.threads, args.repeat)

    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
