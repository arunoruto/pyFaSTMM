"""Concurrency regression tests for the GIL-released FaSTMM2 solve path.

Real threading.Thread concurrency (not multiprocessing, not Dask) --
directly exercises the !f2py threadsafe GIL release plus the removal of
the two process-global Fortran state hazards (truncation_formula, the
OpenMP thread count) that made concurrent calls unsafe before this
refactor. A regression here means two concurrent solve() calls silently
produce wrong physics for at least one of them -- not a crash -- so every
case compares its concurrent result against the same call run serially,
alone.
"""

import threading

import numpy as np
import pytest

from pyfastmm import FaSTMM2


def _case(seed, truncation_formula, n_threads):
    # Deliberately small-ka, touching spheres: this is exactly where
    # truncation_order vs truncation_order2 (the two truncation_formula
    # branches) diverge most -- if truncation_formula were still
    # module-global, concurrent threads racing to overwrite it would show
    # up here as a wrong Nmax for at least one case.
    rng = np.random.default_rng(seed)
    n = 3
    radii = rng.uniform(0.4, 0.7, size=n)
    # Chain the spheres so consecutive ones touch (small ka, strong
    # near-field coupling) instead of scattering them randomly.
    coords = np.zeros((n, 3))
    for i in range(1, n):
        coords[i] = coords[i - 1]
        coords[i, 0] += radii[i - 1] + radii[i]
    eps = rng.uniform(2.5, 4.0, size=n) + 1j * rng.uniform(0.0, 0.2, size=n)
    return dict(
        coords=coords,
        radii=radii,
        eps=eps,
        k=1.0,
        N_theta=15,
        N_phi=1,
        tol=1e-6,
        truncation_formula=truncation_formula,
        n_threads=n_threads,
    )


CASES = [
    _case(0, truncation_formula=0, n_threads=1),
    _case(1, truncation_formula=1, n_threads=2),
    _case(2, truncation_formula=0, n_threads=4),
    _case(3, truncation_formula=1, n_threads=1),
]


def test_concurrent_solves_match_serial_no_cross_contamination():
    serial = [FaSTMM2().solve(**case) for case in CASES]

    concurrent = [None] * len(CASES)

    def run(i, case):
        concurrent[i] = FaSTMM2().solve(**case)

    threads = [threading.Thread(target=run, args=(i, c)) for i, c in enumerate(CASES)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    for i, (s, c) in enumerate(zip(serial, concurrent)):
        assert c is not None, f"case {i}: thread never completed"
        assert c["c_ext"] == pytest.approx(s["c_ext"], rel=1e-10), (
            f"case {i}: concurrent c_ext diverged from its own serial baseline -- "
            "possible cross-contamination of process-global Fortran state "
            "(truncation_formula and/or OpenMP thread count) between "
            "concurrently-running threads"
        )
        assert c["c_abs"] == pytest.approx(s["c_abs"], rel=1e-10), (
            f"case {i}: concurrent c_abs diverged from its own serial baseline"
        )
        np.testing.assert_allclose(
            c["mueller"],
            s["mueller"],
            rtol=1e-8,
            atol=1e-10,
            err_msg=f"case {i}: concurrent mueller diverged from its own serial baseline",
        )

    # A flaky failure above (not every run) is itself diagnostic of a real
    # race, not noise worth retrying past -- thread scheduling is
    # nondeterministic, so a reintroduced module-global-state bug wouldn't
    # necessarily reproduce on every single run of this test.


def test_concurrent_solves_use_different_truncation_formula_correctly():
    """Narrower than the test above: two threads, same geometry, only
    truncation_formula differs -- if it were still module-global, the two
    results could swap or converge to the same (wrong-for-one-of-them)
    value. truncation_order2 (formula 1) gives a strictly larger Nmax than
    truncation_order (formula 0) for small ka, so a real, uncontaminated
    difference between the two results is itself part of the assertion,
    not just an equality check against a serial baseline.
    """
    base = _case(4, truncation_formula=0, n_threads=1)
    kwargs_a = {**base, "truncation_formula": 0}
    kwargs_b = {**base, "truncation_formula": 1}

    result_a = FaSTMM2().solve(**{**kwargs_a, "n_threads": 1})
    result_b = FaSTMM2().solve(**{**kwargs_b, "n_threads": 1})

    results = {}

    def run(key, kwargs):
        results[key] = FaSTMM2().solve(**kwargs)

    threads = [
        threading.Thread(target=run, args=("a", kwargs_a)),
        threading.Thread(target=run, args=("b", kwargs_b)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert results["a"]["c_ext"] == pytest.approx(result_a["c_ext"], rel=1e-10)
    assert results["b"]["c_ext"] == pytest.approx(result_b["c_ext"], rel=1e-10)
    # The two formulas must actually disagree for this geometry -- otherwise
    # this test can't tell a real race apart from "they happen to match".
    assert result_a["c_ext"] != pytest.approx(result_b["c_ext"], rel=1e-6)
