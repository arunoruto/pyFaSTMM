"""Basic sanity tests for FaSTMM2.compute_tmatrix(), no CLI binary required."""

import numpy as np

from pyfastmm import FaSTMM2, get_tmatrix_size


def test_compute_tmatrix_shape_and_finite():
    # Two-sphere cluster, not a single symmetric sphere: for a spherically
    # symmetric single scatterer Tab/Tba are identically zero (no TE/TM
    # coupling), which would make an "any nonzero" check meaningless.
    f = FaSTMM2()
    t_order = 6
    result = f.compute_tmatrix(
        coords=[[-1.2, 0.0, 0.0], [1.2, 0.0, 0.0]],
        radii=[1.0, 0.8],
        eps=[3.0 + 0.1j, 2.5 + 0.05j],
        k=1.0,
        t_order=t_order,
    )

    nm = get_tmatrix_size(t_order)
    assert nm == (t_order + 1) ** 2 - 1
    for key in ("Taa", "Tab", "Tba", "Tbb"):
        m = result[key]
        assert m.shape == (nm, nm)
        assert np.iscomplexobj(m)
        assert not np.any(np.isnan(m))
        assert np.any(m != 0)


def test_get_tmatrix_size():
    assert get_tmatrix_size(0) == 0
    assert get_tmatrix_size(1) == 3
    assert get_tmatrix_size(2) == 8
    assert get_tmatrix_size(16) == 288
