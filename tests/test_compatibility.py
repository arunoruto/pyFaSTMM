"""End-to-end compatibility tests: pyfastmm vs the standalone FaSTMM2 CLI.

For each test case we write a geometry.h5 file, run the CLI binary (built
via `make cli` from the unmodified upstream submodule source, including
io.f90/HDF5 -- unlike the Python extension, which never links HDF5, see
meson.build), read its HDF5 output, run the same configuration through
pyfastmm directly (no files), and compare.

Note on HDF5 array layout: h5py writes/reads numpy's native row-major
order, but FaSTMM2's Fortran HDF5 I/O reads/writes column-major -- so a
dataset an h5py script writes with shape (n, 3) is read by Fortran as
coord(3, n), and a Fortran array Taa(nm, nm) written to a dataset is read
back by h5py with values transposed relative to a naive same-shape
comparison. Both are handled explicitly below (see upstream's own
example.py for the same convention).
"""

import os
import subprocess

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from pyfastmm import FaSTMM2, get_tmatrix_size

_PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_CLI_BIN = os.path.join(_PROJ_ROOT, "build", "cli", "FaSTMM2")

pytestmark = pytest.mark.skipif(
    not os.path.isfile(_CLI_BIN),
    reason=f"FaSTMM2 CLI binary not found at {_CLI_BIN}. Run 'make cli' first.",
)


def _write_geometry(path, coords, radii, eps):
    coords = np.asarray(coords, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    eps = np.asarray(eps, dtype=np.complex128)
    n = len(radii)
    with h5py.File(path, "w") as f:
        f.create_dataset("coord", data=coords)
        f.create_dataset("radius", data=radii)
        f.create_dataset("param_r", data=np.real(eps))
        f.create_dataset("param_i", data=np.imag(eps))
        f.create_dataset("tind", data=np.zeros(n, dtype=np.int32))
        f.create_dataset("angles", data=np.zeros((n, 3)))


def _run_cli(tmp_path, coords, radii, eps, k, extra_args):
    geo_path = os.path.join(tmp_path, "geometry.h5")
    _write_geometry(geo_path, coords, radii, eps)
    args = [_CLI_BIN, "-geometry_file", geo_path, "-k", str(k), *extra_args]
    subprocess.run(args, cwd=tmp_path, capture_output=True, check=True)


# A 3-sphere cluster with real inter-sphere interaction (not a trivial
# single-sphere case that could hide a wrong sphere-to-sphere coupling).
_COORDS = [[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0], [0.0, 2.0, 0.5]]
_RADII = [1.0, 0.8, 0.6]
_EPS = [3.0 + 0.1j, 2.5 + 0.05j, 4.0 + 0.2j]
_K = 1.2


def test_fixed_orientation_matches_cli(tmp_path):
    tmp_path = str(tmp_path)
    s_out = os.path.join(tmp_path, "mueller.h5")
    j_out = os.path.join(tmp_path, "jones.h5")
    _run_cli(
        tmp_path, _COORDS, _RADII, _EPS, _K,
        ["-N_ave", "0", "-N_theta", "15", "-N_phi", "3", "-formulation", "2",
         "-acc", "2", "-tol", "1e-6", "-restart", "10", "-max_iter", "100",
         "-S_out", s_out, "-J_out", j_out],
    )

    with h5py.File(s_out) as f:
        mueller_cli = f["mueller"][()].T
        crs_cli = f["cross_sections"][()]
    with h5py.File(j_out) as f:
        jones_cli = f["A_r"][()].T + 1j * f["A_i"][()].T

    f = FaSTMM2()
    result = f.solve(
        _COORDS, _RADII, _EPS, _K, N_theta=15, N_phi=3,
        formulation=2, acc=2, tol=1e-6, restart=10, max_iter=100,
    )

    np.testing.assert_allclose(result["mueller"], mueller_cli, atol=1e-10)
    np.testing.assert_allclose(result["jones"], jones_cli, atol=1e-10)
    np.testing.assert_allclose(
        [result["c_ext"], result["c_ext_minus_c_abs"], result["c_abs"],
         result["c_sca"], result["asymmetry"]],
        crs_cli, atol=1e-10,
    )


def test_orientation_averaged_matches_cli(tmp_path):
    tmp_path = str(tmp_path)
    s_out = os.path.join(tmp_path, "mueller_avg.h5")
    _run_cli(
        tmp_path, _COORDS, _RADII, _EPS, _K,
        ["-N_ave", "4", "-halton_init", "0", "-N_theta", "15", "-N_phi", "3",
         "-formulation", "2", "-acc", "2", "-tol", "1e-6", "-restart", "10",
         "-max_iter", "100", "-S_out", s_out],
    )

    with h5py.File(s_out) as f:
        mueller_cli = f["mueller"][()].T
        crs_cli = f["cross_sections"][()]

    f = FaSTMM2()
    result = f.solve(
        _COORDS, _RADII, _EPS, _K, N_theta=15, N_phi=3, N_ave=4, halton_init=0,
        formulation=2, acc=2, tol=1e-6, restart=10, max_iter=100,
    )

    np.testing.assert_allclose(result["mueller"], mueller_cli, atol=1e-10)
    np.testing.assert_allclose(
        [result["c_ext"], result["c_ext_minus_c_abs"], result["c_abs"],
         result["c_sca"], result["asymmetry"]],
        crs_cli, atol=1e-10,
    )


def test_tmatrix_matches_cli(tmp_path):
    tmp_path = str(tmp_path)
    t_out = os.path.join(tmp_path, "Tout.h5")
    s_out = os.path.join(tmp_path, "mueller_t.h5")
    _run_cli(
        tmp_path, _COORDS, _RADII, _EPS, _K,
        ["-N_ave", "0", "-Tmatrix", "1", "-T_order", "8", "-formulation", "2",
         "-acc", "2", "-tol", "1e-6", "-restart", "10", "-max_iter", "100",
         "-T_out", t_out, "-S_out", s_out],
    )

    with h5py.File(t_out) as hf:
        Taa_cli = (hf["Taa_r"][()] + 1j * hf["Taa_i"][()]).T
        Tab_cli = (hf["Tab_r"][()] + 1j * hf["Tab_i"][()]).T
        Tba_cli = (hf["Tba_r"][()] + 1j * hf["Tba_i"][()]).T
        Tbb_cli = (hf["Tbb_r"][()] + 1j * hf["Tbb_i"][()]).T

    f = FaSTMM2()
    result = f.compute_tmatrix(
        _COORDS, _RADII, _EPS, _K, t_order=8,
        formulation=2, acc=2, tol=1e-6, restart=10, max_iter=100,
    )

    assert result["Taa"].shape == (get_tmatrix_size(8), get_tmatrix_size(8))
    np.testing.assert_allclose(result["Taa"], Taa_cli, atol=1e-10)
    np.testing.assert_allclose(result["Tab"], Tab_cli, atol=1e-10)
    np.testing.assert_allclose(result["Tba"], Tba_cli, atol=1e-10)
    np.testing.assert_allclose(result["Tbb"], Tbb_cli, atol=1e-10)
