"""
pyFaSTMM Dashboard.

Configure a cluster of spheres, run FaSTMM2 in-process (no files), and
visualize the resulting Mueller-matrix phase function (P11) and degree of
linear polarization (-P12/P11) vs scattering angle. Optionally cross-check
against the standalone FaSTMM2 CLI binary (build/cli/FaSTMM2, via
``make cli``) on the same geometry.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pyfastmm import FaSTMM2

_PROJ_ROOT = Path(__file__).resolve().parent.parent
_CLI_BIN = _PROJ_ROOT / "build" / "cli" / "FaSTMM2"

st.set_page_config(page_title="pyFaSTMM Dashboard", layout="wide")
st.title("pyFaSTMM Dashboard")
st.caption(
    "Compute Mueller-matrix scattering from a cluster of spheres via "
    "FaSTMM2's MLFMM solver, entirely in-process -- no input/output files."
)

# ---------------------------------------------------------------------------
# Sidebar: cluster + solver configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Cluster")
    preset = st.selectbox(
        "Preset", ["Single sphere", "Two spheres", "Random cluster"], index=1
    )

    if preset == "Single sphere":
        default_df = pd.DataFrame(
            {
                "x": [0.0], "y": [0.0], "z": [0.0],
                "radius": [1.0], "eps_re": [3.0], "eps_im": [0.1],
            }
        )
    elif preset == "Two spheres":
        default_df = pd.DataFrame(
            {
                "x": [-1.5, 1.5], "y": [0.0, 0.0], "z": [0.0, 0.0],
                "radius": [1.0, 1.0], "eps_re": [3.0, 3.0], "eps_im": [0.1, 0.1],
            }
        )
    else:
        n_random = st.slider("Number of spheres", 3, 30, 8)
        rng = np.random.default_rng(0)
        pos = rng.uniform(-3.0, 3.0, size=(n_random, 3))
        default_df = pd.DataFrame(
            {
                "x": pos[:, 0], "y": pos[:, 1], "z": pos[:, 2],
                "radius": np.full(n_random, 0.8),
                "eps_re": np.full(n_random, 3.0),
                "eps_im": np.full(n_random, 0.1),
            }
        )

    sphere_df = st.data_editor(default_df, num_rows="dynamic", key="sphere_table")

    st.header("Incident field / solver")
    k = st.number_input("Wavenumber k", value=1.2, min_value=0.01, step=0.1)
    N_theta = st.slider("N_theta", 5, 361, 91, step=2)
    N_phi = st.slider("N_phi", 1, 32, 8)
    N_ave = st.number_input(
        "N_ave (0 = fixed orientation)", value=0, min_value=0, step=1
    )
    formulation = st.selectbox(
        "Formulation", options=[0, 1, 2],
        format_func=lambda v: {0: "STMM", 1: "FaSTMM", 2: "FaSTMM2"}[v],
        index=2,
    )
    acc = st.slider("MLFMM accuracy (digits)", 1, 6, 2)
    tol = st.number_input("GMRES tolerance", value=1e-4, format="%.1e")
    restart = st.number_input("GMRES restart", value=5, min_value=1, step=1)
    max_iter = st.number_input("GMRES max iterations", value=50, min_value=1, step=1)

    compare_cli = st.checkbox(
        "Compare against CLI reference",
        value=False,
        disabled=not _CLI_BIN.is_file(),
        help=(
            "Requires build/cli/FaSTMM2 (run `make cli`) and h5py."
            if not _CLI_BIN.is_file()
            else None
        ),
    )

    run = st.button("Solve", type="primary")


# ---------------------------------------------------------------------------
# CLI cross-check helper
# ---------------------------------------------------------------------------


def _run_cli(coords, radii, eps, k, N_theta, N_phi, N_ave, formulation, acc, tol, restart, max_iter):
    import h5py

    with tempfile.TemporaryDirectory() as tmp:
        geo_path = os.path.join(tmp, "geometry.h5")
        s_out = os.path.join(tmp, "mueller.h5")
        n = len(radii)
        with h5py.File(geo_path, "w") as f:
            f.create_dataset("coord", data=np.asarray(coords))
            f.create_dataset("radius", data=np.asarray(radii))
            f.create_dataset("param_r", data=np.real(eps))
            f.create_dataset("param_i", data=np.imag(eps))
            f.create_dataset("tind", data=np.zeros(n, dtype=np.int32))
            f.create_dataset("angles", data=np.zeros((n, 3)))

        args = [
            str(_CLI_BIN), "-geometry_file", geo_path, "-k", str(k),
            "-N_ave", str(N_ave), "-N_theta", str(N_theta), "-N_phi", str(N_phi),
            "-formulation", str(formulation), "-acc", str(acc), "-tol", str(tol),
            "-restart", str(restart), "-max_iter", str(max_iter), "-S_out", s_out,
        ]
        subprocess.run(args, cwd=tmp, capture_output=True, check=True)

        with h5py.File(s_out) as f:
            return f["mueller"][()].T, f["cross_sections"][()]


# ---------------------------------------------------------------------------
# Solve + plot
# ---------------------------------------------------------------------------

if run:
    coords = sphere_df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    radii = sphere_df["radius"].to_numpy(dtype=np.float64)
    eps = sphere_df["eps_re"].to_numpy(dtype=np.float64) + 1j * sphere_df[
        "eps_im"
    ].to_numpy(dtype=np.float64)

    f = FaSTMM2()
    with st.spinner("Solving..."):
        result = f.solve(
            coords, radii, eps, k,
            N_theta=int(N_theta), N_phi=int(N_phi), N_ave=int(N_ave),
            formulation=int(formulation), acc=int(acc), tol=float(tol),
            restart=int(restart), max_iter=int(max_iter),
        )

    mueller = result["mueller"]
    if N_ave == 0:
        theta_deg = np.degrees(mueller[:, 1])
        p11 = mueller[:, 2]
        p12 = mueller[:, 3]
    else:
        theta_deg = np.degrees(mueller[:, 0])
        p11 = mueller[:, 1]
        p12 = mueller[:, 2]

    dolp = np.where(p11 != 0, -p12 / p11, 0.0)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Cext", f"{result['c_ext']:.4f}")
    col2.metric("Cabs", f"{result['c_abs']:.4f}")
    col3.metric("Csca", f"{result['c_sca']:.4f}")
    col4.metric("Csca (optical thm.)", f"{result['c_ext_minus_c_abs']:.4f}")
    col5.metric("Asymmetry <cos theta>", f"{result['asymmetry']:.4f}")

    cli_theta = cli_p11 = cli_dolp = None
    if compare_cli:
        try:
            cli_mueller, cli_crs = _run_cli(
                coords, radii, eps, k, N_theta, N_phi, N_ave, formulation,
                acc, tol, restart, max_iter,
            )
            st.info(
                f"CLI cross sections: Cext={cli_crs[0]:.4f}, Cabs={cli_crs[2]:.4f}, "
                f"Csca={cli_crs[3]:.4f}, asymmetry={cli_crs[4]:.4f}"
            )
            cli_theta = np.degrees(cli_mueller[:, 1 if N_ave == 0 else 0])
            cli_p11 = cli_mueller[:, 2 if N_ave == 0 else 1]
            cli_p12 = cli_mueller[:, 3 if N_ave == 0 else 2]
            cli_dolp = np.where(cli_p11 != 0, -cli_p12 / cli_p11, 0.0)
        except Exception as exc:  # noqa: BLE001
            st.error(f"CLI run failed: {exc}")

    fig_p11 = go.Figure()
    fig_p11.add_trace(go.Scatter(x=theta_deg, y=p11, mode="lines", name="pyfastmm"))
    if cli_theta is not None:
        fig_p11.add_trace(
            go.Scatter(x=cli_theta, y=cli_p11, mode="markers", name="CLI")
        )
    fig_p11.update_layout(
        title="Phase function P11", xaxis_title="Scattering angle (deg)",
        yaxis_title="P11", yaxis_type="log",
    )
    st.plotly_chart(fig_p11, width='stretch')

    fig_dolp = go.Figure()
    fig_dolp.add_trace(go.Scatter(x=theta_deg, y=dolp, mode="lines", name="pyfastmm"))
    if cli_theta is not None:
        fig_dolp.add_trace(
            go.Scatter(x=cli_theta, y=cli_dolp, mode="markers", name="CLI")
        )
    fig_dolp.update_layout(
        title="Degree of linear polarization (-P12/P11)",
        xaxis_title="Scattering angle (deg)", yaxis_title="-P12/P11",
    )
    st.plotly_chart(fig_dolp, width='stretch')

    with st.expander("Raw Mueller matrix"):
        st.dataframe(mueller)
else:
    st.info("Configure a cluster in the sidebar and click **Solve**.")
