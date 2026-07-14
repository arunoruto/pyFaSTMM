"""
pyFaSTMM Dashboard.

Two modes, as tabs:

- **Single run**: configure a cluster of spheres in the sidebar, run
  FaSTMM2 in-process (no files), and visualize the resulting Mueller-
  matrix phase function (P11) and degree of linear polarization
  (-P12/P11) vs scattering angle. Optionally cross-check against the
  standalone FaSTMM2 CLI binary (build/cli/FaSTMM2, via ``make cli``) on
  the same geometry.
- **Wavelength sweep**: load a TOML sweep config (same shape as pyMSTM's
  -- see pyfastmm._config) pointing at a cluster file (e.g. a PyFracVAL
  fractal aggregate with hundreds of particles) and a wavelength range,
  run FaSTMM2 once per wavelength, and plot the resulting cross-section
  spectrum plus the angular Mueller matrix at any wavelength in the sweep.
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
_DATA_DIR = _PROJ_ROOT / "tests" / "data"

st.set_page_config(page_title="pyFaSTMM Dashboard", layout="wide")
st.title("pyFaSTMM Dashboard")
st.caption(
    "Compute Mueller-matrix scattering from a cluster of spheres via "
    "FaSTMM2's MLFMM solver, entirely in-process -- no input/output files."
)

tab_single, tab_sweep = st.tabs(["Single run", "Wavelength sweep (TOML config)"])

# ---------------------------------------------------------------------------
# Shared helpers
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


def _phi_slices(mueller, n_phi, n_theta):
    """Reshape a fixed-orientation Mueller array into (phi, theta, cols).

    For fixed orientation (N_ave == 0), FaSTMM2 returns the Mueller matrix
    as N_phi separate theta-sweeps concatenated back to back (phi is the
    *outer* loop, theta the inner one -- see mueller_matrix's loop nest in
    external/fastmm2/src/mie.f90). A cluster of spheres generally isn't
    axisymmetric, so those N_phi cuts are genuinely different curves --
    plotting the whole flattened array as a single connected line zig-zags
    back across theta at every phi boundary. Reshaping and picking one phi
    slice (or showing them as separate traces) avoids that.
    """
    return mueller.reshape(n_phi, n_theta, mueller.shape[1])


def render_angular_plots(mueller, N_ave, N_phi, N_theta, cli_mueller=None, key="phi_cut"):
    """Render the P11 phase-function and DoLP plots for one Mueller matrix
    (fixed orientation or orientation-averaged), with a phi-cut selector
    for the fixed-orientation case. `key` disambiguates the selector's
    Streamlit widget key when this is called more than once per page.
    """
    fig_p11 = go.Figure()
    fig_dolp = go.Figure()

    if N_ave == 0:
        mueller_3d = _phi_slices(mueller, N_phi, N_theta)
        phi_deg = np.degrees(mueller_3d[:, 0, 0])
        phi_labels = ["All phi (separate traces)"] + [
            f"phi = {p:.1f} deg" for p in phi_deg
        ]
        phi_choice = st.selectbox("Phi cut to plot", phi_labels, index=1, key=key)

        cli_3d = _phi_slices(cli_mueller, N_phi, N_theta) if cli_mueller is not None else None

        phi_indices = range(N_phi) if phi_choice == phi_labels[0] else [
            phi_labels.index(phi_choice) - 1
        ]
        for i in phi_indices:
            theta_deg = np.degrees(mueller_3d[i, :, 1])
            p11 = mueller_3d[i, :, 2]
            p12 = mueller_3d[i, :, 3]
            dolp = np.where(p11 != 0, -p12 / p11, 0.0)
            label = f"pyfastmm (phi={phi_deg[i]:.1f} deg)"
            fig_p11.add_trace(go.Scatter(x=theta_deg, y=p11, mode="lines", name=label))
            fig_dolp.add_trace(go.Scatter(x=theta_deg, y=dolp, mode="lines", name=label))
            if cli_3d is not None:
                cli_theta = np.degrees(cli_3d[i, :, 1])
                cli_p11 = cli_3d[i, :, 2]
                cli_p12 = cli_3d[i, :, 3]
                cli_dolp = np.where(cli_p11 != 0, -cli_p12 / cli_p11, 0.0)
                cli_label = f"CLI (phi={phi_deg[i]:.1f} deg)"
                fig_p11.add_trace(
                    go.Scatter(x=cli_theta, y=cli_p11, mode="markers", name=cli_label)
                )
                fig_dolp.add_trace(
                    go.Scatter(x=cli_theta, y=cli_dolp, mode="markers", name=cli_label)
                )
    else:
        theta_deg = np.degrees(mueller[:, 0])
        p11 = mueller[:, 1]
        p12 = mueller[:, 2]
        dolp = np.where(p11 != 0, -p12 / p11, 0.0)
        fig_p11.add_trace(go.Scatter(x=theta_deg, y=p11, mode="lines", name="pyfastmm"))
        fig_dolp.add_trace(go.Scatter(x=theta_deg, y=dolp, mode="lines", name="pyfastmm"))
        if cli_mueller is not None:
            cli_theta = np.degrees(cli_mueller[:, 0])
            cli_p11 = cli_mueller[:, 1]
            cli_p12 = cli_mueller[:, 2]
            cli_dolp = np.where(cli_p11 != 0, -cli_p12 / cli_p11, 0.0)
            fig_p11.add_trace(
                go.Scatter(x=cli_theta, y=cli_p11, mode="markers", name="CLI")
            )
            fig_dolp.add_trace(
                go.Scatter(x=cli_theta, y=cli_dolp, mode="markers", name="CLI")
            )

    fig_p11.update_layout(
        title="Phase function P11", xaxis_title="Scattering angle (deg)",
        yaxis_title="P11", yaxis_type="log",
    )
    st.plotly_chart(fig_p11, width="stretch", key=f"{key}_p11")

    fig_dolp.update_layout(
        title="Degree of linear polarization (-P12/P11)",
        xaxis_title="Scattering angle (deg)", yaxis_title="-P12/P11",
    )
    st.plotly_chart(fig_dolp, width="stretch", key=f"{key}_dolp")


# ---------------------------------------------------------------------------
# Tab 1: single run
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

with tab_single:
    # The solve itself (and, if requested, the CLI cross-check) only runs
    # when the Solve button is clicked -- the result is cached in
    # st.session_state so that later widget interactions (e.g. picking
    # which phi cut to plot) just rerun the script without recomputing.
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

        cli_result = None
        if compare_cli:
            try:
                with st.spinner("Running CLI reference..."):
                    cli_mueller, cli_crs = _run_cli(
                        coords, radii, eps, k, N_theta, N_phi, N_ave, formulation,
                        acc, tol, restart, max_iter,
                    )
                cli_result = (cli_mueller, cli_crs)
            except Exception as exc:  # noqa: BLE001
                st.error(f"CLI run failed: {exc}")

        st.session_state["dashboard_result"] = {
            "result": result,
            "cli_result": cli_result,
            "N_theta": int(N_theta),
            "N_phi": int(N_phi),
            "N_ave": int(N_ave),
        }

    if "dashboard_result" in st.session_state:
        state = st.session_state["dashboard_result"]
        result = state["result"]
        cli_result = state["cli_result"]

        mueller = result["mueller"]

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Cext", f"{result['c_ext']:.4f}")
        col2.metric("Cabs", f"{result['c_abs']:.4f}")
        col3.metric("Csca", f"{result['c_sca']:.4f}")
        col4.metric("Csca (optical thm.)", f"{result['c_ext_minus_c_abs']:.4f}")
        col5.metric("Asymmetry <cos theta>", f"{result['asymmetry']:.4f}")

        cli_mueller = None
        if cli_result is not None:
            cli_mueller, cli_crs = cli_result
            st.info(
                f"CLI cross sections: Cext={cli_crs[0]:.4f}, Cabs={cli_crs[2]:.4f}, "
                f"Csca={cli_crs[3]:.4f}, asymmetry={cli_crs[4]:.4f}"
            )

        render_angular_plots(
            mueller, state["N_ave"], state["N_phi"], state["N_theta"],
            cli_mueller=cli_mueller, key="single_phi_cut",
        )

        with st.expander("Raw Mueller matrix"):
            st.dataframe(mueller)
    else:
        st.info("Configure a cluster in the sidebar and click **Solve**.")

# ---------------------------------------------------------------------------
# Tab 2: wavelength sweep from a TOML config
# ---------------------------------------------------------------------------

with tab_sweep:
    from pyfastmm._config import SweepConfig, load_config, run_sweep

    st.subheader("Sweep configuration")

    discovered = sorted(_DATA_DIR.glob("*.toml")) if _DATA_DIR.is_dir() else []
    source_labels = [str(p.relative_to(_PROJ_ROOT)) for p in discovered] + ["Upload custom TOML..."]
    source_choice = st.selectbox("Config source", source_labels, index=0 if discovered else 0)

    cfg: SweepConfig | None = None
    cfg_base_dir: Path = _DATA_DIR
    cfg_error = None

    if source_choice == "Upload custom TOML...":
        uploaded_toml = st.file_uploader("Sweep config (.toml)", type=["toml"])
        uploaded_data_files = st.file_uploader(
            "Cluster position file(s) referenced by the config",
            type=["dat", "txt", "csv", "pos"],
            accept_multiple_files=True,
        )
        if uploaded_toml is not None:
            tmp_dir = Path(tempfile.mkdtemp())
            toml_path = tmp_dir / uploaded_toml.name
            toml_path.write_bytes(uploaded_toml.getvalue())
            for uf in uploaded_data_files or []:
                (tmp_dir / uf.name).write_bytes(uf.getvalue())
            try:
                cfg = load_config(toml_path)
                # If the referenced positions_file wasn't uploaded, fall
                # back to tests/data/ -- lets a custom config (e.g. a
                # different wavelength range) reuse an existing cluster
                # file without re-uploading it.
                if (tmp_dir / cfg.particles.positions_file).is_file():
                    cfg_base_dir = tmp_dir
                else:
                    cfg_base_dir = _DATA_DIR
            except Exception as exc:  # noqa: BLE001
                cfg_error = str(exc)
        else:
            st.info(
                "Upload a .toml sweep config. If it references a cluster "
                "position file not already in tests/data/, upload that "
                "too via particles.positions_file."
            )
    else:
        toml_path = _PROJ_ROOT / source_choice
        try:
            cfg = load_config(toml_path)
            cfg_base_dir = toml_path.parent
        except Exception as exc:  # noqa: BLE001
            cfg_error = str(exc)

    if cfg_error:
        st.error(f"Failed to load config: {cfg_error}")

    if cfg is not None:
        try:
            positions_preview = cfg.particles.load_positions(cfg_base_dir)
            n_particles = positions_preview.shape[0]
            load_error = None
        except Exception as exc:  # noqa: BLE001
            n_particles = None
            load_error = str(exc)

        wl = cfg.wavelengths.get_wavelengths_m()
        info_cols = st.columns(4)
        info_cols[0].metric("Particles", n_particles if n_particles is not None else "?")
        info_cols[1].metric("Wavelengths", len(wl))
        info_cols[2].metric("Range", f"{wl.min():.3g}-{wl.max():.3g}")
        info_cols[3].metric("Formulation", {0: "STMM", 1: "FaSTMM", 2: "FaSTMM2"}[cfg.solver.formulation])

        if load_error:
            st.error(f"Failed to load cluster file: {load_error}")

        with st.expander("Quick overrides (speed vs. accuracy)"):
            ov_theta = st.slider("N_theta", 5, 361, cfg.solver.N_theta, step=2, key="sweep_ntheta")
            ov_phi = st.slider("N_phi", 1, 32, cfg.solver.N_phi, key="sweep_nphi")
            ov_acc = st.slider("MLFMM accuracy (digits)", 1, 6, cfg.solver.acc, key="sweep_acc")
            cfg.solver.N_theta = int(ov_theta)
            cfg.solver.N_phi = int(ov_phi)
            cfg.solver.acc = int(ov_acc)

        run_sweep_clicked = st.button(
            "Run Sweep", type="primary", disabled=(n_particles is None)
        )

        if run_sweep_clicked:
            progress = st.progress(0.0, text="Running sweep...")

            def _on_progress(frac, wavelength_m):
                progress.progress(frac, text=f"Solved wavelength {wavelength_m:.4g} ({frac:.0%})")

            try:
                results = run_sweep(cfg, base_dir=cfg_base_dir, progress_callback=_on_progress)
                st.session_state["sweep_results"] = {
                    "results": results,
                    "N_theta": cfg.solver.N_theta,
                    "N_phi": cfg.solver.N_phi,
                    "N_ave": cfg.solver.N_ave,
                }
            except Exception as exc:  # noqa: BLE001
                st.error(f"Sweep failed: {exc}")
            progress.empty()

    if "sweep_results" in st.session_state:
        sweep_state = st.session_state["sweep_results"]
        results = sweep_state["results"]
        wl_m = np.array([r["wavelength_m"] for r in results])

        st.subheader("Cross-section spectrum")
        fig_spec = go.Figure()
        fig_spec.add_trace(go.Scatter(x=wl_m, y=[r["c_ext"] for r in results], name="Cext", mode="lines+markers"))
        fig_spec.add_trace(go.Scatter(x=wl_m, y=[r["c_abs"] for r in results], name="Cabs", mode="lines+markers"))
        fig_spec.add_trace(go.Scatter(x=wl_m, y=[r["c_sca"] for r in results], name="Csca", mode="lines+markers"))
        fig_spec.update_layout(xaxis_title="Wavelength", yaxis_title="Cross section")
        st.plotly_chart(fig_spec, width="stretch")

        st.subheader("Angular Mueller matrix at a chosen wavelength")
        wl_idx = st.select_slider(
            "Wavelength", options=list(range(len(results))),
            format_func=lambda i: f"{wl_m[i]:.4g}",
        )
        selected = results[wl_idx]
        cols = st.columns(4)
        cols[0].metric("Cext", f"{selected['c_ext']:.4f}")
        cols[1].metric("Cabs", f"{selected['c_abs']:.4f}")
        cols[2].metric("Csca", f"{selected['c_sca']:.4f}")
        cols[3].metric("Asymmetry", f"{selected['asymmetry']:.4f}")

        render_angular_plots(
            selected["mueller"], sweep_state["N_ave"], sweep_state["N_phi"],
            sweep_state["N_theta"], key="sweep_phi_cut",
        )
    elif cfg is None and not discovered:
        st.info("No sweep configs found in tests/data/. Upload one to get started.")
