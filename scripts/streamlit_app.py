"""
pyFaSTMM Dashboard.

Two modes, as tabs -- each tab owns its *own* configuration controls (no
shared sidebar), so switching tabs never leaves stale or irrelevant
widgets from the other mode on screen. Settings live in a row above the
results, not a side column, so plots get the full page width:

- **Single run**: configure a cluster of spheres, run FaSTMM2 in-process
  (no files), and visualize the resulting Mueller-matrix phase function
  (P11) and degree of linear polarization (-P12/P11) vs scattering angle.
- **Wavelength sweep**: load a TOML sweep config (same shape as pyMSTM's
  -- see pyfastmm._config) pointing at a cluster file (e.g. a PyFracVAL
  fractal aggregate with hundreds of particles) and a wavelength range,
  run FaSTMM2 once per wavelength, and plot the resulting cross-section
  spectrum plus the angular Mueller matrix for every wavelength at once
  (color-graded, like pyMSTM's sweep plots).

Both tabs always cross-check against the standalone FaSTMM2 CLI binary
(build/cli/FaSTMM2, via ``make cli``) when it's available -- comparing
against the reference implementation is the whole point of this
dashboard, so it isn't an opt-in checkbox. A quantity always gets the
*same* color for its pyfastmm line and its CLI markers, so the two are
easy to pair up visually.
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
_CLI_AVAILABLE = _CLI_BIN.is_file()

# Fixed colors for the three cross sections, shared between pyfastmm and
# CLI traces of the same quantity (rather than plotly's default per-trace
# color cycling, which would give pyfastmm-Cext and CLI-Cext different,
# unrelated colors).
_QUANTITY_COLORS = {"c_ext": "#1f77b4", "c_abs": "#d62728", "c_sca": "#2ca02c"}

st.set_page_config(page_title="pyFaSTMM Dashboard", layout="wide")
st.title("pyFaSTMM Dashboard")
st.caption(
    "Compute Mueller-matrix scattering from a cluster of spheres via "
    "FaSTMM2's MLFMM solver, entirely in-process -- no input/output files. "
    "Always cross-checked against the standalone CLI reference."
)
if not _CLI_AVAILABLE:
    st.warning(
        "CLI reference binary not found at build/cli/FaSTMM2 -- results "
        "won't be cross-checked. Run `make cli` (needs CMake, LAPACK, "
        "HDF5) to enable it."
    )

tab_single, tab_sweep = st.tabs(["Single run", "Wavelength sweep (TOML config)"])

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _run_cli(coords, radii, eps, k, N_theta, N_phi, N_ave, formulation, acc, tol, restart, max_iter):
    """Run the CLI reference binary on one geometry/wavenumber, return
    (mueller, crs_dict) in the same shape as FaSTMM2.solve()'s result."""
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
            mueller = f["mueller"][()].T
            crs = f["cross_sections"][()]
    return mueller, {
        "c_ext": float(crs[0]), "c_ext_minus_c_abs": float(crs[1]),
        "c_abs": float(crs[2]), "c_sca": float(crs[3]), "asymmetry": float(crs[4]),
    }


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


def _gradient_colors(n):
    """n colors spread across a hue gradient (blue -> red), matching
    pyMSTM's wavelength-sweep plots -- used here for both phi cuts and
    wavelengths so the same index always gets the same color between a
    pyfastmm line and its CLI markers."""
    return [f"hsl({int(260 * (1 - i / max(1, n - 1)))},80%,50%)" for i in range(n)]


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

        phi_indices = list(range(N_phi)) if phi_choice == phi_labels[0] else [
            phi_labels.index(phi_choice) - 1
        ]
        colors = _gradient_colors(len(phi_indices))
        for pos, i in enumerate(phi_indices):
            theta_deg = np.degrees(mueller_3d[i, :, 1])
            p11 = mueller_3d[i, :, 2]
            p12 = mueller_3d[i, :, 3]
            dolp = np.where(p11 != 0, -p12 / p11, 0.0)
            label = f"pyfastmm (phi={phi_deg[i]:.1f} deg)"
            fig_p11.add_trace(go.Scatter(
                x=theta_deg, y=p11, mode="lines", line=dict(color=colors[pos]),
                name=label, legendgroup=f"phi{i}",
            ))
            fig_dolp.add_trace(go.Scatter(
                x=theta_deg, y=dolp, mode="lines", line=dict(color=colors[pos]),
                name=label, legendgroup=f"phi{i}",
            ))
            if cli_3d is not None:
                cli_theta = np.degrees(cli_3d[i, :, 1])
                cli_p11 = cli_3d[i, :, 2]
                cli_p12 = cli_3d[i, :, 3]
                cli_dolp = np.where(cli_p11 != 0, -cli_p12 / cli_p11, 0.0)
                cli_label = f"CLI (phi={phi_deg[i]:.1f} deg)"
                fig_p11.add_trace(go.Scatter(
                    x=cli_theta, y=cli_p11, mode="markers",
                    marker=dict(color=colors[pos], symbol="x"),
                    name=cli_label, legendgroup=f"phi{i}",
                ))
                fig_dolp.add_trace(go.Scatter(
                    x=cli_theta, y=cli_dolp, mode="markers",
                    marker=dict(color=colors[pos], symbol="x"),
                    name=cli_label, legendgroup=f"phi{i}",
                ))
    else:
        theta_deg = np.degrees(mueller[:, 0])
        p11 = mueller[:, 1]
        p12 = mueller[:, 2]
        dolp = np.where(p11 != 0, -p12 / p11, 0.0)
        fig_p11.add_trace(go.Scatter(
            x=theta_deg, y=p11, mode="lines", line=dict(color=_QUANTITY_COLORS["c_ext"]),
            name="pyfastmm",
        ))
        fig_dolp.add_trace(go.Scatter(
            x=theta_deg, y=dolp, mode="lines", line=dict(color=_QUANTITY_COLORS["c_ext"]),
            name="pyfastmm",
        ))
        if cli_mueller is not None:
            cli_theta = np.degrees(cli_mueller[:, 0])
            cli_p11 = cli_mueller[:, 1]
            cli_p12 = cli_mueller[:, 2]
            cli_dolp = np.where(cli_p11 != 0, -cli_p12 / cli_p11, 0.0)
            fig_p11.add_trace(go.Scatter(
                x=cli_theta, y=cli_p11, mode="markers",
                marker=dict(color=_QUANTITY_COLORS["c_ext"], symbol="x"), name="CLI",
            ))
            fig_dolp.add_trace(go.Scatter(
                x=cli_theta, y=cli_dolp, mode="markers",
                marker=dict(color=_QUANTITY_COLORS["c_ext"], symbol="x"), name="CLI",
            ))

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


def render_wavelength_overlay_plots(results, N_ave, N_phi, N_theta, cli_results=None, key="wl_overlay"):
    """P11/DoLP for *every* wavelength in a sweep overlaid on one plot,
    color-graded by wavelength (mirrors pyMSTM's sweep plots) with CLI
    drawn as same-colored markers. For fixed orientation, one phi cut is
    shown at a time (picked below) -- all phi x all wavelengths at once
    would be unreadable.
    """
    wl_m = np.array([r["wavelength_m"] for r in results])
    n = len(results)
    colors = _gradient_colors(n)
    show_every = max(1, n // 8)

    phi_index = 0
    if N_ave == 0:
        sample_3d = _phi_slices(results[0]["mueller"], N_phi, N_theta)
        phi_deg = np.degrees(sample_3d[:, 0, 0])
        phi_index = st.selectbox(
            "Phi cut to plot", list(range(len(phi_deg))), index=0, key=f"{key}_phi",
            format_func=lambda i: f"phi = {phi_deg[i]:.1f} deg",
        )

    fig_p11 = go.Figure()
    fig_dolp = go.Figure()
    for i, r in enumerate(results):
        mueller = r["mueller"]
        if N_ave == 0:
            m3 = _phi_slices(mueller, N_phi, N_theta)
            theta_deg = np.degrees(m3[phi_index, :, 1])
            p11, p12 = m3[phi_index, :, 2], m3[phi_index, :, 3]
        else:
            theta_deg = np.degrees(mueller[:, 0])
            p11, p12 = mueller[:, 1], mueller[:, 2]
        dolp = np.where(p11 != 0, -p12 / p11, 0.0)
        showlegend = i % show_every == 0
        fig_p11.add_trace(go.Scatter(
            x=theta_deg, y=p11, mode="lines", line=dict(color=colors[i], width=1.5),
            name=f"pyfastmm {wl_m[i]:.4g}", legendgroup=f"w{i}", showlegend=showlegend,
        ))
        fig_dolp.add_trace(go.Scatter(
            x=theta_deg, y=dolp, mode="lines", line=dict(color=colors[i], width=1.5),
            name=f"pyfastmm {wl_m[i]:.4g}", legendgroup=f"w{i}", showlegend=showlegend,
        ))
        cli_r = cli_results[i] if cli_results is not None else None
        if cli_r is not None:
            cli_mueller = cli_r["mueller"]
            if N_ave == 0:
                cm3 = _phi_slices(cli_mueller, N_phi, N_theta)
                cli_theta = np.degrees(cm3[phi_index, :, 1])
                cli_p11, cli_p12 = cm3[phi_index, :, 2], cm3[phi_index, :, 3]
            else:
                cli_theta = np.degrees(cli_mueller[:, 0])
                cli_p11, cli_p12 = cli_mueller[:, 1], cli_mueller[:, 2]
            cli_dolp = np.where(cli_p11 != 0, -cli_p12 / cli_p11, 0.0)
            fig_p11.add_trace(go.Scatter(
                x=cli_theta, y=cli_p11, mode="markers",
                marker=dict(color=colors[i], size=4, symbol="x"),
                name=f"CLI {wl_m[i]:.4g}", legendgroup=f"w{i}", showlegend=showlegend,
            ))
            fig_dolp.add_trace(go.Scatter(
                x=cli_theta, y=cli_dolp, mode="markers",
                marker=dict(color=colors[i], size=4, symbol="x"),
                name=f"CLI {wl_m[i]:.4g}", legendgroup=f"w{i}", showlegend=showlegend,
            ))

    fig_p11.update_layout(
        title="Phase function P11 (all wavelengths)", xaxis_title="Scattering angle (deg)",
        yaxis_title="P11", yaxis_type="log", hovermode="x unified", height=500,
    )
    st.plotly_chart(fig_p11, width="stretch", key=f"{key}_p11")

    fig_dolp.update_layout(
        title="Degree of linear polarization (all wavelengths)",
        xaxis_title="Scattering angle (deg)", yaxis_title="-P12/P11",
        hovermode="x unified", height=500,
    )
    st.plotly_chart(fig_dolp, width="stretch", key=f"{key}_dolp")


def render_cross_section_metrics(result, cli_result=None):
    labels = ["Cext", "Cabs", "Csca", "Csca (optical thm.)", "Asymmetry <cos theta>"]
    keys = ["c_ext", "c_abs", "c_sca", "c_ext_minus_c_abs", "asymmetry"]
    cols = st.columns(len(labels))
    for col, label, key in zip(cols, labels, keys):
        delta = None
        if cli_result is not None:
            delta = result[key] - cli_result[key]
        col.metric(label, f"{result[key]:.4f}", delta=f"{delta:+.4f} vs CLI" if delta is not None else None)


def render_accuracy_table(rows_source):
    """rows_source: list of (label, pyfastmm_result_dict, cli_result_dict_or_None)."""
    rows = []
    for label, r, c in rows_source:
        row = {"": label}
        for key, name in [("c_ext", "Cext"), ("c_abs", "Cabs"), ("c_sca", "Csca"), ("asymmetry", "g")]:
            row[f"pyfastmm {name}"] = round(r[key], 6)
            if c is not None:
                row[f"CLI {name}"] = round(c[key], 6)
                row[f"|Delta {name}| %"] = round(
                    abs(r[key] - c[key]) / max(abs(c[key]), 1e-12) * 100, 4
                )
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ---------------------------------------------------------------------------
# Tab 1: single run
# ---------------------------------------------------------------------------

with tab_single:
    st.subheader("Configuration")
    c_cluster, c_field, c_solver1, c_solver2 = st.columns(4)

    with c_cluster:
        preset = st.selectbox(
            "Preset", ["Single sphere", "Two spheres", "Random cluster"], index=1,
            key="single_preset",
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
            n_random = st.slider("Number of spheres", 3, 30, 8, key="single_n_random")
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

    with c_field:
        k = st.number_input("Wavenumber k", value=1.2, min_value=0.01, step=0.1, key="single_k")
        N_theta = st.slider("N_theta", 5, 361, 91, step=2, key="single_ntheta")
        N_phi = st.slider("N_phi", 1, 32, 8, key="single_nphi")
        N_ave = st.number_input(
            "N_ave (0 = fixed orientation)", value=0, min_value=0, step=1, key="single_nave"
        )

    with c_solver1:
        formulation = st.selectbox(
            "Formulation", options=[0, 1, 2],
            format_func=lambda v: {0: "STMM", 1: "FaSTMM", 2: "FaSTMM2"}[v],
            index=2, key="single_formulation",
        )
        acc = st.slider("MLFMM accuracy (digits)", 1, 6, 2, key="single_acc")

    with c_solver2:
        tol = st.number_input("GMRES tolerance", value=1e-4, format="%.1e", key="single_tol")
        restart = st.number_input("GMRES restart", value=5, min_value=1, step=1, key="single_restart")
        max_iter = st.number_input(
            "GMRES max iterations", value=50, min_value=1, step=1, key="single_maxiter"
        )

    run = st.button("Solve", type="primary", key="single_solve")
    st.divider()

    # The solve itself (and the CLI cross-check) only runs when the Solve
    # button is clicked -- the result is cached in st.session_state so
    # that later widget interactions (e.g. picking which phi cut to plot)
    # just rerun the script without recomputing.
    if run:
        coords = sphere_df[["x", "y", "z"]].to_numpy(dtype=np.float64)
        radii = sphere_df["radius"].to_numpy(dtype=np.float64)
        eps = sphere_df["eps_re"].to_numpy(dtype=np.float64) + 1j * sphere_df[
            "eps_im"
        ].to_numpy(dtype=np.float64)

        f = FaSTMM2()
        with st.spinner("Solving (pyfastmm)..."):
            result = f.solve(
                coords, radii, eps, k,
                N_theta=int(N_theta), N_phi=int(N_phi), N_ave=int(N_ave),
                formulation=int(formulation), acc=int(acc), tol=float(tol),
                restart=int(restart), max_iter=int(max_iter),
            )

        cli_mueller = cli_crs = None
        if _CLI_AVAILABLE:
            try:
                with st.spinner("Solving (CLI reference)..."):
                    cli_mueller, cli_crs = _run_cli(
                        coords, radii, eps, k, N_theta, N_phi, N_ave, formulation,
                        acc, tol, restart, max_iter,
                    )
            except Exception as exc:  # noqa: BLE001
                st.error(f"CLI run failed: {exc}")

        st.session_state["dashboard_result"] = {
            "result": result,
            "cli_mueller": cli_mueller,
            "cli_crs": cli_crs,
            "N_theta": int(N_theta),
            "N_phi": int(N_phi),
            "N_ave": int(N_ave),
        }

    if "dashboard_result" in st.session_state:
        state = st.session_state["dashboard_result"]
        result = state["result"]
        cli_mueller = state["cli_mueller"]
        cli_crs = state["cli_crs"]

        render_cross_section_metrics(result, cli_crs)

        render_angular_plots(
            result["mueller"], state["N_ave"], state["N_phi"], state["N_theta"],
            cli_mueller=cli_mueller, key="single_phi_cut",
        )

        with st.expander("Results table and accuracy"):
            render_accuracy_table([("this run", result, cli_crs)])

        with st.expander("Raw Mueller matrix"):
            st.dataframe(result["mueller"])
    else:
        st.info("Configure a cluster above and click **Solve**.")

# ---------------------------------------------------------------------------
# Tab 2: wavelength sweep from a TOML config
# ---------------------------------------------------------------------------

with tab_sweep:
    from pyfastmm._config import SweepConfig, load_config

    st.subheader("Sweep configuration")
    c_source, c_overrides = st.columns([2, 1])

    with c_source:
        discovered = sorted(_DATA_DIR.glob("*.toml")) if _DATA_DIR.is_dir() else []
        source_labels = [str(p.relative_to(_PROJ_ROOT)) for p in discovered] + ["Upload custom TOML..."]
        source_choice = st.selectbox("Config source", source_labels, index=0, key="sweep_source")

        cfg: SweepConfig | None = None
        cfg_base_dir: Path = _DATA_DIR
        cfg_error = None

        if source_choice == "Upload custom TOML...":
            up1, up2 = st.columns(2)
            uploaded_toml = up1.file_uploader("Sweep config (.toml)", type=["toml"], key="sweep_upload_toml")
            uploaded_data_files = up2.file_uploader(
                "Cluster position file(s)",
                type=["dat", "txt", "csv", "pos"],
                accept_multiple_files=True,
                key="sweep_upload_data",
            )
            if uploaded_toml is not None:
                tmp_dir = Path(tempfile.mkdtemp())
                toml_path = tmp_dir / uploaded_toml.name
                toml_path.write_bytes(uploaded_toml.getvalue())
                for uf in uploaded_data_files or []:
                    (tmp_dir / uf.name).write_bytes(uf.getvalue())
                try:
                    cfg = load_config(toml_path)
                    # If the referenced positions_file wasn't uploaded,
                    # fall back to tests/data/ -- lets a custom config
                    # (e.g. a different wavelength range) reuse an
                    # existing cluster file without re-uploading it.
                    if (tmp_dir / cfg.particles.positions_file).is_file():
                        cfg_base_dir = tmp_dir
                    else:
                        cfg_base_dir = _DATA_DIR
                except Exception as exc:  # noqa: BLE001
                    cfg_error = str(exc)
            else:
                st.info(
                    "Upload a .toml sweep config. If it references a "
                    "cluster position file not already in tests/data/, "
                    "upload that too."
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

        n_particles = None
        if cfg is not None:
            try:
                n_particles = cfg.particles.load_positions(cfg_base_dir).shape[0]
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to load cluster file: {exc}")

            wl = cfg.wavelengths.get_wavelengths_m()
            st.caption(
                f"{n_particles if n_particles is not None else '?'} particles, "
                f"{len(wl)} wavelengths in [{wl.min():.3g}, {wl.max():.3g}], "
                f"formulation={ {0: 'STMM', 1: 'FaSTMM', 2: 'FaSTMM2'}[cfg.solver.formulation] }"
            )

    with c_overrides:
        if cfg is not None:
            with st.expander("Quick overrides (speed vs. accuracy)"):
                cfg.solver.N_theta = int(
                    st.slider("N_theta", 5, 361, cfg.solver.N_theta, step=2, key="sweep_ntheta")
                )
                cfg.solver.N_phi = int(st.slider("N_phi", 1, 32, cfg.solver.N_phi, key="sweep_nphi"))
                cfg.solver.acc = int(
                    st.slider("MLFMM accuracy (digits)", 1, 6, cfg.solver.acc, key="sweep_acc")
                )

            if cfg.output.compute_tmatrix:
                st.info("output.compute_tmatrix is set -- CLI cross-check isn't available for T-matrix sweeps.")

            run_sweep_clicked = st.button(
                "Run Sweep", type="primary", disabled=(n_particles is None), key="sweep_run"
            )
        else:
            run_sweep_clicked = False

    st.divider()

    if cfg is not None and run_sweep_clicked:
        positions = cfg.particles.load_positions(cfg_base_dir)
        positions = cfg.incident.rotate_positions(positions)
        coords, radii = positions[:, :3], positions[:, 3]
        eps = np.full(len(radii), cfg.particles.eps())
        wl_m = cfg.wavelengths.get_wavelengths_m()
        k_vals = cfg.wavelengths.get_wavenumbers()
        solver = cfg.solver
        do_cli = _CLI_AVAILABLE and not cfg.output.compute_tmatrix

        n_steps = len(wl_m) * (2 if do_cli else 1)
        progress = st.progress(0.0, text="Running sweep...")
        step = 0
        f = FaSTMM2()
        pyfastmm_results = []
        cli_results = []
        try:
            for wl, kk in zip(wl_m, k_vals):
                if cfg.output.compute_tmatrix:
                    r = f.compute_tmatrix(
                        coords, radii, eps, float(kk), t_order=cfg.output.t_order,
                        formulation=solver.formulation, acc=solver.acc,
                        tol=solver.tolerance, restart=solver.restart,
                        max_iter=solver.max_iterations,
                    )
                else:
                    r = f.solve(
                        coords, radii, eps, float(kk),
                        N_theta=solver.N_theta, N_phi=solver.N_phi,
                        N_ave=solver.N_ave, halton_init=solver.halton_init,
                        formulation=solver.formulation, acc=solver.acc,
                        tol=solver.tolerance, restart=solver.restart,
                        max_iter=solver.max_iterations,
                    )
                r["wavelength_m"] = float(wl)
                r["k"] = float(kk)
                pyfastmm_results.append(r)
                step += 1
                progress.progress(step / n_steps, text=f"pyfastmm: wavelength {wl:.4g} ({step}/{n_steps})")

                if do_cli:
                    try:
                        cli_mueller, cli_crs = _run_cli(
                            coords, radii, eps, float(kk), solver.N_theta, solver.N_phi,
                            solver.N_ave, solver.formulation, solver.acc,
                            solver.tolerance, solver.restart, solver.max_iterations,
                        )
                        cli_crs["mueller"] = cli_mueller
                        cli_results.append(cli_crs)
                    except Exception as exc:  # noqa: BLE001
                        st.warning(f"CLI failed at wavelength {wl:.4g}: {exc}")
                        cli_results.append(None)
                    step += 1
                    progress.progress(step / n_steps, text=f"CLI: wavelength {wl:.4g} ({step}/{n_steps})")

            st.session_state["sweep_results"] = {
                "results": pyfastmm_results,
                "cli_results": cli_results if do_cli else None,
                "N_theta": solver.N_theta,
                "N_phi": solver.N_phi,
                "N_ave": solver.N_ave,
                "is_tmatrix": cfg.output.compute_tmatrix,
            }
        except Exception as exc:  # noqa: BLE001
            st.error(f"Sweep failed: {exc}")
        progress.empty()

    if "sweep_results" in st.session_state:
        sweep_state = st.session_state["sweep_results"]
        results = sweep_state["results"]
        cli_results = sweep_state["cli_results"]
        wl_m = np.array([r["wavelength_m"] for r in results])

        if sweep_state["is_tmatrix"]:
            st.subheader("T-matrix sweep")
            t_idx = st.select_slider(
                "Wavelength", options=list(range(len(results))),
                format_func=lambda i: f"{wl_m[i]:.4g}", key="sweep_t_wl",
            )
            Taa = results[t_idx]["Taa"]
            st.write(f"Taa shape: {Taa.shape}")
            with st.expander("Taa (real part)"):
                st.dataframe(np.real(Taa))
        else:
            st.subheader("Cross-section spectrum")
            fig_spec = go.Figure()
            for key, name in [("c_ext", "Cext"), ("c_abs", "Cabs"), ("c_sca", "Csca")]:
                color = _QUANTITY_COLORS[key]
                fig_spec.add_trace(go.Scatter(
                    x=wl_m, y=[r[key] for r in results], name=f"{name} (pyfastmm)",
                    mode="lines+markers", line=dict(color=color), legendgroup=key,
                ))
                if cli_results is not None:
                    fig_spec.add_trace(go.Scatter(
                        x=wl_m,
                        y=[c[key] if c is not None else None for c in cli_results],
                        name=f"{name} (CLI)", mode="markers",
                        marker=dict(color=color, symbol="x", size=9), legendgroup=key,
                    ))
            fig_spec.update_layout(
                xaxis_title="Wavelength", yaxis_title="Cross section", hovermode="x unified",
            )
            st.plotly_chart(fig_spec, width="stretch")

            st.subheader("Angular Mueller matrix, all wavelengths")
            render_wavelength_overlay_plots(
                results, sweep_state["N_ave"], sweep_state["N_phi"], sweep_state["N_theta"],
                cli_results=cli_results, key="sweep_overlay",
            )

            with st.expander("Results table and accuracy", expanded=True):
                rows_source = [
                    (f"{wl_m[i]:.4g}", results[i], cli_results[i] if cli_results is not None else None)
                    for i in range(len(results))
                ]
                render_accuracy_table(rows_source)
    elif cfg is None:
        st.info("Pick or upload a sweep config above to get started.")
