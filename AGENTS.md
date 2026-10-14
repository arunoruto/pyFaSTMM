# AGENTS.md

Guidance for AI agents (and humans) working on pyFaSTMM.

## What this is

Python bindings for [FaSTMM2](https://bitbucket.org/planetarysystemresearch/fastmm2), an MLFMM-accelerated
superposition T-matrix solver for EM scattering from clusters of spheres. The upstream Fortran code is a
**git submodule** at `external/fastmm2/` and is never edited in place. Geometry goes in as numpy arrays and
Mueller/Jones/cross-section/T-matrix data comes out — unlike the upstream CLI, no HDF5 files are read or written.

Scope (v1): spherical (Lorenz-Mie) monomers only, not upstream's precomputed-per-monomer-T-matrix input path.

## Development commands

Everything runs inside the devenv/nix shell (`devenv shell`), which provides gfortran, LAPACK/BLAS, HDF5,
cmake, pkg-config, and podman, and syncs the venv with all extras via `uv`.

```bash
uv sync --extra dev              # installs deps AND builds the f2py extension
uv run pytest tests/             # full test suite
uv run pytest tests/test_solve.py::test_resolve_n_threads_zero_raises -v   # single test
make cli                         # build the standalone FaSTMM2 CLI reference binary (needs HDF5 + cmake)
make f2py-ext                    # one-off extension build outside an editable install
make clean
uvx cibuildwheel                 # portable manylinux wheels locally (uses [tool.cibuildwheel])
streamlit run scripts/streamlit_app.py   # dashboard (needs the `dashboard` extra)
```

**You normally never need `make f2py-ext`.** meson-python's editable install rebuilds
`src/pyfastmm/_fastmm_ext*.so` automatically on `import pyfastmm` whenever a Fortran source or patch changed.

`tests/test_compatibility.py` **skips itself** if `build/cli/FaSTMM2` doesn't exist — run `make cli` once if
you're touching numerics and want the cross-check against upstream to actually execute.

Editable installs must be built **without** build isolation (`[tool.uv] no-build-isolation-package`):
meson-python bakes the absolute path of the `ninja` it saw at build time into the editable loader and reuses
it on every later import, so an ephemeral isolated build env leaves every subsequent import broken.

## Build architecture

`meson.build` is the single source of truth for the extension build; the `Makefile` target just wraps it.

1. `meson.build` declares `custom_target`s that copy individual upstream `.f90` files and apply a patch from
   `src/pyfastmm/_fortran/patches/` to the **copy** (`tools/apply_patch.py`). The pristine submodule sources stay
   untouched so `make cli` keeps building the unmodified upstream reference binary.
2. The patched copies plus the hand-written wrapper `src/pyfastmm/_fortran/fastmm2_f2py.f90` are handed to
   `tools/build_f2py_ext.py`, which shells out to `numpy.f2py -c` in a scratch dir and copies the resulting `.so`
   to the path meson expects (f2py `-c` can only write to cwd).
3. LAPACK/BLAS link flags come from `pkg-config --libs` (resolved in `meson.build`, passed through as a string),
   not bare `-llapack -lblas` — NixOS has no FHS `/usr/lib` to fall back on.

Only the 16 modules upstream's own CMakeLists compiles are in `f2py_sources`. `matvec.f90`, `matvec_old.f90`,
`matvec_new.f90`, and `octtree.f90` are dead alternates that redeclare the same module names as the files
actually used (`matvec_adaptive.f90`, `octtree_adaptive.f90`) and must not be added. `io.f90` (the only HDF5
user) is deliberately excluded — **the Python extension never links HDF5**; only the CLI does.

### The patches, and why they exist

Each patch in `src/pyfastmm/_fortran/patches/` is one reviewable concern; `mie-gil-release-n-threads.patch` is
chained on top of `mie-far-field-distance-fix.patch`'s output rather than merged into it.

- `gaussquad-f2py-ascii-fix` — f2py's crackfortran parser reads ASCII and chokes on UTF-8 math symbols in
  comments. Comment-only rewrite.
- `mie-far-field-distance-fix` — upstream's far-field radius `RR = 1.0d6 * real(k)` is dimensionally wrong;
  for absorbing media `exp(i*k*RR)` underflows to exact zero and the whole Mueller/Jones matrix comes back NaN
  while cross sections stay finite. Fixed to `RR = 1.0d6 / real(k)`. Also multiplies the far-field `Csca`/`q`
  integrands by `exp(+2*aimag(k)*RR)`: for complex k the propagator's `exp(-Im(k)*RR)` decay isn't cancelled by
  the real `RR**2` factor (Mueller/Jones divide out the full complex exponential; the Csca integral didn't),
  which suppressed `c_sca` by up to e^-600 ≈ 1e-260 in absorbing media. No-op for real k.
- `*-n-threads-threading` (solver, matvec_adaptive, T_matrix, orientation_averaging, mie) — thread `n_threads`
  (and the truncation choice) explicitly through every reachable `!$omp parallel` region, replacing the
  `omp_set_num_threads()` / module-global state upstream relied on. **This is what makes `!f2py threadsafe`
  (GIL release) on the three wrapper entry points correct**; without it, two Python threads racing to set
  different values would silently corrupt each other's physics.
- `solver-deep-rayleigh-gmres-scaling` — chained on `solver-n-threads-threading`'s output. In the
  deep-Rayleigh regime (x ≪ 1, touching spheres) the raw one-center system is numerically singular in double
  precision (measured rcond ~1e-60 at Nmax=13, x≈0.03): any solver's "converged" answer then depends on its
  iteration path, which broke rotational invariance (identical touching pairs along x vs y differed by up to
  12%) and made cross sections solver-path artifacts. GMRES now iterates the Jacobi-balanced similarity
  transform `S⁻¹(I−TA)S`, `S = diag(√|t_n|)` — exact in exact arithmetic, bounded conditioning (same device
  MSTM uses). Side effect: T-matrix elements no longer match the unpatched CLI bit-for-bit (only to GMRES
  iteration noise, ~1e-8 abs) — test_compatibility.py's T-matrix atol reflects that.

Anything that would reintroduce process-global Fortran state breaks concurrency; `tests/test_concurrency.py`
guards this by comparing threaded results against the same calls run serially.

## Python API layers

Three layers, each building on the one below — put shared physics logic at the lowest layer that needs it:

- `_fastmm.py` — `FaSTMM2`: thin, stateless wrapper over the extension. Geometry + `eps` + `k` in, dict out.
  Also the shared host-medium helpers (`eps_relative_to_medium`, `k_in_medium`, `_mstm_medium_factor`,
  `eps_from_index`, `resolve_n_threads`).
- `_cluster.py` — `FaSTMM`: ergonomic class API taking refractive indices and a vacuum wavelength, applying the
  medium transform, incidence rotation, and the cross-section convention. `version` (which solver) is a
  separate axis from `formulation` (0=STMM/1=FaSTMM/2=FaSTMM2 within a version) — two nested `match` checks,
  deliberately not one flat list. Stores each outcome on `self.results` as a validated pydantic model
  (`_results.py`: `SolveResult`/`TMatrixResult` — finiteness-checked, so NaN physics fails loudly at the
  source) while still *returning* the plain dict (`results.to_dict()`) for backward compatibility. pydantic is
  a core dependency for this reason.
- `_config.py` — `SweepConfig` (pydantic + TOML): wavelength sweeps. `run()` = `resolve_inputs()` +
  `run_resolved()`; the split exists so a distributed worker gets pickled arrays and needs no filesystem access.
  `run_resolved` is a `staticmethod` for that reason. `SolverConfig.sweep_threads` runs wavelengths
  concurrently in one process (ThreadPoolExecutor over the GIL-released solves; distinct axis from `n_threads`,
  the OpenMP count *inside* one solve — keep their product ≤ cpu_count).
- `_dask.py` — Dask-distributed sweeps, first-class but strictly optional (`pyfastmm[dask]` extra; the core
  never imports dask, all imports are lazy). `SweepConfig.run_dask()` / `run_resolved_dask()`: scatter
  `ResolvedSweepInputs` once, one task per wavelength, each task a one-wavelength `run_resolved()` so local
  and distributed share one code path. `devenv up` starts a local scheduler + 2 workers (tcp://127.0.0.1:8786)
  for the external-scheduler tier of `tests/test_dask.py`; those tests skip when nothing is listening, and all
  dask tests skip in CI (the `dask` extra is deliberately not part of `dev`).

### Physics conventions that bite

- Inputs are **permittivity** `eps = m**2`, not refractive index — matching upstream's `-eps_r`/`-eps_i` flags.
- FaSTMM2 has no medium input and no incident-angle input. A host medium is emulated by transforming both `eps`
  and `k`; a tilted beam is emulated by **rotating the cluster** (`R = Ry @ Rz` — the other order was measured
  ~1.3% off against MSTM when both angles are nonzero).
- Cross sections are **physical** by default (per unit irradiance in the host; a sphere in a host equals Mie
  with `m/n_host`, `Q_ext -> 2` for large absorbers). `cross_section_convention="mstm"` multiplies all four
  (`c_sca` included) by `_mstm_medium_factor(n_host) = Re(n_host)`, MSTM's medium mode. Raw `FaSTMM2.solve()`
  is always physical.
- Truncation: `truncation_formula` is 0 or 1 (ka-based formulas); `truncation_order` (>= 2) is an explicit
  per-sphere order and overrides it. The Fortran wrapper takes one integer (0/1 formula, >= 2 explicit
  order); `_fortran_truncation()` in `_fastmm.py` does the encoding.
- `n_threads` defaults to **1**, not "all cores", at every layer — callers commonly run sweeps under their own
  process parallelism and would otherwise oversubscribe. It accepts a positive int (absolute), a float in
  `(0, 1]` (fraction of `cpu_count`), or a negative value (offset from `cpu_count`); `0` raises.

## Release pipeline

`test.yml` runs on PRs/pushes to main. `wheels.yml` never runs automatically — trigger it via
`workflow_dispatch` or a `/build-wheels` PR comment (gated on write access). `publish.yml` is
`workflow_dispatch` only: python-semantic-release computes the version from Conventional Commits, tags, and
creates the GitHub Release; wheels then build **from that tag** and publish to PyPI via trusted publishing.

- The filename `publish.yml` is load-bearing (PyPI trusted publisher is configured against it).
- Tag-first: if the wheel build fails after a release, the tag/Release exist with no artifacts and need manual
  cleanup (`gh release delete <tag> --cleanup-tag`).
- All cibuildwheel config lives in `pyproject.toml`, not as `CIBW_*` env vars, so it's reproducible locally.
