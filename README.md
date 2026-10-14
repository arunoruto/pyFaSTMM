# pyFaSTMM

Python bindings for [FaSTMM2](https://bitbucket.org/planetarysystemresearch/fastmm2)
(Fast Superposition T-Matrix Method), a Multi-Level Fast Multipole Method
(MLFMM) accelerated solver for electromagnetic scattering from clusters of
spheres.

Geometry goes in and Mueller/Jones/cross-section/T-matrix data comes out
as plain NumPy arrays -- no input/output files required (unlike the
upstream CLI, which reads/writes HDF5).

## Installation

```bash
pip install pyfastmm
pip install "pyfastmm[dask]"   # optional: distributed wavelength sweeps
```

Binary wheels are published for **Linux x86_64** (CPython 3.11-3.14). On
other platforms pip builds from the sdist, which needs a Fortran compiler
(gfortran), LAPACK/BLAS and `pkg-config` (e.g. on Debian/Ubuntu:
`apt install gfortran liblapack-dev libblas-dev pkg-config`).

## Quick start

```python
from pyfastmm import FaSTMM

coords = [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]  # sphere centres (um)
radii = [1.0, 1.0]                            # sphere radii (um)

cluster = FaSTMM(
    coords,
    radii,
    refractive_index=1.6 + 0.01j,          # particles, vacuum-referenced
    medium_refractive_index=1.33 + 0.0j,   # host medium (default: vacuum)
)
result = cluster.solve(
    wavelength_vacuum=0.55,  # same length unit as coords/radii
    N_theta=181,
    N_phi=32,
    n_threads=4,             # OpenMP threads for this solve (default 1)
)
print(result["c_ext"], result["c_abs"], result["c_sca"], result["asymmetry"])
print(result["mueller"].shape)   # (N_theta*N_phi, 18): phi, theta, P11..P44
```

Notes:

- **Cross sections in a host medium are physical by default**: power per
  unit incident irradiance in the host. A single sphere equals Mie theory
  with the relative index `m / n_host` and the size parameter in the host.
  Pass `cross_section_convention="mstm"` to get MSTM's medium-mode values
  instead (normalised by the vacuum irradiance, i.e. `Re(n_host)` times
  larger). Both agree in vacuum.
- **`n_threads` defaults to 1** (pyfastmm 1.0.0 used all OpenMP cores).
  It accepts a positive count, a negative offset from `os.cpu_count()`
  (`-2` leaves two CPUs free) or a float fraction (`0.5`). The solve
  releases the GIL, so independent solves can also run in Python threads.
- **Truncation**: `truncation_formula=0` (default, `floor(ka + 3 ka^(1/3))`,
  at least 6) or `1` (a more conservative formula, useful for touching
  small spheres); `truncation_order=N` sets an explicit per-sphere order
  and overrides the formula.
- `incident_polar_deg`/`incident_azimuthal_deg` tilt the incident beam
  (emulated by rotating the cluster); `N_ave > 0` averages over
  orientations; `cluster.compute_tmatrix(...)` returns the cluster
  T-matrix.

The low-level `FaSTMM2` class takes permittivities (`eps = m**2`) and a
(complex) wavenumber directly; see its docstrings.

## Wavelength sweeps from TOML

```toml
# sweep.toml
[particles]
positions_file = "cluster.dat"   # columns x y z r (PyFracVAL format works)
scale = 1e-9
refractive_index = [1.5, 0.01]   # or refractive_index_file = "nk.txt"

[medium]
refractive_index = [1.33, 0.0]

[wavelengths]
start = 0.4
stop = 0.8
num = 21
scale = 1e-6

[solver]
N_theta = 181
N_phi = 32
n_threads = 1
sweep_threads = 4                       # wavelengths solved concurrently
truncation_formula = 0                  # or truncation_order = 12
cross_section_convention = "physical"   # or "mstm"
```

```python
from pyfastmm import load_config

cfg = load_config("sweep.toml")
results = cfg.run(base_dir=".")   # one dict per wavelength
# Distributed over Dask (needs pyfastmm[dask]):
results = cfg.run_dask(base_dir=".", scheduler_address="tcp://127.0.0.1:8786")
```

## Scope

Spherical (Lorenz-Mie) monomers only -- not the precomputed
per-monomer-T-matrix ("arbitrarily-shaped constituent particles") input
the upstream CLI also supports. The bundled FaSTMM2 sources carry a few
patches (`src/pyfastmm/_fortran/patches/`), among them fixes for the
far field in absorbing host media and for the solver conditioning of
touching spheres at small size parameters; see `CHANGELOG.md`.

## Development

```bash
git clone --recurse-submodules https://github.com/arunoruto/pyFaSTMM.git
cd pyFaSTMM
uv sync --extra dev   # builds the extension (needs gfortran, LAPACK/BLAS, pkg-config)
make cli              # optional: upstream CLI, for tests/test_compatibility.py (needs cmake, HDF5)
uv run pytest tests/
```

`devenv shell` (see `devenv.nix`) provides all build dependencies.

## Citation

If you use pyFaSTMM, please cite the FaSTMM2 papers:

- J. Markkanen and A. J. Yuffa, "Fast superposition T-matrix solution for
  clusters with arbitrarily-shaped constituent particles", JQSRT 189,
  181-188 (2017), [doi:10.1016/j.jqsrt.2016.11.004](https://doi.org/10.1016/j.jqsrt.2016.11.004)
- J. Markkanen, "Surface-integral-equation solution for solid particles
  with wavelength-scale surface roughness", JQSRT 341, 109467 (2025),
  [doi:10.1016/j.jqsrt.2025.109467](https://doi.org/10.1016/j.jqsrt.2025.109467)

and pyFaSTMM itself (see `CITATION.cff`).

## License

pyFaSTMM is MIT licensed. The bundled FaSTMM2 Fortran code keeps its own
licences (BSD-3-Clause, MIT, and the Zhang & Jin notice), and the wheels
bundle LAPACK/BLAS and the GCC runtime; see `LICENSE` and `LICENSES/`.
