# pyFaSTMM

Python bindings for [FaSTMM2](https://bitbucket.org/planetarysystemresearch/fastmm2)
(Fast Superposition T-Matrix Method), a Multi-Level Fast Multipole Method
(MLFMM) accelerated solver for electromagnetic scattering from clusters of
spheres.

Geometry goes in and Mueller/Jones/cross-section/T-matrix data comes out
as plain NumPy arrays -- no input/output files required (unlike the
upstream CLI, which reads/writes HDF5).

## Quick start

```python
from pyfastmm import FaSTMM2

f = FaSTMM2()

coords = [[-1.5, 0.0, 0.0], [1.5, 0.0, 0.0]]
radii = [1.0, 1.0]
eps = [3.0 + 0.1j, 3.0 + 0.1j]  # permittivity = (refractive index)**2
k = 1.2  # wavenumber

result = f.solve(coords, radii, eps, k, N_theta=91, N_phi=16)
print(f"Cext = {result['c_ext']:.6f}")
print(f"Csca = {result['c_sca']:.6f}")
print(f"Mueller matrix shape: {result['mueller'].shape}")
```

## Features

- **Fixed-orientation scattering**: Mueller and Jones matrices, extinction/
  absorption/scattering cross sections, asymmetry parameter
- **Orientation-averaged scattering**: Halton-sequence orientation averaging
- **T-matrix computation**: full T-matrix output (`Taa`, `Tab`, `Tba`, `Tbb`)
  for a cluster of spheres
- **MLFMM acceleration**: `formulation=2` (FaSTMM2, default) scales to large
  clusters far better than direct superposition T-matrix (`formulation=0`)

Current scope (v1): spherical (Lorenz-Mie) monomers only -- not the
precomputed-per-monomer-T-matrix ("arbitrarily-shaped constituent
particles") input the upstream CLI also supports.

## Installation

```bash
# Clone with submodules
git clone --recurse-submodules https://github.com/arunoruto/pyFaSTMM.git
cd pyFaSTMM

# Build the extension (requires gfortran, LAPACK; see devenv.nix/flake.nix)
make f2py-ext

# Install in development mode
pip install -e .
```

Note: unlike the upstream CLI, the Python extension itself never links
HDF5 -- only `make cli` (the standalone reference binary used by the
compatibility tests) needs it.

## Requirements

- Python >= 3.12
- NumPy >= 1.25
- gfortran (GCC), LAPACK/BLAS

## Running tests

```bash
pip install -e ".[dev]"
make cli  # optional: builds the CLI reference binary for compatibility tests
pytest tests/
```

## Dashboard

```bash
pip install -e ".[dashboard]"
streamlit run scripts/streamlit_app.py
```

## License

MIT. The bundled FaSTMM2 Fortran code retains its original (BSD-3-Clause/
MIT) licenses -- see `LICENSE` and `external/fastmm2/` for details.
