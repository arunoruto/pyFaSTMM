{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

{
  # pyFaSTMM's Python extension (src/pyfastmm/_fastmm_ext*.so) is built by
  # meson-python (see meson.build / pyproject.toml's [build-system]),
  # which shells out to `numpy.f2py -c` (tools/build_f2py_ext.py) against
  # the git submodule at external/fastmm2/. That needs gfortran, LAPACK,
  # and OpenMP (gfortran's -fopenmp covers OpenMP directly, no separate
  # package) at build/link time -- provided below. HDF5 is only needed for
  # the standalone FaSTMM2 CLI reference binary (`make cli`), used as an
  # independent numerical reference by tests/test_compatibility.py -- the
  # Python extension itself never links HDF5 (external/fastmm2/src/io.f90,
  # the only file that uses it, isn't part of the extension's f2py
  # sources; see meson.build). meson, ninja, and meson-python itself are
  # Python-level build requirements pulled in via the `dev` extra in
  # pyproject.toml, installed into this shell's venv by
  # languages.python.uv.sync (allExtras below) rather than as Nix packages
  # -- pyproject.toml's [tool.uv] no-build-isolation-package = ["pyfastmm"]
  # requires them to live in *this* persistent venv (not an ephemeral
  # isolated build env), since meson-python's editable-install rebuild
  # hook records an absolute path to `ninja` the first time it builds and
  # reuses that same path on every later `import pyfastmm`.
  #
  # Once this shell is set up, `import pyfastmm` alone keeps the extension
  # up to date automatically -- `make f2py-ext`/`make cli` (both still
  # present in the Makefile) are only needed for one-off builds outside a
  # full editable install, e.g. producing a standalone .so or the CLI
  # binary without touching the venv.
  packages = [
    pkgs.gfortran
    pkgs.lapack
    pkgs.blas
    pkgs.hdf5-fortran
    pkgs.cmake
  ];

  enterShell = ''
    if [ ! -L "$DEVENV_ROOT/.venv" ]; then
        ln -s "$DEVENV_STATE/venv/" "$DEVENV_ROOT/.venv"
    fi
  '';

  languages.python = {
    enable = true;

    uv = {
      enable = true;
      sync = {
        enable = true;
        allExtras = true;
      };
    };

    libraries = with pkgs; [ zlib ];
  };
}
