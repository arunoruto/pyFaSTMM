#!/usr/bin/env python3
"""Build the pyfastmm f2py extension and place it at the path meson expects.

numpy.f2py's "-c" compile mode always writes its output to the current
working directory, using a filename that includes the Python/platform ABI
tag -- there's no flag to make it write directly to an arbitrary path (see
--build-dir, which only controls where *intermediate* meson files land,
not the final .so). This driver runs the exact same command
`make f2py-ext` runs locally, in a scratch directory, then copies the
resulting extension module to the exact output path meson's custom_target
requires.

FaSTMM2 needs OpenMP (real parallel regions in several modules) and LAPACK
(plain external DGETRF/DGETRI/ZGETRF/ZGETRI/ZGESVD/DSTEV calls, no Fortran
module interface -- link-time only) -- both passed through to f2py below.
In a properly entered devenv/nix shell, `-llapack`/`-lblas` resolve via the
wrapped compiler's NIX_LDFLAGS (from devenv.nix's `packages`); outside one,
LAPACK/BLAS need to be resolvable through the normal linker search path
(e.g. LIBRARY_PATH).

Usage: build_f2py_ext.py <output_path> <comma_separated_only_names> <source...>
"""

import glob
import os
import shutil
import subprocess
import sys
import tempfile


def main():
    output_path = os.path.abspath(sys.argv[1])
    only_names = sys.argv[2].split(",")
    # f2py's meson backend always runs relative to the scratch cwd below;
    # resolve source paths to absolute first so they still resolve there.
    sources = [os.path.abspath(s) for s in sys.argv[3:]]

    env = dict(os.environ)
    ldflags = "-Wl,-z,noexecstack -fopenmp"
    env["LDFLAGS"] = f"{env['LDFLAGS']} {ldflags}" if env.get("LDFLAGS") else ldflags

    with tempfile.TemporaryDirectory() as scratch:
        cmd = [
            sys.executable,
            "-m",
            "numpy.f2py",
            "-c",
            *sources,
            "only:",
            *only_names,
            ":",
            "-m",
            "_fastmm_ext",
            "--f90flags=-fopenmp",
            "-lgomp",
            "-llapack",
            "-lblas",
        ]
        subprocess.run(cmd, cwd=scratch, env=env, check=True)
        matches = glob.glob(os.path.join(scratch, "_fastmm_ext*.so")) + glob.glob(
            os.path.join(scratch, "_fastmm_ext*.pyd")
        )
        if not matches:
            sys.exit("f2py build did not produce _fastmm_ext*.so/.pyd")
        shutil.copy(matches[0], output_path)


if __name__ == "__main__":
    main()
