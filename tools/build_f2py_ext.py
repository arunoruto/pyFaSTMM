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
The LAPACK/BLAS link flags are resolved by the caller via pkg-config (see
meson.build) and passed in as `link_flags` -- explicit `-L<store path>
-llapack` rather than the old bare `-llapack -lblas`, which only linked
outside a Nix shell if NIX_LDFLAGS happened to be set (NixOS has no FHS
/usr/lib for the plain linker search path to fall back on). Falls back to
the bare flags if invoked directly with an empty link_flags argument.

Usage: build_f2py_ext.py <output_path> <comma_separated_only_names> <link_flags> <source...>
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
    link_flags = sys.argv[3].split() if sys.argv[3].strip() else ["-llapack", "-lblas"]
    # f2py's meson backend always runs relative to the scratch cwd below;
    # resolve source paths to absolute first so they still resolve there.
    sources = [os.path.abspath(s) for s in sys.argv[4:]]

    env = dict(os.environ)
    ldflags = "-Wl,-z,noexecstack -fopenmp"
    nix_ldflags = env.get("NIX_LDFLAGS", "")
    if nix_ldflags:
        ldflags = f"{nix_ldflags} {ldflags}"
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
            *link_flags,
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
