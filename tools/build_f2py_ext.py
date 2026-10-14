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


def _driver_safe_ldflags(flags):
    """Translate linker-direct flags into compiler-driver-safe form.

    NIX_LDFLAGS is written for `ld` itself (nix's cc-wrapper normally
    consumes it), so it contains bare `-rpath <path>` pairs -- which
    gfortran's *driver* (what meson/f2py invoke with LDFLAGS) rejects
    with "unrecognized command-line option '-rpath'". Rewrite each pair
    as `-Wl,-rpath,<path>`; everything else (-L..., -l...) passes through
    unchanged. Without this, any rebuild inside a devenv/nix shell that
    exports NIX_LDFLAGS fails at meson's compiler sanity check.
    """
    toks = flags.split()
    out = []
    i = 0
    while i < len(toks):
        if toks[i] == "-rpath" and i + 1 < len(toks):
            out.append(f"-Wl,-rpath,{toks[i + 1]}")
            i += 2
        else:
            out.append(toks[i])
            i += 1
    return " ".join(out)


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
        ldflags = f"{_driver_safe_ldflags(nix_ldflags)} {ldflags}"
    env["LDFLAGS"] = f"{env['LDFLAGS']} {ldflags}" if env.get("LDFLAGS") else ldflags

    # Opt-in extra Fortran flags for *local* builds (e.g.
    # PYFASTMM_F90FLAGS="-march=native"). Never set in CI/cibuildwheel --
    # published wheels must stay portable across x86_64 machines; this is
    # for tuning a build that only ever runs on the machine that built it.
    # f2py's meson backend already builds buildtype=release (-O3), so
    # there's nothing to gain by re-adding -O flags here.
    extra_f90flags = os.environ.get("PYFASTMM_F90FLAGS", "").strip()
    f90flags = f"-fopenmp {extra_f90flags}".strip()

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
            f"--f90flags={f90flags}",
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
