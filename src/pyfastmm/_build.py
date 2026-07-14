"""Build helper for compiling the pyfastmm f2py extension.

Usage:
    python -m pyfastmm._build

Requires gfortran, LAPACK, numpy (with f2py), meson, and ninja (meson/ninja
are invoked internally by numpy.f2py on Python >=3.12), and the fastmm2
submodule at external/fastmm2/.
"""

import glob
import os
import subprocess
import sys


def build_library():
    """Compile the f2py extension via `make f2py-ext`."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    fastmm2_submodule = os.path.join(project_root, "external", "fastmm2", "src")
    if not os.path.isdir(fastmm2_submodule):
        sys.exit(
            "fastmm2 submodule not found at external/fastmm2/\n"
            "Initialize it with:\n"
            "  git submodule update --init --recursive"
        )

    makefile = os.path.join(project_root, "Makefile")
    if not os.path.isfile(makefile):
        sys.exit(f"Makefile not found at {makefile}")

    result = subprocess.run(
        ["make", "-C", project_root, "f2py-ext"],
        capture_output=False,
    )

    if result.returncode != 0:
        sys.exit(result.returncode)

    ext_glob = os.path.join(project_root, "src", "pyfastmm", "_fastmm_ext*.so")
    matches = glob.glob(ext_glob)
    if matches:
        print(f"Built: {matches[0]}")
    else:
        sys.exit("Build succeeded but _fastmm_ext*.so not found")


if __name__ == "__main__":
    build_library()
