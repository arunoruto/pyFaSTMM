#!/usr/bin/env python3
"""Copy a source file and apply a patch to the copy.

Used by meson.build to produce a patched legacy-source copy needed to
work around a real numpy.f2py parser bug (see
src/pyfastmm/_fortran/patches/*.patch for what and why) without touching
the upstream vendored sources the CLI binary also builds from.

Usage: apply_patch.py <source> <patch_file> <output>
"""

import shutil
import subprocess
import sys


def main():
    src, patch_file, dest = sys.argv[1:4]
    shutil.copy(src, dest)
    with open(patch_file, "rb") as f:
        subprocess.run(["patch", dest], stdin=f, check=True)


if __name__ == "__main__":
    main()
