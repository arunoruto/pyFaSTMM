# Makefile for building the pyfastmm f2py extension and the standalone
# FaSTMM2 CLI binary. Requires the fastmm2 submodule at external/fastmm2/

BUILD_DIR = build
FASTMM2 = external/fastmm2/src

CLI_BUILD_DIR = $(BUILD_DIR)/cli
CLI_OUTPUT = $(CLI_BUILD_DIR)/FaSTMM2

# --- f2py extension (the Python-facing pyfastmm backend) ---
#
# The actual build recipe (which sources, the one patch needed to work
# around a real numpy.f2py parser bug, the `only:` function list, the
# -fopenmp/-llapack/-lblas flags) lives in meson.build -- this target is a
# thin wrapper so `make f2py-ext` still works as a quick one-off, without
# needing a full `pip install -e .`. Normal local development doesn't need
# this at all: meson-python's editable install (see pyproject.toml's
# [tool.uv] no-build-isolation-package) rebuilds the extension
# automatically on `import pyfastmm` whenever a source file changed.
F2PY_MESON_BUILD_DIR = $(BUILD_DIR)/f2py-meson

.PHONY: f2py-ext
f2py-ext:
	test -f $(F2PY_MESON_BUILD_DIR)/build.ninja || meson setup $(F2PY_MESON_BUILD_DIR)
	meson compile -C $(F2PY_MESON_BUILD_DIR)
	rm -f src/pyfastmm/_fastmm_ext*.so
	cp $(F2PY_MESON_BUILD_DIR)/_fastmm_ext*.so src/pyfastmm/

# --- Standalone CLI binary (independent numerical reference) ---
#
# Built via CMake from the *unmodified* upstream submodule source
# (including io.f90/HDF5, unlike the Python extension), exactly as
# upstream's own CMakeLists.txt specifies -- used only as an independent
# reference by tests/test_compatibility.py, never shipped in the Python
# package.
.PHONY: cli
cli:
	mkdir -p $(CLI_BUILD_DIR)
	cd $(CLI_BUILD_DIR) && cmake ../../$(FASTMM2) -DCMAKE_BUILD_TYPE=Release -DCMAKE_POLICY_VERSION_MINIMUM=3.5
	cmake --build $(CLI_BUILD_DIR)

.PHONY: all clean

all: f2py-ext cli

clean:
	rm -rf $(BUILD_DIR)
	rm -f src/pyfastmm/_fastmm_ext*.so
	rm -rf .mesonpy-*
	rm -f *.mod *.o
