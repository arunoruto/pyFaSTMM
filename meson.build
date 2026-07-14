project(
  'pyfastmm',
  'c',
  'fortran',
  version: '0.1.0',
  meson_version: '>=1.1.0',
)

py = import('python').find_installation(pure: false)

ext_suffix = run_command(
  py, '-c', 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))',
  check: true,
).stdout().strip()

fastmm2_dir = meson.project_source_root() / 'external' / 'fastmm2' / 'src'
wrapper_dir = meson.project_source_root() / 'src' / 'pyfastmm' / '_fortran'
patches_dir = wrapper_dir / 'patches'

# --- Patched copy of one legacy source file ---
#
# numpy.f2py's Fortran parser (crackfortran) reads source files as ASCII
# and cannot decode gaussquad.f90's UTF-8 math symbols (present only in
# comments -- integral/summation signs, alpha/beta/delta, <=). The patch
# is a pure comment rewrite (ASCII transliteration) -- no algorithm or
# runtime behavior change -- applied only to a scratch copy used for this
# extension. The standalone FaSTMM2 CLI binary keeps building from the
# literal unmodified upstream source.
gaussquad_patched = custom_target(
  'gaussquad-patched',
  input: [
    fastmm2_dir / 'gaussquad.f90',
    patches_dir / 'gaussquad-f2py-ascii-fix.patch',
  ],
  output: 'gaussquad.f90',
  command: [py, files('tools' / 'apply_patch.py'), '@INPUT0@', '@INPUT1@', '@OUTPUT@'],
)

# --- The f2py extension itself ---
#
# Built via numpy.f2py's own "-c" compile mode (the same command
# `make f2py-ext` runs for local dev, see tools/build_f2py_ext.py for why
# a driver script is needed rather than a plain command: line -c always
# writes to the current working directory, with no way to target an
# arbitrary meson-controlled output path directly, and for the
# -fopenmp/-llapack/-lblas flags FaSTMM2 needs that MSTM never did).
#
# Only the 16 modules upstream's own CMakeLists.txt compiles are included
# -- 4 more files in external/fastmm2/src/ (matvec.f90, matvec_old.f90,
# matvec_new.f90, octtree.f90) are dead alternates that redeclare the same
# module names (`matvec`, `octtree`) as the ones actually used
# (matvec_adaptive.f90, octtree_adaptive.f90) and must not be compiled
# alongside them. io.f90 (HDF5 file I/O) is deliberately excluded too --
# it's the only file that uses HDF5, and nothing else has a compile-time
# dependency on it, so the Python extension never needs HDF5 at all; only
# main.f90 (replaced here by fastmm2_f2py.f90) used it upstream.
#
# Grows as more of fastmm2_f2py.f90 is written -- keep in sync with the
# Makefile's F2PY_ONLY.
f2py_only = [
  'fastmm2_solve_fixed', 'fastmm2_solve_averaged', 'fastmm2_compute_tmatrix',
]

f2py_sources = [
  fastmm2_dir / 'common.f90',
  gaussquad_patched,
  fastmm2_dir / 'wigner.f90',
  fastmm2_dir / 'integration_points.f90',
  fastmm2_dir / 'possu.f90',
  fastmm2_dir / 'sfunctions.f90',
  fastmm2_dir / 'translations.f90',
  fastmm2_dir / 'mie.f90',
  fastmm2_dir / 'geometry.f90',
  fastmm2_dir / 'octtree_adaptive.f90',
  fastmm2_dir / 'interpolation.f90',
  fastmm2_dir / 'matvec_adaptive.f90',
  fastmm2_dir / 'solver.f90',
  fastmm2_dir / 'orientation_averaging.f90',
  fastmm2_dir / 'T_matrix.f90',
  wrapper_dir / 'fastmm2_f2py.f90',
]

fastmm_ext = custom_target(
  '_fastmm_ext',
  input: f2py_sources,
  output: '_fastmm_ext' + ext_suffix,
  command: [
    py, files('tools' / 'build_f2py_ext.py'),
    '@OUTPUT@', ','.join(f2py_only), '@INPUT@',
  ],
  install: true,
  install_dir: py.get_install_dir(pure: false) / 'pyfastmm',
)

# --- Pure Python package ---

python_sources = [
  'src' / 'pyfastmm' / '__init__.py',
  'src' / 'pyfastmm' / '_build.py',
  'src' / 'pyfastmm' / '_fastmm.py',
  'src' / 'pyfastmm' / '_config.py',
]

py.install_sources(
  python_sources,
  subdir: 'pyfastmm',
)
