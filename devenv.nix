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
    # meson.build resolves LAPACK/BLAS link flags via `pkg-config --libs`
    # (see meson.build's comment) instead of relying on NIX_LDFLAGS.
    pkgs.pkg-config
    # For building portable manylinux wheels locally via `uvx cibuildwheel`
    # (see .github/workflows/wheels.yml / pyproject.toml's
    # [tool.cibuildwheel]) -- a wheel built directly in this Nix shell
    # links against this system's glibc/dynamic-linker paths
    # (/nix/store/...), which aren't portable off this machine; the
    # manylinux container is what makes cibuildwheel's output portable.
    # Podman over Docker: daemonless, no persistent system service needed
    # for a project-scoped `packages` addition like this one. Note: fully
    # rootless container networking/user namespaces on NixOS sometimes
    # still wants one small *system-level* option
    # (`virtualisation.podman.enable = true`, for subuid/subgid ranges)
    # that a project devenv can't self-contain -- if `podman run` fails
    # with a user-namespace error, that's the fix; most modern NixOS
    # systems already have it.
    pkgs.podman
  ];

  # Podman on NixOS needs a registries.conf (where to resolve short image
  # names like "docker.io/library/hello-world" from) and a policy.json
  # (image signature verification policy) -- normally provided system-wide
  # by `virtualisation.podman.enable`, which a project devenv can't set.
  #
  # registries.conf has a real env var override (below): search docker.io
  # by default. policy.json does not -- containers/image hardcodes
  # $HOME/.config/containers/policy.json (or /etc/containers/policy.json)
  # with no override, so enterShell writes one there instead, only if
  # absent (never clobbers an existing user config). "insecureAccept
  # Anything" (accept all images, no signature verification) is fine for
  # a local build tool pulling public manylinux images -- not meant as a
  # security policy for anything sensitive.
  env.CONTAINERS_REGISTRIES_CONF = pkgs.writeText "registries.conf" ''
    unqualified-search-registries = ["docker.io"]
  '';

  enterShell = ''
    if [ ! -L "$DEVENV_ROOT/.venv" ]; then
        ln -s "$DEVENV_STATE/venv/" "$DEVENV_ROOT/.venv"
    fi
    if [ ! -f "$HOME/.config/containers/policy.json" ]; then
        mkdir -p "$HOME/.config/containers"
        echo '{ "default": [{ "type": "insecureAcceptAnything" }] }' \
            > "$HOME/.config/containers/policy.json"
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
