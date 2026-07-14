{
  description = "Python bindings for FaSTMM2 (Fast Superposition T-Matrix Method, MLFMM-accelerated)";

  inputs = {
    nixpkgs.url = "https://flakehub.com/f/NixOS/nixpkgs/0";
    fastmm2-src = {
      url = "git+https://bitbucket.org/planetarysystemresearch/fastmm2?rev=4b56dc5b30333f3358e205ba8f88a03ee4d2bb3b";
      flake = false;
    };
  };

  outputs =
    { self, nixpkgs, fastmm2-src, ... }@inputs:
    let
      inherit (nixpkgs) lib;

      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "aarch64-darwin"
      ];

      forEachSupportedSystem =
        f:
        lib.genAttrs supportedSystems (
          system:
          f {
            inherit system;
            pkgs = import nixpkgs {
              inherit system;
              config.allowUnfree = true;
            };
          }
        );
    in
    {
      packages = forEachSupportedSystem (
        { pkgs, system }:
        {
          # The standalone FaSTMM2 CLI, built straight from the upstream
          # CMakeLists.txt against the submodule source. This is *not*
          # part of the Python package -- it's an independent numerical
          # reference used by tests/test_compatibility.py (see meson.build
          # for why the Python extension itself skips io.f90/HDF5
          # entirely). Unlike pyMSTM's MSTM CLI, this one needs LAPACK and
          # HDF5(Fortran) at build/link time, matching upstream's own
          # CMakeLists.txt FIND_PACKAGE calls.
          fastmm2 = pkgs.stdenv.mkDerivation {
            pname = "fastmm2";
            version = "unstable-2025";
            src = fastmm2-src + "/src";
            nativeBuildInputs = [ pkgs.cmake pkgs.gfortran ];
            buildInputs = [ pkgs.lapack pkgs.blas pkgs.hdf5-fortran ];
            installPhase = ''
              mkdir -p $out/bin
              cp FaSTMM2 $out/bin/
            '';
          };

          # Note: pyFaSTMM's Python extension (src/pyfastmm/_fastmm_ext*.so)
          # is not built as a Nix derivation -- it's built by meson-python
          # (see meson.build / pyproject.toml) against the local git
          # submodule at external/fastmm2/. gfortran/lapack/blas below
          # cover what that build needs (HDF5 is deliberately not linked
          # into the extension, only into the `fastmm2` CLI derivation
          # above). Unlike devenv.nix, this plain flake shell doesn't run
          # `uv sync` automatically -- run `uv sync --all-extras` once
          # after entering (pulls in meson/ninja/meson-python from the
          # `dev` extra; see pyproject.toml's [tool.uv]
          # no-build-isolation-package comment for why those need to land
          # in the persistent venv, not an ephemeral build env), then
          # `import pyfastmm` rebuilds the extension automatically
          # whenever its Fortran sources change.
        }
      );

      devShells = forEachSupportedSystem (
        { pkgs, system }:
        {
          default = pkgs.mkShellNoCC {
            packages = with pkgs; [
              self.formatter.${system}
              gfortran
              lapack
              blas
              hdf5-fortran
              cmake
            ]
            ++ lib.optionals (!pkgs.stdenv.isDarwin) [
              self.packages.${system}.fastmm2
            ];
          };
        }
      );

      formatter = forEachSupportedSystem ({ pkgs, ... }: pkgs.nixfmt);
    };
}
