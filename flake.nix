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

      pyprojectVersion = (lib.importTOML ./pyproject.toml).project.version;
    in
    {
      packages = forEachSupportedSystem (
        { pkgs, system }:
        rec {
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
            # upstream's CMakeLists.txt declares a `cmake_minimum_required`
            # older than current CMake tolerates without an explicit
            # opt-in -- confirmed via a real build failure ("Configuring
            # incomplete, errors occurred!"), matching the exact same fix
            # already applied to the CLI reference binary built for the
            # test.yml/wheels.yml pipeline (see Makefile's cli target).
            cmakeFlags = [
              "--no-warn-unused-cli"
              (lib.cmakeFeature "CMAKE_POLICY_VERSION_MINIMUM" "3.5")
            ];
            nativeBuildInputs = [ pkgs.cmake pkgs.gfortran ];
            buildInputs = [ pkgs.lapack pkgs.blas pkgs.hdf5-fortran ];
            installPhase = ''
              mkdir -p $out/bin
              cp FaSTMM2 $out/bin/
            '';

            meta = with lib; {
              description = "Fast Superposition T-Matrix Method (MLFMM-accelerated)";
              homepage = "https://bitbucket.org/planetarysystemresearch/fastmm2";
              platforms = platforms.unix;
              maintainers = with maintainers; [ arunoruto ];
              mainProgram = "FaSTMM2";
            };
          };

          # The Python bindings, as a real Nix derivation -- meson-python
          # drives the exact same meson.build/tools/build_f2py_ext.py that
          # `uv sync` triggers locally, so the compiled extension is the
          # same numpy.f2py output either way.
          #
          # external/fastmm2 is a git submodule, and Nix flakes only ever
          # see a git-tracked copy of the flake's own source (via `self`)
          # -- a submodule's checked-out content is invisible to that copy
          # (confirmed directly for pyMSTM's identical setup: `self`'s
          # equivalent submodule directory doesn't even exist, since git
          # records a submodule as a single gitlink entry, not the files
          # inside it). Rather than fetch the source a second time,
          # postPatch repopulates external/fastmm2 from the *same*
          # fastmm2-src input the CLI derivation above already uses.
          pyfastmm = pkgs.python3Packages.buildPythonPackage {
            pname = "pyfastmm";
            version = pyprojectVersion;
            pyproject = true;
            src = self;

            postPatch = ''
              rm -rf external/fastmm2
              mkdir -p external
              cp -r ${fastmm2-src} external/fastmm2
              chmod -R u+w external/fastmm2
            '';

            # nixpkgs' meson-python setup hook pre-runs `meson setup` as
            # its own configurePhase and hands `pypaBuildHook` a
            # `-Cbuild-dir=` pointing at it -- on this nixpkgs/meson-python
            # pairing that pre-configured dir confuses `python -m build`
            # into treating the *build* dir as the source root ("Source
            # .../build does not appear to be a Python project"). Skipping
            # that pre-configure step lets meson-python's own build
            # backend invoke meson itself from the real source root,
            # which is its normal, fully self-contained mode of operation.
            dontUseMesonConfigure = true;

            build-system = [ pkgs.python3Packages.meson-python ];
            nativeBuildInputs = [
              pkgs.meson
              pkgs.ninja
              pkgs.gfortran
              pkgs.gnupatch # applies the f2py-compatibility patch under src/pyfastmm/_fortran/patches/
            ];
            # LAPACK/BLAS: tools/build_f2py_ext.py links -llapack -lblas
            # directly (not declared as a meson dependency() -- see the
            # sdist-build comment in .github/workflows/wheels.yml). OpenMP
            # support (-fopenmp) needs no separate package: gfortran links
            # its own bundled libgomp automatically.
            buildInputs = [ pkgs.lapack pkgs.blas ];
            dependencies = [ pkgs.python3Packages.numpy ];

            # The test suite cross-checks against the standalone
            # `fastmm2` CLI binary (see the derivation above), which
            # isn't wired up as a build input here, so skip pytest during
            # the Nix build; test.yml's CI job covers that.
            doCheck = false;
            pythonImportsCheck = [ "pyfastmm" ];

            meta = with lib; {
              description = "Python bindings for FaSTMM2 (Fast Superposition T-Matrix Method, MLFMM-accelerated)";
              homepage = "https://github.com/arunoruto/pyFaSTMM";
              license = licenses.mit;
              platforms = platforms.unix;
              maintainers = with maintainers; [ arunoruto ];
            };
          };

          default = pyfastmm;
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
