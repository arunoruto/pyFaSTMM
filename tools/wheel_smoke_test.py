"""Post-build smoke test for a freshly built pyfastmm wheel.

Run by cibuildwheel's `test-command` (see [tool.cibuildwheel] in
pyproject.toml) inside the isolated venv it creates for each wheel --
i.e. against the *installed wheel*, not the source tree. Actually solving
a tiny two-sphere cluster (rather than a bare `import pyfastmm`) is the
point: it exercises the compiled _fastmm_ext extension end to end,
catching a wheel that imports fine but whose Fortran extension is
missing, mislinked, or miscompiled for the target Python/platform.

Lives in a real file instead of an inline `python -c "..."` in the
workflow -- see pyMSTM's identical tools/wheel_smoke_test.py for why
(YAML folded scalars turned newlines into spaces, corrupting the
one-liner into an IndentationError).
"""

import pyfastmm


def main() -> None:
    f = pyfastmm.FaSTMM2()
    result = f.solve(
        coords=[[-1.5, 0, 0], [1.5, 0, 0]],
        radii=[1.0, 1.0],
        eps=[2.25 + 0.03j, 2.25 + 0.03j],
        k=1.0,
        N_theta=11,
        N_phi=1,
    )

    c_ext = result["c_ext"]
    assert c_ext > 0, f"expected positive C_ext, got {c_ext!r}"
    print(f"pyfastmm {pyfastmm.__version__} wheel smoke test OK (C_ext = {c_ext:.6g})")


if __name__ == "__main__":
    main()
