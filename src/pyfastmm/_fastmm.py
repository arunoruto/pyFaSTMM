"""
Python bindings for FaSTMM2 (Fast Superposition T-Matrix Method, MLFMM-accelerated).

Uses f2py to call into the compiled extension built from the legacy FaSTMM2
Fortran sources plus src/pyfastmm/_fortran/fastmm2_f2py.f90 (see the
Makefile's f2py-ext target). Unlike the upstream CLI, no files are read or
written -- geometry goes in and Mueller/Jones/cross-section/T-matrix data
comes out as plain numpy arrays. FaSTMM2 keeps no solver state between
calls (unlike MSTM's module-level globals), so this wrapper is stateless:
each call to solve()/compute_tmatrix() is self-contained.

v1 covers spherical (Lorenz-Mie) monomers only -- not the
precomputed-per-monomer-T-matrix ("arbitrarily-shaped constituent
particles") input path upstream also supports.

Note on units: FaSTMM2 works natively in electric *permittivity*
(eps = m**2, where m = n + i*k is the complex refractive index), not
refractive index directly -- matching upstream's own -eps_r/-eps_i CLI
flags and geometry.h5 param_r/param_i datasets. If you have a refractive
index, square it yourself before calling.

Usage:
    import pyfastmm
    import numpy as np

    f = pyfastmm.FaSTMM2()

    coords = [[-1.5, 0, 0], [1.5, 0, 0]]
    radii = [1.0, 1.0]
    eps = np.array([3.0 + 0.1j, 3.0 + 0.1j])

    result = f.solve(coords, radii, eps, k=1.2, N_theta=91, N_phi=16)
    print("Mueller matrix:", result["mueller"].shape)
    print("Cext:", result["c_ext"])
"""

from __future__ import annotations

import os
from typing import Literal

import numpy as np

try:
    from . import _fastmm_ext as _ext
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pyfastmm's compiled extension is not built. Run `make f2py-ext` "
        "or `uv sync` to build it."
    ) from exc


def resolve_n_threads(n_threads: int | float) -> int:
    """Resolve a thread count. Three input shapes are supported:

    - A ``float`` in ``(0, 1]``: a *fraction* of ``os.cpu_count()`` --
      e.g. ``0.5`` (or ``1/2``) means "half the machine's CPUs", ``1.0``
      means "all of them". Rounded to the nearest int, clamped to at
      least 1. This only applies to actual ``float`` values -- the int
      ``1`` is *not* treated as a 100% fraction, it means exactly one
      thread (see the negative case below for why that distinction
      matters: an int is always an absolute/offset count, a float in
      this range is always a fraction).
    - A positive ``int`` (or a ``float`` outside ``(0, 1]``, e.g. ``4.0``):
      an absolute thread count, used as-is (rounded if a float).
    - A negative ``int`` or ``float``: an offset from ``os.cpu_count()``
      -- e.g. ``-2`` means "leave 2 CPUs free", resolving to
      ``cpu_count - 2``. Clamped to at least 1 -- asking to leave more
      CPUs free than the machine has (e.g. ``-6`` on a 4-core machine)
      resolves to 1, not a negative or zero thread count.

    ``0`` is rejected: it isn't a meaningful thread count under any of
    the above readings.

    New in v1.3.
    """
    cpu_count = os.cpu_count() or 1
    if isinstance(n_threads, float) and 0 < n_threads <= 1:
        return max(1, round(cpu_count * n_threads))
    if n_threads == 0:
        raise ValueError(
            "n_threads must not be 0 -- use a positive thread count, a "
            "fraction in (0, 1] of os.cpu_count(), or a negative offset "
            "from it (e.g. -1 to leave one CPU free)"
        )
    if n_threads > 0:
        return round(n_threads)
    return max(1, round(cpu_count + n_threads))


def get_tmatrix_size(t_order):
    """Row/column dimension of a T-matrix block for a given truncation order.

    Pure Python -- (t_order + 1)**2 - 1, matching upstream's own ordering
    convention (-1(1),0(1),1(1),-2(2),...,N(N)).
    """
    return (t_order + 1) ** 2 - 1


def eps_from_index(n):
    """Electric permittivity ``eps = m**2`` from a complex refractive index.

    FaSTMM2 works natively in permittivity (see module docstring) -- this
    is the one-line squaring every caller otherwise has to write by hand.
    Broadcasts over array-likes.
    """
    m = np.asarray(n, dtype=np.complex128)
    return m**2


# --- Host-medium helpers ---------------------------------------------------
#
# FaSTMM2 has no separate "medium" input -- solve()/compute_tmatrix() just
# take a per-particle eps and a single k. A (possibly absorbing) host
# medium is modeled by transforming both: the particle's permittivity
# *relative to the medium* instead of relative to vacuum, and a complex
# wavenumber *in* the medium instead of the vacuum one. These three
# functions are the shared implementation of that transform, used by
# FaSTMM (and through it SweepConfig) and available to callers that drive
# FaSTMM2 directly.


def eps_relative_to_medium(n_particle, n_medium):
    """Particle permittivity relative to a host medium:
    ``eps_eff = (n_particle / n_medium)**2``. With a vacuum medium
    (``n_medium == 1+0j``) this is exactly ``eps_from_index(n_particle)``.
    """
    return eps_from_index(np.asarray(n_particle, dtype=np.complex128) / n_medium)


def k_in_medium(k_vacuum, n_medium):
    """Wavenumber inside a host medium: ``k_medium = k_vacuum * n_medium``
    -- complex (absorbing) whenever ``n_medium`` has a nonzero imaginary
    part. With a vacuum medium this is exactly ``k_vacuum`` unchanged.
    """
    return k_vacuum * n_medium


def medium_flux_norm(n_medium):
    """MSTM medium-mode cross-section factor, ``Re(n_medium)``.

    ``FaSTMM2.solve()`` returns physical cross sections: power removed
    from the beam per unit incident irradiance *in the host medium*. For a
    single sphere in a non-absorbing host they equal Mie theory with the
    relative index ``m / n_medium`` and the size parameter in the host
    (Bohren & Huffman), so a large absorbing sphere has ``Q_ext -> 2``.

    MSTM's medium mode normalises by the vacuum irradiance of the same
    field amplitude instead, which makes its cross sections larger by
    ``Re(n_medium)``. Multiply physical cross sections by this factor to
    compare with MSTM, or pass ``cross_section_convention="mstm"`` to
    ``FaSTMM.solve()``. With a vacuum medium this returns exactly ``1.0``.
    """
    return complex(n_medium).real


def _prepare_geometry(coords, radii, eps):
    coords = np.asarray(coords, dtype=np.float64)
    if coords.ndim != 2 or 3 not in coords.shape:
        raise ValueError("coords must have shape (n, 3) or (3, n)")
    if coords.shape[0] == 3 and coords.shape[1] != 3:
        coords_f = np.asfortranarray(coords)
    else:
        coords_f = np.asfortranarray(coords.T)

    n = coords_f.shape[1]
    radii = np.ascontiguousarray(radii, dtype=np.float64)
    eps = np.ascontiguousarray(eps, dtype=np.complex128)
    if radii.shape != (n,):
        raise ValueError(f"radii must have shape ({n},), got {radii.shape}")
    if eps.shape != (n,):
        raise ValueError(f"eps must have shape ({n},), got {eps.shape}")
    # Guard before Fortran, which doesn't validate: a non-positive (or
    # non-finite) radius can *hang* the Mie/truncation-order loops rather
    # than error out -- discovered the hard way by a test feeding a
    # negative radius to a solve that then never returned.
    if not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
        raise ValueError("radii must all be positive and finite")

    return n, coords_f, radii, np.real(eps).copy(), np.imag(eps).copy()


def _fortran_truncation(truncation_formula, truncation_order):
    """Encode the truncation choice as the single integer the Fortran
    wrapper takes: 0/1 select a ka-based formula, a value >= 2 is an
    explicit per-sphere order."""
    if truncation_order is not None:
        if isinstance(truncation_order, bool) or int(truncation_order) != truncation_order:
            raise ValueError(f"truncation_order must be an integer, got {truncation_order!r}")
        if truncation_order < 2:
            raise ValueError(f"truncation_order must be >= 2, got {truncation_order}")
        return int(truncation_order)
    if truncation_formula is None:
        return 0
    if truncation_formula not in (0, 1):
        raise ValueError(
            f"truncation_formula must be 0 or 1, got {truncation_formula!r} "
            "(use truncation_order for an explicit expansion order)"
        )
    return int(truncation_formula)


def _cross_sections_dict(crs):
    return {
        "c_ext": float(crs[0]),
        "c_ext_minus_c_abs": float(crs[1]),
        "c_abs": float(crs[2]),
        "c_sca": float(crs[3]),
        "asymmetry": float(crs[4]),
    }


class FaSTMM2:
    """Main interface to the FaSTMM2 MLFMM T-matrix solver."""

    def __init__(self):
        self._ext = _ext

    def solve(
        self,
        coords,
        radii,
        eps,
        k,
        N_theta=181,
        N_phi=32,
        N_ave=0,
        halton_init=0,
        formulation=2,
        acc=2,
        tol=1e-4,
        restart=5,
        max_iter=50,
        truncation_formula: Literal[0, 1] | None = None,
        truncation_order: int | None = None,
        n_threads: int | float | None = None,
    ):
        """Solve for the Mueller matrix (and, for fixed orientation, Jones
        matrix) and cross sections of a cluster of spherical monomers.

        Parameters
        ----------
        coords : array-like, shape (n, 3) or (3, n)
            Sphere center coordinates.
        radii : array-like, shape (n,)
            Sphere radii.
        eps : array-like of complex, shape (n,)
            Electric permittivity of each sphere (eps = m**2, m = complex
            refractive index -- see module docstring).
        k : float or complex
            Wavenumber (real or complex; a complex value models a
            (possibly absorbing) background medium, n_medium = k * lambda /
            (2*pi)).  Equivalent to upstream's k with a non-trivial
            background refractive index.
        N_theta, N_phi : int
            Angular resolution of the returned Mueller/Jones matrices.
        N_ave : int
            Number of orientations to average over via a Halton sequence.
            0 (default) means fixed orientation -- no averaging, and a
            Jones matrix is also returned.
        halton_init : int
            Starting point of the Halton sequence (only used if N_ave > 0).
        formulation : int
            0: STMM, 1: FaSTMM, 2: FaSTMM2 (default, fastest for large
            clusters).
        acc : int
            Desired MLFMM accuracy, in number of significant digits.
        tol, restart, max_iter : float, int, int
            GMRES solver parameters.
        truncation_formula : {0, 1}, optional
            Per-sphere Mie truncation formula: 0 (default) --
            ``truncation_order``, ``floor(ka + 3*ka^(1/3))``, min 6; 1 --
            ``truncation_order2``, a more conservative formula that gives
            larger expansion orders, especially for small ``ka`` where the
            default can under-resolve near-field coupling of touching
            spheres. ``None`` resolves to 0.
        truncation_order : int, optional
            Explicit per-sphere expansion order (>= 2); overrides
            ``truncation_formula`` when given. The cluster-level order is
            raised to at least this value. Touching spheres at small size
            parameter converge slowly in order, which ka-based formulas
            cannot express; orders beyond ~20 start losing precision in
            the translation arithmetic at very small ka. Note that both
            formulas never go below 6.

            Both truncation arguments are real per-call arguments passed
            straight through to the Fortran extension (not global/process
            state), so concurrent calls with different values from
            different Python threads are safe.
        n_threads : int, optional
            OpenMP thread count used by the GMRES matrix-vector product
            (the per-iteration hot path), the solver, and far-field
            evaluation. May be negative -- see ``resolve_n_threads()`` --
            to mean an offset from ``os.cpu_count()`` rather than an
            absolute count, or a float in (0, 1] for a fraction of it.
            ``None`` (default) resolves to 1 -- deliberately conservative,
            since a caller solving many clusters across its own process-
            or thread-level parallelism (e.g. a wavelength sweep) would
            otherwise oversubscribe the machine by workers * OpenMP's
            default thread count without asking for it. Raise it
            explicitly for a single, one-off solve: measured on a
            128-sphere touching aggregate (16 logical cores), near-linear
            speedup through 8 threads (4.7x over 1 thread), but *worse*
            at 16 than at 8 (oversubscription) -- picking a good value for
            your hardware matters more than "more is better". Like the
            truncation arguments, a real per-call argument -- safe under
            concurrent calls with different values.

        Returns
        -------
        dict with keys:
            mueller : ndarray
                Fixed orientation (N_ave == 0): shape (N_theta*N_phi, 18),
                columns [phi, theta, P11, P12, ..., P44].
                Orientation-averaged (N_ave > 0): shape (N_theta, 17),
                columns [scattering angle, P11, P12, ..., P44].
            jones : ndarray of complex, shape (N_theta*N_phi, 6), optional
                Only present for fixed orientation. Columns
                [phi, theta, S1, S2, S3, S4].
            c_ext, c_abs, c_sca : float
                Extinction, absorption, scattering cross sections
                (c_sca via far-field integration).
            c_ext_minus_c_abs : float
                Cext - Cabs (scattering cross section via optical theorem).
            asymmetry : float
                Asymmetry parameter <cos(theta)>.
        """
        resolved_truncation_formula = _fortran_truncation(
            truncation_formula, truncation_order
        )
        resolved_n_threads = 1 if n_threads is None else resolve_n_threads(n_threads)

        _, coords_f, radii_f, eps_r, eps_i = _prepare_geometry(coords, radii, eps)

        if N_ave == 0:
            mueller, jones, crs = self._ext.fastmm2_f2py_bindings.fastmm2_solve_fixed(
                coords_f,
                radii_f,
                eps_r,
                eps_i,
                complex(k),
                N_theta,
                N_phi,
                formulation,
                acc,
                tol,
                restart,
                max_iter,
                resolved_truncation_formula,
                resolved_n_threads,
            )
            result = {"mueller": mueller, "jones": jones}
        else:
            mueller, crs = self._ext.fastmm2_f2py_bindings.fastmm2_solve_averaged(
                coords_f,
                radii_f,
                eps_r,
                eps_i,
                complex(k),
                N_theta,
                N_phi,
                N_ave,
                halton_init,
                formulation,
                acc,
                tol,
                restart,
                max_iter,
                resolved_truncation_formula,
                resolved_n_threads,
            )
            result = {"mueller": mueller}

        result.update(_cross_sections_dict(crs))
        return result

    def compute_tmatrix(
        self,
        coords,
        radii,
        eps,
        k,
        t_order=16,
        formulation=2,
        acc=2,
        tol=1e-4,
        restart=5,
        max_iter=50,
        truncation_formula: Literal[0, 1] | None = None,
        truncation_order: int | None = None,
        n_threads: int | float | None = None,
    ):
        """Compute the T-matrix of a cluster of spherical monomers.

        Parameters
        ----------
        coords, radii, eps, k, formulation, acc, tol, restart, max_iter :
            Same meaning as in solve().
        t_order : int
            Requested T-matrix truncation order.
        truncation_formula, truncation_order, n_threads : optional
            Same meaning as in solve().

        Returns
        -------
        dict with keys Taa, Tab, Tba, Tbb : ndarray of complex, each shape
        (nm, nm) where nm = get_tmatrix_size(t_order), ordered as
        (-1(1),0(1),1(1),-2(2),...,N(N)).
        """
        resolved_truncation_formula = _fortran_truncation(
            truncation_formula, truncation_order
        )
        resolved_n_threads = 1 if n_threads is None else resolve_n_threads(n_threads)

        _, coords_f, radii_f, eps_r, eps_i = _prepare_geometry(coords, radii, eps)
        nm = get_tmatrix_size(t_order)

        Taa, Tab, Tba, Tbb = self._ext.fastmm2_f2py_bindings.fastmm2_compute_tmatrix(
            coords_f,
            radii_f,
            eps_r,
            eps_i,
            complex(k),
            t_order,
            nm,
            formulation,
            acc,
            tol,
            restart,
            max_iter,
            resolved_truncation_formula,
            resolved_n_threads,
        )
        return {"Taa": Taa, "Tab": Tab, "Tba": Tba, "Tbb": Tbb}
