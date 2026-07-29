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

import numpy as np

try:
    from . import _fastmm_ext as _ext
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pyfastmm's compiled extension is not built. Run `make f2py-ext` "
        "or `uv sync` to build it."
    ) from exc


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
# functions are the shared implementation of that transform -- used by
# SweepConfig.run() (this package) and t-bench's fastmm2-python adapter,
# which independently need the exact same formulas since neither one goes
# through the other.


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
    """Cross-section correction factor for a host medium.

    FaSTMM2's ``cross_sections()``/``sphere_absorbtion2()`` (which produce
    ``c_ext``/``c_abs``/``c_ext_minus_c_abs``) implicitly assume vacuum-like
    incident-flux normalization (Poynting flux ~ ``|E|**2``); a real host
    medium's flux instead scales as ``Re(n_medium) * |E|**2``, so those
    three cross sections need multiplying by this factor. Confirmed against
    MSTM (which models this correctly) for iron-in-olivine clusters: up to
    ~46% off without this factor, <0.03% with it, for single sphere,
    touching pair, 4-chain, and a real 32-sphere fractal aggregate.

    Deliberately *not* applicable to the far-field-integrated ``c_sca``
    (from ``compute_cluster_fields_far``) -- that value has its own,
    separate reliability problem in the deep-Rayleigh regime, unrelated to
    and not addressed by this correction (see space-weathering's PLAN.md).
    With a vacuum medium this returns exactly ``1.0`` (a no-op).
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

    return n, coords_f, radii, np.real(eps).copy(), np.imag(eps).copy()


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

    def set_truncation_formula(self, formula: int = 0) -> None:
        """Choose the per-sphere Mie truncation formula.

        0 (default): ``truncation_order``  -- ``floor(ka + 3*ka^(1/3))``, min 6.
        1: ``truncation_order2`` -- more conservative formula that gives
           larger expansion orders, especially for small ``ka`` where
           FaSTMM2's default can under-resolve near-field coupling of
           touching spheres.

        Must be called *before* ``solve()`` or ``compute_tmatrix()`` --
        this sets a module-level Fortran variable, so it affects all
        subsequent calls (not just this instance, not thread-safe).

        New in v1.1.
        """
        self._ext.fastmm2_f2py_bindings.truncation_formula = formula

    def set_num_threads(self, n: int) -> None:
        """Set the OpenMP thread count used by the parallel regions in the
        GMRES matrix-vector product (the per-iteration hot path), the
        solver, and far-field evaluation -- all already built with
        -fopenmp, but previously only controllable via the OMP_NUM_THREADS
        environment variable, which libgomp reads once at first use and
        ignores further changes to (confirmed: setting it from Python
        mid-process has no effect). This calls the standard
        omp_set_num_threads() runtime function directly, so it takes
        effect on the very next solve()/compute_tmatrix() call.

        Measured on a real 128-sphere touching iron-in-olivine aggregate
        (16 logical cores): near-linear speedup through 8 threads (4.7x
        over 1 thread), but *worse* at 16 than at 8 (oversubscription) --
        picking a good value for your hardware matters more than "more is
        better". Also relevant when composing with process-level
        parallelism (e.g. running many solves across a multiprocessing
        pool): each worker process gets its own OpenMP thread pool, so
        workers * n_threads can oversubscribe the machine if you don't
        account for it.

        Must be called *before* solve()/compute_tmatrix() -- this affects
        the whole process (OpenMP has no per-call thread count), same
        process-global caveat as set_truncation_formula().

        New in v1.2.
        """
        self._ext.fastmm2_f2py_bindings.fastmm2_set_num_threads(n)

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
        truncation_formula: int | None = None,
        n_threads: int | None = None,
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
        truncation_formula : int, optional
            Per-sphere Mie truncation formula (0=default, 1=conservative).
            Convenience wrapper around ``set_truncation_formula()``.  When
            None (default), does not change the current setting.
        n_threads : int, optional
            OpenMP thread count.  Convenience wrapper around
            ``set_num_threads()``.  When None (default), does not change
            the current setting (which, absent any call, is whatever
            OMP_NUM_THREADS was at process start, or OpenMP's own default
            -- typically all visible cores).

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
        if truncation_formula is not None:
            self.set_truncation_formula(truncation_formula)
        if n_threads is not None:
            self.set_num_threads(n_threads)

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
        n_threads: int | None = None,
    ):
        """Compute the T-matrix of a cluster of spherical monomers.

        Parameters
        ----------
        coords, radii, eps, k, formulation, acc, tol, restart, max_iter :
            Same meaning as in solve().
        t_order : int
            Requested T-matrix truncation order.
        n_threads : int, optional
            Same meaning as in solve() -- convenience wrapper around
            set_num_threads().

        Returns
        -------
        dict with keys Taa, Tab, Tba, Tbb : ndarray of complex, each shape
        (nm, nm) where nm = get_tmatrix_size(t_order), ordered as
        (-1(1),0(1),1(1),-2(2),...,N(N)).
        """
        if n_threads is not None:
            self.set_num_threads(n_threads)

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
        )
        return {"Taa": Taa, "Tab": Tab, "Tba": Tba, "Tbb": Tbb}
