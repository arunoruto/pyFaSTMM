"""FaSTMM -- an ergonomic, class-driven single-cluster API, version-aware.

FaSTMM2.solve() itself is deliberately minimal (geometry in, arrays out,
no state) -- but every real caller ends up needing the same handful of
things around it: transforming a particle/medium refractive index pair
into what solve() actually wants (eps relative to the medium, a complex
medium wavenumber), correcting for the medium's incident-flux
normalization, and rotating the cluster to emulate tilted incidence
(FaSTMM2's beam is fixed along +z). Two independent call sites --
SweepConfig.run() here and t-bench's fastmm2-python adapter -- used to
each reimplement all of that by hand. FaSTMM is the one shared
implementation both now build on, so a bug in this logic only needs
fixing once.

Named after the software family (FaSTMM), not the specific solver
version (FaSTMM2) -- `version` picks which one, defaulting to 2, the
only one that exists today. This is deliberately NOT the same axis as
`formulation` (0=STMM, 1=FaSTMM, 2=FaSTMM2/MLFMM), which selects an
algorithm *within* a given version: a hypothetical FaSTMM3 would almost
certainly ship its own compiled backend and quite possibly its own set
of formulations, which is exactly why version dispatch and formulation
validation are two separate, nested match/case checks below rather than
one flat list of "valid combinations".
"""

from __future__ import annotations

import numpy as np

from pyfastmm._fastmm import (
    FaSTMM2,
    eps_relative_to_medium,
    k_in_medium,
    medium_flux_norm,
)


def rotate_for_incidence(coords, polar_deg, azimuthal_deg):
    """Rotate ``coords`` so FaSTMM2's fixed +z incident beam is physically
    equivalent to illuminating the *unrotated* cluster from
    ``(polar_deg, azimuthal_deg)``. Same validated rotation as
    ``pyfastmm._config.IncidentConfig.rotate_positions`` -- see that
    method's docstring for the order derivation and the cross-tool
    validation against MSTM (R = Ry @ Rz, not Rz @ Ry, once both angles
    are simultaneously nonzero). Public (unlike that method) because a
    caller that isn't going through SweepConfig/FaSTMM -- e.g. writing
    geometry for the standalone FaSTMM2 CLI binary, which has no
    incident-angle flag of its own -- needs the exact same rotation."""
    if polar_deg == 0.0 and azimuthal_deg == 0.0:
        return coords

    theta = np.radians(polar_deg)
    phi = np.radians(azimuthal_deg)
    cos_t, sin_t = np.cos(-theta), np.sin(-theta)
    Ry = np.array([[cos_t, 0, sin_t], [0, 1, 0], [-sin_t, 0, cos_t]])
    cos_p, sin_p = np.cos(-phi), np.sin(-phi)
    Rz = np.array([[cos_p, -sin_p, 0], [sin_p, cos_p, 0], [0, 0, 1]])
    R = Ry @ Rz

    coords = np.asarray(coords, dtype=np.float64)
    return coords @ R.T


def radiative_transfer_phase_function(result, n_theta=None):
    """Rescale a ``FaSTMM.solve()`` result's raw Mueller S11/S12 to the
    radiative-transfer phase-function convention (``integral(S11 dOmega)
    == 4*pi``).

    FaSTMM2's raw S11 (``mie.f90``: ``(|S1|^2+...+|S4|^2)/2``) follows the
    Bohren-Huffman ``dCsca/dOmega = S11/k^2`` convention instead -- fine
    for internal use, but not what callers plotting a phase function or
    computing DoLP actually want. Requires ``result`` to carry
    ``"mueller"``, ``"k_medium"``, and ``"c_ext_minus_c_abs"`` (all present
    in ``FaSTMM.solve()``'s return value).

    Parameters
    ----------
    result : dict
        A ``FaSTMM.solve()`` result (fixed orientation, i.e. ``N_ave=0``
        -- the raw mueller layout this function assumes only applies then).
    n_theta : int, optional
        Only the first ``n_theta`` rows (the phi=0 cut) are returned;
        defaults to all rows in ``result["mueller"]``.

    Returns
    -------
    ndarray, shape (n_theta, 3)
        Columns ``[theta_deg, S11, S12]``, phase-function-normalized.
    """
    mueller = result["mueller"]
    rows = mueller[:n_theta] if n_theta is not None else mueller
    phase_norm = 4.0 * np.pi / (abs(result["k_medium"]) ** 2 * result["c_ext_minus_c_abs"])
    theta_deg = np.degrees(rows[:, 1])
    s11 = rows[:, 2] * phase_norm
    s12 = rows[:, 3] * phase_norm
    return np.column_stack([theta_deg, s11, s12])


def _unsupported_version(version) -> ValueError:
    return ValueError(
        f"pyfastmm.FaSTMM only supports version=2 (FaSTMM2); got version={version!r}"
    )


def _validate_formulation_v2(formulation) -> None:
    match formulation:
        case 0 | 1 | 2:
            return
        case _:
            raise ValueError(
                f"formulation={formulation!r} is not supported by FaSTMM "
                "version 2 (valid: 0=STMM, 1=FaSTMM, 2=FaSTMM2/MLFMM)"
            )


class FaSTMM:
    """A cluster of spherical monomers, optionally embedded in a host
    medium, solved with a chosen FaSTMM version.

    Parameters
    ----------
    coords : array-like, shape (n, 3)
        Sphere center coordinates.
    radii : array-like, shape (n,)
        Sphere radii, same length unit as ``coords`` and (via
        ``wavelength_vacuum``) the wavelength.
    refractive_index : complex or array-like of complex, shape (n,)
        Per-particle refractive index, vacuum-referenced (i.e. as you'd
        look it up from a materials database -- *not* pre-divided by the
        medium). A single value broadcasts to every particle.
    medium_refractive_index : complex, default ``1.0+0.0j`` (vacuum)
        Host medium refractive index, vacuum-referenced. A nonzero
        imaginary part models an absorbing host.
    version : int, default 2
        Which FaSTMM solver version to use. Only 2 (FaSTMM2) exists
        today; any other value raises ValueError at call time (not
        construction time -- see ``solve()``). Not the same thing as
        ``formulation`` (see module docstring).
    """

    def __init__(
        self,
        coords,
        radii,
        refractive_index,
        medium_refractive_index: complex = 1.0 + 0.0j,
        version: int = 2,
    ):
        self.coords = np.asarray(coords, dtype=np.float64)
        self.radii = np.asarray(radii, dtype=np.float64)
        n = len(self.radii)
        self.refractive_index = np.broadcast_to(
            np.asarray(refractive_index, dtype=np.complex128), (n,)
        ).copy()
        self.medium_refractive_index = complex(medium_refractive_index)
        self.version = version

    def _eps_and_k(self, wavelength_vacuum):
        k_medium = k_in_medium(
            2.0 * np.pi / wavelength_vacuum, self.medium_refractive_index
        )
        eps = eps_relative_to_medium(
            self.refractive_index, self.medium_refractive_index
        )
        return eps, complex(k_medium)

    def solve(
        self,
        wavelength_vacuum,
        *,
        incident_polar_deg=0.0,
        incident_azimuthal_deg=0.0,
        N_theta=181,
        N_phi=32,
        N_ave=0,
        halton_init=0,
        formulation=2,
        acc=2,
        tol=1e-4,
        restart=5,
        max_iter=50,
        truncation_formula=0,
        n_threads: int | float = 1,
    ):
        """Solve for the Mueller/Jones matrix and cross sections at
        ``wavelength_vacuum`` (same length unit as ``coords``/``radii``).

        Returns the same dict ``FaSTMM2.solve()`` does (``mueller``,
        ``jones`` when ``N_ave==0``, ``c_ext``, ``c_abs``,
        ``c_ext_minus_c_abs``, ``c_sca``, ``asymmetry``), with
        ``c_ext``/``c_abs``/``c_ext_minus_c_abs`` already corrected for
        the host medium's incident-flux normalization (see
        ``medium_flux_norm()``) -- deliberately *not* applied to ``c_sca``
        (the far-field-integrated value), which has its own, separate
        reliability problem in the deep-Rayleigh regime. Also adds
        ``"k_medium"`` (the complex wavenumber actually used) and
        ``"eps"`` (the per-particle permittivity actually used), mainly
        so ``radiative_transfer_phase_function()`` can use them without
        the caller re-deriving them.

        ``n_threads`` defaults to 1 (*not* "use all cores") -- deliberately
        conservative, since a caller solving many clusters across its own
        process- or thread-level parallelism (e.g. a campaign sweep) would
        otherwise oversubscribe the machine by workers * OpenMP's default
        thread count without asking for it. Raise it explicitly for a
        single, one-off solve (see ``set_num_threads()``'s docstring for
        measured scaling -- more isn't always better past your physical
        core count). May also be negative (an offset from
        ``os.cpu_count()``, e.g. ``-2`` to leave 2 CPUs free) or a float
        in ``(0, 1]`` (a fraction of ``os.cpu_count()``, e.g. ``0.5`` for
        half the machine) -- see ``resolve_n_threads()``.

        Raises
        ------
        ValueError
            If ``self.version`` isn't a supported FaSTMM version, or
            ``formulation`` isn't valid for that version.
        """
        match self.version:
            case 2:
                return self._solve_v2(
                    wavelength_vacuum,
                    incident_polar_deg=incident_polar_deg,
                    incident_azimuthal_deg=incident_azimuthal_deg,
                    N_theta=N_theta,
                    N_phi=N_phi,
                    N_ave=N_ave,
                    halton_init=halton_init,
                    formulation=formulation,
                    acc=acc,
                    tol=tol,
                    restart=restart,
                    max_iter=max_iter,
                    truncation_formula=truncation_formula,
                    n_threads=n_threads,
                )
            case _:
                raise _unsupported_version(self.version)

    def _solve_v2(
        self,
        wavelength_vacuum,
        *,
        incident_polar_deg,
        incident_azimuthal_deg,
        N_theta,
        N_phi,
        N_ave,
        halton_init,
        formulation,
        acc,
        tol,
        restart,
        max_iter,
        truncation_formula,
        n_threads,
    ):
        _validate_formulation_v2(formulation)

        eps, k_medium = self._eps_and_k(wavelength_vacuum)
        coords = rotate_for_incidence(
            self.coords, incident_polar_deg, incident_azimuthal_deg
        )

        result = FaSTMM2().solve(
            coords,
            self.radii,
            eps,
            k_medium,
            N_theta=N_theta,
            N_phi=N_phi,
            N_ave=N_ave,
            halton_init=halton_init,
            formulation=formulation,
            acc=acc,
            tol=tol,
            restart=restart,
            max_iter=max_iter,
            truncation_formula=truncation_formula,
            n_threads=n_threads,
        )
        flux_norm = medium_flux_norm(self.medium_refractive_index)
        result["c_ext"] *= flux_norm
        result["c_abs"] *= flux_norm
        result["c_ext_minus_c_abs"] *= flux_norm
        result["k_medium"] = k_medium
        result["eps"] = eps
        return result

    def compute_tmatrix(
        self,
        wavelength_vacuum,
        *,
        t_order=16,
        formulation=2,
        acc=2,
        tol=1e-4,
        restart=5,
        max_iter=50,
        n_threads: int | float = 1,
    ):
        """Compute the cluster's T-matrix at ``wavelength_vacuum``. Same
        medium transform as ``solve()``; no flux-normalization correction
        applies here (the T-matrix has no cross sections to correct).
        ``n_threads`` defaults to 1 -- see ``solve()``'s docstring.

        Raises
        ------
        ValueError
            If ``self.version`` isn't a supported FaSTMM version, or
            ``formulation`` isn't valid for that version.
        """
        match self.version:
            case 2:
                return self._compute_tmatrix_v2(
                    wavelength_vacuum,
                    t_order=t_order,
                    formulation=formulation,
                    acc=acc,
                    tol=tol,
                    restart=restart,
                    max_iter=max_iter,
                    n_threads=n_threads,
                )
            case _:
                raise _unsupported_version(self.version)

    def _compute_tmatrix_v2(
        self,
        wavelength_vacuum,
        *,
        t_order,
        formulation,
        acc,
        tol,
        restart,
        max_iter,
        n_threads,
    ):
        _validate_formulation_v2(formulation)

        eps, k_medium = self._eps_and_k(wavelength_vacuum)
        result = FaSTMM2().compute_tmatrix(
            self.coords,
            self.radii,
            eps,
            k_medium,
            t_order=t_order,
            formulation=formulation,
            acc=acc,
            tol=tol,
            restart=restart,
            max_iter=max_iter,
            n_threads=n_threads,
        )
        result["k_medium"] = k_medium
        result["eps"] = eps
        return result
