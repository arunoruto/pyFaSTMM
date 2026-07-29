"""Cluster -- an ergonomic, class-driven single-cluster API for FaSTMM2.

FaSTMM2.solve() itself is deliberately minimal (geometry in, arrays out,
no state) -- but every real caller ends up needing the same handful of
things around it: transforming a particle/medium refractive index pair
into what solve() actually wants (eps relative to the medium, a complex
medium wavenumber), correcting for the medium's incident-flux
normalization, and rotating the cluster to emulate tilted incidence
(FaSTMM2's beam is fixed along +z). Two independent call sites --
SweepConfig.run() here and t-bench's fastmm2-python adapter -- used to
each reimplement all of that by hand. Cluster is the one shared
implementation both now build on, so a bug in this logic only needs
fixing once.
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
    caller that isn't going through SweepConfig/Cluster -- e.g. writing
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
    """Rescale a ``Cluster.solve()`` result's raw Mueller S11/S12 to the
    radiative-transfer phase-function convention (``integral(S11 dOmega)
    == 4*pi``).

    FaSTMM2's raw S11 (``mie.f90``: ``(|S1|^2+...+|S4|^2)/2``) follows the
    Bohren-Huffman ``dCsca/dOmega = S11/k^2`` convention instead -- fine
    for internal use, but not what callers plotting a phase function or
    computing DoLP actually want. Requires ``result`` to carry
    ``"mueller"``, ``"k_medium"``, and ``"c_ext_minus_c_abs"`` (all present
    in ``Cluster.solve()``'s return value).

    Parameters
    ----------
    result : dict
        A ``Cluster.solve()`` result (fixed orientation, i.e. ``N_ave=0``
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


class Cluster:
    """A cluster of spherical monomers, optionally embedded in a host
    medium.

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
    """

    def __init__(
        self,
        coords,
        radii,
        refractive_index,
        medium_refractive_index: complex = 1.0 + 0.0j,
    ):
        self.coords = np.asarray(coords, dtype=np.float64)
        self.radii = np.asarray(radii, dtype=np.float64)
        n = len(self.radii)
        self.refractive_index = np.broadcast_to(
            np.asarray(refractive_index, dtype=np.complex128), (n,)
        ).copy()
        self.medium_refractive_index = complex(medium_refractive_index)

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
        """
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
    ):
        """Compute the cluster's T-matrix at ``wavelength_vacuum``. Same
        medium transform as ``solve()``; no flux-normalization correction
        applies here (the T-matrix has no cross sections to correct)."""
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
        )
        result["k_medium"] = k_medium
        result["eps"] = eps
        return result
