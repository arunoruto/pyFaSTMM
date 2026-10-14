"""FaSTMM -- an ergonomic, class-driven single-cluster API, version-aware.

FaSTMM2.solve() itself is deliberately minimal (geometry in, arrays out,
no state) -- but every real caller ends up needing the same handful of
things around it: transforming a particle/medium refractive index pair
into what solve() actually wants (eps relative to the medium, a complex
medium wavenumber), choosing the cross-section convention in a host
medium, and rotating the cluster to emulate tilted incidence (FaSTMM2's
beam is fixed along +z). FaSTMM is the one shared implementation of that
(SweepConfig builds on it too), so a bug in this logic only needs fixing
once.

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

from typing import Literal

import numpy as np

from pyfastmm._fastmm import (
    FaSTMM2,
    eps_relative_to_medium,
    k_in_medium,
    medium_flux_norm,
)
from pyfastmm._results import SolveResult, TMatrixResult


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


CrossSectionConvention = Literal["physical", "mstm"]


def _validate_convention(cross_section_convention) -> None:
    if cross_section_convention not in ("physical", "mstm"):
        raise ValueError(
            "cross_section_convention must be 'physical' or 'mstm', got "
            f"{cross_section_convention!r}"
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

    Attributes
    ----------
    results : SolveResult | TMatrixResult | None
        The most recent ``solve()``/``compute_tmatrix()`` outcome as a
        validated pydantic model (see :mod:`pyfastmm._results` -- the
        validation catches non-finite Mueller/Jones/T-matrix output
        loudly at the source). ``None`` until the first call. The
        methods also still *return* the plain-dict form
        (``results.to_dict()``) for backward compatibility, so both
        access styles work:

        >>> f = FaSTMM(coords, radii, 1.5 + 0.01j)
        >>> f.solve(0.55e-6)["c_ext"]      # dict style, unchanged
        >>> f.results.c_ext                # attribute style, validated

        Note: ``results`` holds the *last* call's outcome. Concurrent
        ``solve()`` calls on one shared instance will race on it -- give
        each thread its own FaSTMM (cheap: it holds only input arrays)
        the way ``SweepConfig.run_resolved()`` does, and use the
        per-call return values.
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
        self.results: SolveResult | TMatrixResult | None = None

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
        truncation_formula: Literal[0, 1] = 0,
        truncation_order: int | None = None,
        n_threads: int | float = 1,
        cross_section_convention: CrossSectionConvention = "physical",
    ):
        """Solve for the Mueller/Jones matrix and cross sections at
        ``wavelength_vacuum`` (same length unit as ``coords``/``radii``).

        Returns the same dict ``FaSTMM2.solve()`` does (``mueller``,
        ``jones`` when ``N_ave==0``, ``c_ext``, ``c_abs``,
        ``c_ext_minus_c_abs``, ``c_sca``, ``asymmetry``). Also adds
        ``"k_medium"`` (the complex wavenumber actually used) and
        ``"eps"`` (the per-particle permittivity actually used), mainly
        so ``radiative_transfer_phase_function()`` can use them without
        the caller re-deriving them.

        ``cross_section_convention`` selects how cross sections in a host
        medium are normalised:

        - ``"physical"`` (default): power per unit incident irradiance in
          the host, exactly what ``FaSTMM2.solve()`` returns. A single
          sphere in a non-absorbing host equals Mie theory with
          ``m / n_host`` and the size parameter in the host, and a large
          absorbing sphere has ``Q_ext -> 2``.
        - ``"mstm"``: MSTM's medium mode, normalised by the vacuum
          irradiance of the same field amplitude, i.e. the physical values
          multiplied by ``Re(n_host)`` (see ``medium_flux_norm()``). Use it
          to compare with MSTM output.

        Both conventions agree in vacuum.

        ``truncation_formula``/``truncation_order`` select the per-sphere
        expansion order -- see ``FaSTMM2.solve()``.

        ``n_threads`` defaults to 1 (*not* "use all cores") -- deliberately
        conservative, since a caller solving many clusters across its own
        process- or thread-level parallelism (e.g. a wavelength sweep) would
        otherwise oversubscribe the machine by workers * OpenMP's default
        thread count without asking for it. Raise it explicitly for a
        single, one-off solve (see ``FaSTMM2.solve()``'s docstring for
        measured scaling -- more isn't always better past your physical
        core count). May also be negative (an offset from
        ``os.cpu_count()``, e.g. ``-2`` to leave 2 CPUs free) or a float
        in ``(0, 1]`` (a fraction of ``os.cpu_count()``, e.g. ``0.5`` for
        half the machine) -- see ``resolve_n_threads()``.

        Raises
        ------
        ValueError
            If ``self.version`` isn't a supported FaSTMM version,
            ``formulation`` isn't valid for that version, or
            ``cross_section_convention`` is unknown.
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
                    truncation_order=truncation_order,
                    n_threads=n_threads,
                    cross_section_convention=cross_section_convention,
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
        truncation_order,
        n_threads,
        cross_section_convention,
    ):
        _validate_formulation_v2(formulation)
        _validate_convention(cross_section_convention)

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
            truncation_order=truncation_order,
            n_threads=n_threads,
        )
        if cross_section_convention == "mstm":
            factor = medium_flux_norm(self.medium_refractive_index)
            for key in ("c_ext", "c_abs", "c_ext_minus_c_abs", "c_sca"):
                result[key] *= factor
        result["k_medium"] = k_medium
        result["eps"] = eps
        # Store the validated model on the instance (the class-driven
        # access path); return the plain dict (the backward-compatible
        # one). Validation happens here -- a non-finite Mueller/Jones
        # raises now, at the source, instead of poisoning downstream
        # analysis silently.
        validated = SolveResult(wavelength_vacuum=float(wavelength_vacuum), **result)
        self.results = validated
        return validated.to_dict()

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
        truncation_formula: Literal[0, 1] = 0,
        truncation_order: int | None = None,
        n_threads: int | float = 1,
    ):
        """Compute the cluster's T-matrix at ``wavelength_vacuum``. Same
        medium transform as ``solve()`` (the cross-section convention does
        not apply: a T-matrix has no cross sections).
        ``truncation_formula``/``truncation_order``/``n_threads`` default
        the same way as ``solve()`` -- see its docstring.

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
                    truncation_formula=truncation_formula,
                    truncation_order=truncation_order,
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
        truncation_formula,
        truncation_order,
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
            truncation_formula=truncation_formula,
            truncation_order=truncation_order,
            n_threads=n_threads,
        )
        result["k_medium"] = k_medium
        result["eps"] = eps
        # Same store-model / return-dict split as _solve_v2.
        validated = TMatrixResult(wavelength_vacuum=float(wavelength_vacuum), **result)
        self.results = validated
        return validated.to_dict()
