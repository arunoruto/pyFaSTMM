"""
Declarative sweep configuration for pyfastmm.

Uses the same TOML shape as pyMSTM's sweep configs
(https://github.com/arunoruto/pyMSTM) for the tool-agnostic parts --
``[particles]``, ``[wavelengths]``, ``[medium]``, ``[incident]`` -- so the
same cluster/wavelength-sweep config file can drive either tool. A single
particle-position file (e.g. a PyFracVAL fractal aggregate) and a single
TOML sweep config are shared as-is (see tests/data/).

``[solver]``/``[output]`` are necessarily FaSTMM2-specific (different
Fortran backend, different knobs) -- unrecognized fields from a pyMSTM-
authored config are silently ignored (pydantic's default behavior), and
FaSTMM2-specific fields just fall back to their own defaults if absent.

Two physical differences from MSTM worth knowing:

- FaSTMM2 supports a (possibly absorbing) background medium since
  v1.1 -- the ``[medium].refractive_index`` is used to compute the
  effective wavenumber ``k = 2*pi * n_medium / wavelength_vac`` and
  particle permittivity contrast ``eps_eff = (n_particle / n_medium)**2``.
  A complex ``refractive_index`` (n, k) on the medium leads to complex
  wavenumber ``k`` and, therefore, an absorbing (lossy) host.
- FaSTMM2 always illuminates along +z; there is no direct incident-beam-
  angle control like MSTM's ``incident_alpha_deg``/``incident_beta_deg``.
  ``[incident]`` is honored by rotating the cluster geometry the
  equivalent amount before solving instead (physically equivalent, since
  only the *relative* orientation of beam and cluster matters).

Example
-------
.. code-block:: toml

    [particles]
    positions_file = "cluster.dat"
    scale = 1e-9
    refractive_index = [1.5, 0.01]

    [medium]
    refractive_index_file = "nk_olivine.csv"

    [wavelengths]
    start = 0.4
    stop = 0.8
    num = 21
    scale = 1e-6

    [solver]
    formulation = 2
    tolerance = 1e-4
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, model_validator

from pyfastmm._fastmm import FaSTMM2

# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


def _resolve_path(base_dir: str | os.PathLike[str], filepath: str) -> Path:
    p = Path(filepath)
    if p.is_absolute():
        return p
    return Path(base_dir) / p


class _RefractiveIndexSpec(BaseModel):
    """Shared ``refractive_index``/``refractive_index_file`` handling used
    by both :class:`ParticlesConfig` and :class:`MediumConfig`."""

    refractive_index: tuple[float, float] = (1.0, 0.0)
    refractive_index_file: str | None = None

    def get_nk(
        self,
        wavelengths_raw: np.ndarray,
        base_dir: str | os.PathLike[str] = "",
    ) -> np.ndarray:
        """Complex refractive index ``n + i*k`` at each wavelength.

        If *refractive_index_file* is set, loads the file
        (whitespace-delimited columns ``wavelength  n  k``, same units as
        the ``[wavelengths]`` config before *scale*) and linearly
        interpolates onto the requested wavelengths.  Otherwise broadcasts
        *refractive_index* as a fixed (wavelength-independent) value.
        """
        if self.refractive_index_file:
            path = _resolve_path(base_dir, self.refractive_index_file)
            data = np.loadtxt(str(path))
            n_interp = np.interp(wavelengths_raw, data[:, 0], data[:, 1])
            k_interp = np.interp(wavelengths_raw, data[:, 0], data[:, 2])
            return n_interp + 1j * k_interp
        n_re, n_im = self.refractive_index
        return np.full_like(wavelengths_raw, complex(n_re, n_im), dtype=complex)


class ParticlesConfig(_RefractiveIndexSpec):
    """Particle cluster specification (same shape as pyMSTM's)."""

    positions_file: str = ""
    scale: float = 1.0
    gap_factor: float = 1.0
    refractive_index: tuple[float, float] = (1.5, 0.0)

    def load_positions(self, base_dir: str | os.PathLike[str] = "") -> np.ndarray:
        """Return (N, 4) array ``[x, y, z, radius]`` in physical units.

        Supports ``.dat``, ``.txt``, ``.csv``, ``.pos`` files.
        Automatically strips comment lines (``#``) for PyFracVAL format.

        *gap_factor* stretches positions (not radii) to separate touching
        spheres.
        """
        path = _resolve_path(base_dir, self.positions_file)
        suffix = Path(path).suffix.lower()
        if suffix == ".csv":
            data = np.loadtxt(str(path), delimiter=",")
        else:  # .dat, .txt, .pos -- whitespace
            data = np.loadtxt(str(path))
        if data.ndim == 1:
            data = data.reshape(-1, 4)
        data = data * self.scale
        if self.gap_factor != 1.0:
            data[:, :3] *= self.gap_factor
        return data

    def eps(self) -> complex:
        """Electric permittivity (eps = m**2) from the refractive index."""
        n_re, n_im = self.refractive_index
        return complex(n_re, n_im) ** 2


class WavelengthsConfig(BaseModel):
    """Wavelength sweep specification (same shape as pyMSTM's).

    Exactly one of *values* or *start / stop / num* must be given.
    *scale* converts the user-specified numbers to meters.
    """

    values: list[float] | None = None
    start: float | None = None
    stop: float | None = None
    num: int | None = None
    scale: float = 1.0  # 1e-6 = micrometers, 1e-9 = nanometers

    @model_validator(mode="after")
    def _check_spec(self) -> WavelengthsConfig:
        explicit = self.values is not None
        ranged = (
            self.start is not None and self.stop is not None and self.num is not None
        )
        if not explicit and not ranged:
            raise ValueError(
                "Specify either wavelengths.values or wavelengths.{start,stop,num}"
            )
        return self

    def get_wavelengths_m(self) -> np.ndarray:
        """Wavelengths in meters."""
        if self.values is not None:
            return np.asarray(self.values, dtype=float) * self.scale
        return np.linspace(self.start, self.stop, self.num) * self.scale  # type: ignore[arg-type]

    def get_wavenumbers(self) -> np.ndarray:
        """Vacuum wavenumbers k = 2*pi/wavelength, in the same length unit
        as the particle positions/radii (see ParticlesConfig.scale)."""
        return 2.0 * np.pi / self.get_wavelengths_m()


class MediumConfig(_RefractiveIndexSpec):
    """Surrounding medium. FaSTMM2 uses the medium refractive index to
    compute the effective wavenumber ``k = 2*pi * n_medium / wavelength_vac``
    and particle permittivity contrast ``eps_eff = (n_particle / n_medium)**2``,
    fully supporting (possibly absorbing) background media since v1.1."""


class IncidentConfig(BaseModel):
    """Incident plane-wave direction, honored via a geometry pre-rotation
    (see module docstring)."""

    polar_angle_deg: float = 0.0
    azimuthal_angle_deg: float = 0.0
    direction: int = 1

    def rotate_positions(self, positions: np.ndarray) -> np.ndarray:
        """Rotate cluster coordinates (in place on a copy) so that solving
        with FaSTMM2's fixed +z incident beam is equivalent to illuminating
        the *unrotated* cluster from (polar_angle_deg, azimuthal_angle_deg).
        """
        if self.polar_angle_deg == 0.0 and self.azimuthal_angle_deg == 0.0:
            return positions

        theta = np.radians(self.polar_angle_deg)
        phi = np.radians(self.azimuthal_angle_deg)

        # Rotate the cluster by -theta about the (rotated) y-axis and then
        # -phi about z, the inverse of tilting the beam by (theta, phi) --
        # equivalent because only the relative orientation matters.
        cos_t, sin_t = np.cos(-theta), np.sin(-theta)
        Ry = np.array([[cos_t, 0, sin_t], [0, 1, 0], [-sin_t, 0, cos_t]])
        cos_p, sin_p = np.cos(-phi), np.sin(-phi)
        Rz = np.array([[cos_p, -sin_p, 0], [sin_p, cos_p, 0], [0, 0, 1]])
        R = Rz @ Ry

        out = positions.copy()
        out[:, :3] = positions[:, :3] @ R.T
        return out


class SolverConfig(BaseModel):
    """FaSTMM2 solver settings."""

    formulation: int = 2  # 0: STMM, 1: FaSTMM, 2: FaSTMM2
    acc: int = 2  # MLFMM accuracy, significant digits
    tolerance: float = 1e-4
    restart: int = 5
    max_iterations: int = 50
    N_theta: int = 181
    N_phi: int = 32
    N_ave: int = 0  # 0 = fixed orientation
    halton_init: int = 0
    truncation_formula: int = 0
    """Per-sphere Mie truncation formula (see FaSTMM2.set_truncation_formula).
    0 (default, unchanged from pre-v1.1 behavior): floor(ka + 3*ka**(1/3)),
    min 6. 1: the more conservative formula -- needed for touching,
    small-ka, high-refractive-index-contrast clusters (e.g. nanometer-scale
    metal monomers), where formula 0 under-resolves near-field coupling and
    under-converges Cext by double-digit percentages."""


class OutputConfig(BaseModel):
    """Output control."""

    compute_tmatrix: bool = False
    t_order: int = 16


class SweepConfig(BaseModel):
    """Complete sweep configuration."""

    particles: ParticlesConfig = Field(default_factory=ParticlesConfig)
    wavelengths: WavelengthsConfig
    medium: MediumConfig = Field(default_factory=MediumConfig)
    incident: IncidentConfig = Field(default_factory=IncidentConfig)
    solver: SolverConfig = Field(default_factory=SolverConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    def run(
        self,
        base_dir: str | os.PathLike[str] = "",
        progress_callback: Any = None,
    ) -> list[dict]:
        """Run FaSTMM2.solve() once per wavelength in the sweep.

        Parameters
        ----------
        base_dir : path used to resolve a relative particles.positions_file
        progress_callback : optional callable(fraction_done: float, wavelength_m: float)
            called after each wavelength, e.g. to drive a progress bar.

        Returns
        -------
        list of dicts, one per wavelength, each the same dict solve() (or
        compute_tmatrix(), if output.compute_tmatrix is set) returns, plus
        "wavelength_m" and "k" keys.
        """
        positions = self.particles.load_positions(base_dir)
        positions = self.incident.rotate_positions(positions)
        coords = positions[:, :3]
        radii = positions[:, 3]
        n_spheres = len(radii)

        wavelengths_m = self.wavelengths.get_wavelengths_m()
        wavelengths_raw = wavelengths_m / self.wavelengths.scale

        nk_particle = self.particles.get_nk(wavelengths_raw, base_dir)
        nk_medium = self.medium.get_nk(wavelengths_raw, base_dir)

        f = FaSTMM2()
        solver = self.solver
        f.set_truncation_formula(solver.truncation_formula)
        results = []
        for i, wl in enumerate(wavelengths_m):
            n_part = nk_particle[i]
            n_med = nk_medium[i]
            eps_eff = (n_part / n_med) ** 2
            eps = np.full(n_spheres, eps_eff)
            k = 2.0 * np.pi * n_med / wl
            if self.output.compute_tmatrix:
                result = f.compute_tmatrix(
                    coords,
                    radii,
                    eps,
                    complex(k),
                    t_order=self.output.t_order,
                    formulation=solver.formulation,
                    acc=solver.acc,
                    tol=solver.tolerance,
                    restart=solver.restart,
                    max_iter=solver.max_iterations,
                )
            else:
                result = f.solve(
                    coords,
                    radii,
                    eps,
                    complex(k),
                    N_theta=solver.N_theta,
                    N_phi=solver.N_phi,
                    N_ave=solver.N_ave,
                    halton_init=solver.halton_init,
                    formulation=solver.formulation,
                    acc=solver.acc,
                    tol=solver.tolerance,
                    restart=solver.restart,
                    max_iter=solver.max_iterations,
                )
            result["wavelength_m"] = float(wl)
            result["k"] = complex(k)
            results.append(result)
            if progress_callback is not None:
                progress_callback((i + 1) / len(wavelengths_m), float(wl))

        return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_config(path: str | os.PathLike[str]) -> SweepConfig:
    """Load a sweep TOML configuration file."""
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    return SweepConfig(**raw)


def run_sweep(
    config: SweepConfig,
    base_dir: str | os.PathLike[str] = "",
    progress_callback: Any = None,
) -> list[dict]:
    """Run a sweep. Thin wrapper around :meth:`SweepConfig.run`."""
    return config.run(base_dir, progress_callback)
