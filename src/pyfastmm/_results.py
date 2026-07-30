"""Typed result containers for the class-driven FaSTMM API.

``FaSTMM.solve()``/``compute_tmatrix()`` store their latest result on the
instance (``FaSTMM.results``) as one of these pydantic models -- attribute
access with integrated validation, instead of a stringly-typed dict. The
methods' *return values* stay plain dicts (``to_dict()``'s output),
keeping every existing caller -- SweepConfig.run_resolved(), spcwth,
t-bench's adapter -- working unchanged.

The validation is not decorative: ``mueller``/``jones``/T-matrix blocks
are checked finite at construction, so a solve that silently produced
NaN physics (exactly what the pre-``mie-far-field-distance-fix`` RR bug
did for every absorbing medium -- all-NaN Mueller while cross sections
stayed plausible) now fails loudly at the source instead of surfacing as
corrupt science three scripts downstream. Deliberately *not* validated:
cross-section positivity -- the known ~0.1-0.2% deep-Rayleigh precision
floor can put c_abs marginally above c_ext (see space-weathering's
PLAN.md), and rejecting that would turn a characterized numerical floor
into spurious hard errors.
"""

from __future__ import annotations

import numpy as np
from pydantic import BaseModel, ConfigDict, field_validator


class _ResultBase(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def to_dict(self) -> dict:
        """The plain-dict form of this result -- the keys the pre-Results
        dict API returned (``jones`` omitted when absent), plus the
        additive ``wavelength_vacuum``."""
        d = {name: getattr(self, name) for name in type(self).model_fields}
        if d.get("jones") is None:
            d.pop("jones", None)
        return d


class SolveResult(_ResultBase):
    """One ``FaSTMM.solve()`` outcome. ``jones`` is only present for
    fixed orientation (``N_ave == 0``); cross sections carry the host-
    medium flux normalization already applied (see ``FaSTMM.solve()``)."""

    mueller: np.ndarray
    jones: np.ndarray | None = None
    c_ext: float
    c_ext_minus_c_abs: float
    c_abs: float
    c_sca: float
    asymmetry: float
    k_medium: complex
    eps: np.ndarray
    wavelength_vacuum: float

    @field_validator("mueller", "jones")
    @classmethod
    def _finite(cls, v):
        if v is not None and not np.all(np.isfinite(v)):
            raise ValueError(
                "contains non-finite values -- the solve produced NaN/inf "
                "physics (historically the signature of far-field underflow "
                "in absorbing media; see the mie-far-field-distance-fix patch)"
            )
        return v


class TMatrixResult(_ResultBase):
    """One ``FaSTMM.compute_tmatrix()`` outcome: the four T-matrix blocks
    (ordering ``(-1(1),0(1),1(1),-2(2),...,N(N))``, see
    ``get_tmatrix_size()``) plus the wavenumber/permittivity actually
    used."""

    Taa: np.ndarray
    Tab: np.ndarray
    Tba: np.ndarray
    Tbb: np.ndarray
    k_medium: complex
    eps: np.ndarray
    wavelength_vacuum: float

    @field_validator("Taa", "Tab", "Tba", "Tbb")
    @classmethod
    def _finite(cls, v):
        if not np.all(np.isfinite(v)):
            raise ValueError("contains non-finite values")
        return v
