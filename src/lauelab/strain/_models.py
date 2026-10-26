# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Owned strain results and aligned notebook tables."""

from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

from lauelab.indexing.errors import InputError
from ._inputs import StrainReflections, _provenance, _readonly
from ._layout import STATUSES
from ._tensor import (
    _cell_parameters,
    _coordinates,
    _mask,
    _orientation,
    _strain,
    _matrix,
)


@dataclass(frozen=True)
class StrainResult:
    """Independent evidence for one pattern's deviatoric-strain refinement.

    ``tensor`` is (3,3), dimensionless tensor shear in ``coordinates``.
    ``reciprocal`` and ``reference_reciprocal`` are rows in 1/nm including
    2π. ``rotation`` is the fitted canonical-cell orientation. Failed fits
    have a NaN tensor; a finite candidate lattice may remain for diagnosis.
    ``measurements`` preserves selected inputs and source identities.
    Predictions/residuals follow that same row order. Arrays are owned and
    read-only; provenance access returns a copy. Convergence/support are
    numerical diagnostics, not calibrated experimental confidence.

    Attributes
    ----------
    status, message : str
        Numerical outcome and diagnostic explanation. ``success`` is true
        only for ``"converged"``.
    coordinates, fit_mask : str
        Tensor frame and six effective cell-parameter flags.
    tensor, rotation : numpy.ndarray
        Dimensionless strain and proper fitted canonical-cell rotation,
        each shape ``(3, 3)``.
    reciprocal, reference_reciprocal : numpy.ndarray
        Refined and reference reciprocal rows, shape ``(3, 3)``, 1/nm with 2π.
    cell_parameters, working_cell_parameters : numpy.ndarray
        Final volume-normalized and pre-normalization fitted cell values,
        respectively: a,b,c in nm, alpha,beta,gamma in degrees, shape ``(6,)``.
    measurements : StrainReflections
        Original prepared inputs, identities and calibration snapshot.
    predicted_qhat, predicted_xy, angular_residual_deg : numpy.ndarray
        Aligned unit directions ``(n, 3)``, zero-based frame pixels ``(n, 2)``
        and nonnegative angular errors in degrees ``(n,)``. Unavailable
        predictions are NaN.
    n_parameters, rank, nfev : int
        Free parameter count, joint Jacobian rank and optimizer evaluations.
    condition_number : float
        Joint Jacobian condition number; infinity when unavailable.
    provenance_json : str
        JSON snapshot of effective model, optimizer settings and conventions.
    """

    status: str
    message: str
    coordinates: str
    fit_mask: str
    tensor: np.ndarray
    reciprocal: np.ndarray
    reference_reciprocal: np.ndarray
    rotation: np.ndarray
    cell_parameters: np.ndarray
    working_cell_parameters: np.ndarray
    measurements: StrainReflections
    predicted_qhat: np.ndarray
    predicted_xy: np.ndarray
    angular_residual_deg: np.ndarray
    n_parameters: int
    rank: int
    condition_number: float
    nfev: int
    provenance_json: str = field(default="{}", repr=False)

    def __post_init__(self):
        if self.status not in STATUSES:
            raise InputError(f"unknown strain status {self.status!r}")
        _coordinates(self.coordinates)
        mask, _ = _mask(self.fit_mask)
        if not isinstance(self.measurements, StrainReflections):
            raise TypeError("measurements must be StrainReflections")
        n = len(self.measurements.hkl)
        shapes = {
            "tensor": (3, 3),
            "reciprocal": (3, 3),
            "reference_reciprocal": (3, 3),
            "rotation": (3, 3),
            "cell_parameters": (6,),
            "working_cell_parameters": (6,),
            "predicted_qhat": (n, 3),
            "predicted_xy": (n, 2),
            "angular_residual_deg": (n,),
        }
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name))
            if (
                value.dtype.kind not in "iuf"
                or value.shape != shape
                or np.isinf(value).any()
            ):
                raise InputError(
                    f"{name} must be a real array with shape {shape}, without infinities"
                )
            object.__setattr__(self, name, _readonly(value))
        _matrix(self.reference_reciprocal, "reference_reciprocal")
        for name in ("n_parameters", "rank", "nfev"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < 0
            ):
                raise InputError(f"{name} must be a nonnegative integer")
        if self.n_parameters != 3 + int(mask.sum()) or self.rank > self.n_parameters:
            raise InputError("parameter/rank diagnostics disagree with the fit mask")
        if np.isnan(self.condition_number) or self.condition_number < 1:
            raise InputError("condition_number must be >=1 or infinity")
        _provenance(self.provenance_json)
        if self.success:
            _matrix(self.reciprocal)
            if (
                self.measurements.initial_reciprocal is None
                or 2 * n <= self.n_parameters
            ):
                raise InputError(
                    "a converged result requires a starting lattice and redundant measurements"
                )
            for name in (
                "tensor",
                "rotation",
                "cell_parameters",
                "working_cell_parameters",
                "predicted_qhat",
            ):
                if not np.isfinite(getattr(self, name)).all():
                    raise InputError(f"converged result {name} must be finite")
            if not np.allclose(
                self.rotation, _orientation(self.reciprocal), rtol=0, atol=1e-10
            ) or not np.allclose(
                self.rotation @ self.rotation.T, np.eye(3), rtol=0, atol=1e-10
            ):
                raise InputError(
                    "recorded rotation disagrees with the fitted canonical-cell orientation"
                )
            if (
                abs(
                    np.linalg.slogdet(self.reciprocal)[1]
                    - np.linalg.slogdet(self.reference_reciprocal)[1]
                )
                > 1e-10
            ):
                raise InputError("refined cell does not preserve reference volume")
            if not np.allclose(
                self.cell_parameters,
                _cell_parameters(self.reciprocal),
                rtol=1e-11,
                atol=1e-11,
            ):
                raise InputError(
                    "recorded cell parameters disagree with the refined reciprocal lattice"
                )
            predicted = self.measurements.hkl @ self.reciprocal
            predicted /= np.linalg.norm(predicted, axis=1)[:, None]
            if not np.allclose(self.predicted_qhat, predicted, rtol=0, atol=1e-11):
                raise InputError(
                    "predicted directions disagree with fitted lattice and HKLs"
                )
            angles = np.degrees(
                np.arctan2(
                    np.linalg.norm(np.cross(predicted, self.measurements.qhat), axis=1),
                    np.sum(predicted * self.measurements.qhat, axis=1),
                )
            )
            if not np.allclose(self.angular_residual_deg, angles, rtol=0, atol=1e-10):
                raise InputError(
                    "angular residuals disagree with measured/predicted directions"
                )
            expected = _strain(
                self.reciprocal, self.reference_reciprocal, self.coordinates
            )
            if not np.allclose(self.tensor, expected, rtol=0, atol=1e-11):
                raise InputError(
                    "tensor disagrees with the recorded lattice/reference/frame"
                )
            if (
                self.rank != self.n_parameters
                or not np.isfinite(self.angular_residual_deg).all()
            ):
                raise InputError(
                    "a converged result requires full rank and finite residuals"
                )
        elif not np.isnan(self.tensor).all():
            raise InputError(
                "unsuccessful results must mark the tensor unavailable with NaNs"
            )

    @property
    def success(self):
        return self.status == "converged"

    @property
    def provenance(self):
        return _provenance(self.provenance_json)

    @property
    def frame_id(self):
        return self.measurements.frame_id

    @property
    def pattern_index(self):
        return self.measurements.pattern_index

    @property
    def rms_error_deg(self):
        return (
            float(np.sqrt(np.mean(self.angular_residual_deg**2)))
            if len(self.angular_residual_deg)
            else np.nan
        )

    def tensor_in(self, coordinates="beamline"):
        """Return tensor components in a named Igor frame without refitting."""
        _coordinates(coordinates)
        if not self.success:
            return np.full((3, 3), np.nan)
        return _strain(self.reciprocal, self.reference_reciprocal, coordinates)

    @property
    def von_mises_strain(self):
        """Return Igor's scalar ``sqrt(3/2 * sum(epsilon_dev**2))``."""
        return float(np.sqrt(1.5 * np.sum(self.tensor**2)))

    @property
    def sum_abs_epsilon_crystal(self):
        """Crystal-frame component sum; this quantity is not rotation invariant."""
        return float(np.sum(np.abs(self.tensor_in("crystal"))))

    @property
    def predicted_energy_kev(self):
        """Calculated reflection energies using Igor's hc=1.239841857 keV nm.

        These are predictions, not measured energies. Invalid incident-beam
        scattering directions and unavailable fitted lattices return NaN.
        """
        q = self.measurements.hkl @ self.reciprocal
        sine = -self.predicted_qhat[:, 2]
        energy = np.full(len(q), np.nan)
        np.divide(
            1.239841857 * np.linalg.norm(q, axis=1),
            4 * np.pi * sine,
            out=energy,
            where=sine > 0,
        )
        energy.setflags(write=False)
        return energy

    @property
    def reflections(self):
        """Return an immutable notebook Table of aligned inputs/predictions."""
        from lauelab.visualization.tables import Table

        data = self.measurements
        columns = {
            "peak_index": data.peak_indices,
            "angular_residual_deg": self.angular_residual_deg,
            "predicted_energy_kev": self.predicted_energy_kev,
        }
        for i, name in enumerate(("h", "k", "l")):
            columns[name] = data.hkl[:, i]
        for i, name in enumerate(("x", "y")):
            columns[f"measured_{name}_px"] = data.xy[:, i]
            columns[f"predicted_{name}_px"] = self.predicted_xy[:, i]
            columns[f"residual_{name}_px"] = self.predicted_xy[:, i] - data.xy[:, i]
        for i, name in enumerate("xyz"):
            columns[f"measured_qhat_{name}"] = data.qhat[:, i]
            columns[f"predicted_qhat_{name}"] = self.predicted_qhat[:, i]
        return Table(columns)
