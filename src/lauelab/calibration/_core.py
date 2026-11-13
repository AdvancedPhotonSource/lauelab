# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Fit of a detector's rotation and translation to measured reflections."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import metadata
from numbers import Real
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from lauelab.indexing import Crystal, DetectorGeometry, InputError, load_crystal
from lauelab.indexing._liblaue import NativeCrystal

from ._model import HC_KEV_NM, Residuals, calculated_energy_kev, rotation_vector
from ._table import CalibrationTable

STATUSES = ("ok", "insufficient_data", "energy_mismatch", "not_converged")

# A measured energy further than this fraction from the calculated energy of
# its hkl most likely belongs to another harmonic of the reflection.
_ENERGY_MISMATCH = 0.25
# Igor rounds the fitted detector rotation vector to 1e-8 rad.
_ROTATION_STEP = 1e-8
# Largest accepted condition number of the scaled Jacobian at the solution; a
# larger one means some parameter combination is not determined by the data.
_CONDITION_LIMIT = 1e8


@dataclass(frozen=True)
class DetectorCalibration:
    """Result of :func:`calibrate_detector`.

    Lengths are in µm, rotation vectors are axis times angle in radians, and
    rotation matrices are proper, dimensionless ``(3, 3)`` arrays. Per-row
    arrays follow the rows of :attr:`table`.

    Attributes
    ----------
    status
        One of ``"ok"``, ``"insufficient_data"``, ``"energy_mismatch"`` and
        ``"not_converged"``. Only ``"ok"`` results have refined values; the
        others repeat the initial values.
    message
        Explanation of the status.
    table
        The measured reflections.
    initial, refined
        The detector before and after the fit. ``refined.rotation_vector`` and
        ``refined.translation`` are rounded as Igor rounds them: the rotation
        vector to 1e-8 rad, and the translation to the power of ten nearest
        1 % of a pixel.
    initial_sample_rotation, refined_sample_rotation
        Rotation from the crystal reference basis to the laboratory frame, so
        that ``q = rotation @ reference.T @ hkl``. They are equal when the
        sample rotation was not fitted.
    fit_sample_rotation
        Whether the sample rotation was fitted.
    azimuth_deg
        Required detector azimuth about the incident beam in degrees, or
        `None` when the sample rotation was not fitted.
    rotation_vector_sigma, translation_sigma_um, sample_rotation_sigma
        Standard errors of the detector rotation vector (rad), translation
        (µm), and sample rotation about the laboratory x, y and z axes (rad),
        from the fit residuals and Jacobian. They depend on Igor's relative
        weights and are not a calibrated measurement uncertainty. ``NaN`` when
        not fitted or not determined.
    error_start, error_end
        Igor's error, the root of the weighted mean square of the reflection
        errors (dimensionless, near radians) combined with the azimuth error,
        before and after the fit.
    n_evaluations
        Number of residual evaluations.
    predicted_pixels
        Full-detector ``(x, y)`` position of each reflection on the refined
        detector, with shape ``(n, 2)``; ``NaN`` off the detector.
    pixel_error
        ``predicted_pixels - table.pixels`` with shape ``(n, 2)``.
    angle_error_deg
        Angle between the calculated and the measured scattering vector of each
        row with a pixel, shape ``(n,)``.
    calculated_energy_kev
        Energy of each reflection calculated from the refined sample rotation
        and the crystal, shape ``(n,)``, as Igor's ``calculated_keV`` column.
    energy_error_ev
        Measured minus calculated energy in eV, shape ``(n,)``.
    reference_reciprocal
        Reciprocal basis of the crystal reference orientation, rows in 1/nm
        including 2π.
    """

    status: str
    message: str
    table: CalibrationTable
    initial: DetectorGeometry
    refined: DetectorGeometry
    initial_sample_rotation: np.ndarray
    refined_sample_rotation: np.ndarray
    fit_sample_rotation: bool
    azimuth_deg: float | None
    rotation_vector_sigma: np.ndarray
    translation_sigma_um: np.ndarray
    sample_rotation_sigma: np.ndarray
    error_start: float
    error_end: float
    n_evaluations: int
    predicted_pixels: np.ndarray
    pixel_error: np.ndarray
    angle_error_deg: np.ndarray
    calculated_energy_kev: np.ndarray
    energy_error_ev: np.ndarray
    reference_reciprocal: np.ndarray
    crystal_name: str
    crystal_source: str | None

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                value = np.array(value, copy=True)
                value.setflags(write=False)
                object.__setattr__(self, name, value)

    def __repr__(self) -> str:
        return (
            f"DetectorCalibration(status={self.status!r}, detector_id={self.refined.detector_id!r}, "
            f"rms_pixel_error={self.rms_pixel_error:.4g}, error_end={self.error_end:.4g})"
        )

    @property
    def success(self) -> bool:
        """Whether the fit produced refined values."""
        return self.status == "ok"

    @property
    def refined_reciprocal(self) -> np.ndarray:
        """Reciprocal basis rows (1/nm, including 2π) of the refined sample orientation.

        Pass it to :func:`calibrate_detector` to calibrate another detector
        with the sample rotation fixed.
        """
        return self.reference_reciprocal @ self.refined_sample_rotation.T

    @property
    def sample_rotation_vector(self) -> np.ndarray:
        """Refined sample rotation as an axis-angle vector in radians (Igor's rho)."""
        return rotation_vector(self.refined_sample_rotation)

    @property
    def rms_pixel_error(self) -> float:
        """Root-mean-square distance in pixels between predicted and measured positions."""
        distance = np.linalg.norm(self.pixel_error, axis=1)
        distance = distance[np.isfinite(distance)]
        return float(np.sqrt(np.mean(distance**2))) if len(distance) else float("nan")

    @property
    def rms_energy_error_ev(self) -> float:
        """Root-mean-square measured minus calculated energy in eV."""
        values = self.energy_error_ev[np.isfinite(self.energy_error_ev)]
        return float(np.sqrt(np.mean(values**2))) if len(values) else float("nan")

    def summary_note(self) -> str:
        """Return a one-line ASCII summary for the ``<note>`` of a geometry file.

        The text is shorter than 100 characters, the length of Igor's
        ``geoNote`` field.
        """
        note = (
            f"lauelab {_package_version()} {self.crystal_name} {len(self.table)}pk "
            f"{self.table.n_energies}E rms {self.rms_pixel_error:.3g}px"
        )
        note = note.encode("ascii", "replace").decode("ascii").replace("<", "").replace(">", "").replace("&", "")
        return note[:99]

    def provenance(self) -> dict:
        """Return the fit inputs and settings as a JSON-compatible dictionary."""
        return json.loads(json.dumps({
            "objective": "LaueGo Calibration.ipf CalibrationErrorRPinternal (version 1.01)",
            "optimizer": "scipy.optimize.least_squares/trf",
            "hc_kev_nm": HC_KEV_NM,
            "fit_sample_rotation": self.fit_sample_rotation,
            "azimuth_deg": self.azimuth_deg,
            "crystal_name": self.crystal_name,
            "crystal_source": self.crystal_source,
            "detector_id": self.initial.detector_id,
            "initial_rotation_vector_rad": self.initial.rotation_vector.tolist(),
            "initial_translation_um": self.initial.translation.tolist(),
            "initial_sample_rotation": self.initial_sample_rotation.tolist(),
            "hkl": self.table.hkl.tolist(),
            "pixels": _json_floats(self.table.pixels),
            "energy_kev": _json_floats(self.table.energy_kev),
            "energy_source": list(self.table.energy_source),
            "empirical_igor_parity": "pending",
        }))


def _package_version() -> str:
    try:
        return metadata.version("lauelab")
    except metadata.PackageNotFoundError:
        return "unknown"


def _json_floats(values: np.ndarray):
    return [None if not np.isfinite(v) else float(v) for v in np.ravel(values)]


def _reference(crystal: Crystal) -> np.ndarray:
    return NativeCrystal.create(crystal).reciprocal()


def _round_translation(translation: np.ndarray, detector: DetectorGeometry) -> np.ndarray:
    step = min(detector.size_x / detector.nx, detector.size_y / detector.ny) / 100.0
    step = 10.0 ** np.round(np.log10(step))
    return np.round(translation / step) * step + 0.0


def calibrate_detector(
    table: CalibrationTable,
    crystal: Crystal | str | Path,
    reciprocal,
    detector: DetectorGeometry,
    *,
    fit_sample_rotation: bool = True,
    azimuth_deg: float = 90.0,
    max_nfev: int = 2000,
) -> DetectorCalibration:
    """Fit a detector's rotation and translation to reflections of a known crystal.

    This is LaueGo's detector calibration (``OptimizeAll`` in
    ``Calibration.ipf``). For each row of ``table``, the scattering vector
    calculated from the crystal and the sample rotation is compared with the
    measured one: its direction from the pixel position and its length from the
    measured energy. Peaks with a measured energy have a larger weight,
    ``min((n / n_energy)**2, 40)``.

    When ``fit_sample_rotation`` is true (detector 0 in LaueGo), the sample
    rotation is fitted with the detector. A rotation of the sample and the
    detector together about the incident beam changes no measured value, so an
    extra term holds the detector azimuth at ``azimuth_deg``, weighted 10 times.
    When it is false (the other detectors), the sample rotation of
    ``reciprocal`` is fixed; use the ``refined_reciprocal`` of a calibration of
    detector 0.

    Parameters
    ----------
    table
        Measured reflections, such as the result of :func:`calibration_table`.
    crystal
        The calibration crystal, or the path of its crystal XML file.
    reciprocal
        Reciprocal basis rows in 1/nm, including 2π, of the sample orientation,
        such as ``Pattern.reciprocal``. Its rotation from the crystal reference
        basis is the starting sample rotation.
    detector
        Starting detector geometry, from :meth:`Geometry.detector
        <lauelab.indexing.Geometry.detector>`.
    fit_sample_rotation
        Fit the sample rotation with the detector. At least one measured energy
        is then required, as in LaueGo.
    azimuth_deg
        Detector azimuth about the incident beam in degrees, used only when
        ``fit_sample_rotation`` is true. 90 is the 34-ID-E value of LaueGo's
        beamline preference.
    max_nfev
        Maximum number of residual evaluations.

    Returns
    -------
    DetectorCalibration
        The refined detector, sample rotation, per-row errors, and status. A
        fit that cannot run or does not converge returns a result whose status
        says why.

    Raises
    ------
    InputError
        If an argument is malformed.
    """
    if not isinstance(table, CalibrationTable):
        raise InputError("table must be a CalibrationTable")
    if isinstance(crystal, (str, Path)):
        crystal = load_crystal(crystal)
    if not isinstance(crystal, Crystal):
        raise InputError("crystal must be a Crystal or a crystal file path")
    if not isinstance(detector, DetectorGeometry):
        raise InputError("detector must be a DetectorGeometry")
    if not isinstance(azimuth_deg, Real) or not np.isfinite(azimuth_deg):
        raise InputError("azimuth_deg must be a finite number")
    if isinstance(max_nfev, bool) or not isinstance(max_nfev, (int, np.integer)) or max_nfev < 1:
        raise InputError("max_nfev must be a positive integer")
    reciprocal = np.asarray(reciprocal, dtype=np.float64)
    if reciprocal.shape != (3, 3) or not np.isfinite(reciprocal).all():
        raise InputError("reciprocal must be a finite (3, 3) array")

    reference = _reference(crystal)
    sample0 = np.linalg.solve(reference, reciprocal).T
    if not np.allclose(sample0 @ sample0.T, np.eye(3), atol=1e-5) or np.linalg.det(sample0) <= 0:
        raise InputError(
            "reciprocal is not a rotation of the crystal's reference basis; "
            "use a pattern indexed with this crystal"
        )
    sample0 = Rotation.from_matrix(sample0).as_matrix()
    azimuth = float(azimuth_deg) if fit_sample_rotation else None
    residuals = Residuals(
        detector, reference, table.hkl, table.pixels, table.energy_kev, azimuth
    )
    r0 = detector.rotation_vector.copy()
    p0 = detector.translation.copy()

    def build(status, message, r, p, sample, sigmas, error_end, nfev):
        detector_fit, q_calc, direction, _, _, _ = residuals.parts(r, p, sample)
        predicted = detector_fit.q_to_pixel(q_calc, on_detector=True)
        pixel_error = np.where(residuals.has_pixel[:, None], predicted - table.pixels, np.nan)
        cos_angle = np.sum(direction * q_calc, axis=1) / np.linalg.norm(q_calc, axis=1)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        energy = calculated_energy_kev(q_calc)
        return DetectorCalibration(
            status, message, table, detector, detector_fit, sample0, sample,
            bool(fit_sample_rotation), azimuth, *sigmas,
            float(np.linalg.norm(residuals(r0, p0, sample0))), float(error_end), int(nfev),
            predicted, pixel_error, np.where(residuals.has_pixel, angle, np.nan), energy,
            (table.energy_kev - energy) * 1e3, reference, crystal.name, crystal.source,
        )

    no_sigma = (np.full(3, np.nan),) * 3
    error_start = float(np.linalg.norm(residuals(r0, p0, sample0)))

    def unchanged(status, message):
        return build(status, message, r0, p0, sample0, no_sigma, error_start, 0)

    n_parameters = 9 if fit_sample_rotation else 6
    if fit_sample_rotation and table.n_energies == 0:
        return unchanged(
            "insufficient_data",
            "fitting the sample rotation needs at least one measured energy",
        )
    if residuals.n_measurements <= n_parameters:
        return unchanged(
            "insufficient_data",
            f"{residuals.n_measurements} measured values cannot determine {n_parameters} parameters",
        )
    start_energy = calculated_energy_kev(table.hkl @ reference @ sample0.T)
    with np.errstate(invalid="ignore"):
        ratio = table.energy_kev / start_energy
    far = np.flatnonzero(np.isfinite(ratio) & (np.abs(ratio - 1.0) > _ENERGY_MISMATCH))
    if len(far):
        row = int(far[0])
        return unchanged(
            "energy_mismatch",
            f"row {row}: measured {table.energy_kev[row]:.4g} keV for hkl "
            f"{tuple(int(v) for v in table.hkl[row])}, calculated {start_energy[row]:.4g} keV; "
            "the energy may belong to another harmonic (see CalibrationTable.with_hkl)",
        )

    def sample_of(x):
        if not fit_sample_rotation:
            return sample0
        return Rotation.from_rotvec(x[6:9]).as_matrix() @ sample0

    def objective(x):
        return residuals(x[0:3], x[3:6], sample_of(x))

    x0 = np.concatenate([r0, p0, np.zeros(3)])[:n_parameters]
    pixel_um = min(detector.size_x / detector.nx, detector.size_y / detector.ny)
    scale = np.array([1e-3] * 3 + [pixel_um] * 3 + [1e-3] * 3)[:n_parameters]
    fit = least_squares(
        objective, x0, method="trf", x_scale=scale, max_nfev=int(max_nfev),
        xtol=1e-14, ftol=1e-14, gtol=1e-14,
    )
    if fit.status <= 0 or not np.isfinite(fit.x).all():
        return unchanged("not_converged", f"the fit did not converge: {fit.message}")

    r = np.round(fit.x[0:3] / _ROTATION_STEP) * _ROTATION_STEP + 0.0
    p = _round_translation(fit.x[3:6], detector)
    sample = sample_of(fit.x)
    # Parameters in units of their typical step, so that the condition number
    # compares rotations, translations and the sample rotation fairly.
    scaled = fit.jac * scale
    singular = np.linalg.svd(scaled, compute_uv=False)
    condition = singular[0] / singular[-1] if singular[-1] > 0 else np.inf
    if not np.isfinite(condition) or condition > _CONDITION_LIMIT:
        return unchanged(
            "insufficient_data",
            f"the reflections do not determine all {n_parameters} parameters "
            f"(condition number {condition:.3g})",
        )
    dof = residuals.n_measurements - n_parameters
    covariance = np.linalg.inv(scaled.T @ scaled) * (2.0 * fit.cost / dof)
    variance = np.diag(covariance) * scale**2
    sigma = np.sqrt(np.where(variance >= 0, variance, np.nan))
    sigmas = (sigma[0:3], sigma[3:6], sigma[6:9] if fit_sample_rotation else np.full(3, np.nan))
    error_end = float(np.linalg.norm(residuals(r, p, sample)))
    message = f"converged after {fit.nfev} evaluations: {fit.message}"
    return build("ok", message, r, p, sample, sigmas, error_end, fit.nfev)
