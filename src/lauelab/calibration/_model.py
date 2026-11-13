# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Detector and reflection model of the LaueGo detector calibration.

The functions follow LaueGo ``microN/Calibration.ipf`` (version 1.01). Rotation
vectors are axis times angle in radians, as in the geometry file, not the
Rodrigues vectors (axis times tan(angle/2)) used elsewhere in lauelab.
"""

from __future__ import annotations

import re

import numpy as np
from scipy.spatial.transform import Rotation

from lauelab.indexing import DetectorGeometry, InputError

#: Planck constant times speed of light in keV nm, the value of ``hc`` in
#: LaueGo ``Calibration.ipf``.
HC_KEV_NM = 1.239841984

_INCIDENT = np.array([0.0, 0.0, 1.0])
# Igor multiplies the azimuth error by 10 so that it outweighs the peak errors.
_AZIMUTH_WEIGHT = 10.0
# Igor caps the squared-error weight of a peak with a measured energy at 40.
_MAX_ENERGY_WEIGHT = 40.0


def rotation_matrix(rotation_vector) -> np.ndarray:
    """Return the rotation matrix of an axis-angle rotation vector in radians."""
    vector = np.asarray(rotation_vector, dtype=np.float64)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        raise InputError("a rotation vector must contain three finite values")
    return Rotation.from_rotvec(vector).as_matrix()


def rotation_vector(matrix) -> np.ndarray:
    """Return the axis-angle rotation vector in radians of a rotation matrix.

    The angle is in ``[0, pi]``, as in Igor's ``axisOfMatrix``.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise InputError("a rotation matrix must be a finite (3, 3) array")
    if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-6) or np.linalg.det(matrix) <= 0:
        raise InputError("the matrix is not a proper rotation")
    return Rotation.from_matrix(matrix).as_rotvec()


def matrix_from_igor(text: str) -> np.ndarray:
    """Read a 3 x 3 matrix that Igor wrote in a wave note.

    Igor writes a matrix such as ``rotation_matrix0`` or ``recip_lattice0`` as
    ``{{m00,m10,m20}{m01,m11,m21}{m02,m12,m22}}``: each inner group is one
    column. Reading the groups as rows gives the transpose, which for a
    rotation is the inverse rotation.

    Parameters
    ----------
    text
        The matrix text, with or without spaces and commas between groups.

    Returns
    -------
    numpy.ndarray
        The matrix with shape ``(3, 3)``, so that ``matrix[i, j]`` is Igor's
        ``m[i][j]``.

    Raises
    ------
    InputError
        If the text does not contain three groups of three numbers.
    """
    groups = re.findall(r"\{([^{}]*)\}", text)
    try:
        columns = [[float(value) for value in re.split(r"[,\s]+", group.strip())] for group in groups]
    except ValueError as error:
        raise InputError(f"cannot read an Igor matrix from {text!r}: {error}") from error
    if len(columns) != 3 or any(len(column) != 3 for column in columns):
        raise InputError(f"an Igor matrix needs three groups of three numbers; received {text!r}")
    return np.asarray(columns, dtype=np.float64).T


def detector_with(detector: DetectorGeometry, rotation_vector, translation) -> DetectorGeometry:
    """Return ``detector`` with another rotation vector (rad) and translation (µm)."""
    rotation_vector = np.asarray(rotation_vector, dtype=np.float64)
    return DetectorGeometry(
        detector.nx, detector.ny, detector.size_x, detector.size_y, detector.detector_id,
        np.asarray(translation, dtype=np.float64), rotation_vector, rotation_matrix(rotation_vector),
    )


def peak_directions(detector: DetectorGeometry, pixels) -> np.ndarray:
    """Return unit scattering vectors of full-detector pixels for a source at the origin.

    Parameters
    ----------
    detector
        Detector geometry.
    pixels
        Zero-based full-detector ``(x, y)`` coordinates with shape ``(n, 2)``.

    Returns
    -------
    numpy.ndarray
        Unit scattering vectors in the 34-ID-E laboratory frame with shape
        ``(n, 3)``. For a sample at depth zero these are the vectors that
        :meth:`~lauelab.indexing.Geometry.pixels_to_q` returns.
    """
    pixels = np.asarray(pixels, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise InputError("pixels must have shape (n, 2)")
    outgoing = detector.pixel_to_lab(pixels)
    outgoing /= np.linalg.norm(outgoing, axis=1)[:, None]
    q = outgoing - _INCIDENT
    return q / np.linalg.norm(q, axis=1)[:, None]


def detector_azimuth_deg(detector: DetectorGeometry) -> float:
    """Return the azimuth of a detector about the incident beam in degrees.

    This is Igor's ``AzimuthAngle``. For a detector far from the incident beam
    it is the azimuth of the detector's z axis; for a detector close to forward
    or back scattering it is the azimuth of its +x pixel direction.
    """
    rho = detector.rotation
    tan_size = min(detector.size_x, detector.size_y) / abs(detector.translation[2])
    center = rho @ detector.translation
    tan_center = np.hypot(center[0], center[1]) / abs(center[2])
    if tan_size > tan_center:
        angle = np.arctan2(rho[1, 0], rho[0, 0])
    else:
        sign = np.sign(detector.translation[2])
        angle = np.arctan2(sign * rho[1, 2], sign * rho[0, 2])
    return float(np.degrees(angle))


def calculated_energy_kev(q: np.ndarray) -> np.ndarray:
    """Return the photon energy that diffracts into each reciprocal vector ``q`` (1/nm)."""
    q = np.asarray(q, dtype=np.float64)
    length = np.linalg.norm(q, axis=-1)
    sin_theta = -q[..., 2] / length
    with np.errstate(divide="ignore", invalid="ignore"):
        energy = HC_KEV_NM * length / (4.0 * np.pi * sin_theta)
    return np.where(sin_theta > 0, energy, np.nan)


class Residuals:
    """Igor's ``CalibrationErrorRPinternal`` as a residual vector.

    The sum of squares of :meth:`__call__` is the square of the error that
    Igor's ``CalibrationErrorRPrho`` (``azimuth_deg`` set) or
    ``CalibrationErrorRP`` (``azimuth_deg`` is `None`) returns, so both reach
    the same minimum.
    """

    def __init__(self, detector, reference, hkl, pixels, energy_kev, azimuth_deg):
        self.detector = detector
        self.hkl_q = np.asarray(hkl, dtype=np.float64) @ reference
        pixels = np.asarray(pixels, dtype=np.float64)
        inside = (
            np.isfinite(pixels).all(axis=1)
            & (pixels[:, 0] >= 0) & (pixels[:, 0] <= detector.nx - 1)
            & (pixels[:, 1] >= 0) & (pixels[:, 1] <= detector.ny - 1)
        )
        energy = np.asarray(energy_kev, dtype=np.float64)
        self.has_energy = np.isfinite(energy) & (energy > 0)
        self.has_pixel = inside
        self.used = self.has_energy | self.has_pixel
        self.pixels = np.where(inside[:, None], pixels, 0.0)
        self.energy = np.where(self.has_energy, energy, np.nan)
        # Igor counts every row of the table here (DimSize), used or not.
        n = len(energy)
        n_energy = int(self.has_energy.sum())
        self.energy_weight = min((n / max(n_energy, 1)) ** 2, _MAX_ENERGY_WEIGHT)
        self.weight_sum = float(np.where(self.has_energy, self.energy_weight, 1.0)[self.used].sum())
        self.azimuth_deg = azimuth_deg

    @property
    def n_measurements(self) -> int:
        """Independent measured values: two per pixel, one per energy, one azimuth."""
        return (
            2 * int(self.has_pixel.sum()) + int(self.has_energy.sum())
            + (self.azimuth_deg is not None)
        )

    def parts(self, rotation_vector, translation, sample_rotation):
        """Return the per-peak model quantities for one set of parameters."""
        detector = detector_with(self.detector, rotation_vector, translation)
        q_calc = self.hkl_q @ sample_rotation.T
        q_calc_length = np.linalg.norm(q_calc, axis=1)
        direction = q_calc / q_calc_length[:, None]
        if self.has_pixel.any():
            measured = peak_directions(detector, self.pixels)
            direction = np.where(self.has_pixel[:, None], measured, direction)
        sin_theta = -direction[:, 2]
        q_length = np.where(
            self.has_energy, 4.0 * np.pi * sin_theta * self.energy / HC_KEV_NM, q_calc_length
        )
        dq = q_calc - q_length[:, None] * direction
        radial = np.sum(dq * direction, axis=1)
        perpendicular = dq - radial[:, None] * direction
        return detector, q_calc, direction, q_length, radial, perpendicular

    def __call__(self, rotation_vector, translation, sample_rotation) -> np.ndarray:
        detector, _, _, q_length, radial, perpendicular = self.parts(
            rotation_vector, translation, sample_rotation
        )
        scale = 1.0 / np.sqrt(self.weight_sum)
        with np.errstate(divide="ignore", invalid="ignore"):
            perpendicular = perpendicular / q_length[:, None]
            radial = np.sqrt(self.energy_weight) * radial / q_length
        valid = self.used & np.isfinite(q_length) & (q_length > 0)
        perpendicular = np.where((valid & self.has_pixel)[:, None], perpendicular, 0.0)
        radial = np.where(valid & self.has_energy, radial, 0.0)
        parts = [scale * perpendicular.ravel(), scale * radial]
        if self.azimuth_deg is not None:
            difference = (detector_azimuth_deg(detector) - self.azimuth_deg + 180.0) % 360.0 - 180.0
            parts.append(np.array([_AZIMUTH_WEIGHT * np.radians(difference)]))
        return np.concatenate(parts)
