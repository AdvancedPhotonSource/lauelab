# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""ABI-mode bindings for the in-process Laue indexing library."""

from __future__ import annotations

from dataclasses import dataclass
from os import fspath
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

import numpy as np

from lauelab._native import _load_library, ffi, get_library

from ._frame import roi_to_detector_pixels


class NativeCrystal:
    """Compiled crystal structure retained for repeated indexing calls."""

    def __init__(self, handle, library):
        self._library = library
        self._handle = ffi.gc(handle, library.laue_crystal_free)

    @classmethod
    def create(cls, crystal):
        library = get_library()
        error = ffi.new("char[256]")
        atoms = ffi.new("laue_atom[]", len(crystal.atoms))
        for index, atom in enumerate(crystal.atoms):
            atoms[index].name = atom.symbol.encode()
            atoms[index].x, atoms[index].y, atoms[index].z = atom.position
            atoms[index].occupancy = atom.occupancy
        cell = crystal.cell.in_angstrom
        handle = library.laue_crystal_create(
            crystal.name.encode(), crystal.space_group,
            cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma,
            atoms, len(crystal.atoms), error, 256,
        )
        if handle == ffi.NULL:
            message = ffi.string(error).decode(errors="replace")
            raise ValueError(f"Failed to initialize crystal: {message}")
        return cls(handle, library)

    def reciprocal(self) -> np.ndarray:
        """Return native reciprocal basis rows in ``1/nm``."""
        values = ffi.new("double[3][3]")
        if self._library.laue_crystal_reciprocal(self._handle, values):
            raise RuntimeError("Failed to read native reciprocal lattice")
        return np.asarray([[values[row][column] for column in range(3)] for row in range(3)])


@dataclass(frozen=True)
class WireGeometry:
    """Wire geometry in micrometres and beamline rotation conventions."""

    diameter_um: float
    F_um: float
    origin_um: np.ndarray
    axis: np.ndarray
    axis_rotated: np.ndarray
    rotation: np.ndarray
    rotation_magnitude_deg: float

    def __post_init__(self) -> None:
        for name, shape in (("origin_um", (3,)), ("axis", (3,)),
                            ("axis_rotated", (3,)), ("rotation", (3, 3))):
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite array with shape {shape}")
            value.setflags(write=False)
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class DetectorGeometry:
    """Validated metadata for one detector in a geometry file.

    Parameters
    ----------
    nx, ny
        Detector dimensions in unbinned pixels.
    size_x, size_y
        Detector dimensions in micrometres.
    detector_id
        Detector identifier stored in the geometry file.
    translation
        Detector-frame translation vector in micrometres.
    rotation_vector
        Axis-angle rotation vector in radians.
    rotation
        Rotation matrix from detector coordinates to beamline coordinates.

    Notes
    -----
    Instances are returned by :meth:`Geometry.detector`. Detector slots may be
    sparse, so a detector's physical slot cannot be inferred from
    :attr:`Geometry.detector_count`.
    """

    nx: int
    ny: int
    size_x: float
    size_y: float
    detector_id: str
    translation: np.ndarray
    rotation_vector: np.ndarray
    rotation: np.ndarray

    def __post_init__(self) -> None:
        shapes = {
            "translation": (3,),
            "rotation_vector": (3,),
            "rotation": (3, 3),
        }
        for name, shape in shapes.items():
            value = np.array(getattr(self, name), dtype=np.float64, copy=True)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite array with shape {shape}")
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def pixel_to_lab(self, pixels: np.ndarray) -> np.ndarray:
        """Convert full-detector pixels to beamline coordinates.

        Parameters
        ----------
        pixels
            Zero-based pixel coordinates with shape ``(..., 2)`` in ``(x, y)``
            order.

        Returns
        -------
        numpy.ndarray
            Beamline coordinates in micrometres with shape ``(..., 3)``.
        """
        pixels = np.asarray(pixels, dtype=np.float64)
        if pixels.shape[-1:] != (2,):
            raise ValueError("pixels must have shape (..., 2)")
        if not np.isfinite(pixels).all():
            raise ValueError("pixel coordinates must be finite")

        detector = np.empty(pixels.shape[:-1] + (3,), dtype=np.float64)
        detector[..., 0] = (
            (pixels[..., 0] - 0.5 * (self.nx - 1)) * self.size_x / self.nx
            + self.translation[0]
        )
        detector[..., 1] = (
            (pixels[..., 1] - 0.5 * (self.ny - 1)) * self.size_y / self.ny
            + self.translation[1]
        )
        detector[..., 2] = self.translation[2]
        return detector @ self.rotation.T

    def q_to_pixel(
        self,
        q: np.ndarray,
        *,
        depth: float | None = None,
        on_detector: bool = False,
    ) -> np.ndarray:
        """Project reciprocal-space vectors onto this detector.

        Invalid rays and, when requested, off-detector intersections are
        returned as ``NaN`` coordinates.

        Parameters
        ----------
        q
            Reciprocal-space vectors with shape ``(..., 3)``.
        depth
            Sample depth in micrometres. `None` places the source at the
            beamline origin.
        on_detector
            Replace intersections outside the detector with ``NaN``.

        Returns
        -------
        numpy.ndarray
            Zero-based full-detector ``(x, y)`` coordinates with shape
            ``(..., 2)``.
        """
        q = np.asarray(q, dtype=np.float64)
        if q.shape[-1:] != (3,):
            raise ValueError("q must have shape (..., 3)")
        if depth is not None and not np.isfinite(depth):
            raise ValueError("depth must be finite when provided")

        flat = q.reshape((-1, 3))
        result = np.full((len(flat), 2), np.nan, dtype=np.float64)
        norms = np.linalg.norm(flat, axis=1)
        finite = np.isfinite(flat).all(axis=1) & (norms > 0)
        if np.any(finite):
            qhat = flat[finite] / norms[finite, None]
            q_length = -2.0 * qhat[:, 2]
            outgoing = qhat * q_length[:, None] + np.array([0.0, 0.0, 1.0])
            source = np.array([0.0, 0.0, 0.0 if depth is None else depth])
            source_detector = self.rotation.T @ source
            outgoing_detector = outgoing @ self.rotation
            denominator = outgoing_detector[:, 2]
            valid_ray = (q_length >= 0) & (np.abs(denominator) > 1e-15)
            distance = np.full(len(qhat), np.nan)
            distance[valid_ray] = (
                self.translation[2] - source_detector[2]
            ) / denominator[valid_ray]
            valid_ray &= distance >= 0

            intersection = source_detector + distance[:, None] * outgoing_detector
            xy = np.empty((len(qhat), 2), dtype=np.float64)
            xy[:, 0] = (
                (intersection[:, 0] - self.translation[0]) * self.nx / self.size_x
                + 0.5 * (self.nx - 1)
            )
            xy[:, 1] = (
                (intersection[:, 1] - self.translation[1]) * self.ny / self.size_y
                + 0.5 * (self.ny - 1)
            )
            valid_ray &= np.isfinite(xy).all(axis=1)
            if on_detector:
                valid_ray &= (
                    (xy[:, 0] >= 0)
                    & (xy[:, 0] <= self.nx - 1)
                    & (xy[:, 1] >= 0)
                    & (xy[:, 1] <= self.ny - 1)
                )
            finite_indices = np.flatnonzero(finite)
            result[finite_indices[valid_ray]] = xy[valid_ray]
        return result.reshape(q.shape[:-1] + (2,))


class Geometry:
    """Parsed detector geometry retained for repeated in-process conversion.

    Parameters
    ----------
    path
        Path to a Laue geometry XML file. Only detector geometry is loaded by
        this public interface.

    Raises
    ------
    ValueError
        If the detector declarations are malformed, incomplete, duplicated, or
        contain invalid dimensions or physical parameters.
    ImportError
        If the native ``liblaue`` library is unavailable.

    Notes
    -----
    Detector indices are physical slots from the geometry file, not ordinal
    positions among active detectors. Slots may therefore be sparse.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        try:
            detector_ids = [
                (node.findtext("{*}ID") or "").strip()
                for node in ET.parse(self.path).findall(".//{*}Detectors/{*}Detector")
            ]
        except OSError as error:
            raise ValueError(f"Failed to load geometry {path}: {error}") from error
        except ET.ParseError:
            # Not an XML document. The native reader also accepts the older
            # "$tag value" geometry format, which has at most one declaration
            # per detector slot, so the duplicate-ID scan does not apply.
            detector_ids = []
        duplicates = sorted({value for value in detector_ids if value and detector_ids.count(value) > 1})
        if duplicates:
            raise ValueError(f"Failed to load geometry {path}: duplicate detector ID {duplicates[0]!r}")
        error = ffi.new("char[256]")
        library = get_library()
        handle = library.laue_geometry_from_file(fspath(path).encode(), error, 256)
        if handle == ffi.NULL:
            message = ffi.string(error).decode(errors="replace")
            raise ValueError(f"Failed to load geometry {path}: {message}")
        self._library = library
        self._handle = ffi.gc(handle, library.laue_geometry_free)

    def __repr__(self) -> str:
        return f"Geometry(path={str(self.path)!r}, detector_count={self.detector_count})"

    @property
    def detector_count(self) -> int:
        """Number of active detectors in the geometry."""
        return self._library.laue_geometry_detector_count(self._handle)

    @property
    def wire(self) -> WireGeometry | None:
        """Wire geometry, or `None` when the file has no wire section."""
        info = ffi.new("laue_wire_info *")
        error = ffi.new("char[256]")
        status = self._library.laue_geometry_wire_info(self._handle, info, error, 256)
        if status:
            raise RuntimeError(ffi.string(error).decode(errors="replace"))
        if not info.has_wire:
            return None
        return WireGeometry(
            info.dia,
            info.F,
            np.asarray(list(info.origin)),
            np.asarray(list(info.axis)),
            np.asarray(list(info.axisR)),
            np.asarray([list(row) for row in info.R]),
            info.Rmag,
        )

    def find_detector(self, detector_id: str) -> int:
        """Find the physical slot for a detector identifier.

        Parameters
        ----------
        detector_id
            Exact detector identifier from the geometry file.

        Returns
        -------
        int
            Physical detector slot, or ``-1`` if the identifier is absent.
        """
        return self._library.laue_geometry_find_detector(self._handle, detector_id.encode())

    def detector(self, detector_index: int = 0) -> DetectorGeometry:
        """Return metadata for an active detector slot.

        Parameters
        ----------
        detector_index
            Physical detector slot from the geometry file.

        Returns
        -------
        DetectorGeometry
            Validated detector dimensions and identity.

        Raises
        ------
        ValueError
            If the slot is outside the supported range or is inactive.
        """
        info = ffi.new("laue_detector_info *")
        error = ffi.new("char[256]")
        status = self._library.laue_geometry_detector_info(
            self._handle, detector_index, info, error, 256
        )
        if status:
            message = ffi.string(error).decode(errors="replace")
            raise ValueError(message)
        return DetectorGeometry(
            info.nx,
            info.ny,
            info.size_x,
            info.size_y,
            ffi.string(info.detector_id).decode(errors="replace"),
            np.asarray(list(info.translation)),
            np.asarray(list(info.rotation_vector)),
            np.asarray([list(row) for row in info.rotation]),
        )

    def pixels_to_q(
        self,
        peaks: np.ndarray,
        *,
        detector_index: int = 0,
        start: tuple[int, int] = (0, 0),
        group: tuple[int, int] = (1, 1),
        depth: Optional[float] = None,
    ) -> np.ndarray:
        """Convert peak coordinates to unit scattering vectors.

        Parameters
        ----------
        peaks
            Array-like peak coordinates with shape ``(n, 2)``, whose columns
            are zero-based ``(x, y)`` coordinates in the supplied frame, or a
            structured array with shape ``(n,)`` and ``fit_x`` and ``fit_y``
            fields, such as ``FrameResult.peaks`` or ``PeakSearch.peaks``.
        detector_index
            Physical detector slot from the geometry file.
        start
            Zero-based detector ``(x, y)`` origin of the frame.
        group
            Positive detector-pixel grouping factors as ``(x, y)``.
        depth
            Optional finite sample depth in micrometres passed to the geometry
            calculation.

        Returns
        -------
        numpy.ndarray
            Unit scattering vectors with shape ``(n, 3)`` and ``float64``
            dtype.

        Raises
        ------
        ValueError
            If coordinates, region parameters, depth, or detector selection
            are invalid.
        RuntimeError
            If native pixel-to-q conversion fails.

        Notes
        -----
        Grouped coordinates are mapped to the center of the corresponding
        detector-pixel group before bounds validation and conversion.
        """
        names = getattr(getattr(peaks, "dtype", None), "names", None)
        if names is not None:
            if np.ndim(peaks) != 1 or not {"fit_x", "fit_y"} <= set(names):
                raise ValueError("structured peaks must have shape (n,) and fit_x and fit_y fields")
            peaks = np.column_stack([peaks["fit_x"], peaks["fit_y"]])
        pixels = np.asarray(peaks, dtype=np.float64)
        if pixels.ndim != 2 or pixels.shape[1] != 2:
            raise ValueError("peaks must have shape (n, 2)")
        if len(start) != 2 or len(group) != 2:
            raise ValueError("start and group must each contain two integers")
        if not all(isinstance(value, (int, np.integer)) for value in (*start, *group)):
            raise ValueError("start and group values must be integers")
        if min(start) < 0 or min(group) < 1:
            raise ValueError("start values must be nonnegative and group values positive")
        if depth is not None and not np.isfinite(depth):
            raise ValueError("depth must be finite when provided")
        if not np.isfinite(pixels).all():
            raise ValueError("peak coordinates must be finite")
        detector = self.detector(detector_index)
        if len(pixels):
            detector_pixels = roi_to_detector_pixels(pixels, start, group)
            if (detector_pixels < 0).any() or (detector_pixels[:, 0] > detector.nx - 1).any() or (
                detector_pixels[:, 1] > detector.ny - 1
            ).any():
                raise ValueError(
                    f"peak coordinates fall outside detector bounds {detector.nx}x{detector.ny}"
                )

        qhat, status, message = self._pixels_to_q(pixels, detector_index, start, group, depth)
        if status:
            raise RuntimeError(message)
        return qhat

    def _pixels_to_q(self, pixels, detector_index, start, group, depth):
        """Convert ``(n, 2)`` coordinates without Python-side validation.

        Returns the ``(n, 3)`` unit scattering vectors, the native status, and
        the native message, which is empty on success.
        """
        c_peaks = ffi.new("laue_peak[]", len(pixels))
        for index, (x, y) in enumerate(pixels):
            c_peaks[index].fit_x = x
            c_peaks[index].fit_y = y

        result = ffi.new("laue_frame_result *")
        result.startx, result.starty = start
        result.groupx, result.groupy = group
        result.depth = np.nan if depth is None else depth
        result.n_peaks = len(pixels)
        result.peaks = c_peaks
        status = self._library.laue_pixels_to_q(self._handle, detector_index, result)
        message = ffi.string(result.message).decode(errors="replace") if status else ""
        qhat = np.asarray(
            [[c_peaks[i].qhat[j] for j in range(3)] for i in range(len(pixels))],
            dtype=np.float64,
        ).reshape((-1, 3))
        return qhat, status, message


def load_geometry(path: str | Path) -> Geometry:
    """Load detector geometry for reuse.

    Parameters
    ----------
    path
        Path to a Laue geometry XML file.

    Returns
    -------
    Geometry
        Parsed detector geometry backed by native state.

    Raises
    ------
    ValueError
        If the detector geometry is invalid.
    ImportError
        If the native ``liblaue`` library is unavailable.
    """
    return Geometry(path)


def version() -> str:
    return ffi.string(get_library().laue_version()).decode()
