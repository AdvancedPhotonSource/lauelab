# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Shared frame input and detector ROI conversions."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
import os
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ScanFrame:
    """Reference to a reconstructed depth image for indexing.

    Supply the point file path, point ID, and depth index to select a stored
    frame. The reference can be pickled, sent to a worker, and saved with
    indexing results. Each process opens the point file when it reads the
    frame; a scan catalog is unnecessary.

    Parameters
    ----------
    path : pathlib.Path or str
        A point file, such as ``points/<input stem>.h5`` of a scan directory
        written by :func:`~lauelab.reconstruct.reconstruct_scan`, or a
        standalone file written by :func:`~lauelab.reconstruct.reconstruct_point`.
        :meth:`ScanReader.point_path <lauelab.reconstruct.ScanReader.point_path>`
        resolves a catalog point ID to this path.
    point_id : str
        Expected point ID. Reading fails if the ID in the file differs.
    depth_index : int
        Zero-based position of the frame in the point's depth stack. Its
        physical depth in µm is read from the file.
    """

    path: str
    point_id: str
    depth_index: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", os.fspath(self.path))
        if not isinstance(self.point_id, str) or not self.point_id:
            raise ValueError("point_id must be a non-empty string")
        if isinstance(self.depth_index, (bool, np.bool_)) or not isinstance(self.depth_index, Integral):
            raise TypeError("depth_index must be an integer")
        if self.depth_index < 0:
            raise ValueError("depth_index must be nonnegative")
        object.__setattr__(self, "depth_index", int(self.depth_index))


def read_scan_frame(frame: ScanFrame):
    """Read one stored frame, its metadata, and its processing values.

    Returns the same ``(image, metadata, processing)`` triple as
    :func:`read_h5_frame`. Only the selected frame is read from the file.
    """
    from lauelab.reconstruct import PointReader

    with PointReader(frame.path) as point:
        if point.point_id != frame.point_id:
            raise ValueError(f"{frame.path} holds point {point.point_id!r}, not {frame.point_id!r}")
        image = point.frame(frame.depth_index)
        metadata = {
            "scan_number": point.scan_number,
            "energy_kev": point.energy_kev,
            "detector_id": point.detector_id or None,
            "sample_position": (
                point.sample_position if np.isfinite(point.sample_position).all() else None
            ),
        }
        metadata = {name: value for name, value in metadata.items() if value is not None}
        processing = {
            "start": point.start,
            "group": point.group,
            "depth": float(point.depth_um[frame.depth_index]),
        }
    return image, metadata, processing


def roi_to_detector_pixels(points, start, group):
    """Convert ROI pixel centers to full-detector pixel coordinates."""
    points = np.asarray(points)
    start = np.asarray(start)
    group = np.asarray(group)
    return start + points * group + (group - 1) / 2.0


def detector_to_roi_pixels(points, start, group):
    """Convert full-detector pixel coordinates to ROI pixel centers."""
    points = np.asarray(points)
    start = np.asarray(start)
    group = np.asarray(group)
    return (points - start - (group - 1) / 2.0) / group


def roi_inclusive_end(image_shape, start, group):
    """Return the inclusive full-detector ``(x, y)`` endpoint of an ROI."""
    size = np.asarray(image_shape)[::-1]
    return tuple(np.asarray(start) + size * np.asarray(group) - 1)


def read_h5_frame(path: str | Path):
    """Read the supported HDF5 image, metadata, and optional ROI settings."""
    try:
        import h5py
    except ImportError as error:
        raise ImportError("h5py is required to read an HDF5 frame") from error

    def scalar(source, name):
        if name not in source:
            return None
        value = source[name][()]
        if isinstance(value, np.ndarray):
            value = value.flat[0]
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return value.item() if isinstance(value, np.generic) else value

    def integer(source, name):
        value = scalar(source, name)
        if value is None:
            return None
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise ValueError(f"HDF5 field {name!r} must be an integer")
        converted = int(value)
        if not np.isfinite(value) or value != converted:
            raise ValueError(f"HDF5 field {name!r} must be an integer")
        return converted

    def finite_real(source, name):
        # The LaueGo reader takes the first element of a numeric dataset and
        # treats a non-finite value as "no value".
        if name not in source:
            return None
        values = np.asarray(source[name]).ravel()
        if values.size == 0:
            return None
        if values.dtype.kind not in "iuf":
            raise ValueError(f"HDF5 field {name!r} must be a real number")
        value = float(values[0])
        return value if np.isfinite(value) else None

    with h5py.File(path, "r") as source:
        image = source["entry1/data/data"][...]
        shutter = scalar(source, "entry1/microDiffraction/CCDshutter")
        position = tuple(
            scalar(source, name) for name in (
                "entry1/sample/sampleX",
                "entry1/sample/sampleY",
                "entry1/sample/sampleZ",
            )
        )
        file_time = source.attrs.get("file_time")
        if isinstance(file_time, bytes):
            file_time = file_time.decode("utf-8")
        metadata = {
            "title": scalar(source, "entry1/title"),
            "sample_name": scalar(source, "entry1/sample/name"),
            "user_name": scalar(source, "entry1/user/name"),
            "beamline": scalar(source, "Facility/facility_beamline"),
            "scan_number": scalar(source, "entry1/scanNum"),
            "date_exposed": file_time,
            "beam_bad": scalar(source, "entry1/microDiffraction/BeamBad"),
            "ccd_shutter": (
                ("out" if shutter else "in") if shutter is not None else None
            ),
            "light_on": scalar(source, "entry1/microDiffraction/LightOn"),
            "mono_mode": scalar(source, "entry1/microDiffraction/MonoMode"),
            "sample_position": (
                position if all(value is not None for value in position) else None
            ),
            "energy_kev": scalar(source, "entry1/sample/incident_energy"),
            "hutch_temperature": scalar(source, "entry1/microDiffraction/HutchTemperature"),
            "sample_distance": scalar(source, "entry1/sample/distance"),
            "detector_id": scalar(source, "entry1/detector/ID"),
            "exposure_seconds": scalar(source, "entry1/detector/exposure"),
        }
        metadata = {name: value for name, value in metadata.items() if value is not None}

        processing = {}
        start = (
            integer(source, "entry1/detector/startx"),
            integer(source, "entry1/detector/starty"),
        )
        group = (
            integer(source, "entry1/detector/binx"),
            integer(source, "entry1/detector/biny"),
        )
        if all(value is not None for value in start):
            processing["start"] = start
        if all(value is not None for value in group):
            processing["group"] = group
        depth = finite_real(source, "entry1/depth")
        if depth is not None:
            processing["depth"] = depth

    return image, metadata, processing


def load_mask(path: str | Path) -> np.ndarray:
    """Load a peak-search mask stored as a 34-ID-E HDF5 image.

    Parameters
    ----------
    path
        HDF5 file whose ``entry1/data/data`` dataset holds the mask image. Any
        numeric dtype is accepted.

    Returns
    -------
    numpy.ndarray
        Two-dimensional boolean array. `True` marks a pixel excluded from peak
        search. This follows the LaueGo ``peaksearch -K`` convention: nonzero
        mask pixels are excluded and zero pixels remain available.

    Raises
    ------
    OSError
        If the file cannot be opened.
    KeyError
        If the image dataset is missing.
    ValueError
        If the dataset is not a two-dimensional numeric array.

    Notes
    -----
    The mask shape is checked against the frame when it is passed to
    :meth:`Indexer.index`, not here.
    """
    try:
        import h5py
    except ImportError as error:
        raise ImportError("h5py is required to read an HDF5 mask") from error

    with h5py.File(path, "r") as source:
        values = source["entry1/data/data"][...]
    if values.ndim != 2 or values.dtype.kind not in "iufb":
        raise ValueError(
            f"mask {path} must be a two-dimensional numeric image; "
            f"received shape={values.shape}, dtype={values.dtype}"
        )
    return values != 0
