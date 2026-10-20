# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Shared frame input and detector ROI conversions."""

from __future__ import annotations

from numbers import Real
from pathlib import Path

import numpy as np


def _energy_fields(source):
    """Read single-frame physics metadata without guessing a vector element.

    Indexing reads the first element of each metadata array. Energy callers
    opt in and validate the effective values after explicit metadata overrides.
    """
    fields = {
        "energy_kev": "entry1/sample/incident_energy",
        "mono_mode": "entry1/microDiffraction/MonoMode",
        "exposure_seconds": "entry1/detector/exposure",
        "i0_counts": "entry1/monitor/I0",
        "scaler_count_time_seconds": "entry1/monitor/ScalerCountTime",
    }
    metadata = {}
    for name, path in fields.items():
        if path not in source:
            continue
        value = source[path][()]
        if isinstance(value, np.ndarray) and value.size == 1:
            value = value.reshape(-1)[0]
        if isinstance(value, np.generic):
            value = value.item()
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        metadata[name] = value
    units = source["entry1/data/data"].attrs.get("units")
    if units is not None:
        metadata["intensity_units"] = units.decode("utf-8") if isinstance(units, bytes) else str(units)
    return metadata


def read_scan_frame(frame, *, energy_metadata=False):
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
        if energy_metadata:
            metadata.update(_energy_fields(point._file))
            metadata["prior_processing"] = {**point.settings, "norm_rescale": point.norm_rescale}
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


def read_h5_frame(path: str | Path, *, energy_metadata=False):
    """Read the supported HDF5 image, metadata, and optional ROI settings.

    With energy_metadata=True, stored origin and binning values must contain
    both axes or neither.
    """
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

        if energy_metadata:
            metadata.update(_energy_fields(source))

        processing = {}
        start = (
            integer(source, "entry1/detector/startx"),
            integer(source, "entry1/detector/starty"),
        )
        group = (
            integer(source, "entry1/detector/binx"),
            integer(source, "entry1/detector/biny"),
        )
        for name, values, fields in (
            ("start", start, "startx and starty"),
            ("group", group, "binx and biny"),
        ):
            present = [value is not None for value in values]
            if energy_metadata and any(present) and not all(present):
                raise ValueError(f"HDF5 detector {name} requires both {fields} when either is supplied")
            if all(present):
                processing[name] = values
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
