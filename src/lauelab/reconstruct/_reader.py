# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""HDF5 input conventions for in-process wire-scan reconstruction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import re

import h5py
import numpy as np

from lauelab.indexing.errors import InputError


@dataclass(frozen=True)
class ImageGeometry:
    """Detector ROI geometry in unbinned-pixel coordinates.

    Parameters
    ----------
    nx_full : int
        Full detector width in unbinned pixels.
    ny_full : int
        Full detector height in unbinned pixels.
    start : tuple of int
        Zero-based ``(x, y)`` coordinate of the ROI origin in unbinned pixels.
        The default is ``(0, 0)``.
    group : tuple of int
        ``(x, y)`` detector binning factors. The default is ``(1, 1)``.
    n_rows : int or None
        Binned image height. The default is ``None``, which uses
        ``ny_full // group[1]``.
    n_cols : int or None
        Binned image width. The default is ``None``, which uses
        ``nx_full // group[0]``.
    """

    nx_full: int
    ny_full: int
    start: tuple[int, int] = (0, 0)
    group: tuple[int, int] = (1, 1)
    n_rows: int | None = None
    n_cols: int | None = None

    @property
    def shape(self) -> tuple[int, int]:
        """Binned image shape as ``(rows, columns)``."""
        return (
            self.n_rows if self.n_rows is not None else self.ny_full // self.group[1],
            self.n_cols if self.n_cols is not None else self.nx_full // self.group[0],
        )


@dataclass(frozen=True)
class ScanInfo:
    image_geometry: ImageGeometry
    shape: tuple[int, int, int]
    dtype: np.dtype
    intensity_map: np.ndarray | None
    wire_xyz: np.ndarray
    positioner: str
    file_time: str | None
    scale: np.ndarray | None
    scan_number: int | None
    sample_position: tuple[float, float, float]
    energy_kev: float | None


def _scalar(source: h5py.File, name: str, default=None):
    if name not in source:
        return default
    values = np.asarray(source[name]).ravel()
    if not len(values):
        return default
    value = values[0]
    return value.item() if isinstance(value, np.generic) else value


def _text(value) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def positioner_from_file_time(value: str | None) -> str:
    """Match the executable, which does not accept the ISO ``T`` separator."""
    if not value:
        return "none"
    match = re.match(r"^(\d{4})-(\d{2})-(\d{2})\s*(\d{1,2}):(\d{1,2}):(\d{1,2})", value)
    if match is None:
        return "none"
    try:
        taken = datetime(*map(int, match.groups()))
    except ValueError:
        return "none"
    if taken < datetime(2006, 5, 1):
        return "none"
    if taken < datetime(2009, 10, 1):
        return "pm500"
    return "alio"


def read_scan_info(source: h5py.File, normalization: str | None = None, *,
                   intensity_map: bool = True) -> ScanInfo:
    """Read metadata and aligned wire positions from an open scan file.

    ``intensity_map=False`` skips reading the intensity-map frame and leaves
    ``ScanInfo.intensity_map`` as ``None``; scan preparation needs only the
    metadata of each input.
    """
    try:
        return _read_scan_info(source, normalization, intensity_map)
    except (ValueError, TypeError, OverflowError, KeyError) as error:
        raise InputError(f"invalid metadata in {source.filename}: {error}") from error


def _read_scan_info(source, normalization, read_intensity_map) -> ScanInfo:
    name = "entry1/data/data"
    if name not in source:
        raise InputError(f"input file has no {name!r} dataset")
    data = source[name]
    if not isinstance(data, h5py.Dataset):
        raise InputError(f"{name!r} must be a dataset")
    if data.ndim != 3 or data.shape[0] < 5:
        raise InputError(
            "entry1/data/data needs at least 5 stored slices "
            "(1 skipped, 3 differenced, 1 unused)"
        )
    if not np.issubdtype(data.dtype, np.number):
        raise InputError(f"input images must have a numeric dtype, not {data.dtype}")

    rows, cols = data.shape[1:]
    nx = int(_scalar(source, "entry1/detector/Nx", cols))
    ny = int(_scalar(source, "entry1/detector/Ny", rows))
    start = (
        int(_scalar(source, "entry1/detector/startx", 0)),
        int(_scalar(source, "entry1/detector/starty", 0)),
    )
    group = (
        int(_scalar(source, "entry1/detector/binx", 1)),
        int(_scalar(source, "entry1/detector/biny", 1)),
    )

    # Slice 0 is bookkeeping. Slice 1 is the intensity map and first scan
    # frame; the executable reads but never differences the final stored slice.
    n_images = data.shape[0] - 2
    defaults = [float(_scalar(source, f"entry1/wire/wirebase{name}", 0.0)) for name in "XYZ"]
    vectors = []
    for name, default in zip("XYZ", defaults):
        path = f"entry1/wire/wire{name}"
        values = np.asarray(source[path], dtype=np.float64).ravel() if path in source else np.empty(0)
        aligned = np.full(n_images + 1, default if np.isfinite(default) else 0.0)
        available = values[2:n_images + 3]
        aligned[:len(available)] = available
        vectors.append(aligned)
    wire_xyz = np.ascontiguousarray(np.column_stack(vectors), dtype=np.float64)

    scale = None
    if normalization:
        path = f"entry1/{normalization}"
        if path not in source:
            raise InputError(f"normalization vector {normalization!r} is missing")
        values = np.asarray(source[path], dtype=np.float64).ravel()[1:n_images + 1]
        if len(values) < n_images:
            raise InputError(f"normalization vector {normalization!r} has fewer than {n_images + 1} entries")
        divisor = {"mA": 102.0, "cnt3": 88100.0}.get(normalization, 1.0)
        scale = np.ascontiguousarray(values / divisor)

    sample = tuple(float(_scalar(source, f"entry1/sample/sample{name}", np.nan)) for name in "XYZ")
    scan_number = _scalar(source, "entry1/scanNum")
    file_time = _text(source.attrs.get("file_time"))
    energy = _scalar(source, "entry1/sample/incident_energy")
    return ScanInfo(
        image_geometry=ImageGeometry(nx, ny, start, group, rows, cols),
        shape=(n_images, rows, cols),
        dtype=data.dtype,
        intensity_map=np.asarray(data[1], dtype=np.float64) if read_intensity_map else None,
        wire_xyz=wire_xyz,
        positioner=positioner_from_file_time(file_time),
        file_time=file_time,
        scale=scale,
        scan_number=None if scan_number is None else int(scan_number),
        sample_position=sample,
        energy_kev=None if energy is None else float(energy),
    )


def cutoff_mask(intensity_map: np.ndarray, percent_brightest: float) -> np.ndarray:
    """Compute the executable-compatible bright-pixel mask."""
    if not 0 < percent_brightest <= 100:
        raise InputError("percent_brightest must be greater than 0 and at most 100")
    ordered = np.sort(np.asarray(intensity_map, dtype=np.float64).ravel())
    index = min(
        len(ordered) - 1,
        int(np.floor(len(ordered) * min((100.0 - percent_brightest) / 100.0, 1.0))),
    )
    cutoff = max(1, int(ordered[index]))
    return np.ascontiguousarray(np.asarray(intensity_map) >= cutoff, dtype=np.uint8)


def normalization_plane(intensity_map: np.ndarray, exponent: float | None,
                        threshold: float | None) -> tuple[np.ndarray | None, float | None]:
    """Compute exponent normalization with the executable's arithmetic order."""
    if exponent is None:
        return None, None
    if not 0 < exponent <= 5:
        raise InputError("norm_exponent must be greater than 0 and at most 5")
    exponent = float(np.float32(exponent))
    image = np.asarray(intensity_map, dtype=np.float64)
    if threshold is None:
        ordered = np.sort(image.ravel())
        count = len(ordered) // 2
        if count < 10:
            raise InputError("intensity map has too few pixels for automatic normalization")
        values = ordered[:count]
        total = np.add.accumulate(values)[-1]
        total_squared = np.add.accumulate(values * values)[-1]
        mean = total / count
        sigma = np.sqrt((total_squared - count * mean * mean) / (count - 1))
        threshold = float(np.float32(mean + 5 * sigma))
    else:
        threshold = float(np.float32(threshold))
    if not threshold > 0:
        raise InputError("norm_threshold must be positive")
    plane = np.where(image < threshold, threshold**-exponent, image**-exponent)
    return np.ascontiguousarray(plane, dtype=np.float64), threshold
