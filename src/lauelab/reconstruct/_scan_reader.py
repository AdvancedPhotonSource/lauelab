# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Reader and validator for reconstruction-scan HDF5 files."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Iterator

import h5py
import numpy as np

from lauelab._hdf5 import check_format_version
from lauelab.indexing.errors import InputError, InvalidScanFile

from . import _scan_layout as layout
from ._scan_layout import PointStatus, RunStatus
from ._writer import PIXEL_DTYPES

REFERENCE_IMAGES = ("first_raw", "sum_raw", "sum_reconstructed")


def _text(value) -> str:
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def _index(value, length: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    if not 0 <= value < length:
        raise IndexError(f"{name} {value} is outside 0 to {length - 1}")
    return int(value)


def _region_bounds(bounds, shape) -> tuple[int, int, int, int]:
    if len(bounds) != 4 or any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
        for value in bounds
    ):
        raise InputError("bounds must be four integers (y0, y1, x0, x1)")
    y0, y1, x0, x1 = (int(value) for value in bounds)
    _, rows, columns = shape
    if not (0 <= y0 < y1 <= rows and 0 <= x0 < x1 <= columns):
        raise InputError(
            f"bounds {(y0, y1, x0, x1)} are outside the {rows} by {columns} image"
        )
    return y0, y1, x0, x1


@dataclass(frozen=True)
class PointEntry:
    """Metadata for one point from the scan catalog, loaded without reading pixels.

    Attributes
    ----------
    index : int
        Zero-based manifest index.
    point_id : str
        Stable identity of the point within the run.
    status : str
        ``"complete"``, ``"failed"``, ``"interrupted"``, or ``"unattempted"``
        in a published file. Only a complete point has pixels.
    error : str
        Message of a failed point, otherwise empty.
    source_path : str
        Input path as it was given to the run.
    scan_number : int or None
        Acquisition scan number.
    sample_position : tuple of float
        Sample ``(x, y, z)`` in µm in the acquisition coordinate system. A
        component the input did not provide is NaN.
    energy_kev : float or None
        Incident energy in keV.
    shape : tuple of int
        ``(n_depths, rows, columns)``; all zero for a point whose input could
        not be read.
    depth_bounds_um : tuple of float
        First and last physical depth in µm; NaN when ``shape`` is zero.
    dtype : numpy.dtype or None
        Stored pixel dtype.
    """

    index: int
    point_id: str
    status: str
    error: str
    source_path: str
    scan_number: int | None
    sample_position: tuple[float, float, float]
    energy_kev: float | None
    shape: tuple[int, int, int]
    depth_bounds_um: tuple[float, float]
    dtype: np.dtype | None

    @property
    def complete(self) -> bool:
        """Whether the point has pixels, reference images, and reductions."""
        return self.status == "complete"


def _read_catalog(source: h5py.File) -> tuple[PointEntry, ...]:
    catalog = {name.rsplit("/", 1)[1]: source[name][...] for name in layout.CATALOG_DATASETS}
    entries = []
    for index in range(len(catalog["status"])):
        code = int(catalog["pixel_types"][index])
        energy = float(catalog["energies_kev"][index])
        scan_number = int(catalog["scan_numbers"][index])
        rows, columns = (int(value) for value in catalog["image_shapes"][index])
        entries.append(PointEntry(
            index=index,
            point_id=_text(catalog["point_ids"][index]),
            status=PointStatus(int(catalog["status"][index])).name.lower(),
            error=_text(catalog["errors"][index]),
            source_path=_text(catalog["source_paths"][index]),
            scan_number=None if scan_number < 0 else scan_number,
            sample_position=tuple(float(value) for value in catalog["sample_positions"][index]),
            energy_kev=None if np.isnan(energy) else energy,
            shape=(int(catalog["n_depths"][index]), rows, columns),
            depth_bounds_um=tuple(float(value) for value in catalog["depth_bounds"][index]),
            dtype=PIXEL_DTYPES.get(code),
        ))
    return tuple(entries)


class PointReader:
    """Bounded access to one complete point of an open :class:`ScanReader`.

    Every method returns a new array that the caller owns. A method reads only
    the frames and region it is asked for. The reader is valid until its
    :class:`ScanReader` closes.

    Attributes
    ----------
    entry : PointEntry
        Catalog row of the point.
    shape : tuple of int
        ``(n_depths, rows, columns)``.
    dtype : numpy.dtype
        Stored pixel dtype.
    depth_um : numpy.ndarray
        Physical depth of each frame in µm, shape ``(n_depths,)``.
    detector_id : str
        Detector identifier from the input, or an empty string.
    detector_size : tuple of int
        Full detector ``(x, y)`` size in unbinned pixels.
    start, group : tuple of int
        Zero-based ``(x, y)`` frame origin in unbinned pixels and ``(x, y)``
        binning factors, as :meth:`lauelab.indexing.Indexer.index` takes them.
    norm_rescale : float
        Factor applied to computed values before they were stored.
    norm_threshold : float or None
        Exponent-normalization threshold that was used.
    raw_slices : tuple of int
        Half-open range of input slices behind the raw reference images.
    """

    def __init__(self, group: h5py.Group, entry: PointEntry) -> None:
        self._group = group
        self._data = group["data"]
        self.entry = entry
        self.shape = tuple(int(value) for value in self._data.shape)
        self.dtype = self._data.dtype
        self.depth_um = group["depth_um"][...]
        self.detector_id = _text(group["detector/id"][()])
        self.detector_size = tuple(int(value) for value in group["detector/size"][...])
        self.start = tuple(int(value) for value in group["detector/roi_start"][...])
        self.group = tuple(int(value) for value in group["detector/roi_group"][...])
        self.norm_rescale = float(group["normalization/rescale"][()])
        threshold = float(group["normalization/threshold"][()])
        self.norm_threshold = None if np.isnan(threshold) else threshold
        self.raw_slices = tuple(int(value) for value in group["reference/raw_slices"][...])

    def frame(self, depth_index: int) -> np.ndarray:
        """Return the stored frame at zero-based ``depth_index``, shape ``(rows, columns)``."""
        return self._data[_index(depth_index, self.shape[0], "depth_index")]

    def region(self, bounds, depths: slice | None = None) -> np.ndarray:
        """Return stored pixels in half-open ``bounds`` through a range of depths.

        Parameters
        ----------
        bounds : tuple of int
            ``(y0, y1, x0, x1)`` in stored-image pixels. The region must lie
            inside the image and contain at least one pixel.
        depths : slice or None
            Depth indices with a step of 1. The default is every depth.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_selected_depths, y1 - y0, x1 - x0)`` in the stored dtype.
        """
        y0, y1, x0, x1 = self._bounds(bounds)
        start, stop, step = (depths or slice(None)).indices(self.shape[0])
        if step != 1:
            raise InputError("depths must have a step of 1")
        return self._data[start:stop, y0:y1, x0:x1]

    def _bounds(self, bounds) -> tuple[int, int, int, int]:
        return _region_bounds(bounds, self.shape)

    def iter_blocks(self, max_bytes: int) -> Iterator[tuple[int, np.ndarray]]:
        """Yield ``(first_depth_index, frames)`` blocks of at most ``max_bytes``.

        Blocks cover every depth in order. ``max_bytes`` must hold at least one
        frame.
        """
        frame_bytes = self.shape[1] * self.shape[2] * self.dtype.itemsize
        if (isinstance(max_bytes, (bool, np.bool_)) or not isinstance(max_bytes, Integral)
                or max_bytes < frame_bytes):
            raise InputError(f"max_bytes must be at least one frame ({frame_bytes} bytes)")
        count = int(max_bytes // frame_bytes)
        for first in range(0, self.shape[0], count):
            yield first, self._data[first:first + count]

    def reference(self, name: str) -> np.ndarray:
        """Return reference image ``"first_raw"``, ``"sum_raw"``, or ``"sum_reconstructed"``.

        The raw images hold detector counts before filtering and
        normalization. ``"sum_reconstructed"`` is the sum of the stored frames.
        """
        if name not in REFERENCE_IMAGES:
            raise InputError(f"reference image must be one of {REFERENCE_IMAGES}; received {name!r}")
        return self._group[f"reference/{name}"][...]

    def depth_intensity(self) -> np.ndarray:
        """Return the sum of the stored pixels of each frame, shape ``(n_depths,)``.

        The dtype is ``numpy.int64`` for integer pixels and ``numpy.float64``
        otherwise.
        """
        return self._group["reductions/depth_intensity"][...]

    def computed_depth_intensity(self) -> np.ndarray:
        """Return the sum of each computed frame before scaling and conversion.

        These ``numpy.float64`` values equal ``ReconstructionResult.
        depth_intensity``. They differ from :meth:`depth_intensity` whenever
        storage changed a pixel.
        """
        return self._group["computed/depth_intensity"][...]


class ScanReader:
    """Read a published reconstruction-scan file.

    Use it as a context manager and open it in the process that reads from it;
    it holds an open HDF5 file and cannot be sent to another process.

    Parameters
    ----------
    path : pathlib.Path or str
        A file written by :func:`~lauelab.reconstruct.reconstruct_scan`.

    Attributes
    ----------
    path : pathlib.Path
    run_status : str
        ``"finished"`` or ``"cancelled"``.
    points : tuple of PointEntry
        Catalog in manifest order.

    Raises
    ------
    OSError
        If the file cannot be opened as HDF5.
    ValueError
        If the file is not a supported reconstruction-scan file.
    """

    def __init__(self, path) -> None:
        self.path = Path(path)
        self._file = h5py.File(self.path, "r")
        try:
            check_format_version(
                self._file, format_name=layout.FORMAT,
                supported_versions=layout.SUPPORTED_VERSIONS,
            )
            self.run_status = RunStatus(int(self._file["run/status"][()])).name.lower()
            self.points = _read_catalog(self._file)
        except BaseException:
            self._file.close()
            raise
        self._by_id = {entry.point_id: entry for entry in self.points}

    def __enter__(self) -> "ScanReader":
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.close()

    def close(self) -> None:
        """Close the file. Safe to call more than once."""
        self._file.close()

    @property
    def point_ids(self) -> tuple[str, ...]:
        """Point IDs in manifest order, including incomplete points."""
        return tuple(entry.point_id for entry in self.points)

    def point(self, point_id: str) -> PointReader:
        """Return the reader of a complete point.

        Raises
        ------
        KeyError
            If the run has no point with this ID.
        InputError
            If the point is incomplete and its image data may be missing or
            partially written.
        """
        entry = self._by_id[point_id]
        if not entry.complete:
            detail = f": {entry.error}" if entry.error else ""
            raise InputError(f"point {point_id!r} is {entry.status}{detail}")
        return PointReader(self._file[layout.POINT_GROUP.format(index=entry.index)], entry)


@dataclass(frozen=True)
class ScanFileSummary:
    """Run status and point counts returned by :func:`validate_scan_file`.

    Attributes
    ----------
    run_status : str
        ``"finished"`` or ``"cancelled"``.
    n_points, n_complete, n_failed, n_interrupted, n_unattempted : int
        Point counts by status.
    complete : bool
        ``True`` when every point is complete. A valid file can be incomplete.
    lauelab_version : str
        Package version that wrote the file.
    """

    run_status: str
    n_points: int
    n_complete: int
    n_failed: int
    n_interrupted: int
    n_unattempted: int
    lauelab_version: str

    @property
    def complete(self) -> bool:
        return self.n_complete == self.n_points


def _is_variable_text(dtype) -> bool:
    info = h5py.check_string_dtype(np.dtype(dtype))
    return info is not None and info.length is None


def _check_dataset(source, path: str, dtype, shape, label: str):
    if path not in source or not isinstance(source[path], h5py.Dataset):
        raise InvalidScanFile(f"{label}: dataset {path!r} is missing")
    dataset = source[path]
    if _is_variable_text(dtype):
        matches = _is_variable_text(dataset.dtype)
    else:
        matches = dataset.dtype == dtype
    if not matches:
        raise InvalidScanFile(f"{label}: {path!r} has dtype {dataset.dtype}, expected {np.dtype(dtype)}")
    if dataset.shape != tuple(shape):
        raise InvalidScanFile(f"{label}: {path!r} has shape {dataset.shape}, expected {tuple(shape)}")
    return dataset


def validate_scan_file(path) -> ScanFileSummary:
    """Check the structure of a closed reconstruction-scan file with bounded reads.

    Parameters
    ----------
    path : pathlib.Path or str
        File to check. Its writer must have closed it.

    Returns
    -------
    ScanFileSummary
        Run status and point counts. Structural validity does not mean that
        every point is complete; check :attr:`ScanFileSummary.complete`.

    Raises
    ------
    OSError
        If the file cannot be opened as HDF5.
    InvalidScanFile
        If the format marker or version is wrong; the run is still running or
        failed; a dataset is missing or has the wrong dtype or shape; catalog
        rows disagree in length; a status is unknown or not terminal; point IDs
        are empty or repeat; a complete point has no group; or a point group
        disagrees with its catalog row.

    Notes
    -----
    The check reads dataset shapes and dtypes, the catalog, and each point's
    depth vector. Pixel values are excluded from validation, and the file
    remains unchanged.
    """
    with h5py.File(path, "r") as source:
        try:
            check_format_version(
                source, format_name=layout.FORMAT, supported_versions=layout.SUPPORTED_VERSIONS
            )
        except ValueError as error:
            raise InvalidScanFile(str(error)) from error
        for name, spec in layout.RUN_DATASETS.items():
            _check_dataset(source, name, spec.dtype, spec.shape, "run")
        try:
            run_status = RunStatus(int(source["run/status"][()]))
        except ValueError as error:
            raise InvalidScanFile(f"run: {error}") from error
        if run_status not in layout.PUBLISHABLE_RUN_STATUSES:
            raise InvalidScanFile(f"run status is {run_status.name.lower()}")

        if "catalog/status" not in source:
            raise InvalidScanFile("catalog: dataset '/catalog/status' is missing")
        n_points = source["catalog/status"].shape[0]
        for name, spec in layout.CATALOG_DATASETS.items():
            _check_dataset(source, name, spec.dtype, (n_points, *spec.shape[1:]), "catalog")
        codes = source["catalog/status"][...]
        unknown = set(codes.tolist()) - {int(status) for status in layout.TERMINAL_POINT_STATUSES}
        if unknown:
            raise InvalidScanFile(f"catalog: point status codes {sorted(unknown)} are not terminal")
        point_ids = [_text(value) for value in source["catalog/point_ids"][...]]
        if any(not value for value in point_ids) or len(set(point_ids)) != n_points:
            raise InvalidScanFile("catalog: point IDs must be non-empty and unique")

        image_shapes = source["catalog/image_shapes"][...]
        n_depths = source["catalog/n_depths"][...]
        depth_bounds = source["catalog/depth_bounds"][...]
        pixel_types = source["catalog/pixel_types"][...]
        for index in range(n_points):
            name = layout.POINT_GROUP.format(index=index)
            label = f"point {index}"
            if name not in source:
                if codes[index] == PointStatus.COMPLETE:
                    raise InvalidScanFile(f"{label}: complete point has no group")
                continue
            group = source[name]
            stored = PIXEL_DTYPES.get(int(pixel_types[index]))
            if stored is None:
                raise InvalidScanFile(f"{label}: pixel type {int(pixel_types[index])} is unknown")
            rows, columns = (int(value) for value in image_shapes[index])
            dims = {"n_depths": int(n_depths[index]), "rows": rows, "columns": columns}
            if min(dims.values()) < 1:
                raise InvalidScanFile(f"{label}: catalog shape {dims} is empty")
            raw = group["reference/first_raw"].dtype if "reference/first_raw" in group else None
            for path, spec in layout.POINT_DATASETS.items():
                shape = tuple(dims[value] if isinstance(value, str) else value for value in spec.shape)
                dtype = layout.resolve_dtype(spec, stored=stored, input=raw)
                _check_dataset(group, path, dtype, shape, label)
            if layout.POINT_SOURCE_GROUP not in group:
                raise InvalidScanFile(f"{label}: group {layout.POINT_SOURCE_GROUP!r} is missing")
            depth_um = group["depth_um"][...]
            if not np.isfinite(depth_um).all() or (np.diff(depth_um) <= 0).any():
                raise InvalidScanFile(f"{label}: depth_um must be finite and increasing")
            if not np.array_equal(depth_bounds[index], (depth_um[0], depth_um[-1])):
                raise InvalidScanFile(f"{label}: depth_bounds disagree with depth_um")

        counts = {status: int((codes == status).sum()) for status in layout.TERMINAL_POINT_STATUSES}
        return ScanFileSummary(
            run_status=run_status.name.lower(),
            n_points=n_points,
            n_complete=counts[PointStatus.COMPLETE],
            n_failed=counts[PointStatus.FAILED],
            n_interrupted=counts[PointStatus.INTERRUPTED],
            n_unattempted=counts[PointStatus.UNATTEMPTED],
            lauelab_version=_text(source.attrs.get("lauelab_version", "")),
        )
