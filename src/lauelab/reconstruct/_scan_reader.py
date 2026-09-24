# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Readers and validator for reconstruction scans and point files."""

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


def _metadata_equal(left, right) -> bool:
    """Compare scalar or array metadata, including matching missing NaNs."""
    left, right = np.asarray(left), np.asarray(right)
    numeric = left.dtype.kind in "biufc" and right.dtype.kind in "biufc"
    return np.array_equal(left, right, equal_nan=numeric)


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
        Stable identity of the point within the scan.
    status : str
        ``"pending"``, ``"writing"``, ``"complete"``, ``"failed"``,
        ``"interrupted"``, or ``"unattempted"``. The catalog makes a point
        available for reading once its status is ``"complete"``.
    error : str
        Error message for a failed point; empty otherwise.
    path : str
        Location of the point file relative to the directory of ``scan.h5``,
        with ``/`` separators. Assigned during preparation, including for
        points that later fail.
    source_path : str
        Absolute path of the input file.
    scan_number : int or None
        Acquisition scan number.
    sample_position : tuple of float
        Sample ``(x, y, z)`` in µm in the acquisition coordinate system. Missing
        components are NaN.
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
    path: str
    source_path: str
    scan_number: int | None
    sample_position: tuple[float, float, float]
    energy_kev: float | None
    shape: tuple[int, int, int]
    depth_bounds_um: tuple[float, float]
    dtype: np.dtype | None

    @property
    def complete(self) -> bool:
        """Whether the catalog lists this point as complete."""
        return self.status == "complete"


def check_scan_format(source: h5py.File) -> None:
    """Raise ``InvalidScanFile`` unless ``source`` is a supported ``scan.h5``."""
    name = source.attrs.get("format")
    name = name.decode("utf-8") if isinstance(name, bytes) else name
    version = source.attrs.get("version")
    if name == layout.SCAN_FORMAT and version in layout.RETIRED_SCAN_VERSIONS:
        raise InvalidScanFile(
            f"{source.filename} uses the retired single-file reconstruction-scan layout "
            f"(version {int(version)}), which this lauelab release cannot read. "
            f"Reconstruct the points again to write a scan directory."
        )
    try:
        check_format_version(source, format_name=layout.SCAN_FORMAT,
                             supported_versions={layout.SCAN_VERSION})
    except ValueError as error:
        raise InvalidScanFile(f"{source.filename}: {error}") from error


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


def _check_row(index: int, values: dict) -> None:
    """Check the catalog row of one point for internal consistency."""
    label = f"point {index}"
    status = PointStatus(int(values["status"][index]))
    shape = tuple(int(value) for value in values["image_shapes"][index])
    n_depths = int(values["n_depths"][index])
    if n_depths == 0:
        # A point whose input could not be inspected: no metadata, an error.
        if status != PointStatus.FAILED or shape != (0, 0):
            raise InvalidScanFile(f"{label}: only a failed point can have no depths")
    else:
        if min(shape) < 1 or n_depths < 0:
            raise InvalidScanFile(f"{label}: catalog shape {(n_depths, *shape)} is empty")
        first, last = (float(value) for value in values["depth_bounds"][index])
        if not (np.isfinite(first) and np.isfinite(last) and first <= last):
            raise InvalidScanFile(f"{label}: depth_bounds must be finite and ordered")
        if int(values["pixel_types"][index]) not in PIXEL_DTYPES:
            raise InvalidScanFile(f"{label}: pixel type {int(values['pixel_types'][index])} is unknown")
        start, stop = (int(value) for value in values["raw_slices"][index])
        if not 0 <= start < stop:
            raise InvalidScanFile(f"{label}: raw_slices {(start, stop)} is not a slice range")
    if status == PointStatus.FAILED and not _text(values["errors"][index]):
        raise InvalidScanFile(f"{label}: a failed point has no error text")


def _load_catalog(source: h5py.File) -> tuple[RunStatus, tuple[PointEntry, ...]]:
    """Validate an open ``scan.h5`` and return its run status and entries."""
    check_scan_format(source)
    for name, spec in {**layout.SCAN_RUN_DATASETS, **layout.SETTINGS_DATASETS}.items():
        _check_dataset(source, name, spec.dtype, spec.shape, "run")
    try:
        run_status = RunStatus(int(source["run/status"][()]))
    except ValueError as error:
        raise InvalidScanFile(f"run: {error}") from error

    if "catalog/status" not in source:
        raise InvalidScanFile("catalog: dataset '/catalog/status' is missing")
    n_points = source["catalog/status"].shape[0] if source["catalog/status"].ndim == 1 else -1
    if n_points < 1:
        raise InvalidScanFile("catalog: a scan has at least one point")
    for name, spec in layout.CATALOG_DATASETS.items():
        _check_dataset(source, name, spec.dtype, (n_points, *spec.shape[1:]), "catalog")
    values = {name.rsplit("/", 1)[1]: source[name][...] for name in layout.CATALOG_DATASETS}

    codes = set(values["status"].tolist())
    unknown = codes - {int(status) for status in PointStatus}
    if unknown:
        raise InvalidScanFile(f"catalog: point status codes {sorted(unknown)} are unknown")
    if run_status in layout.SETTLED_RUN_STATUSES:
        unsettled = codes - {int(status) for status in layout.TERMINAL_POINT_STATUSES}
        if unsettled:
            raise InvalidScanFile(
                f"catalog: a {run_status.name.lower()} run has unsettled point status codes "
                f"{sorted(unsettled)}"
            )
    point_ids = [_text(value) for value in values["point_ids"]]
    if any(not value for value in point_ids) or len(set(point_ids)) != n_points:
        raise InvalidScanFile("catalog: point IDs must be non-empty and unique")
    paths = [_text(value) for value in values["point_paths"]]
    for index, path in enumerate(paths):
        directory, _, name = path.partition("/")
        stem = name[:-len(layout.POINT_SUFFIX)] if name.endswith(layout.POINT_SUFFIX) else ""
        if (directory != layout.POINT_DIRECTORY or layout.unsafe_stem(stem)
                or path != layout.point_path(stem)):
            raise InvalidScanFile(f"point {index}: point path {path!r} is not a point-file path")
    if len({path.casefold() for path in paths}) != n_points:
        raise InvalidScanFile("catalog: point paths must be unique, ignoring case")
    for index in range(n_points):
        _check_row(index, values)

    entries = []
    for index in range(n_points):
        energy = float(values["energies_kev"][index])
        scan_number = int(values["scan_numbers"][index])
        rows, columns = (int(value) for value in values["image_shapes"][index])
        entries.append(PointEntry(
            index=index,
            point_id=point_ids[index],
            status=PointStatus(int(values["status"][index])).name.lower(),
            error=_text(values["errors"][index]),
            path=paths[index],
            source_path=_text(values["source_paths"][index]),
            scan_number=None if scan_number < 0 else scan_number,
            sample_position=tuple(float(value) for value in values["sample_positions"][index]),
            energy_kev=None if np.isnan(energy) else energy,
            shape=(int(values["n_depths"][index]), rows, columns),
            depth_bounds_um=tuple(float(value) for value in values["depth_bounds"][index]),
            dtype=PIXEL_DTYPES.get(int(values["pixel_types"][index])),
        ))
    return run_status, tuple(entries)


class ScanReader:
    """Read the catalog of a reconstruction scan.

    On creation, the reader loads a snapshot of ``scan.h5`` and closes the
    catalog file. Create a new reader to see later updates. You can list
    points and their metadata without opening any point files.

    Parameters
    ----------
    path : pathlib.Path or str
        ``scan.h5`` of a directory written by
        :func:`~lauelab.reconstruct.prepare_scan`.

    Attributes
    ----------
    path : pathlib.Path
    directory : pathlib.Path
        Directory containing ``scan.h5``; point paths are relative to it.
    run_status : str
        ``"running"``, ``"finished"``, ``"cancelled"``, or ``"failed"``.
    points : tuple of PointEntry
        Catalog in manifest order.

    Raises
    ------
    OSError
        If the file cannot be opened as HDF5.
    InvalidScanFile
        If the file is not a valid catalog of a supported version, including a
        file in the retired single-file layout.
    """

    def __init__(self, path) -> None:
        self.path = Path(path)
        self.directory = self.path.parent
        with h5py.File(self.path, "r") as source:
            run_status, self.points = _load_catalog(source)
        self.run_status = run_status.name.lower()
        self._by_id = {entry.point_id: entry for entry in self.points}
        self._opened: list[PointReader] = []

    def __enter__(self) -> "ScanReader":
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.close()

    def close(self) -> None:
        """Close every point reader returned by :meth:`point`. Safe to call more than once."""
        for point in self._opened:
            point.close()
        self._opened.clear()

    @property
    def point_ids(self) -> tuple[str, ...]:
        """Point IDs in manifest order, including incomplete points."""
        return tuple(entry.point_id for entry in self.points)

    def point_path(self, point_id: str) -> Path:
        """Return the point file of ``point_id``, resolved against :attr:`directory`.

        The path is assigned during preparation; the file may not yet exist.

        Raises
        ------
        KeyError
            If the scan has no point with this ID.
        """
        return self.directory / self._by_id[point_id].path

    def point(self, point_id: str) -> "PointReader":
        """Open the point file of a complete point.

        Only this point's file is opened. The returned reader belongs to the
        caller, who may close it; closing this ``ScanReader`` also closes it.

        Raises
        ------
        KeyError
            If the scan has no point with this ID.
        InputError
            If the catalog snapshot does not list the point as complete.
        InvalidScanFile
            If the point file is invalid or its metadata disagrees with the
            catalog.
        """
        entry = self._by_id[point_id]
        if not entry.complete:
            detail = f": {entry.error}" if entry.error else ""
            raise InputError(f"point {point_id!r} is {entry.status}{detail}")
        point = PointReader(self.directory / entry.path)
        identity_matches = (point.point_id, point.manifest_index, point.shape, point.dtype) == (
            entry.point_id, entry.index, entry.shape, entry.dtype,
        )
        metadata_matches = all(_metadata_equal(actual, expected) for actual, expected in (
            (point.depth_um[[0, -1]], entry.depth_bounds_um),
            (point.source_path, entry.source_path),
            (point.scan_number, entry.scan_number),
            (point.sample_position, entry.sample_position),
            (point.energy_kev, entry.energy_kev),
        ))
        if not identity_matches or not metadata_matches:
            point.close()
            raise InvalidScanFile(
                f"{point.path} does not hold point {point_id!r} as the catalog describes it"
            )
        self._opened = [opened for opened in self._opened if opened._file.id.valid]
        self._opened.append(point)
        return point


@dataclass(frozen=True)
class ScanFileSummary:
    """Run status and point counts returned by :func:`validate_scan_file`.

    Attributes
    ----------
    run_status : str
        ``"running"``, ``"finished"``, ``"cancelled"``, or ``"failed"``.
    n_points, n_pending, n_writing, n_complete, n_failed, n_interrupted, n_unattempted : int
        Point counts by status.
    complete : bool
        ``True`` when every point is complete. A valid catalog can be
        incomplete.
    lauelab_version : str
        Package version that wrote the catalog.
    """

    run_status: str
    n_points: int
    n_pending: int
    n_writing: int
    n_complete: int
    n_failed: int
    n_interrupted: int
    n_unattempted: int
    lauelab_version: str

    @property
    def complete(self) -> bool:
        return self.n_complete == self.n_points


def validate_scan_file(path) -> ScanFileSummary:
    """Check the structure of a closed ``scan.h5`` with bounded reads.

    Parameters
    ----------
    path : pathlib.Path or str
        Catalog to check. Its writer must have closed it.

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
        If the format marker or version is wrong, including the retired
        single-file layout; a dataset is missing or has the wrong dtype or
        shape; catalog rows disagree in length; a status code is unknown, or a
        finished or cancelled run has a pending or writing point; point IDs or
        point paths are empty, unsafe, or repeat; or a catalog row is
        inconsistent.

    Notes
    -----
    The check reads ``scan.h5`` only. Point files are not opened, and the
    file remains unchanged.
    """
    with h5py.File(path, "r") as source:
        run_status, entries = _load_catalog(source)
        version = _text(source.attrs.get("lauelab_version", ""))
    counts = {status: 0 for status in PointStatus}
    for entry in entries:
        counts[PointStatus[entry.status.upper()]] += 1
    return ScanFileSummary(
        run_status=run_status.name.lower(),
        n_points=len(entries),
        n_pending=counts[PointStatus.PENDING],
        n_writing=counts[PointStatus.WRITING],
        n_complete=counts[PointStatus.COMPLETE],
        n_failed=counts[PointStatus.FAILED],
        n_interrupted=counts[PointStatus.INTERRUPTED],
        n_unattempted=counts[PointStatus.UNATTEMPTED],
        lauelab_version=version,
    )


def check_point_format(source: h5py.File) -> None:
    """Raise ``InvalidScanFile`` unless ``source`` is a supported point file."""
    name = source.attrs.get("format")
    name = name.decode("utf-8") if isinstance(name, bytes) else name
    if name == layout.SCAN_FORMAT:
        check_scan_format(source)
        raise InvalidScanFile(
            f"{source.filename} is a reconstruction-scan catalog, not a point file; "
            f"open it with ScanReader"
        )
    try:
        check_format_version(source, format_name=layout.POINT_FORMAT,
                             supported_versions={layout.POINT_VERSION})
    except ValueError as error:
        raise InvalidScanFile(f"{source.filename}: {error}") from error


@dataclass(frozen=True)
class PointHeader:
    """Identity and output description of a validated point file."""

    point_id: str
    manifest_index: int | None
    shape: tuple[int, int, int]
    dtype: np.dtype
    depth_bounds_um: tuple[float, float]


def validate_point(source: h5py.File) -> PointHeader:
    """Check the structure of an open point file with bounded reads.

    Dataset dtypes and shapes are checked against the layout, the completion
    marker must be set, and the depths must be finite and increasing. NeXus
    classes, signals, axes, units, and internal depth links are checked. Pixel
    values are not read.
    """
    check_point_format(source)
    label = f"point file {source.filename}"
    data = source.get("entry1/data/data")
    if not isinstance(data, h5py.Dataset) or data.ndim != 3:
        raise InvalidScanFile(f"{label}: dataset '/entry1/data/data' is missing or not three-dimensional")
    stored = data.dtype
    if stored not in PIXEL_DTYPES.values():
        raise InvalidScanFile(f"{label}: pixel dtype {stored} is not a stored pixel type")
    n_depths, rows, columns = data.shape
    if min(n_depths, rows, columns) < 1:
        raise InvalidScanFile(f"{label}: '/entry1/data/data' is empty")
    raw_data = source.get("entry1/reconstruction/first_raw/data")
    if not isinstance(raw_data, h5py.Dataset):
        raise InvalidScanFile(f"{label}: first_raw/data must be a dataset")
    raw = raw_data.dtype
    dims = {"n_depths": n_depths, "rows": rows, "columns": columns}
    for path, spec in {**layout.POINT_SETTINGS_DATASETS, **layout.POINT_DATASETS}.items():
        shape = tuple(dims[value] if isinstance(value, str) else value for value in spec.shape)
        dtype = layout.resolve_dtype(spec, stored=stored, input=raw)
        dataset = _check_dataset(source, path, dtype, shape, label)
        for name, expected in {**({"units": spec.units} if spec.units else {}), **spec.attrs}.items():
            if not _metadata_equal(dataset.attrs.get(name), expected):
                raise InvalidScanFile(f"{label}: {path} attribute {name!r} must be {expected!r}")
    for path, attributes in layout.POINT_GROUP_ATTRIBUTES.items():
        if path not in source or not isinstance(source[path], h5py.Group):
            raise InvalidScanFile(f"{label}: {path!r} must be a group")
        for name, expected in attributes.items():
            if not _metadata_equal(source[path].attrs.get(name), expected):
                raise InvalidScanFile(f"{label}: {path} attribute {name!r} must be {expected!r}")
    for path in layout.POINT_DEPTH_LINKS:
        if (not isinstance(source.get(path, getlink=True), h5py.HardLink)
                or source[path].id != source["entry1/depth"].id):
            raise InvalidScanFile(f"{label}: {path} must link to /entry1/depth")
    if int(source["entry1/reconstruction/point/complete"][()]) != 1:
        raise InvalidScanFile(f"{label}: the point is not complete")
    point_id = _text(source["entry1/reconstruction/point/id"][()])
    if not point_id:
        raise InvalidScanFile(f"{label}: the point ID is empty")
    depth_um = source["entry1/depth"][...]
    if not np.isfinite(depth_um).all() or (np.diff(depth_um) <= 0).any():
        raise InvalidScanFile(f"{label}: depth_um must be finite and increasing")
    index = int(source["entry1/reconstruction/point/manifest_index"][()])
    return PointHeader(
        point_id=point_id,
        manifest_index=None if index < 0 else index,
        shape=(n_depths, rows, columns),
        dtype=stored,
        depth_bounds_um=(float(depth_um[0]), float(depth_um[-1])),
    )


def _setting(source, path):
    value = source[path][()]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    elif isinstance(value, np.ndarray):
        return tuple(value.tolist())
    elif isinstance(value, np.generic):
        value = value.item()
    if path == "/entry1/reconstruction/settings/cosmic_filter":
        return bool(value)
    missing = layout.POINT_SETTINGS_DATASETS[path].attrs.get("missing")
    if (missing == "nan" and np.isnan(value)) or (missing is not None and value == missing):
        return None
    return value


class PointReader:
    """Read frames, regions, and metadata from a reconstructed point file.

    Open the reader in the process that will use it, and close it with a
    context manager or :meth:`close`. The open reader cannot be sent to
    another process. All required metadata is in the point file, so you can
    read it without the catalog, raw input, or original geometry file.
    Image reads load only the requested frames or region and return arrays
    that remain usable after the reader closes.

    Parameters
    ----------
    path : pathlib.Path or str
        A point file, for example ``points/Twin2_wire_1.h5`` of a scan.

    Attributes
    ----------
    path : pathlib.Path
    point_id : str
        Point ID recorded in the file.
    manifest_index : int or None
        Zero-based position in the scan the point was prepared in, or
        ``None`` for a standalone point.
    shape : tuple of int
        ``(n_depths, ny, nx)``. ``frame(i)[y, x]`` is the pixel at ``(x, y)``.
    dtype : numpy.dtype
        Stored pixel dtype.
    depth_um : numpy.ndarray
        Physical depth of each frame in µm along the incident beam relative to
        the geometry's Si origin, shape ``(n_depths,)``.
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
        Half-open range of input slices used for the raw reference images.
    source_path : str
        Absolute path of the input file when the point was reconstructed.
    scan_number : int or None
        Acquisition scan number.
    sample_position : tuple of float
        Sample ``(x, y, z)`` in µm in the acquisition coordinate system. Missing
        components are NaN.
    energy_kev : float or None
        Incident energy in keV.
    settings : dict
        Requested reconstruction settings by name, such as ``"depth_range"``
        and ``"wire_edge"``, and ``"geometry_path"``. A setting that was not
        given is ``None``.

    Raises
    ------
    OSError
        If the file cannot be opened as HDF5.
    InvalidScanFile
        If the file is not a complete point file of a supported version,
        including a scan catalog or a file in the retired single-file layout.
    """

    def __init__(self, path) -> None:
        self.path = Path(path)
        self._file = h5py.File(self.path, "r")
        try:
            self._load(self._file)
        except BaseException:
            self._file.close()
            raise

    def _load(self, source: h5py.File) -> None:
        header = validate_point(source)
        self._data = source["entry1/data/data"]
        self.point_id = header.point_id
        self.manifest_index = header.manifest_index
        self.shape = header.shape
        self.dtype = header.dtype
        self.depth_um = source["entry1/depth"][...]
        self.detector_id = _text(source["entry1/reconstruction/detector/id"][()])
        self.detector_size = tuple(int(value) for value in source["entry1/reconstruction/detector/size"][...])
        self.start = tuple(int(value) for value in source["entry1/reconstruction/detector/roi_start"][...])
        self.group = tuple(int(value) for value in source["entry1/reconstruction/detector/roi_group"][...])
        self.norm_rescale = float(source["entry1/reconstruction/normalization/rescale"][()])
        threshold = float(source["entry1/reconstruction/normalization/threshold"][()])
        self.norm_threshold = None if np.isnan(threshold) else threshold
        self.raw_slices = tuple(int(value) for value in source["entry1/reconstruction/acquisition/raw_slices"][...])
        self.source_path = _text(source["entry1/reconstruction/acquisition/source_path"][()])
        scan_number = int(source["entry1/reconstruction/acquisition/scan_number"][()])
        self.scan_number = None if scan_number < 0 else scan_number
        self.sample_position = tuple(float(value) for value in source["entry1/reconstruction/acquisition/sample_position"][...])
        energy = float(source["entry1/reconstruction/acquisition/energy_kev"][()])
        self.energy_kev = None if np.isnan(energy) else energy
        self.settings = {
            path.rsplit("/", 1)[1] if path.startswith("/entry1/reconstruction/settings/") else "geometry_path":
            _setting(source, path)
            for path in layout.POINT_SETTINGS_DATASETS if path != "/entry1/reconstruction/geometry/xml"
        }

    def __enter__(self) -> "PointReader":
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.close()

    def close(self) -> None:
        """Close the file. Safe to call more than once."""
        self._file.close()

    def geometry_xml(self) -> str:
        """Return the complete geometry XML used during reconstruction."""
        return _text(self._file["entry1/reconstruction/geometry/xml"][()])

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
        return self._file[f"entry1/reconstruction/{name}/data"][...]

    def depth_intensity(self) -> np.ndarray:
        """Return the sum of the stored pixels of each frame, shape ``(n_depths,)``.

        The dtype is ``numpy.int64`` for integer pixels and ``numpy.float64``
        otherwise.
        """
        return self._file["entry1/reconstruction/stored_depth_intensity/data"][...]

    def computed_depth_intensity(self) -> np.ndarray:
        """Return the sum of each computed frame before scaling and conversion.

        These ``numpy.float64`` values equal ``ReconstructionResult.
        depth_intensity``. They differ from :meth:`depth_intensity` whenever
        storage changed a pixel.
        """
        return self._file["entry1/reconstruction/computed_depth_intensity/data"][...]
