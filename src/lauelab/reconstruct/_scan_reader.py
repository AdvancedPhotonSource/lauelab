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
from ._reader import _scalar, read_acquisition, read_image_geometry
from ._scan_layout import PointStatus, RunStatus
from ._writer import PIXEL_DTYPES, pixel_type

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


def _is_fixed_text(dtype) -> bool:
    info = h5py.check_string_dtype(np.dtype(dtype))
    return info is not None and info.length is not None


def _check_dataset(source, path: str, dtype, shape, label: str):
    if path not in source or not isinstance(source[path], h5py.Dataset):
        raise InvalidScanFile(f"{label}: dataset {path!r} is missing")
    dataset = source[path]
    if isinstance(dtype, str):
        matches, expected = _is_fixed_text(dataset.dtype), "fixed-length text"
    elif _is_variable_text(dtype):
        matches, expected = _is_variable_text(dataset.dtype), np.dtype(dtype)
    else:
        matches, expected = dataset.dtype == dtype, np.dtype(dtype)
    if not matches:
        raise InvalidScanFile(f"{label}: {path!r} has dtype {dataset.dtype}, expected {expected}")
    if dataset.shape != tuple(shape):
        raise InvalidScanFile(f"{label}: {path!r} has shape {dataset.shape}, expected {tuple(shape)}")
    return dataset


def _check_spec(source, path: str, spec, shape, label: str):
    """Check a catalog dataset against its layout entry."""
    dataset = _check_dataset(source, path, spec.dtype, shape, label)
    enum = h5py.check_enum_dtype(spec.dtype)
    if enum is not None and h5py.check_enum_dtype(dataset.dtype) != enum:
        raise InvalidScanFile(f"{label}: {path!r} must have the enumerated type {enum}")
    if dataset.attrs.get("units") != spec.units:
        raise InvalidScanFile(f"{label}: {path!r} must have units {spec.units!r}")
    if (spec.fill is not None and h5py.check_string_dtype(spec.dtype) is None
            and not _metadata_equal(dataset.fillvalue, spec.fill)):
        raise InvalidScanFile(f"{label}: {path!r} must have the fill value {spec.fill!r}")
    return dataset


def _check_row(index: int, values: dict) -> None:
    """Check the catalog row of one point for internal consistency."""
    label = f"point {index}"
    status = PointStatus(int(values["statuses"][index]))
    shape = tuple(int(value) for value in values["shapes"][index])
    if shape[0] == 0:
        # A point whose input could not be inspected: no metadata, an error.
        if status != PointStatus.FAILED or shape != (0, 0, 0):
            raise InvalidScanFile(f"{label}: only a failed point can have no depths")
    else:
        if min(shape) < 1:
            raise InvalidScanFile(f"{label}: catalog shape {shape} is empty")
        first, last = (float(value) for value in values["depth_bounds"][index])
        if not (np.isfinite(first) and np.isfinite(last) and first <= last):
            raise InvalidScanFile(f"{label}: depth_bounds must be finite and ordered")
        if int(values["dtypes"][index]) not in PIXEL_DTYPES:
            raise InvalidScanFile(f"{label}: pixel type {int(values['dtypes'][index])} is unknown")
    if status == PointStatus.FAILED and not _text(values["errors"][index]):
        raise InvalidScanFile(f"{label}: a failed point has no error text")


def _check_links(source, paths, statuses) -> None:
    """Check that ``/point_files`` links exactly the complete points."""
    links = source.get(layout.POINT_LINKS)
    if not isinstance(links, h5py.Group):
        raise InvalidScanFile(f"catalog: group {layout.POINT_LINKS!r} is missing")
    found = {}
    for name in links:
        link = links.get(name, getlink=True)
        found[name] = (link.filename, link.path) if isinstance(link, h5py.ExternalLink) else None
    expected = {Path(path).stem: (path, "/") for path, status in zip(paths, statuses)
                if status == PointStatus.COMPLETE}
    if found != expected:
        raise InvalidScanFile(
            f"catalog: {layout.POINT_LINKS} must hold one external link to each complete point"
        )


def _load_catalog(source: h5py.File) -> tuple[RunStatus, tuple[PointEntry, ...]]:
    """Validate an open ``scan.h5`` and return its run status and entries."""
    check_scan_format(source)
    for name in layout.SCAN_ATTRIBUTES:
        if name not in source.attrs:
            raise InvalidScanFile(f"run: root attribute {name!r} is missing")
    try:
        run_status = RunStatus(int(source.attrs["status"]))
    except (TypeError, ValueError) as error:
        raise InvalidScanFile(f"run: {error}") from error
    for name, spec in layout.SCAN_DATASETS.items():
        _check_spec(source, name, spec, spec.shape, "parameters")
    if "path" not in source["geometry"].attrs:
        raise InvalidScanFile("parameters: '/geometry' has no path attribute")

    statuses = source.get("points/statuses")
    if not isinstance(statuses, h5py.Dataset):
        raise InvalidScanFile("catalog: dataset '/points/statuses' is missing")
    n_points = statuses.shape[0] if statuses.ndim == 1 else -1
    if n_points < 0:
        raise InvalidScanFile("catalog: point statuses must be one-dimensional")
    if n_points == 0 and run_status == RunStatus.FINISHED:
        raise InvalidScanFile("catalog: a finished scan has at least one point")
    for name, spec in layout.POINT_COLUMNS.items():
        _check_spec(source, name, spec, (n_points, *spec.shape[1:]), "catalog")
    values = {name.rsplit("/", 1)[1]: source[name][...] for name in layout.POINT_COLUMNS}

    codes = set(values["statuses"].tolist())
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
    paths = [_text(value) for value in values["paths"]]
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
    _check_links(source, paths, values["statuses"])

    entries = []
    for index in range(n_points):
        energy = float(values["energies_kev"][index])
        scan_number = int(values["scan_numbers"][index])
        entries.append(PointEntry(
            index=index,
            point_id=point_ids[index],
            status=PointStatus(int(values["statuses"][index])).name.lower(),
            error=_text(values["errors"][index]),
            path=paths[index],
            source_path=_text(values["source_paths"][index]),
            scan_number=None if scan_number < 0 else scan_number,
            sample_position=tuple(float(value) for value in values["sample_positions"][index]),
            energy_kev=None if np.isnan(energy) else energy,
            shape=tuple(int(value) for value in values["shapes"][index]),
            depth_bounds_um=tuple(float(value) for value in values["depth_bounds"][index]),
            dtype=PIXEL_DTYPES.get(int(values["dtypes"][index])),
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
        If the file is not a valid catalog of a supported version.
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
            If the point file is invalid, or its point ID, manifest index,
            shape, dtype, or depth bounds disagree with the catalog.
        """
        entry = self._by_id[point_id]
        if not entry.complete:
            detail = f": {entry.error}" if entry.error else ""
            raise InputError(f"point {point_id!r} is {entry.status}{detail}")
        point = PointReader(self.directory / entry.path)
        found = (point.point_id, point.manifest_index, point.shape, point.dtype,
                 tuple(float(value) for value in point.depth_um[[0, -1]]))
        if found != (entry.point_id, entry.index, entry.shape, entry.dtype, entry.depth_bounds_um):
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
        ``True`` when at least one point exists and every point is complete. A valid catalog can be
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
        return self.n_points > 0 and self.n_complete == self.n_points


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
        If the format marker or version is wrong; a root attribute or dataset
        is missing or has the wrong dtype, enumerated type, shape, units, or
        fill value; catalog rows disagree in length; a status code is
        unknown, or a finished or cancelled run has a pending or writing
        point; point IDs or point paths are empty, unsafe, or repeat; a
        catalog row is inconsistent; or ``/point_files`` does not contain
        exactly one link to each complete point.

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


def _first(source, path: str):
    """Return the first element of the dataset at ``path`` as a Python value."""
    value = source[path][0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value.item() if isinstance(value, np.generic) else value


def _optional(source, path: str):
    return _first(source, path) if path in source else None


def validate_point(source: h5py.File) -> PointHeader:
    """Check the structure of an open point file with bounded reads.

    Every dataset that lauelab adds is checked against the layout: presence
    unless optional, dtype, shape, units, and fixed attributes. The
    normalization datasets are present together or not at all, and the
    depths must be finite and increasing. Metadata copied from the input is
    not checked. Pixel values are not read.
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
    first_raw = layout.RECONSTRUCTION + "first_raw"
    if not isinstance(source.get(first_raw), h5py.Dataset):
        raise InvalidScanFile(f"{label}: dataset {first_raw!r} is missing")
    raw = source[first_raw].dtype
    dims = {"n_depths": n_depths, "rows": rows, "columns": columns}
    for path, spec in layout.POINT_DATASETS.items():
        if spec.optional and path not in source:
            continue
        shape = tuple(dims[value] if isinstance(value, str) else value for value in spec.shape)
        dtype = spec.dtype if spec.dtype == layout.TEXT else layout.resolve_dtype(
            spec, stored=stored, input=raw)
        dataset = _check_dataset(source, path, dtype, shape, label)
        for name, expected in {**({"units": spec.units} if spec.units else {}), **spec.attrs}.items():
            if not _metadata_equal(dataset.attrs.get(name), expected):
                raise InvalidScanFile(f"{label}: {path} attribute {name!r} must be {expected!r}")
    if len({path in source for path in layout.NORMALIZATION_DATASETS}) != 1:
        raise InvalidScanFile(f"{label}: the normalization datasets must be written together")
    point_id = _first(source, layout.RECONSTRUCTION + "point_id")
    if not point_id:
        raise InvalidScanFile(f"{label}: the point ID is empty")
    depth_um = source["entry1/depth"][...]
    if not np.isfinite(depth_um).all() or (np.diff(depth_um) <= 0).any():
        raise InvalidScanFile(f"{label}: depth_um must be finite and increasing")
    index = _optional(source, layout.RECONSTRUCTION + "manifest_index")
    if index is not None and index < 0:
        raise InvalidScanFile(f"{label}: the manifest index is negative")
    return PointHeader(
        point_id=point_id,
        manifest_index=index,
        shape=(n_depths, rows, columns),
        dtype=stored,
        depth_bounds_um=(float(depth_um[0]), float(depth_um[-1])),
    )


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
        The detector fields are read from the input metadata copied into the
        point file, with the defaults that reconstruction applied to the
        input.
    norm_rescale : float
        Factor applied to computed values before they were stored; 1 without
        exponent normalization.
    norm_threshold : float or None
        Exponent-normalization threshold that was used, or ``None`` without
        exponent normalization.
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
        Reconstruction settings used for this point by name, such as
        ``"depth_range"`` and ``"wire_edge"``, and ``"detector"`` and
        ``"geometry_path"``. Settings chosen automatically hold the value
        chosen: ``"output_pixel_type"`` is the code of :attr:`dtype`,
        ``"norm_threshold"`` the threshold used, and ``"rows_per_stripe"``
        the stripe height used. A setting that does not apply is ``None``.

    Raises
    ------
    OSError
        If the file cannot be opened as HDF5.
    InvalidScanFile
        If the file is not a complete point file of a supported version,
        including a scan catalog.
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
        geometry = read_image_geometry(source, *self.shape[1:])
        detector_id = _scalar(source, "entry1/detector/ID", "")
        self.detector_id = _text(detector_id)
        self.detector_size = (geometry.nx_full, geometry.ny_full)
        self.start, self.group = geometry.start, geometry.group
        self.scan_number, self.sample_position, self.energy_kev = read_acquisition(source)
        exponent, threshold, rescale = (_optional(source, path) for path in layout.NORMALIZATION_DATASETS)
        self.norm_rescale = 1.0 if rescale is None else float(rescale)
        self.norm_threshold = None if threshold is None else float(threshold)
        group = layout.RECONSTRUCTION
        self.raw_slices = tuple(int(value) for value in source[group + "raw_slices"][...])
        self.source_path = _first(source, group + "source_path")
        self.settings = {
            "detector": int(_first(source, group + "detector")),
            "depth_range": tuple(float(value) for value in source[group + "depth_range"][...]),
            "resolution": float(_first(source, group + "resolution")),
            "wire_edge": _first(source, group + "wire_edge"),
            "percent_brightest": float(_first(source, group + "percent_brightest")),
            "normalization": _optional(source, group + "normalization"),
            "norm_exponent": None if exponent is None else float(exponent),
            "norm_threshold": self.norm_threshold,
            "cosmic_filter": "entry1/microDiffraction/cosmic_filter" in source,
            "frame_skip": int(_first(source, group + "frame_skip")),
            "wire_skip": int(_first(source, group + "wire_skip")),
            "output_pixel_type": pixel_type(self.dtype),
            "rows_per_stripe": int(_first(source, group + "rows_per_stripe")),
            "memory_limit_mb": int(_first(source, group + "memory_limit_mb")),
            "geometry_path": _first(source, group + "geometry_path"),
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
        return _first(self._file, layout.RECONSTRUCTION + "geometry_xml")

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
        return self._file[layout.RECONSTRUCTION + name][...]

    def depth_intensity(self) -> np.ndarray:
        """Return the sum of the stored pixels of each frame, shape ``(n_depths,)``.

        The dtype is ``numpy.int64`` for integer pixels and ``numpy.float64``
        otherwise.
        """
        return self._file[layout.RECONSTRUCTION + "depth_intensity"][...]

    def computed_depth_intensity(self) -> np.ndarray:
        """Return the sum of each computed frame before scaling and conversion.

        These ``numpy.float64`` values equal ``ReconstructionResult.
        depth_intensity``. They differ from :meth:`depth_intensity` whenever
        storage changed a pixel.
        """
        return self._file[layout.RECONSTRUCTION + "computed_depth_intensity"][...]
