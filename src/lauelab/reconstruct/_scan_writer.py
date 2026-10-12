# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Writers for reconstruction scans: catalog snapshots and point files.

The scan coordinator is the only writer of ``scan.h5``. It keeps the catalog in
memory and periodically publishes pending changes as a complete snapshot: it is
written to a private file, closed, validated, and moved over ``scan.h5``. A
reader that already holds the file keeps its earlier snapshot.

Each worker writes its own point file. Before computation, it creates all
datasets in a private file, then fills them stripe by stripe through
:class:`PointSink`, which writes the reductions last. The file is published
only after it is closed and validated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import h5py
import numpy as np

from lauelab._hdf5 import set_units, write_root_attributes
from lauelab._native import ffi, get_library
from lauelab._publish import partial_path, publish_file

from . import _scan_layout as layout
from ._scan_layout import PointStatus, RunStatus
from ._writer import PIXEL_DTYPES, _copy_metadata

_F8 = np.dtype("<f8")

# Chunk shape of the pixel datasets as (depths, rows, columns), chosen from
# tests/perf_testing/run_scan_storage_perf.py: write time hardly depends on it,
# one depth per chunk reads a frame fastest but an ROI through depth slowest,
# and this shape keeps both well under a second. A stripe of 256 rows covers
# whole chunks. It is not part of the format contract.
DATA_CHUNKS = (4, 128, 128)
COMPRESSION = {None: {}, "gzip": {"compression": "gzip", "compression_opts": 1, "shuffle": True}}


def _stored_type_code(dtype) -> int:
    library = get_library()
    codes = {
        "<f4": library.LAUE_PIXEL_F32, "<i4": library.LAUE_PIXEL_I32,
        "<i2": library.LAUE_PIXEL_I16, "<u2": library.LAUE_PIXEL_U16,
        "<f8": library.LAUE_PIXEL_F64, "|i1": library.LAUE_PIXEL_I8, "|u1": library.LAUE_PIXEL_U8,
    }
    return codes[np.dtype(dtype).str]


def store_stripe(values: np.ndarray, dtype, rescale: float = 1.0, n_threads: int = 1):
    """Convert one computed stripe to its stored form and reduce it.

    Parameters
    ----------
    values : numpy.ndarray
        Computed ``(n_depths, rows, columns)`` float64 stripe.
    dtype
        Stored pixel dtype, one of ``PIXEL_DTYPES``.
    rescale : float
        Factor applied before conversion.
    n_threads : int
        OpenMP threads for the pass.

    Returns
    -------
    tuple of numpy.ndarray
        ``(stored, depth_sums, pixel_sums)``: the stored stripe; the total of
        each depth, shape ``(n_depths,)``; and the total through depth of each
        pixel, shape ``(rows, columns)``. Sums are ``numpy.int64`` for an
        integer ``dtype`` and ``numpy.float64`` otherwise, and do not depend on
        ``n_threads``.

    Notes
    -----
    Values are multiplied by ``rescale`` before conversion. Floating-point
    output matches a NumPy cast; integer output truncates toward zero and
    saturates at the dtype limits. NaN becomes 0 in integer output. Finite
    values and infinities match HDF5 conversion; integer NaN conversion can
    differ from HDF5.
    """
    values = np.ascontiguousarray(values, dtype=_F8)
    if values.ndim != 3:
        raise ValueError("values must have shape (n_depths, rows, columns)")
    dtype = np.dtype(dtype)
    stored = np.empty(values.shape, dtype)
    depth_sums = np.empty(values.shape[0], layout.accumulator_dtype(dtype))
    pixel_sums = np.empty(values.shape[1:], depth_sums.dtype)
    status = get_library().laue_recon_store_stripe(
        ffi.from_buffer("double[]", values), values.shape[0], values[0].size, float(rescale),
        _stored_type_code(dtype), ffi.from_buffer(stored), ffi.from_buffer(depth_sums),
        ffi.from_buffer(pixel_sums), int(n_threads),
    )
    if status:
        raise ValueError("stripe conversion rejected its arguments")
    return stored, depth_sums, pixel_sums


def _create(group, path, spec, *, shape=None, dtype=None, data=None, **options):
    dtype = spec.dtype if dtype is None else dtype
    if spec.fill is not None and h5py.check_string_dtype(np.dtype(dtype)) is None:
        options["fillvalue"] = spec.fill
    dataset = group.create_dataset(path.lstrip("/"), shape=shape, dtype=dtype, data=data, **options)
    set_units(dataset, spec.units)
    for name, value in spec.attrs.items():
        dataset.attrs[name] = value
    return dataset


def parameter_values(detector: int, settings: Mapping) -> dict:
    """Return the ``/parameters`` values of a scan catalog.

    ``settings`` holds the :class:`~lauelab.reconstruct.Reconstructor` keyword
    arguments of a prepared task. A setting that was not given is stored as
    its fill value.
    """
    values = {}
    for path, spec in layout.SCAN_DATASETS.items():
        if path.startswith("/parameters/"):
            name = path.rsplit("/", 1)[1]
            value = detector if name == "detector" else settings[name]
            values[path] = spec.fill if value is None else value
    return values


def native_version() -> str:
    """Return the version string of the loaded native library."""
    return ffi.string(get_library().laue_version()).decode()


def write_catalog(path: Path, *, created: str, run_status: RunStatus, native_version: str,
                  parameters: Mapping, geometry_path: str, geometry_xml: str,
                  points: Mapping, replace: bool) -> None:
    """Publish one complete catalog snapshot at ``path``.

    Parameters
    ----------
    path
        ``scan.h5`` of the scan directory.
    created
        Creation time of the scan, kept by every snapshot.
    run_status
        Run status of this snapshot.
    native_version
        Version of the native library that prepared the scan.
    parameters
        Values of the ``/parameters`` datasets, from :func:`parameter_values`.
    geometry_path, geometry_xml
        Geometry file path as given, and its complete text.
    points
        Values of every ``POINT_COLUMNS`` entry, one row per point.
    replace
        ``False`` for the first snapshot, which refuses to replace an existing
        file; ``True`` for later snapshots.

    Raises
    ------
    OSError
        If the snapshot cannot be written or moved into place. The previous
        snapshot, if any, is unchanged.
    InvalidScanFile
        If the snapshot does not validate; it is left at its ``.partial``
        name.
    """
    from ._scan_reader import validate_scan_file

    partial = partial_path(path)
    with h5py.File(partial, "w") as target:
        write_root_attributes(target, format_name=layout.SCAN_FORMAT,
                              version=layout.SCAN_VERSION, created=created)
        target.attrs["program"] = layout.PROGRAM
        target.attrs["native_version"] = native_version
        target.attrs.create("status", int(run_status), dtype=layout.RUN_STATUS_TYPE)
        values = {**parameters, "/geometry/xml": geometry_xml}
        for name, spec in layout.SCAN_DATASETS.items():
            _create(target, name, spec, data=np.asarray(values[name], dtype=spec.dtype))
        target["geometry"].attrs["path"] = geometry_path
        for name, spec in layout.POINT_COLUMNS.items():
            _create(target, name, spec, data=np.asarray(points[name], dtype=spec.dtype))
        links = target.create_group(layout.POINT_LINKS)
        for point_path, status in zip(points["/points/paths"], points["/points/statuses"]):
            if status == PointStatus.COMPLETE:
                links[Path(point_path).stem] = h5py.ExternalLink(point_path, "/")
    validate_scan_file(partial)
    publish_file(partial, path, overwrite=replace)


def _text(value: str) -> np.ndarray:
    """Return ``value`` as a ``(1,)`` fixed-length UTF-8 array."""
    encoded = value.encode("utf-8")
    return np.array([encoded], dtype=h5py.string_dtype("utf-8", max(1, len(encoded))))


def write_point_metadata(file: h5py.File, task, *, info, depth_um, output_type: int, first_raw,
                         source_metadata, source_stat, num_threads: int) -> None:
    """Create every dataset of one point in its open private file.

    Parameters
    ----------
    file
        The private point file, newly created.
    task
        The :class:`~lauelab.reconstruct.PointTask` of the point.
    info
        ``ScanInfo`` of the input.
    depth_um
        Physical depths of the point in µm.
    output_type
        Stored pixel-type code.
    first_raw
        First selected raw frame, already read from the input.
    source_metadata
        Input metadata to copy, as the reconstruction executable copies it.
    source_stat
        ``os.stat_result`` of the input, or ``None``.
    num_threads
        OpenMP threads of the reconstruction.
    """
    _copy_metadata(source_metadata, file, exclude_entry=("depth",))
    write_root_attributes(file, format_name=layout.POINT_FORMAT, version=layout.POINT_VERSION)
    _, rows, columns = info.shape
    stored = PIXEL_DTYPES[output_type]
    settings = task.settings
    reconstruction = layout.RECONSTRUCTION
    values = {
        "/entry1/depth": depth_um,
        **{reconstruction + name: settings[name] for name in (
            "depth_range", "resolution", "wire_edge", "percent_brightest", "frame_skip",
            "wire_skip", "memory_limit_mb",
        )},
        reconstruction + "detector": task.detector,
        reconstruction + "geometry_path": task.geometry_path,
        reconstruction + "geometry_xml": task.geometry_xml,
        reconstruction + "point_id": task.point_id,
        reconstruction + "source_path": task.source,
        reconstruction + "raw_slices": task.raw_slices,
        reconstruction + "native_version": native_version(),
        reconstruction + "num_threads": num_threads,
        reconstruction + "first_raw": first_raw,
    }
    if settings["normalization"]:
        values[reconstruction + "normalization"] = settings["normalization"]
    if task.index is not None:
        values[reconstruction + "manifest_index"] = task.index
    if source_stat is not None:
        values[reconstruction + "source_size"] = source_stat.st_size
        values[reconstruction + "source_mtime_ns"] = source_stat.st_mtime_ns
    if settings["cosmic_filter"]:
        values["/entry1/microDiffraction/cosmic_filter"] = 1
    if settings["norm_exponent"] is not None:
        # The sink records the threshold and rescale that are used.
        values.update(zip(layout.NORMALIZATION_DATASETS, (settings["norm_exponent"], np.nan, np.nan)))
    dims = {"n_depths": len(depth_um), "rows": rows, "columns": columns}
    for path, spec in layout.POINT_DATASETS.items():
        if spec.optional and path not in values:
            continue
        if spec.dtype == layout.TEXT:
            text = _text(values[path])
            _create(file, path, spec, dtype=text.dtype, data=text)
            continue
        dtype = layout.resolve_dtype(spec, stored=stored, input=info.dtype)
        shape = tuple(dims[name] if isinstance(name, str) else name for name in spec.shape)
        if path == "/entry1/data/data":
            _create(file, path, spec, shape=shape, dtype=dtype,
                    chunks=tuple(min(chunk, size) for chunk, size in zip(DATA_CHUNKS, shape)),
                    **COMPRESSION[task.compression])
        elif path in values:
            _create(file, path, spec, dtype=dtype, data=np.asarray(values[path], dtype=dtype).reshape(shape))
        else:
            _create(file, path, spec, shape=shape, dtype=dtype)


class PointSink:
    """Stripe sink that writes one point into its private file.

    The reconstructor reports failures after the first stripe as failed point
    results. The sink retains its first output exception in ``error`` so the
    caller can detect output failures and stop the run. Input and computation
    failures affect only the current point. Stripes arrive already loaded;
    the sink performs no input reads.
    """

    output_files: list = []

    def __init__(self, file: h5py.File, n_threads: int) -> None:
        self._file = file
        self._data = file["entry1/data/data"]
        self._n_threads = n_threads
        self._rescale = 1.0
        self.error: Exception | None = None

    def _guard(self, operation, *args, **kwargs):
        try:
            return operation(*args, **kwargs)
        except Exception as error:
            if self.error is None:
                self.error = error
            raise

    def stripe_buffer_bytes(self, n_depths, cols) -> tuple[int, int]:
        # Stored pixels, depth totals, and a per-pixel projection. The raw
        # stripe reduction has the same per-pixel cost and runs separately.
        return n_depths * 8, cols * (n_depths * self._data.dtype.itemsize + 8)

    def begin(self, **kwargs) -> None:
        self._guard(self._begin, **kwargs)

    def _begin(self, *, depth_um, shape, output_type, threshold, rescale) -> None:
        if self._data.shape != (len(depth_um), *shape) or self._data.dtype != PIXEL_DTYPES[output_type]:
            raise ValueError("the point file does not match the reconstruction")
        normalized = layout.NORMALIZATION_DATASETS[1] in self._file
        if normalized != (threshold is not None):
            raise ValueError("the point file does not match the normalization")
        accumulator = layout.accumulator_dtype(self._data.dtype)
        self._depth_intensity = np.zeros(len(depth_um), dtype=accumulator)
        self._sum_reconstructed = np.zeros(shape, dtype=accumulator)
        self._sum_raw = np.zeros(shape, dtype=self._file[layout.RECONSTRUCTION + "sum_raw"].dtype)
        self._rescale = rescale
        if normalized:
            _, threshold_path, rescale_path = layout.NORMALIZATION_DATASETS
            self._file[threshold_path][...] = threshold
            self._file[rescale_path][...] = rescale

    def raw(self, row0, stripe) -> None:
        # A float64 sum of integers of at most 4 bytes is exact below 2**53.
        rows = stripe.shape[1]
        self._sum_raw[row0:row0 + rows] = stripe.sum(
            axis=0, dtype=np.int64 if stripe.dtype.kind in "iu" else _F8
        )

    def write(self, row0, values) -> None:
        self._guard(self._write, row0, values)

    def _write(self, row0, values) -> None:
        stored, depth_sums, pixel_sums = store_stripe(
            values, self._data.dtype, self._rescale, self._n_threads
        )
        row1 = row0 + stored.shape[1]
        self._data[:, row0:row1, :] = stored
        self._depth_intensity += depth_sums
        self._sum_reconstructed[row0:row1] = pixel_sums

    def finish(self, **kwargs) -> None:
        self._guard(self._finish, **kwargs)

    def _finish(self, *, totals, stripe_rows, elapsed, **_) -> None:
        group = self._file[layout.RECONSTRUCTION]
        group["sum_raw"][...] = self._sum_raw
        group["sum_reconstructed"][...] = self._sum_reconstructed
        group["depth_intensity"][...] = self._depth_intensity
        group["computed_depth_intensity"][...] = totals
        group["rows_per_stripe"][...] = stripe_rows
        group["elapsed_seconds"][...] = elapsed

    def close(self) -> None:
        pass


def _missing(value, fill):
    return fill if value is None else value
