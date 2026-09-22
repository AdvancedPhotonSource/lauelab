# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Writer for reconstruction-scan HDF5 files.

One :class:`ScanWriter` owns the file for a whole run. It creates every group
and dataset from the frozen manifest before any point is computed, hands one
stripe sink to the reconstructor for each point, and publishes the closed,
validated file. No other object writes to the file.
"""

from __future__ import annotations

import os
from pathlib import Path

import h5py
import numpy as np

from lauelab._hdf5 import set_units, write_root_attributes
from lauelab._native import ffi, get_library
from lauelab._publish import partial_path, publish_file
from lauelab.indexing.errors import InputError

from . import _scan_layout as layout
from ._scan_layout import PointStatus, RunStatus
from ._writer import PIXEL_DTYPES, _copy_metadata, pixel_type

_F8 = np.dtype("<f8")

# Chunk shape of the pixel datasets as (depths, rows, columns), chosen from
# tests/perf_testing/run_scan_storage_perf.py: write time hardly depends on it,
# one depth per chunk reads a frame fastest but an ROI through depth slowest,
# and this shape keeps both well under a second. A stripe of 256 rows covers
# whole chunks. It is not part of the format contract.
DATA_CHUNKS = (4, 128, 128)
COMPRESSION = {None: {}, "gzip": {"compression": "gzip", "compression_opts": 1, "shuffle": True}}

_FILL = {"nan": np.nan, "": "", -1: -1}


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
    A stored value equals what an HDF5 dataset of ``dtype`` stores for
    ``values * rescale``: the rounded value for a floating-point type, and for
    an integer type the value truncated toward zero and saturated at the
    limits of the type. NaN stores 0 in an integer type.
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


def _error_text(message: str) -> bytes:
    """Encode ``message`` for the fixed-width error dataset without splitting a character."""
    width = layout.MUTABLE_TEXT.itemsize
    return str(message).encode("utf-8")[:width].decode("utf-8", errors="ignore").encode("utf-8")


def _create(group, path, spec, *, shape=None, dtype=None, data=None, **options):
    dataset = group.create_dataset(
        path.lstrip("/"), shape=shape, dtype=spec.dtype if dtype is None else dtype,
        data=data, **options,
    )
    set_units(dataset, spec.units)
    for name, value in spec.attrs.items():
        dataset.attrs[name] = value
    return dataset


class ScanWriter:
    """Single owner of one reconstruction-scan file.

    Parameters
    ----------
    path
        Final path of the file. The writer works on ``<path>.partial`` and
        renames it into place in :meth:`close`.
    reconstructor
        The configured :class:`~lauelab.reconstruct.Reconstructor`; its options
        are the run settings.
    n_points
        Manifest length.
    overwrite
        Replace an existing ``path``. The default raises ``FileExistsError``
        before any work is done.
    compression
        ``None`` or ``"gzip"`` for the pixel datasets.
    """

    def __init__(self, path, *, reconstructor, n_points: int, overwrite: bool = False,
                 compression: str | None = None) -> None:
        if compression not in COMPRESSION:
            raise InputError(f"compression must be None or 'gzip'; received {compression!r}")
        self.path = Path(path)
        if self.path.exists() and not overwrite:
            raise FileExistsError(f"{self.path} exists; pass overwrite=True to replace it")
        self._overwrite = overwrite
        self._partial = partial_path(self.path)
        self._storage = COMPRESSION[compression]
        self._n_points = int(n_points)
        self._ids: list[str] = []
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = h5py.File(self._partial, "w")
        try:
            self._write_run(reconstructor)
        except BaseException:
            self._file.close()
            raise

    # -- file creation ---------------------------------------------------------

    def _write_run(self, reconstructor) -> None:
        from lauelab._native import ffi, get_library

        target = self._file
        write_root_attributes(target, format_name=layout.FORMAT, version=layout.VERSION)
        geometry_path = reconstructor.geometry_path
        values = {
            "/run/status": int(RunStatus.RUNNING),
            "/run/native_version": ffi.string(get_library().laue_version()).decode(),
            "/settings/detector": reconstructor.detector,
            "/settings/depth_range": reconstructor.depth_range,
            "/settings/resolution": reconstructor.resolution,
            "/settings/wire_edge": reconstructor.wire_edge,
            "/settings/percent_brightest": reconstructor.percent_brightest,
            "/settings/normalization": reconstructor.normalization or "",
            "/settings/norm_exponent": _missing(reconstructor.norm_exponent, np.nan),
            "/settings/norm_threshold": _missing(reconstructor.norm_threshold, np.nan),
            "/settings/cosmic_filter": int(reconstructor.cosmic_filter),
            "/settings/output_pixel_type": _missing(reconstructor.output_pixel_type, -1),
            "/settings/memory_limit_mb": reconstructor.memory_limit_mb,
            "/geometry/path": "" if geometry_path is None else os.fspath(geometry_path),
            "/geometry/xml": "" if geometry_path is None else Path(geometry_path).read_text(),
        }
        for path, spec in layout.RUN_DATASETS.items():
            _create(target, path, spec, data=np.asarray(values[path], dtype=spec.dtype))
        for path, spec in layout.CATALOG_DATASETS.items():
            shape = (self._n_points,) + tuple(spec.shape[1:])
            dataset = _create(target, path, spec, shape=shape)
            fill = spec.attrs.get("missing")
            if fill is not None and h5py.check_string_dtype(spec.dtype) is None:
                dataset[...] = _FILL[fill]
        target.create_group("points")
        target.flush()

    def add_point(self, index: int, point_id: str, source_path, *, source=None, info=None,
                  stored_dtype=None, depth_um=None, error: str | None = None) -> None:
        """Write the catalog row of one point and, unless it failed, its group.

        Call once for each manifest index in order, before any point is
        computed. ``error`` records a point whose input could not be read; it
        gets a ``failed`` row and no group.
        """
        if index != len(self._ids):
            raise ValueError("points must be added in manifest order")
        if not isinstance(point_id, str) or not point_id or point_id in self._ids:
            raise ValueError(f"point ID {point_id!r} must be a unique, non-empty string")
        self._ids.append(point_id)
        catalog = self._file["catalog"]
        catalog["point_ids"][index] = point_id
        catalog["source_paths"][index] = os.fspath(source_path)
        try:
            status = os.stat(source_path)
            catalog["source_sizes"][index] = status.st_size
            catalog["source_mtimes_ns"][index] = status.st_mtime_ns
        except OSError:
            pass
        if error is not None:
            catalog["image_shapes"][index] = (0, 0)
            catalog["n_depths"][index] = 0
            catalog["status"][index] = int(PointStatus.FAILED)
            catalog["errors"][index] = _error_text(error)
            return

        n_images, rows, columns = info.shape
        stored = np.dtype(stored_dtype)
        depth_um = np.asarray(depth_um, dtype=_F8)
        if info.scan_number is not None:
            catalog["scan_numbers"][index] = info.scan_number
        catalog["sample_positions"][index] = info.sample_position
        if info.energy_kev is not None:
            catalog["energies_kev"][index] = info.energy_kev
        catalog["image_shapes"][index] = (rows, columns)
        catalog["n_depths"][index] = len(depth_um)
        catalog["depth_bounds"][index] = (depth_um[0], depth_um[-1])
        catalog["pixel_types"][index] = pixel_type(stored)

        group = self._file.create_group(layout.POINT_GROUP.format(index=index))
        geometry = info.image_geometry
        dims = {"n_depths": len(depth_um), "rows": rows, "columns": columns}
        values = {
            "depth_um": depth_um,
            "detector/id": _detector_id(source),
            "detector/size": (geometry.nx_full, geometry.ny_full),
            "detector/roi_start": geometry.start,
            "detector/roi_group": geometry.group,
            "normalization/threshold": np.nan,
            "normalization/rescale": 1.0,
            "reference/raw_slices": (1, n_images + 1),
        }
        for path, spec in layout.POINT_DATASETS.items():
            dtype = layout.resolve_dtype(spec, stored=stored, input=info.dtype)
            shape = tuple(dims[name] if isinstance(name, str) else name for name in spec.shape)
            if path == "data":
                _create(
                    group, path, spec, shape=shape, dtype=dtype,
                    chunks=tuple(min(chunk, size) for chunk, size in zip(DATA_CHUNKS, shape)),
                    **self._storage,
                )
            elif path in values:
                _create(group, path, spec, dtype=dtype, data=np.asarray(values[path], dtype=dtype))
            else:
                _create(group, path, spec, shape=shape, dtype=dtype)
        _copy_metadata(source, group.create_group(layout.POINT_SOURCE_GROUP))

    # -- points ----------------------------------------------------------------

    def sink(self, index: int, source, n_threads: int = 1) -> "_ScanSink":
        """Return the stripe sink for the point at ``index``."""
        return _ScanSink(self, index, source, n_threads)

    def fail_point(self, index: int, error: str) -> None:
        """Record an expected failure of one point."""
        self._file["catalog/errors"][index] = _error_text(error)
        self._set_status(index, PointStatus.FAILED)

    def _set_status(self, index: int, status: PointStatus) -> None:
        self._file["catalog/status"][index] = int(status)
        self._file.flush()

    # -- end of run ------------------------------------------------------------

    def close(self, *, cancelled: bool = False) -> Path:
        """Finish the run, close and validate the file, and publish it.

        Points that never started become ``unattempted`` and a point left in
        ``writing`` becomes ``interrupted``.
        """
        from ._scan_reader import validate_scan_file

        status = self._file["catalog/status"]
        codes = status[...]
        codes[codes == PointStatus.PENDING] = PointStatus.UNATTEMPTED
        codes[codes == PointStatus.WRITING] = PointStatus.INTERRUPTED
        status[...] = codes
        self._file["run/status"][()] = int(RunStatus.CANCELLED if cancelled else RunStatus.FINISHED)
        self._file.close()
        validate_scan_file(self._partial)
        return publish_file(self._partial, self.path, overwrite=self._overwrite)

    def abort(self) -> None:
        """Close after a shared failure. The file keeps its ``.partial`` name."""
        try:
            self._file["run/status"][()] = int(RunStatus.FAILED)
        except Exception:
            pass
        try:
            self._file.close()
        except Exception:
            pass


class _ScanSink:
    """Stripe sink that writes one point into the run file.

    ``error`` holds the exception of a failed file operation. The reconstructor
    reports any failure after its first stripe as a failed point; the run
    reads ``error`` to tell a failure of the shared file from a failure of the
    point's input.
    """

    output_files: list = []

    def __init__(self, writer: ScanWriter, index: int, source, n_threads: int) -> None:
        self._writer = writer
        self._index = index
        self._source = source
        self._n_threads = n_threads
        self._rescale = 1.0
        self._group = writer._file[layout.POINT_GROUP.format(index=index)]
        self._data = self._group["data"]
        self.error: Exception | None = None

    def _guard(self, operation, *args, **kwargs):
        try:
            return operation(*args, **kwargs)
        except Exception as error:
            self.error = error
            raise

    def stripe_buffer_bytes(self, n_depths, cols) -> tuple[int, int]:
        # Stored pixels, depth totals, and a per-pixel projection. The raw
        # stripe reduction has the same per-pixel cost and runs separately.
        return n_depths * 8, cols * (n_depths * self._data.dtype.itemsize + 8)

    def begin(self, *, depth_um, shape, output_type, threshold, rescale) -> None:
        # A mismatch means the input changed after the manifest was frozen. It
        # is a failure of this point, not of the file.
        if (
            self._data.dtype != PIXEL_DTYPES[output_type]
            or self._data.shape != (len(depth_um), *shape)
            or not np.array_equal(self._group["depth_um"][...], depth_um)
        ):
            raise InputError(
                f"point {self._index} changed after the run prepared its output"
            )
        self._guard(self._begin, depth_um=depth_um, shape=shape, output_type=output_type,
                    threshold=threshold, rescale=rescale)

    def _begin(self, *, depth_um, shape, output_type, threshold, rescale) -> None:
        group = self._group
        stored = PIXEL_DTYPES[output_type]
        raw = self._source["entry1/data/data"]
        accumulator = layout.accumulator_dtype(stored)
        self._depth_intensity = np.zeros(len(depth_um), dtype=accumulator)
        self._sum_reconstructed = np.zeros(shape, dtype=accumulator)
        self._sum_raw = np.zeros(shape, dtype=group["reference/sum_raw"].dtype)
        self._rescale = rescale
        self._writer._set_status(self._index, PointStatus.WRITING)
        group["normalization/threshold"][()] = np.nan if threshold is None else threshold
        group["normalization/rescale"][()] = rescale
        group["reference/first_raw"][...] = raw[1]

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

    def _finish(self, *, totals, **_) -> None:
        group = self._group
        group["reference/sum_raw"][...] = self._sum_raw
        group["reference/sum_reconstructed"][...] = self._sum_reconstructed
        group["reductions/depth_intensity"][...] = self._depth_intensity
        group["computed/depth_intensity"][...] = totals
        self._writer._set_status(self._index, PointStatus.COMPLETE)

    def close(self) -> None:
        pass


def _missing(value, fill):
    return fill if value is None else value


def _detector_id(source) -> str:
    if "entry1/detector/ID" not in source:
        return ""
    values = np.asarray(source["entry1/detector/ID"]).ravel()
    if not len(values):
        return ""
    value = values[0]
    return value.decode(errors="replace") if isinstance(value, bytes) else str(value)
