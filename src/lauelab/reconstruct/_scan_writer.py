# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Writers for reconstruction scans: catalog snapshots and point files.

The scan coordinator is the only writer of ``scan.h5``. It keeps the catalog in
memory and periodically publishes pending changes as a complete snapshot: it is
written to a private file, closed, validated, and moved over ``scan.h5``. A
reader that already holds the file keeps its earlier snapshot.

Each worker writes its own point file. Before computation, it creates all
datasets in a private file, then fills them stripe by stripe through
:class:`PointSink`, and sets ``/entry1/reconstruction/point/complete`` last.
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
from ._scan_layout import RunStatus
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
    dataset = group.create_dataset(
        path.lstrip("/"), shape=shape, dtype=spec.dtype if dtype is None else dtype,
        data=data, **options,
    )
    set_units(dataset, spec.units)
    for name, value in spec.attrs.items():
        dataset.attrs[name] = value
    return dataset


def settings_values(detector: int, settings: Mapping, geometry_path: str,
                    geometry_xml: str) -> dict:
    """Return the ``SETTINGS_DATASETS`` values for one scan or point.

    ``settings`` holds the :class:`~lauelab.reconstruct.Reconstructor` keyword
    arguments of a prepared task.
    """
    return {
        "/settings/detector": detector,
        "/settings/depth_range": settings["depth_range"],
        "/settings/resolution": settings["resolution"],
        "/settings/wire_edge": settings["wire_edge"],
        "/settings/percent_brightest": settings["percent_brightest"],
        "/settings/normalization": settings["normalization"] or "",
        "/settings/norm_exponent": _missing(settings["norm_exponent"], np.nan),
        "/settings/norm_threshold": _missing(settings["norm_threshold"], np.nan),
        "/settings/cosmic_filter": int(settings["cosmic_filter"]),
        "/settings/output_pixel_type": _missing(settings["output_pixel_type"], -1),
        "/settings/rows_per_stripe": _missing(settings["rows_per_stripe"], -1),
        "/settings/memory_limit_mb": settings["memory_limit_mb"],
        "/geometry/path": geometry_path,
        "/geometry/xml": geometry_xml,
    }


def native_version() -> str:
    """Return the version string of the loaded native library."""
    return ffi.string(get_library().laue_version()).decode()


def write_catalog(path: Path, *, created: str, run_status: RunStatus,
                  run_values: Mapping, catalog: Mapping, replace: bool) -> None:
    """Publish one complete catalog snapshot at ``path``.

    Parameters
    ----------
    path
        ``scan.h5`` of the scan directory.
    created
        Creation time of the scan, kept by every snapshot.
    run_status
        Run status of this snapshot.
    run_values
        Values of ``SETTINGS_DATASETS`` and ``/run/native_version``.
    catalog
        Values of every ``CATALOG_DATASETS`` entry, one row per point.
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
        values = {**run_values, "/run/status": int(run_status)}
        for name, spec in {**layout.SCAN_RUN_DATASETS, **layout.SETTINGS_DATASETS}.items():
            _create(target, name, spec, data=np.asarray(values[name], dtype=spec.dtype))
        for name, spec in layout.CATALOG_DATASETS.items():
            _create(target, name, spec, data=np.asarray(catalog[name], dtype=spec.dtype))
    validate_scan_file(partial)
    publish_file(partial, path, overwrite=replace)


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
        Input metadata to preserve at its original paths.
    source_stat
        ``os.stat_result`` of the input, or ``None``.
    num_threads
        OpenMP threads of the reconstruction.
    """
    _copy_metadata(source_metadata, file, exclude_entry=("depth",), exclude_data=("depth",))
    # Acquisition axes describe the raw frames, not the reconstructed stack.
    for name in list(file["entry1/data"].attrs):
        if name in ("signal", "axes", "default", "auxiliary_signals") or name.endswith("_indices"):
            del file["entry1/data"].attrs[name]
    write_root_attributes(file, format_name=layout.POINT_FORMAT, version=layout.POINT_VERSION)
    for path, attributes in layout.POINT_GROUP_ATTRIBUTES.items():
        file.require_group(path).attrs.update(attributes)
    n_images, rows, columns = info.shape
    stored = PIXEL_DTYPES[output_type]
    geometry = info.image_geometry
    values = {
        **{"/entry1/reconstruction" + path: value for path, value in
           settings_values(task.detector, task.settings, task.geometry_path, task.geometry_xml).items()},
        "/entry1/reconstruction/program": "lauelab",
        "/entry1/reconstruction/version": file.attrs["lauelab_version"],
        "/entry1/reconstruction/date": file.attrs["created"],
        "/entry1/reconstruction/point/id": task.point_id,
        "/entry1/reconstruction/point/manifest_index": _missing(task.index, -1),
        "/entry1/reconstruction/point/complete": 0,
        "/entry1/reconstruction/point/native_version": native_version(),
        "/entry1/reconstruction/execution/num_threads": num_threads,
        "/entry1/reconstruction/execution/rows_per_stripe": 0,
        "/entry1/reconstruction/acquisition/source_path": task.source,
        "/entry1/reconstruction/acquisition/source_size": -1 if source_stat is None else source_stat.st_size,
        "/entry1/reconstruction/acquisition/source_mtime_ns": -1 if source_stat is None else source_stat.st_mtime_ns,
        "/entry1/reconstruction/acquisition/scan_number": _missing(info.scan_number, -1),
        "/entry1/reconstruction/acquisition/sample_position": info.sample_position,
        "/entry1/reconstruction/acquisition/energy_kev": _missing(info.energy_kev, np.nan),
        "/entry1/depth": depth_um,
        "/entry1/reconstruction/detector/id": _detector_id(source_metadata),
        "/entry1/reconstruction/detector/size": (geometry.nx_full, geometry.ny_full),
        "/entry1/reconstruction/detector/roi_start": geometry.start,
        "/entry1/reconstruction/detector/roi_group": geometry.group,
        "/entry1/reconstruction/normalization/threshold": np.nan,
        "/entry1/reconstruction/normalization/rescale": 1.0,
        "/entry1/reconstruction/first_raw/data": first_raw,
        "/entry1/reconstruction/acquisition/raw_slices": (1, n_images + 1),
    }
    dims = {"n_depths": len(depth_um), "rows": rows, "columns": columns}
    for path, spec in {**layout.POINT_SETTINGS_DATASETS, **layout.POINT_DATASETS}.items():
        dtype = layout.resolve_dtype(spec, stored=stored, input=info.dtype)
        shape = tuple(dims[name] if isinstance(name, str) else name for name in spec.shape)
        if path == "/entry1/data/data":
            _create(file, path, spec, shape=shape, dtype=dtype,
                    chunks=tuple(min(chunk, size) for chunk, size in zip(DATA_CHUNKS, shape)),
                    **COMPRESSION[task.compression])
        elif path in values:
            _create(file, path, spec, dtype=dtype, data=np.asarray(values[path], dtype=dtype))
        else:
            _create(file, path, spec, shape=shape, dtype=dtype)
    for path in layout.POINT_DEPTH_LINKS:
        file[path] = file["entry1/depth"]


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
        accumulator = layout.accumulator_dtype(self._data.dtype)
        self._depth_intensity = np.zeros(len(depth_um), dtype=accumulator)
        self._sum_reconstructed = np.zeros(shape, dtype=accumulator)
        self._sum_raw = np.zeros(shape, dtype=self._file["entry1/reconstruction/sum_raw/data"].dtype)
        self._rescale = rescale
        self._file["entry1/reconstruction/normalization/threshold"][()] = np.nan if threshold is None else threshold
        self._file["entry1/reconstruction/normalization/rescale"][()] = rescale

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

    def _finish(self, *, totals, stripe_rows, **_) -> None:
        file = self._file
        file["entry1/reconstruction/sum_raw/data"][...] = self._sum_raw
        file["entry1/reconstruction/sum_reconstructed/data"][...] = self._sum_reconstructed
        file["entry1/reconstruction/stored_depth_intensity/data"][...] = self._depth_intensity
        file["entry1/reconstruction/computed_depth_intensity/data"][...] = totals
        file["entry1/reconstruction/execution/rows_per_stripe"][()] = stripe_rows
        file["entry1/reconstruction/point/complete"][()] = 1

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
