# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Canonical dataset layouts for reconstruction scans and point files.

Scan output consists of a ``scan.h5`` catalog and one HDF5 file per point
under ``points/``. The catalog stores run settings and lists points in manifest
order. Writers, readers, validators, and the format documentation in
``docs/development/reconstruction-scan-format.md`` share these layout tables.

Shapes use symbolic dimensions: ``n_points`` is the manifest length, and
``n_depths``, ``rows``, and ``columns`` can vary between points. Dtypes that
depend on the point use the following rules:

``stored``
    The point's output pixel type, one of ``lauelab.reconstruct._writer.
    PIXEL_DTYPES``.
``input``
    The dtype of the point's raw frames.
``raw_accumulator``
    ``<i8`` when ``input`` is an integer dtype of at most 4 bytes, otherwise
    ``<f8``.
``stored_accumulator``
    The same rule applied to ``stored``. Every integer pixel type is at most
    4 bytes, so integer output always reduces exactly in ``<i8``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import IntEnum
from types import MappingProxyType
from typing import Mapping

import numpy as np

from lauelab._hdf5 import UTF8

SCAN_FORMAT = "lauelab-reconstruction-scan"
SCAN_VERSION = 2
# Version 1 kept every point inside one file. It was never released for
# production use and is not read; readers name it explicitly instead of
# reporting a generic version mismatch.
RETIRED_SCAN_VERSIONS = frozenset({1})

POINT_FORMAT = "lauelab-reconstruction-point"
POINT_VERSION = 2

SCAN_FILENAME = "scan.h5"
POINT_DIRECTORY = "points"
POINT_SUFFIX = ".h5"

RAW_SELECTION = "34ide-multi-image: stored slices [1, n_stored - 1)"


class PointStatus(IntEnum):
    """Point state. ``PENDING`` is zero so that an unwritten entry reads as
    pending, never as complete."""

    PENDING = 0
    WRITING = 1
    COMPLETE = 2
    FAILED = 3
    INTERRUPTED = 4
    UNATTEMPTED = 5


class RunStatus(IntEnum):
    """Run state recorded in each catalog snapshot."""

    RUNNING = 0
    FINISHED = 1
    CANCELLED = 2
    FAILED = 3


# A finished or cancelled run holds terminal point states only.
TERMINAL_POINT_STATUSES = frozenset({
    PointStatus.COMPLETE, PointStatus.FAILED,
    PointStatus.INTERRUPTED, PointStatus.UNATTEMPTED,
})
SETTLED_RUN_STATUSES = frozenset({RunStatus.FINISHED, RunStatus.CANCELLED})


@dataclass(frozen=True)
class DatasetSpec:
    """Storage convention for one dataset."""

    dtype: object
    shape: tuple = ()
    units: str | None = None
    attrs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.dtype, str):
            object.__setattr__(self, "dtype", np.dtype(self.dtype))
        object.__setattr__(self, "attrs", MappingProxyType(dict(self.attrs)))


F8 = np.dtype("<f8")
I4 = np.dtype("<i4")
I8 = np.dtype("<i8")
U1 = np.dtype("u1")


def _spec(dtype, shape=(), units=None, **attrs):
    return DatasetSpec(dtype=dtype, shape=shape, units=units, attrs=attrs)


# Requested reconstruction settings and the geometry, identical in the catalog
# and in every point file of a scan.
SETTINGS_DATASETS = MappingProxyType({
    "/settings/detector": _spec(I4),
    "/settings/depth_range": _spec(F8, (2,), "um"),
    "/settings/resolution": _spec(F8, units="um"),
    "/settings/wire_edge": _spec(UTF8),
    "/settings/percent_brightest": _spec(F8),
    "/settings/normalization": _spec(UTF8, missing=""),
    "/settings/norm_exponent": _spec(F8, missing="nan"),
    "/settings/norm_threshold": _spec(F8, missing="nan"),
    "/settings/cosmic_filter": _spec(U1),
    "/settings/output_pixel_type": _spec(I4, missing=-1),
    "/settings/rows_per_stripe": _spec(I4, missing=-1),
    "/settings/memory_limit_mb": _spec(I4),
    "/geometry/path": _spec(UTF8, missing=""),
    "/geometry/xml": _spec(UTF8),
})

# ``scan.h5`` holds these, SETTINGS_DATASETS, and CATALOG_DATASETS.
SCAN_RUN_DATASETS = MappingProxyType({
    "/run/status": _spec(U1),
    "/run/native_version": _spec(UTF8),
})

# One row per manifest entry, in manifest order. The catalog alone answers
# every question the point table asks; no point file is opened to list a scan.
CATALOG_DATASETS = MappingProxyType({
    "/catalog/point_ids": _spec(UTF8, ("n_points",)),
    "/catalog/point_paths": _spec(UTF8, ("n_points",)),
    "/catalog/source_paths": _spec(UTF8, ("n_points",)),
    "/catalog/source_sizes": _spec(I8, ("n_points",), missing=-1),
    "/catalog/source_mtimes_ns": _spec(I8, ("n_points",), missing=-1),
    "/catalog/raw_slices": _spec(I8, ("n_points", 2), missing=-1, selection=RAW_SELECTION),
    "/catalog/scan_numbers": _spec(I4, ("n_points",), missing=-1),
    "/catalog/sample_positions": _spec(F8, ("n_points", 3), "um", missing="nan"),
    "/catalog/energies_kev": _spec(F8, ("n_points",), "keV", missing="nan"),
    "/catalog/image_shapes": _spec(I4, ("n_points", 2), order="rows,columns"),
    "/catalog/n_depths": _spec(I4, ("n_points",)),
    "/catalog/depth_bounds": _spec(F8, ("n_points", 2), "um", missing="nan"),
    "/catalog/pixel_types": _spec(I4, ("n_points",), missing=-1),
    "/catalog/status": _spec(U1, ("n_points",)),
    "/catalog/errors": _spec(UTF8, ("n_points",), missing=""),
})

# Point files contain these datasets and POINT_SETTINGS_DATASETS. All are
# created before computation; the point/complete marker is set last, before
# closing and publishing the file.
POINT_DATASETS = MappingProxyType({
    "/entry1/reconstruction/program": _spec(UTF8),
    "/entry1/reconstruction/version": _spec(UTF8),
    "/entry1/reconstruction/date": _spec(UTF8),
    "/entry1/reconstruction/point/id": _spec(UTF8),
    "/entry1/reconstruction/point/manifest_index": _spec(I8, missing=-1),
    "/entry1/reconstruction/point/complete": _spec(U1),
    "/entry1/reconstruction/point/native_version": _spec(UTF8),
    "/entry1/reconstruction/execution/num_threads": _spec(I4),
    "/entry1/reconstruction/execution/rows_per_stripe": _spec(I4),
    "/entry1/reconstruction/acquisition/source_path": _spec(UTF8),
    "/entry1/reconstruction/acquisition/source_size": _spec(I8, units="bytes", missing=-1),
    "/entry1/reconstruction/acquisition/source_mtime_ns": _spec(I8, missing=-1),
    "/entry1/reconstruction/acquisition/scan_number": _spec(I4, missing=-1),
    "/entry1/reconstruction/acquisition/sample_position": _spec(F8, (3,), "um", missing="nan"),
    "/entry1/reconstruction/acquisition/energy_kev": _spec(F8, units="keV", missing="nan"),
    "/entry1/data/data": _spec("stored", ("n_depths", "rows", "columns"), values="stored"),
    "/entry1/depth": _spec(F8, ("n_depths",), "um"),
    "/entry1/reconstruction/detector/id": _spec(UTF8, missing=""),
    "/entry1/reconstruction/detector/size": _spec(I4, (2,), "pixel", order="x,y", binning="unbinned"),
    "/entry1/reconstruction/detector/roi_start": _spec(I4, (2,), "pixel", order="x,y", binning="unbinned"),
    "/entry1/reconstruction/detector/roi_group": _spec(I4, (2,), order="x,y"),
    "/entry1/reconstruction/normalization/threshold": _spec(F8, missing="nan"),
    "/entry1/reconstruction/normalization/rescale": _spec(F8),
    "/entry1/reconstruction/first_raw/data": _spec("input", ("rows", "columns"), values="raw"),
    "/entry1/reconstruction/sum_raw/data": _spec("raw_accumulator", ("rows", "columns"), values="raw"),
    "/entry1/reconstruction/acquisition/raw_slices": _spec(I8, (2,), selection=RAW_SELECTION),
    "/entry1/reconstruction/sum_reconstructed/data": _spec("stored_accumulator", ("rows", "columns"),
                                          values="stored"),
    "/entry1/reconstruction/stored_depth_intensity/data": _spec("stored_accumulator", ("n_depths",), values="stored"),
    "/entry1/reconstruction/computed_depth_intensity/data": _spec(F8, ("n_depths",), values="computed"),
})

# The catalog retains its settings paths; points keep the same fields under NXprocess.
POINT_SETTINGS_DATASETS = MappingProxyType({
    "/entry1/reconstruction" + path: (replace(spec, units="MiB")
        if path == "/settings/memory_limit_mb" else spec)
    for path, spec in SETTINGS_DATASETS.items()
})

POINT_GROUP_ATTRIBUTES = {
    "/": {"default": "entry1"},
    "/entry1": {"NX_class": "NXentry", "default": "data"},
    "/entry1/data": {"NX_class": "NXdata", "signal": "data", "axes": ["depth", ".", "."]},
    "/entry1/reconstruction": {"NX_class": "NXprocess"},
}
for name in ("point", "settings", "execution", "geometry", "acquisition", "detector", "normalization"):
    POINT_GROUP_ATTRIBUTES[f"/entry1/reconstruction/{name}"] = {"NX_class": "NXparameters"}
for name in ("first_raw", "sum_raw", "sum_reconstructed", "stored_depth_intensity", "computed_depth_intensity"):
    POINT_GROUP_ATTRIBUTES[f"/entry1/reconstruction/{name}"] = {
        "NX_class": "NXdata", "signal": "data",
        "axes": ["depth"] if name.endswith("depth_intensity") else [".", "."],
    }
POINT_DEPTH_LINKS = (
    "/entry1/data/depth",
    "/entry1/reconstruction/stored_depth_intensity/depth",
    "/entry1/reconstruction/computed_depth_intensity/depth",
)


def accumulator_dtype(dtype) -> np.dtype:
    """Return the reduction dtype for pixels of ``dtype``.

    An ``<i8`` sum of ``n`` values of at most 4 bytes is exact while
    ``n < 2**31``, which exceeds the pixel count of a 4096 by 4096 frame.
    """
    dtype = np.dtype(dtype)
    return I8 if dtype.kind in "iu" and dtype.itemsize <= 4 else F8


def resolve_dtype(spec: DatasetSpec, *, stored=None, input=None) -> np.dtype:
    """Return the concrete dtype of ``spec`` for one point."""
    if not isinstance(spec.dtype, str):
        return spec.dtype
    if spec.dtype == "stored":
        return np.dtype(stored)
    if spec.dtype == "input":
        return np.dtype(input)
    if spec.dtype == "stored_accumulator":
        return accumulator_dtype(stored)
    if spec.dtype == "raw_accumulator":
        return accumulator_dtype(input)
    raise ValueError(f"unknown dtype rule {spec.dtype!r}")


def point_path(stem: str) -> str:
    """Return the catalog path of the point file for input ``stem``."""
    return f"{POINT_DIRECTORY}/{stem}{POINT_SUFFIX}"


def unsafe_stem(stem: str) -> str | None:
    """Return a validation error for an unsafe input stem, or ``None`` if valid.

    Any name the filesystem accepts is allowed except an empty or hidden name
    and one with a path separator or control character, so that the stems of
    read-only beamline data can be used unchanged.
    """
    if not stem:
        return "it is empty"
    if stem.startswith("."):
        return "it starts with '.'"
    if "/" in stem or "\\" in stem:
        return "it contains a path separator"
    if any(ord(character) < 32 or ord(character) == 127 for character in stem):
        return "it contains a control character"
    return None
