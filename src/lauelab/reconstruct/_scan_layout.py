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
``text``
    Fixed-length UTF-8 text whose length is that of the value, as in the
    34-ID-E acquisition files. Point files only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from types import MappingProxyType
from typing import Mapping

import h5py
import numpy as np

from lauelab._hdf5 import UTF8

from ._writer import PIXEL_DTYPES

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
POINT_LINKS = "/point_files"


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

# Codes are stored as HDF5 enumerated types, so that HDF5 tools show the names.
POINT_STATUS_TYPE = h5py.enum_dtype({status.name.lower(): status.value for status in PointStatus},
                                    basetype="u1")
RUN_STATUS_TYPE = h5py.enum_dtype({status.name.lower(): status.value for status in RunStatus},
                                  basetype="u1")
# LaueGo pixel-type codes of the stored frames; "none" before a point has an
# output type.
PIXEL_TYPE = h5py.enum_dtype({**{dtype.name: code for code, dtype in PIXEL_DTYPES.items()}, "none": -1},
                             basetype="i1")


@dataclass(frozen=True)
class DatasetSpec:
    """Storage convention for one dataset."""

    dtype: object
    shape: tuple = ()
    units: str | None = None
    optional: bool = False
    fill: object = None
    attrs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.dtype, str):
            object.__setattr__(self, "dtype", np.dtype(self.dtype))
        object.__setattr__(self, "attrs", MappingProxyType(dict(self.attrs)))


# Fixed-length UTF-8 text sized to its value, as acquisition strings are.
TEXT = "text"

F8 = np.dtype("<f8")
I4 = np.dtype("<i4")
I8 = np.dtype("<i8")
U1 = np.dtype("u1")


def _spec(dtype, shape=(), units=None, *, optional=False, fill=None, **attrs):
    return DatasetSpec(dtype=dtype, shape=shape, units=units, optional=optional, fill=fill,
                       attrs=attrs)


# Root attributes of ``scan.h5`` in addition to the common ones: the program,
# the native library version, and the run status as a RUN_STATUS_TYPE value.
SCAN_ATTRIBUTES = ("program", "native_version", "status")
PROGRAM = "lauelab"

# Requested reconstruction settings, named as the Reconstructor arguments, and
# the geometry text. ``/geometry`` also has a ``path`` attribute. A setting
# that was not given is stored as ``fill``, which is also the HDF5 fill value of a
# numeric dataset; text uses the empty string.
SCAN_DATASETS = MappingProxyType({
    "/parameters/detector": _spec(I4),
    "/parameters/depth_range": _spec(F8, (2,), "um"),
    "/parameters/resolution": _spec(F8, units="um"),
    "/parameters/wire_edge": _spec(UTF8),
    "/parameters/percent_brightest": _spec(F8),
    "/parameters/normalization": _spec(UTF8, fill=""),
    "/parameters/norm_exponent": _spec(F8, fill=np.nan),
    "/parameters/norm_threshold": _spec(F8, fill=np.nan),
    "/parameters/cosmic_filter": _spec(U1),
    "/parameters/frame_skip": _spec(I4),
    "/parameters/wire_skip": _spec(I4),
    "/parameters/output_pixel_type": _spec(I4, fill=-1),
    "/parameters/rows_per_stripe": _spec(I4, fill=-1),
    "/parameters/memory_limit_mb": _spec(I4, units="MiB"),
    "/geometry/xml": _spec(UTF8),
})

# One row per manifest entry, in manifest order. The catalog alone answers
# every question the point table asks; no point file is opened to list a scan.
# ``fill`` is the value of a row that has no value.
POINT_COLUMNS = MappingProxyType({
    "/points/point_ids": _spec(UTF8, ("n_points",)),
    "/points/paths": _spec(UTF8, ("n_points",)),
    "/points/statuses": _spec(POINT_STATUS_TYPE, ("n_points",)),
    "/points/errors": _spec(UTF8, ("n_points",), fill=""),
    "/points/source_paths": _spec(UTF8, ("n_points",)),
    "/points/scan_numbers": _spec(I4, ("n_points",), fill=-1),
    "/points/sample_positions": _spec(F8, ("n_points", 3), "um", fill=np.nan),
    "/points/energies_kev": _spec(F8, ("n_points",), "keV", fill=np.nan),
    "/points/shapes": _spec(I4, ("n_points", 3), fill=0),
    "/points/depth_bounds": _spec(F8, ("n_points", 2), "um", fill=np.nan),
    "/points/dtypes": _spec(PIXEL_TYPE, ("n_points",), fill=-1),
})

# Point files copy the input tree except ``entry1/wire``, as the reconstruction
# executable does, and add the datasets below. Single values are ``(1,)``
# arrays and strings are fixed-length UTF-8, in the style of the 34-ID-E
# acquisition files. Optional datasets are written only when they apply.
RECONSTRUCTION = "/entry1/reconstruction/"
POINT_DATASETS = MappingProxyType({
    "/entry1/data/data": _spec("stored", ("n_depths", "rows", "columns"), signal=np.int32(1)),
    "/entry1/depth": _spec(F8, ("n_depths",), "micron"),
    # Written by the executable at these paths, only when the step ran.
    "/entry1/microDiffraction/cosmic_filter": _spec(I4, (1,), optional=True),
    "/entry1/microDiffraction/norm_exponent": _spec(F8, (1,), optional=True),
    "/entry1/microDiffraction/norm_threshold": _spec(F8, (1,), optional=True),
    "/entry1/microDiffraction/norm_rescale": _spec(F8, (1,), optional=True),
    **{RECONSTRUCTION + name: spec for name, spec in {
        "detector": _spec(I4, (1,)),
        "depth_range": _spec(F8, (2,), "micron"),
        "resolution": _spec(F8, (1,), "micron"),
        "wire_edge": _spec(TEXT, (1,)),
        "percent_brightest": _spec(F8, (1,)),
        "normalization": _spec(TEXT, (1,), optional=True),
        "frame_skip": _spec(I4, (1,)),
        "wire_skip": _spec(I4, (1,)),
        "geometry_path": _spec(TEXT, (1,)),
        "geometry_xml": _spec(TEXT, (1,)),
        "point_id": _spec(TEXT, (1,)),
        "manifest_index": _spec(I4, (1,), optional=True),
        "source_path": _spec(TEXT, (1,)),
        "source_size": _spec(I8, (1,), "bytes", optional=True),
        "source_mtime_ns": _spec(I8, (1,), optional=True),
        "raw_slices": _spec(I4, (2,)),
        "native_version": _spec(TEXT, (1,)),
        "num_threads": _spec(I4, (1,)),
        "rows_per_stripe": _spec(I4, (1,)),
        "memory_limit_mb": _spec(I4, (1,), "MiB"),
        "elapsed_seconds": _spec(F8, (1,), "s"),
        "first_raw": _spec("input", ("rows", "columns")),
        "sum_raw": _spec("raw_accumulator", ("rows", "columns")),
        "sum_reconstructed": _spec("stored_accumulator", ("rows", "columns")),
        "depth_intensity": _spec("stored_accumulator", ("n_depths",)),
        "computed_depth_intensity": _spec(F8, ("n_depths",)),
    }.items()},
})

NORMALIZATION_DATASETS = tuple(
    "/entry1/microDiffraction/" + name for name in ("norm_exponent", "norm_threshold", "norm_rescale")
)

# A file holding any of these was written by a reconstruction, so it is not
# raw input.
RESERVED_INPUT_PATHS = (
    RECONSTRUCTION.rstrip("/"), "/entry1/microDiffraction/cosmic_filter", *NORMALIZATION_DATASETS,
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
