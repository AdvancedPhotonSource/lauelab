# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Canonical dataset layout for reconstruction-scan HDF5 files.

One file holds one reconstruction run: a point catalog in manifest order and
one independent group per point. This module is the single definition of the
layout; the writer, reader, validator, and ``docs/development/
reconstruction-scan-format.md`` all follow this table.

Shapes use symbolic dimensions: ``n_points`` is the manifest length, and
``n_depths``, ``rows``, and ``columns`` belong to one point. Points do not
share them. A dtype that depends on the point is named by rule:

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

from dataclasses import dataclass, field
from enum import IntEnum
from types import MappingProxyType
from typing import Mapping

import numpy as np

from lauelab._hdf5 import UTF8

FORMAT = "lauelab-reconstruction-scan"
VERSION = 1
SUPPORTED_VERSIONS = frozenset({VERSION})

POINT_GROUP = "/points/{index:06d}"
RAW_SELECTION = "34ide-multi-image: stored slices [1, n_stored - 1)"

# Text that a writer changes after the file is created uses a fixed width, so
# that no object is allocated at point completion.
MUTABLE_TEXT = np.dtype("S1024")


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
    """Run state. Only ``FINISHED`` and ``CANCELLED`` files are published."""

    RUNNING = 0
    FINISHED = 1
    CANCELLED = 2
    FAILED = 3


# A closed, published file holds terminal point states only.
TERMINAL_POINT_STATUSES = frozenset({
    PointStatus.COMPLETE, PointStatus.FAILED,
    PointStatus.INTERRUPTED, PointStatus.UNATTEMPTED,
})
PUBLISHABLE_RUN_STATUSES = frozenset({RunStatus.FINISHED, RunStatus.CANCELLED})


@dataclass(frozen=True)
class DatasetSpec:
    """Storage convention for one reconstruction-scan dataset."""

    dtype: object
    shape: tuple = ()
    units: str | None = None
    mutable: bool = False
    attrs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.dtype, str):
            object.__setattr__(self, "dtype", np.dtype(self.dtype))
        object.__setattr__(self, "attrs", MappingProxyType(dict(self.attrs)))


F8 = np.dtype("<f8")
I4 = np.dtype("<i4")
I8 = np.dtype("<i8")
U1 = np.dtype("u1")


def _spec(dtype, shape=(), units=None, *, mutable=False, **attrs):
    return DatasetSpec(dtype=dtype, shape=shape, units=units, mutable=mutable, attrs=attrs)


# Written once when the file is created, except ``/run/status``.
RUN_DATASETS = MappingProxyType({
    "/run/status": _spec(U1, mutable=True),
    "/run/native_version": _spec(UTF8),
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
    "/settings/memory_limit_mb": _spec(I4),
    "/geometry/path": _spec(UTF8, missing=""),
    "/geometry/xml": _spec(UTF8),
})

# One row per manifest entry, in manifest order. The catalog alone answers
# every question the point table asks; no point group is opened to list a run.
CATALOG_DATASETS = MappingProxyType({
    "/catalog/point_ids": _spec(UTF8, ("n_points",)),
    "/catalog/source_paths": _spec(UTF8, ("n_points",)),
    "/catalog/source_sizes": _spec(I8, ("n_points",), missing=-1),
    "/catalog/source_mtimes_ns": _spec(I8, ("n_points",), missing=-1),
    "/catalog/scan_numbers": _spec(I4, ("n_points",), missing=-1),
    "/catalog/sample_positions": _spec(F8, ("n_points", 3), "um", missing="nan"),
    "/catalog/energies_kev": _spec(F8, ("n_points",), "keV", missing="nan"),
    "/catalog/image_shapes": _spec(I4, ("n_points", 2), order="rows,columns"),
    "/catalog/n_depths": _spec(I4, ("n_points",)),
    "/catalog/depth_bounds": _spec(F8, ("n_points", 2), "um", missing="nan"),
    "/catalog/pixel_types": _spec(I4, ("n_points",), missing=-1),
    "/catalog/status": _spec(U1, ("n_points",), mutable=True),
    "/catalog/errors": _spec(MUTABLE_TEXT, ("n_points",), mutable=True, missing=""),
})

# Paths below are relative to one point group, ``POINT_GROUP``. Every dataset
# is created before computation starts. A point whose catalog entry failed has
# no group.
POINT_DATASETS = MappingProxyType({
    "data": _spec("stored", ("n_depths", "rows", "columns"), mutable=True,
                  values="stored"),
    "depth_um": _spec(F8, ("n_depths",), "um"),
    "detector/id": _spec(UTF8, missing=""),
    "detector/size": _spec(I4, (2,), "pixel", order="x,y", binning="unbinned"),
    "detector/roi_start": _spec(I4, (2,), "pixel", order="x,y", binning="unbinned"),
    "detector/roi_group": _spec(I4, (2,), order="x,y"),
    "normalization/threshold": _spec(F8, mutable=True, missing="nan"),
    "normalization/rescale": _spec(F8, mutable=True),
    "reference/first_raw": _spec("input", ("rows", "columns"), mutable=True,
                                 values="raw"),
    "reference/sum_raw": _spec("raw_accumulator", ("rows", "columns"), mutable=True,
                               values="raw"),
    "reference/raw_slices": _spec(I8, (2,), selection=RAW_SELECTION),
    "reference/sum_reconstructed": _spec("stored_accumulator", ("rows", "columns"),
                                         mutable=True, values="stored"),
    "reductions/depth_intensity": _spec("stored_accumulator", ("n_depths",),
                                        mutable=True, values="stored"),
    "computed/depth_intensity": _spec(F8, ("n_depths",), mutable=True,
                                      values="computed"),
})

# A copy of the point's 34-ID-E source objects, without ``entry1/data/data``
# and ``entry1/wire``. lauelab does not define its contents; per-depth export
# copies it back unchanged.
POINT_SOURCE_GROUP = "source"


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
