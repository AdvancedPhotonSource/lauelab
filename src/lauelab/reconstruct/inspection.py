# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Numerical inspection of one reconstructed point: reference images, depth traces, and square ROIs.

Inspection results contain independent NumPy arrays, depth coordinates, and
pixel-value metadata. Use them with the figure builders in
:mod:`lauelab.visualization` or with custom plotting code.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Mapping

import numpy as np

from lauelab.indexing.errors import InputError

from ._scan_layout import accumulator_dtype
from ._scan_reader import REFERENCE_IMAGES

Bounds = tuple[int, int, int, int]

REFERENCE_LABELS = {
    "first_raw": "First raw frame",
    "sum_raw": "Sum of raw frames",
    "sum_reconstructed": "Sum of reconstructed frames",
}


def _readonly(values: np.ndarray) -> np.ndarray:
    values = np.array(values, copy=True)
    values.setflags(write=False)
    return values


# --- Square ROIs ---------------------------------------------------------------

def square_bounds(size: int, x: float, y: float, image_shape: tuple[int, int]) -> Bounds:
    """Place a square ROI of ``size`` pixels at a click and return its bounds.

    Parameters
    ----------
    size : int
        Side of the square in stored-image pixels; a positive integer.
    x, y : float
        Click position in zero-based stored-image pixel coordinates, where an
        integer is a pixel centre and pixel ``(x, y)`` covers ``x - 0.5`` to
        ``x + 0.5``.
    image_shape : tuple of int
        ``(rows, columns)`` of the stored image.

    Returns
    -------
    tuple of int
        Half-open ``(y0, y1, x0, x1)`` selecting ``image[y0:y1, x0:x1]``,
        which holds exactly ``size * size`` pixels.

    Raises
    ------
    InputError
        If ``size`` is not a positive integer, the click is not finite, or
        the square would extend outside the image.

    Notes
    -----
    The square's centre is the legal centre nearest the click: an integer
    for an odd size and a half-integer for an even size. A click at equal
    distance from two legal centres takes the lower coordinate. On each axis
    the first pixel is ``ceil(click - size / 2)``.
    """
    if isinstance(size, (bool, np.bool_)) or not isinstance(size, Integral) or size < 1:
        raise InputError(f"ROI size must be a positive integer; received {size!r}")
    size = int(size)
    rows, columns = (int(value) for value in image_shape)
    try:
        x, y = float(x), float(y)
    except (TypeError, ValueError) as error:
        raise InputError("ROI position must be two numbers") from error
    if not np.isfinite([x, y]).all():
        raise InputError("ROI position must be finite")
    x0 = int(np.ceil(x - size / 2))
    y0 = int(np.ceil(y - size / 2))
    if x0 < 0 or y0 < 0 or x0 + size > columns or y0 + size > rows:
        raise InputError(
            f"a {size} by {size} ROI at ({x:g}, {y:g}) does not fit inside the "
            f"{rows} by {columns} image"
        )
    return (y0, y0 + size, x0, x0 + size)


def bounds_center(bounds: Bounds) -> tuple[float, float]:
    """Return the ``(x, y)`` centre of half-open ``(y0, y1, x0, x1)`` bounds."""
    y0, y1, x0, x1 = _bounds(bounds)
    return ((x0 + x1 - 1) / 2, (y0 + y1 - 1) / 2)


def _bounds(bounds) -> Bounds:
    if len(bounds) != 4 or any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) for value in bounds
    ):
        raise InputError("bounds must be four integers (y0, y1, x0, x1)")
    y0, y1, x0, x1 = (int(value) for value in bounds)
    if not (0 <= y0 < y1 and 0 <= x0 < x1):
        raise InputError(f"bounds {(y0, y1, x0, x1)} must select at least one pixel")
    return y0, y1, x0, x1


# --- Prepared data --------------------------------------------------------------

@dataclass(frozen=True)
class DepthTrace:
    """Intensity of one region through the reconstructed depths of a point.

    Attributes
    ----------
    values : numpy.ndarray
        Sum of the stored pixels in the region at each depth, shape
        ``(n_depths,)``. ``numpy.int64`` for an integer stored dtype and
        ``numpy.float64`` otherwise; exact in the integer case. Values are
        signed, and a sum can be zero or negative.
    depth_um : numpy.ndarray
        Physical depth of each sample in µm, shape ``(n_depths,)``.
    point_id : str
        Point the trace belongs to.
    bounds : tuple of int or None
        Half-open ``(y0, y1, x0, x1)`` of the region, or `None` for the full
        frame.
    label : str
        Caller-facing name of the region.
    """

    values: np.ndarray
    depth_um: np.ndarray
    point_id: str
    bounds: Bounds | None
    label: str

    def __post_init__(self):
        values = _readonly(self.values)
        depth_um = _readonly(np.asarray(self.depth_um, dtype=np.float64))
        if values.ndim != 1 or depth_um.shape != values.shape:
            raise ValueError("values and depth_um must be one-dimensional and the same length")
        if values.dtype.kind not in "if":
            raise ValueError("values must be numeric")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "depth_um", depth_um)
        object.__setattr__(self, "bounds", None if self.bounds is None else _bounds(self.bounds))
        if not self.label:
            raise ValueError("label cannot be empty")

    @property
    def depth_index(self) -> np.ndarray:
        """Zero-based depth index of each sample, the alternative x axis."""
        return np.arange(len(self.values))

    @property
    def n_pixels(self) -> int:
        """Number of stored pixels summed at each depth, or 0 for the full frame."""
        if self.bounds is None:
            return 0
        y0, y1, x0, x1 = self.bounds
        return (y1 - y0) * (x1 - x0)

    def normalized(self) -> "NormalizedTrace":
        """Divide the trace by its own maximum. See :class:`NormalizedTrace`."""
        maximum = self.values.max() if len(self.values) else 0
        if not maximum > 0:
            return NormalizedTrace(
                None, f"{self.label}: maximum {maximum:g} is not positive, so the trace "
                "cannot be normalized"
            )
        return NormalizedTrace(self.values / maximum, "")

    def log_samples(self) -> "LogSamples":
        """Select the samples a logarithmic axis can show. See :class:`LogSamples`."""
        kept = np.flatnonzero(self.values > 0)
        return LogSamples(kept, len(self.values) - len(kept))


@dataclass(frozen=True)
class NormalizedTrace:
    """Normalized trace and an explanation when normalization is unavailable.

    Attributes
    ----------
    values : numpy.ndarray or None
        ``trace / max(trace)`` as ``numpy.float64``, with the sign of each
        sample preserved and the maximum equal to 1. `None` when the maximum
        is zero or negative.
    reason : str
        Why ``values`` is `None`, or an empty string.
    """

    values: np.ndarray | None
    reason: str

    def __post_init__(self):
        if self.values is not None:
            object.__setattr__(self, "values", _readonly(np.asarray(self.values, dtype=np.float64)))

    @property
    def available(self) -> bool:
        return self.values is not None


@dataclass(frozen=True)
class LogSamples:
    """Positive-sample indices and the count of omitted nonpositive samples.

    Attributes
    ----------
    kept : numpy.ndarray
        Indices of the samples that are positive, in order.
    n_omitted : int
        Number of zero or negative samples omitted from a logarithmic plot.
    """

    kept: np.ndarray
    n_omitted: int

    def __post_init__(self):
        object.__setattr__(self, "kept", _readonly(np.asarray(self.kept, dtype=np.int64)))


@dataclass(frozen=True)
class ReferenceImage:
    """Reference image with its source kind and pixel-value convention.

    Attributes
    ----------
    image : numpy.ndarray
        ``(rows, columns)`` array the caller owns.
    kind : str
        ``"first_raw"``, ``"sum_raw"``, or ``"sum_reconstructed"``.
    values : str
        ``"raw"`` for detector counts before filtering and normalization, or
        ``"stored"`` for a sum of the stored reconstructed frames.
    point_id : str
    label : str
        Caller-facing description of ``kind``.
    """

    image: np.ndarray
    kind: str
    values: str
    point_id: str
    label: str

    def __post_init__(self):
        image = _readonly(self.image)
        if image.ndim != 2:
            raise ValueError("a reference image must be two-dimensional")
        object.__setattr__(self, "image", image)
        if self.kind not in REFERENCE_IMAGES:
            raise ValueError(f"kind must be one of {REFERENCE_IMAGES}")
        if self.values not in ("raw", "stored"):
            raise ValueError("values must be 'raw' or 'stored'")

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.image.shape)


# --- Point access ---------------------------------------------------------------

class ArrayPoint:
    """The point-access interface over an in-memory ``(depth, y, x)`` stack.

    Use it to inspect ``ReconstructionResult.images`` or any stack the same
    way as a point in a scan file. Inspection uses the array's existing
    dtype and values.

    Parameters
    ----------
    images : numpy.ndarray
        Numeric array with shape ``(n_depths, rows, columns)``.
    depth_um : numpy.ndarray
        Physical depth of each frame in µm, shape ``(n_depths,)``.
    point_id : str
        Name for the point in prepared data. The default is ``"array"``.

    Notes
    -----
    The available reference image is ``"sum_reconstructed"``. Requests for
    raw reference images raise :class:`~lauelab.indexing.InputError`.
    """

    def __init__(self, images, depth_um, point_id: str = "array") -> None:
        images = np.asarray(images)
        if images.ndim != 3 or images.dtype.kind not in "iuf":
            raise InputError("images must be a numeric array with shape (n_depths, rows, columns)")
        depth_um = np.asarray(depth_um, dtype=np.float64)
        if depth_um.shape != (len(images),) or not np.isfinite(depth_um).all():
            raise InputError("depth_um must hold one finite value per frame")
        if not isinstance(point_id, str) or not point_id:
            raise InputError("point_id must be a non-empty string")
        self._images = images
        self.depth_um = _readonly(depth_um)
        self.point_id = point_id
        self.shape = tuple(int(value) for value in images.shape)
        self.dtype = images.dtype

    def frame(self, depth_index: int) -> np.ndarray:
        """Return frame ``depth_index`` as a new array."""
        if isinstance(depth_index, (bool, np.bool_)) or not isinstance(depth_index, Integral):
            raise TypeError("depth_index must be an integer")
        if not 0 <= depth_index < self.shape[0]:
            raise IndexError(f"depth_index {depth_index} is outside 0 to {self.shape[0] - 1}")
        return np.array(self._images[int(depth_index)])

    def region(self, bounds, depths: slice | None = None) -> np.ndarray:
        """Return pixels in half-open ``bounds`` through ``depths`` as a new array."""
        y0, y1, x0, x1 = _bounds(bounds)
        _, rows, columns = self.shape
        if y1 > rows or x1 > columns:
            raise InputError(f"bounds {(y0, y1, x0, x1)} are outside the {rows} by {columns} image")
        start, stop, step = (depths or slice(None)).indices(self.shape[0])
        if step != 1:
            raise InputError("depths must have a step of 1")
        return np.array(self._images[start:stop, y0:y1, x0:x1])

    def depth_intensity(self) -> np.ndarray:
        """Return the sum of each frame in the accumulator dtype."""
        return self._images.sum(axis=(1, 2), dtype=accumulator_dtype(self.dtype))

    def iter_blocks(self, max_bytes: int):
        """Yield owned ``(first_depth_index, frames)`` blocks under a byte budget.

        The budget must hold one frame; the original caller-owned array is
        additional. Callers control how many returned blocks they retain.
        """
        frame_bytes = self.shape[1] * self.shape[2] * self.dtype.itemsize
        if (isinstance(max_bytes, (bool, np.bool_)) or not isinstance(max_bytes, Integral)
                or max_bytes < max(1, frame_bytes)):
            raise InputError(f"max_bytes must be at least one frame ({frame_bytes} bytes)")
        count = int(max_bytes // max(1, frame_bytes))
        for first in range(0, self.shape[0], count):
            yield first, np.array(self._images[first:first + count])

    def reference(self, name: str) -> np.ndarray:
        if name == "sum_reconstructed":
            return self._images.sum(axis=0, dtype=accumulator_dtype(self.dtype))
        if name in REFERENCE_IMAGES:
            raise InputError(f"reference image {name!r} is not available for an in-memory point")
        raise InputError(f"reference image must be one of {REFERENCE_IMAGES}; received {name!r}")


def _point_id(point) -> str:
    entry = getattr(point, "entry", None)
    return entry.point_id if entry is not None else point.point_id


def reference_image(point, kind: str = "sum_reconstructed") -> ReferenceImage:
    """Return a reference image with its source kind and pixel-value convention.

    Parameters
    ----------
    point : PointReader, PerDepthReader, or ArrayPoint
    kind : str
        ``"sum_reconstructed"`` (the default), ``"first_raw"``, or
        ``"sum_raw"``.

    Raises
    ------
    InputError
        If ``kind`` is unknown or the point does not hold that image.
    """
    if kind not in REFERENCE_IMAGES:
        raise InputError(f"kind must be one of {REFERENCE_IMAGES}; received {kind!r}")
    return ReferenceImage(
        point.reference(kind), kind, "stored" if kind == "sum_reconstructed" else "raw",
        _point_id(point), REFERENCE_LABELS[kind],
    )


def depth_trace(point, bounds: Bounds | None = None, *, label: str | None = None,
                max_bytes: int = 64 * 2**20) -> DepthTrace:
    """Return the stored intensity of a region through depth.

    Parameters
    ----------
    point : PointReader, PerDepthReader, or ArrayPoint
    bounds : tuple of int or None
        Half-open ``(y0, y1, x0, x1)`` in stored-image pixels, or `None` for
        the full frame. A scan point uses its embedded reduction without
        reading pixels; per-depth files and arrays are reduced on demand.
        An ROI reads only its selected pixels in bounded depth blocks.
    label : str or None
        Name for the trace. The default names the full frame or the bounds.
    max_bytes : int
        Maximum pixel bytes read at once for an ROI; default 64 MiB. Must
        hold at least one depth plane of the ROI. The returned 1D trace and
        storage-library buffers are additional. Ignored for the full-frame
        trace, which uses the reader's reduction.

    Raises
    ------
    InputError
        If the bounds are not four integers selecting at least one pixel
        inside the image.
    """
    point_id = _point_id(point)
    if bounds is None:
        values = point.depth_intensity()
        return DepthTrace(values, point.depth_um, point_id, None, label or "Full frame")
    bounds = _bounds(bounds)
    y0, y1, x0, x1 = bounds
    if y1 > point.shape[1] or x1 > point.shape[2]:
        raise InputError(f"bounds {bounds} are outside the {point.shape[1:]} image")
    plane_bytes = (y1 - y0) * (x1 - x0) * point.dtype.itemsize
    if (isinstance(max_bytes, (bool, np.bool_)) or not isinstance(max_bytes, Integral)
            or max_bytes < plane_bytes):
        raise InputError(f"max_bytes must hold at least one ROI plane ({plane_bytes} bytes)")
    count = int(max_bytes // plane_bytes)
    values = np.empty(point.shape[0], dtype=accumulator_dtype(point.dtype))
    for first in range(0, len(values), count):
        stop = min(first + count, len(values))
        pixels = point.region(bounds, slice(first, stop))
        values[first:stop] = pixels.sum(axis=(1, 2), dtype=values.dtype)
        del pixels
    default = f"ROI [{y0}:{y1}, {x0}:{x1}]"
    return DepthTrace(values, point.depth_um, point_id, bounds, label or default)


def roi_traces(point, rois: Mapping[str, Bounds], *,
               max_bytes: int = 64 * 2**20) -> dict[str, DepthTrace]:
    """Return one :class:`DepthTrace` per named region, in the given order.

    ``rois`` maps a caller-chosen name to half-open bounds. An empty mapping
    returns an empty dict and is not an error. ``max_bytes`` bounds each
    ROI's pixel reads as in :func:`depth_trace`; ROIs are reduced in sequence.
    """
    return {name: depth_trace(point, bounds, label=name, max_bytes=max_bytes)
            for name, bounds in rois.items()}
