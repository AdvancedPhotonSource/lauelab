# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Point access over the existing one-HDF5-file-per-depth output."""

from numbers import Integral
from pathlib import Path

import h5py
import numpy as np

from lauelab.indexing.errors import InputError

from ._reader import _scalar
from ._scan_layout import accumulator_dtype
from ._scan_reader import REFERENCE_IMAGES, _index, _region_bounds, _text
from ._writer import pixel_type


class PerDepthReader:
    """Inspect one point stored as individual reconstructed HDF5 frames.

    Parameters
    ----------
    paths
        Iterable of per-depth HDF5 paths from ``Reconstructor``, ``reconstruct``, or
        ``export_per_depth``. Supply only one point's files, without its
        summary text file. Files are ordered by their embedded physical
        depth, not their filename. Depths must be finite and unique; frame
        shape, dtype, detector mapping and normalization must agree.
    point_id : str
        Caller-assigned point identity; default ``"per-depth"``.

    Notes
    -----
    Use as a context manager. Metadata inspection reads no pixels, and each
    data operation opens and closes at most one file at a time. Closing the
    reader prevents further reads. Treat the source files as immutable while
    using it. Every returned array is owned by the caller.

    ``shape`` is ``(depth, y, x)``; ``dtype`` describes stored pixels, and
    ``depth_um`` holds physical depths in µm. ``detector_size``, ``start`` and
    ``group`` use the same unbinned ``(x, y)`` convention as :class:`PointReader`.
    ``norm_rescale`` and ``norm_threshold`` describe the stored conversion;
    reading never applies it again. Only the reconstructed sum reference is
    available. Raw references raise :class:`~lauelab.indexing.InputError`.
    """

    def __init__(self, paths, point_id: str = "per-depth") -> None:
        if not isinstance(point_id, str) or not point_id:
            raise InputError("point_id must be a non-empty string")
        if isinstance(paths, (str, Path)):
            raise InputError("paths must be a sequence of per-depth HDF5 files")
        records = []
        expected = None
        for path in paths:
            path = Path(path)
            try:
                with h5py.File(path, "r") as source:
                    data = source["entry1/data/data"]
                    if not isinstance(data, h5py.Dataset) or data.ndim != 2 or min(data.shape) < 1:
                        raise InputError(f"{path}: expected a nonempty 2D reconstructed frame")
                    pixel_type(data.dtype)
                    depth = float(_scalar(source, "entry1/depth"))
                    if not np.isfinite(depth):
                        raise InputError(f"{path}: depth must be finite")
                    rows, cols = data.shape
                    metadata = {
                        "dtype": data.dtype,
                        "detector_id": _text(_scalar(source, "entry1/detector/ID", "")),
                        "detector_size": tuple(int(_scalar(source, f"entry1/detector/N{axis}", size))
                                               for axis, size in zip("xy", (cols, rows))),
                        "start": tuple(int(_scalar(source, f"entry1/detector/start{axis}", 0))
                                       for axis in "xy"),
                        "group": tuple(int(_scalar(source, f"entry1/detector/bin{axis}", 1))
                                       for axis in "xy"),
                        "norm_rescale": float(_scalar(source, "entry1/microDiffraction/norm_rescale", 1.0)),
                        "norm_threshold": _scalar(source, "entry1/microDiffraction/norm_threshold"),
                    }
                    if metadata["norm_threshold"] is not None:
                        threshold = float(metadata["norm_threshold"])
                        metadata["norm_threshold"] = None if np.isnan(threshold) else threshold
                    signature = (data.shape, metadata)
                    if expected is not None and signature != expected:
                        raise InputError(f"{path}: frame shape, dtype or metadata disagrees with the point")
                    expected = signature
                    records.append((depth, path))
            except (KeyError, TypeError, ValueError, OverflowError) as error:
                raise InputError(f"invalid per-depth frame {path}: {error}") from error
        if not records:
            raise InputError("paths must contain at least one per-depth frame")
        records.sort(key=lambda record: record[0])
        self.depth_um = np.array([record[0] for record in records])
        if (np.diff(self.depth_um) <= 0).any():
            raise InputError("per-depth files must have unique depths")
        self.paths = tuple(record[1] for record in records)
        self.shape = (len(records), *expected[0])
        for name, value in expected[1].items():
            setattr(self, name, value)
        self.point_id = point_id
        self._closed = False

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

    def close(self) -> None:
        """Prevent further reads. Safe to call more than once."""
        self._closed = True

    def _check_open(self):
        if self._closed:
            raise InputError("per-depth reader is closed")

    def _read_into(self, index, bounds, destination, selection=None):
        self._check_open()
        y0, y1, x0, x1 = bounds
        with h5py.File(self.paths[index], "r") as source:
            data = source["entry1/data/data"]
            if data.shape != self.shape[1:] or data.dtype != self.dtype:
                raise InputError(f"{self.paths[index]} changed after opening the point")
            data.read_direct(destination, np.s_[y0:y1, x0:x1], selection)

    def frame(self, depth_index: int) -> np.ndarray:
        """Return one stored ``(y, x)`` frame at a zero-based depth index."""
        index = _index(depth_index, self.shape[0], "depth_index")
        frame = np.empty(self.shape[1:], dtype=self.dtype)
        self._read_into(index, (0, self.shape[1], 0, self.shape[2]), frame)
        return frame

    def region(self, bounds, depths: slice | None = None) -> np.ndarray:
        """Return a region through depth, as :meth:`PointReader.region` does."""
        self._check_open()
        y0, y1, x0, x1 = _region_bounds(bounds, self.shape)
        start, stop, step = (depths or slice(None)).indices(self.shape[0])
        if step != 1:
            raise InputError("depths must have a step of 1")
        values = np.empty((max(0, stop - start), y1 - y0, x1 - x0), dtype=self.dtype)
        for offset, index in enumerate(range(start, stop)):
            self._read_into(index, bounds, values, np.s_[offset, :, :])
        return values

    def iter_blocks(self, max_bytes: int):
        """Yield ``(first_depth_index, frames)`` under a pixel byte budget.

        ``max_bytes`` must hold at least one frame. Only one returned block
        is allocated per iteration; callers control how many blocks they keep.
        """
        frame_bytes = self.shape[1] * self.shape[2] * self.dtype.itemsize
        if (isinstance(max_bytes, (bool, np.bool_)) or not isinstance(max_bytes, Integral)
                or max_bytes < frame_bytes):
            raise InputError(f"max_bytes must be at least one frame ({frame_bytes} bytes)")
        count = int(max_bytes // frame_bytes)
        for first in range(0, self.shape[0], count):
            yield first, self.region((0, self.shape[1], 0, self.shape[2]), slice(first, first + count))

    def depth_intensity(self) -> np.ndarray:
        """Sum stored pixels per depth, reading one frame at a time."""
        self._check_open()
        values = np.empty(self.shape[0], dtype=accumulator_dtype(self.dtype))
        for index in range(len(values)):
            values[index] = self.frame(index).sum(dtype=values.dtype)
        return values

    def reference(self, name: str) -> np.ndarray:
        """Sum reconstructed frames; raw references are unavailable."""
        self._check_open()
        if name not in REFERENCE_IMAGES:
            raise InputError(f"reference image must be one of {REFERENCE_IMAGES}")
        if name != "sum_reconstructed":
            raise InputError(f"reference image {name!r} is not available for per-depth files")
        values = np.zeros(self.shape[1:], dtype=accumulator_dtype(self.dtype))
        for index in range(self.shape[0]):
            values += self.frame(index)
        return values
