# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Shared frame input and detector ROI conversions."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import os

import numpy as np

from .errors import InputError

from lauelab._frame import (
    detector_to_roi_pixels, load_mask, read_h5_frame, read_scan_frame,
    roi_inclusive_end, roi_to_detector_pixels,
)


@dataclass(frozen=True)
class ScanFrame:
    """Reference to one reconstructed depth image for processing.

    Supply the point file path, point ID, and depth index to select a stored
    frame. The reference can be pickled, sent to a worker, and saved with
    indexing results. Each process opens the point file when it reads the
    frame; a scan catalog is unnecessary.

    Parameters
    ----------
    path : pathlib.Path or str
        A point file, such as ``points/<input stem>.h5`` of a scan directory
        written by :func:`~lauelab.reconstruct.reconstruct_scan`, or a
        standalone file written by :func:`~lauelab.reconstruct.reconstruct_point`.
        :meth:`ScanReader.point_path <lauelab.reconstruct.ScanReader.point_path>`
        resolves a catalog point ID to this path.
    point_id : str
        Expected point ID. Reading fails if the ID in the file differs.
    depth_index : int
        Zero-based position of the frame in the point's depth stack. Its
        physical depth in µm is read from the file.
    """

    path: str
    point_id: str
    depth_index: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", os.fspath(self.path))
        if not isinstance(self.point_id, str) or not self.point_id:
            raise ValueError("point_id must be a non-empty string")
        if isinstance(self.depth_index, (bool, np.bool_)) or not isinstance(self.depth_index, Integral):
            raise TypeError("depth_index must be an integer")
        if self.depth_index < 0:
            raise ValueError("depth_index must be nonnegative")
        object.__setattr__(self, "depth_index", int(self.depth_index))


def read_frame(frame: str | os.PathLike[str] | ScanFrame):
    """Read one stored frame without processing it.

    Parameters
    ----------
    frame
        Path to a 34-ID-E HDF5 frame, or a :class:`ScanFrame` selecting a
        reconstructed depth image.

    Returns
    -------
    tuple
        ``(image, metadata, processing)``. ``image`` is the two-dimensional
        pixel array indexed ``[y, x]``, in the type stored in the file.
        ``metadata`` is a dict of the :class:`FrameMetadata` field names found
        in the file. ``processing`` holds any of ``start`` and ``group``, each
        an ``(x, y)`` pair giving the full-detector ROI origin and pixel
        grouping, and ``depth`` in µm. A key is absent when the file does not
        record it.

    Raises
    ------
    InputError
        If the file cannot be read or is not a supported frame.
    """
    if isinstance(frame, ScanFrame):
        try:
            return read_scan_frame(frame)
        except (OSError, KeyError, ValueError, IndexError) as error:
            raise InputError(f"cannot read {frame}: {error}") from error
    try:
        return read_h5_frame(frame)
    except (OSError, KeyError, ValueError) as error:
        raise InputError(f"cannot read HDF5 frame {frame}: {error}") from error
