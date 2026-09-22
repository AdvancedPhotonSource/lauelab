# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Reconstruct many wire-scan points into one reconstruction-scan file."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from time import perf_counter
from typing import Callable, Sequence

import h5py

from lauelab.indexing.errors import InputError, LaueError, ReconstructionError

from ._reader import read_scan_info
from ._scan_writer import ScanWriter
from ._writer import PIXEL_DTYPES
from .reconstructor import Reconstructor

# Failures of one point's input or reconstruction. Anything else, and any
# failure of the shared output file, stops the run.
EXPECTED_POINT_ERRORS = (LaueError, MemoryError, OSError)


@dataclass(frozen=True)
class PointOutcome:
    """Status and reconstruction result for one point.

    Attributes
    ----------
    index : int
        Zero-based manifest index.
    point_id : str
        Stable identity of the point within the run.
    status : str
        ``"complete"``, ``"failed"``, or ``"unattempted"``.
    error : str or None
        Message of a failed point.
    seconds : float
        Wall time spent on the point, including reading and writing.
    """

    index: int
    point_id: str
    status: str
    error: str | None = None
    seconds: float = 0.0


@dataclass(frozen=True)
class ScanResult:
    """Result of :func:`reconstruct_scan`.

    Attributes
    ----------
    path : pathlib.Path
        The published file.
    cancelled : bool
        ``True`` when ``should_stop`` ended the run early.
    outcomes : tuple of PointOutcome
        One outcome for each manifest entry, in manifest order.
    """

    path: Path
    cancelled: bool
    outcomes: tuple[PointOutcome, ...]

    @property
    def complete(self) -> bool:
        """Whether every point is complete."""
        return all(outcome.status == "complete" for outcome in self.outcomes)


def reconstruct_scan(
    paths: Sequence[str | os.PathLike],
    output: str | os.PathLike,
    *,
    geometry,
    detector: int,
    point_ids: Sequence[str] | None = None,
    overwrite: bool = False,
    compression: str | None = None,
    progress: Callable[[PointOutcome], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    **reconstructor_kwargs,
) -> ScanResult:
    """Reconstruct point files into one reconstruction-scan HDF5 file.

    Points are reconstructed one after another, each with the OpenMP threads
    of one :class:`Reconstructor`. Image buffers are reused between
    points. Catalog metadata grows with the number of points.

    Parameters
    ----------
    paths
        Input 34-ID-E multi-image HDF5 point files. Their order is the
        manifest order of the file.
    output : pathlib.Path or str
        Destination path. Output is written to ``<output>.partial``, closed,
        and validated before being renamed to this path.
    geometry : Geometry, pathlib.Path, or str
        Parsed geometry or path to a geometry XML file with a wire section.
    detector : int
        Active physical detector slot in ``geometry``.
    point_ids
        Sequence of string identifiers corresponding to ``paths``, or
        ``None``. IDs must be non-empty and unique. The default is the manifest
        index in decimal.
    overwrite : bool
        Replace an existing ``output``. The default raises ``FileExistsError``
        before any point is read.
    compression : str or None
        ``"gzip"`` compresses the stored frames losslessly; the default is
        ``None``. Compression shrinks the file and lengthens the run.
    progress
        Callable or ``None``. It is called with the :class:`PointOutcome` of
        each point as soon as the point ends.
    should_stop
        Callable or ``None``, polled before each point. When it returns
        ``True`` the run starts no further point; a point in progress always
        finishes. The remaining points are recorded as unattempted and the
        file is still published.
    **reconstructor_kwargs
        Keyword arguments for :class:`Reconstructor`, including the required
        ``depth_range``.

    Returns
    -------
    ScanResult
        The published path and one outcome per point. An expected failure of
        one point's input or reconstruction is recorded for that point and
        does not stop the run. Check every outcome.

    Raises
    ------
    InputError
        If the shared configuration, ``point_ids``, or ``compression`` is
        invalid.
    FileExistsError
        If ``output`` exists and ``overwrite`` is false.
    ReconstructionError
        If the output file fails. The unfinished file keeps its ``.partial``
        name and is not a valid result.
    """
    paths = [Path(path) for path in paths]
    if point_ids is None:
        point_ids = [str(index) for index in range(len(paths))]
    point_ids = list(point_ids)
    if len(point_ids) != len(paths):
        raise InputError("point_ids must have one entry for each path")
    if any(not isinstance(value, str) or not value for value in point_ids) or (
        len(set(point_ids)) != len(point_ids)
    ):
        raise InputError("point_ids must be unique, non-empty strings")
    if progress is not None and not callable(progress):
        raise InputError("progress must be callable or None")
    if should_stop is not None and not callable(should_stop):
        raise InputError("should_stop must be callable or None")
    reconstructor = Reconstructor(geometry, detector, **reconstructor_kwargs)

    writer = ScanWriter(output, reconstructor=reconstructor, n_points=len(paths),
                        overwrite=overwrite, compression=compression)
    outcomes: list[PointOutcome | None] = [None] * len(paths)

    def report(outcome: PointOutcome) -> None:
        outcomes[outcome.index] = outcome
        if progress is not None:
            progress(outcome)

    try:
        ready = _freeze_manifest(writer, reconstructor, paths, point_ids, report)
        cancelled = False
        for index in ready:
            if should_stop is not None and should_stop():
                cancelled = True
                break
            report(_reconstruct_point(writer, reconstructor, index, point_ids[index], paths[index]))
        for index, outcome in enumerate(outcomes):
            if outcome is None:
                outcomes[index] = PointOutcome(index, point_ids[index], "unattempted")
        published = writer.close(cancelled=cancelled)
    except BaseException:
        writer.abort()
        raise
    return ScanResult(published, cancelled, tuple(outcomes))


def _freeze_manifest(writer, reconstructor, paths, point_ids, report) -> list[int]:
    """Create every catalog row and point group; return the indices to compute."""
    ready = []
    for index, (path, point_id) in enumerate(zip(paths, point_ids)):
        source = None
        try:
            if not path.is_file():
                raise InputError(f"input file does not exist: {path}")
            source = h5py.File(path, "r")
            info = read_scan_info(source, reconstructor.normalization)
            stored_dtype = PIXEL_DTYPES[reconstructor._output_type(info.dtype)]
            depth_um = reconstructor._depth_grid(info.image_geometry)
        except EXPECTED_POINT_ERRORS as error:
            if source is not None:
                source.close()
            writer.add_point(index, point_id, path, error=str(error))
            report(PointOutcome(index, point_id, "failed", str(error)))
            continue
        with source:
            writer.add_point(index, point_id, path, source=source, info=info,
                             stored_dtype=stored_dtype, depth_um=depth_um)
        ready.append(index)
    return ready


def _reconstruct_point(writer, reconstructor, index, point_id, path) -> PointOutcome:
    started = perf_counter()
    sink = None
    try:
        with h5py.File(path, "r") as source:
            info = read_scan_info(source, reconstructor.normalization)
            sink = writer.sink(index, source, reconstructor.num_threads)
            result = reconstructor._run_file(source, info, sink, False)
        error = None if result.success else result.error
    except EXPECTED_POINT_ERRORS as caught:
        error = str(caught)
    if sink is not None and sink.error is not None:
        raise ReconstructionError(
            f"writing point {point_id!r} to {writer.path} failed: {sink.error}"
        ) from sink.error
    seconds = perf_counter() - started
    if error is None:
        return PointOutcome(index, point_id, "complete", seconds=seconds)
    writer.fail_point(index, error)
    return PointOutcome(index, point_id, "failed", error, seconds)
