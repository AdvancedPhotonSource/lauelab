# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Reconstruction scans: one standalone file per point plus a catalog."""

from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import io
import logging
import math
import multiprocessing
from numbers import Real
import os
from pathlib import Path
import signal
import tempfile
import threading
from time import perf_counter
from typing import Callable, Sequence

import h5py
import numpy as np

from lauelab._publish import partial_path, publish_file
from lauelab.indexing import Geometry
from lauelab.indexing.errors import InputError, InvalidScanFile, LaueError, ReconstructionError

from . import _scan_layout as layout
from ._scan_layout import PointStatus, RunStatus
from ._scan_reader import PointReader, _metadata_equal
from ._scan_writer import (COMPRESSION, PointSink, _missing, native_version, parameter_values,
                           write_catalog, write_point_metadata)
from ._writer import PIXEL_DTYPES, _copy_metadata
from .reconstructor import Reconstructor, physical_core_count

_LOG = logging.getLogger(__name__)
_SNAPSHOT_SECONDS = 5.0

# Failures of one point's input or reconstruction. Anything else, and any
# failure of an output file, stops the run.
EXPECTED_POINT_ERRORS = (LaueError, MemoryError, OSError)

# Reconstructor keyword arguments carried by a task. ``num_threads`` is not
# one of them: it belongs to the process that executes the task.
SETTING_NAMES = (
    "depth_range", "resolution", "wire_edge", "percent_brightest", "normalization",
    "norm_exponent", "norm_threshold", "cosmic_filter", "frame_skip", "wire_skip",
    "output_pixel_type", "rows_per_stripe", "memory_limit_mb",
)

# Settings that a point file contains as given. The others can be chosen
# automatically, and the point file contains the value chosen.
RECORDED_SETTINGS = (
    "depth_range", "resolution", "wire_edge", "percent_brightest", "normalization",
    "norm_exponent", "cosmic_filter", "frame_skip", "wire_skip", "memory_limit_mb",
)


@dataclass(frozen=True)
class PointTask:
    """Serializable inputs and settings for reconstructing one point.

    Tasks contain strings, numbers, and plain containers, so they can be
    pickled for a process pool or MPI. To serialize as JSON, first convert
    with :func:`dataclasses.asdict`; rebuild with ``PointTask(**values)``.
    Each worker opens its own input and creates its own native state. Open
    files, native handles, and callbacks are never included in a task.

    Attributes
    ----------
    point_id : str
        Identifier assigned to the point.
    index : int or None
        Zero-based manifest index in its scan, or ``None`` for a standalone
        point.
    source : str
        Absolute path of the input 34-ID-E multi-image HDF5 file.
    output : str
        Absolute path of the point file to publish.
    detector : int
        Physical detector slot in the geometry.
    settings : dict
        :class:`~lauelab.reconstruct.Reconstructor` keyword arguments other
        than ``num_threads``, validated and normalized. Treat as read-only.
    geometry_path : str
        Geometry file path as given, for provenance.
    geometry_xml : str
        Complete geometry XML captured during preparation and used for every
        point, even if the original geometry file changes later.
    compression : str or None
        ``None`` or ``"gzip"`` for the stored frames.
    image_shape : tuple of int
        ``(rows, columns)`` of the input frames at preparation.
    n_depths : int
        Number of reconstructed depths at preparation.
    depth_bounds_um : tuple of float
        First and last physical depth in µm at preparation.
    pixel_type : int
        Stored pixel-type code at preparation.
    raw_slices : tuple of int
        Half-open range of selected input slices at preparation.
    scan_number : int or None
        Acquisition scan number.
    sample_position : tuple of float or None
        Sample ``(x, y, z)`` in µm; missing components are ``None``.
    energy_kev : float or None
        Incident energy in keV, when available.

    Notes
    -----
    Before reconstruction, the worker checks that the output description,
    input selection, and acquisition metadata still match the prepared task.
    Raw pixel content is not verified.
    """

    point_id: str
    index: int | None
    source: str
    output: str
    detector: int
    settings: dict
    geometry_path: str
    geometry_xml: str
    compression: str | None
    image_shape: tuple[int, int]
    n_depths: int
    depth_bounds_um: tuple[float, float]
    pixel_type: int
    raw_slices: tuple[int, int]
    scan_number: int | None
    sample_position: tuple[float | None, float | None, float | None]
    energy_kev: float | None

    def __post_init__(self) -> None:
        # Rebuilding from JSON turns tuples into lists; normalize so that a
        # round trip gives an equal task.
        if set(self.settings) != set(SETTING_NAMES):
            raise ValueError(f"settings must have exactly the keys {SETTING_NAMES}")
        settings = dict(self.settings)
        settings["depth_range"] = tuple(float(value) for value in settings["depth_range"])
        object.__setattr__(self, "settings", settings)
        object.__setattr__(self, "image_shape", tuple(int(value) for value in self.image_shape))
        object.__setattr__(self, "depth_bounds_um",
                           tuple(float(value) for value in self.depth_bounds_um))
        object.__setattr__(self, "raw_slices", tuple(int(value) for value in self.raw_slices))
        object.__setattr__(self, "sample_position", tuple(self.sample_position))


@dataclass(frozen=True)
class PointOutcome:
    """Status and output of one point.

    Attributes
    ----------
    index : int or None
        Zero-based manifest index, or ``None`` for a standalone point.
    point_id : str
        Identifier assigned to the point.
    status : str
        ``"complete"`` or ``"failed"`` from :func:`reconstruct_point`. A scan
        result also reports ``"interrupted"`` and ``"unattempted"`` points;
        a running coordinator reports ``"pending"`` and ``"writing"`` too.
    error : str or None
        Error message when reconstruction failed.
    seconds : float
        Wall time spent on the point, including reading and writing.
    output : str or None
        Absolute output path when the point completed successfully.
    """

    index: int | None
    point_id: str
    status: str
    error: str | None = None
    seconds: float = 0.0
    output: str | None = None


@dataclass(frozen=True)
class ScanResult:
    """Result of a reconstruction scan.

    Attributes
    ----------
    path : pathlib.Path
        Path to the ``scan.h5`` catalog.
    cancelled : bool
        ``True`` when a stop request or an interrupt ended the run early.
    outcomes : tuple of PointOutcome
        One outcome for each manifest entry, in manifest order, including
        points whose input failed inspection.
    """

    path: Path
    cancelled: bool
    outcomes: tuple[PointOutcome, ...]

    @property
    def complete(self) -> bool:
        """Whether at least one point exists and every point is complete."""
        return bool(self.outcomes) and all(outcome.status == "complete" for outcome in self.outcomes)


def _shared_configuration(geometry, detector, reconstructor_kwargs):
    """Validate shared settings; return the reconstructor and task fields."""
    if "num_threads" in reconstructor_kwargs:
        raise InputError("num_threads is chosen when a point is executed, not when it is prepared")
    reconstructor = Reconstructor(geometry, detector, num_threads=1, **reconstructor_kwargs)
    settings = {name: getattr(reconstructor, name) for name in SETTING_NAMES}
    if settings["norm_exponent"] is not None:
        settings["norm_exponent"] = float(settings["norm_exponent"])
    if settings["norm_threshold"] is not None:
        settings["norm_threshold"] = float(settings["norm_threshold"])
    geometry_path = os.fspath(reconstructor.geometry_path)
    return reconstructor, {
        "detector": int(detector),
        "settings": settings,
        "geometry_path": geometry_path,
        "geometry_xml": Path(geometry_path).read_text(),
    }


def _check_compression(compression) -> None:
    if compression not in COMPRESSION:
        raise InputError(f"compression must be None or 'gzip'; received {compression!r}")


def _check_raw_input(file: h5py.File) -> None:
    """Reject an input that already contains reconstruction output."""
    for path in layout.RESERVED_INPUT_PATHS:
        if path in file:
            raise InputError(
                f"input contains {path}, which reconstruction writes; reconstruct the raw "
                f"scan file instead"
            )


def _inspect(source: Path, reconstructor: Reconstructor) -> dict:
    """Read the metadata of one input; raise an expected error if it is unusable."""
    if not source.is_file():
        raise InputError(f"input file does not exist: {source}")
    with h5py.File(source, "r") as file:
        _check_raw_input(file)
        info = reconstructor._scan_info(file, intensity_map=False)
    if info.scan_number is not None and not 0 <= info.scan_number <= np.iinfo(np.int32).max:
        raise InputError(f"invalid metadata in {source}: scan number {info.scan_number} "
                         f"is outside 0 to {np.iinfo(np.int32).max}")
    depth_um = reconstructor._depth_grid(info.image_geometry)
    return _input_description(info, depth_um, reconstructor)


def _input_description(info, depth_um, reconstructor) -> dict:
    """Metadata frozen in a task and checked again before reconstruction."""
    n_images, rows, columns = info.shape
    return {
        "image_shape": (rows, columns),
        "n_depths": len(depth_um),
        "depth_bounds_um": (float(depth_um[0]), float(depth_um[-1])),
        "pixel_type": reconstructor._output_type(info.dtype),
        "raw_slices": (reconstructor.frame_skip, reconstructor.frame_skip + n_images),
        "scan_number": info.scan_number,
        "sample_position": tuple(None if np.isnan(value) else value
                                 for value in info.sample_position),
        "energy_kev": info.energy_kev,
    }


def _point_task(source, output, *, geometry, detector, point_id=None, compression=None,
                **reconstructor_kwargs) -> PointTask:
    """Prepare one standalone point with the rules :func:`prepare_scan` uses."""
    _check_compression(compression)
    reconstructor, shared = _shared_configuration(geometry, detector, reconstructor_kwargs)
    source = Path(os.path.abspath(source))
    point_id = source.stem if point_id is None else point_id
    if not isinstance(point_id, str) or not point_id:
        raise InputError("point_id must be a non-empty string")
    return PointTask(
        point_id=point_id, index=None, source=os.fspath(source),
        output=os.path.abspath(output), compression=compression, **shared,
        **_inspect(source, reconstructor),
    )


def _reconstructor(task: PointTask, num_threads: int | None) -> Reconstructor:
    """Build a reconstructor from the geometry text embedded in ``task``."""
    return _configured_reconstructor({
        "geometry_path": task.geometry_path, "geometry_xml": task.geometry_xml,
        "detector": task.detector, "settings": task.settings,
    }, num_threads)


def _configured_reconstructor(configuration, num_threads=1) -> Reconstructor:
    """Inspect/execute with frozen geometry even after its source is removed."""
    suffix = Path(configuration["geometry_path"]).suffix or ".xml"
    with tempfile.TemporaryDirectory(prefix="lauelab-geometry-") as directory:
        path = Path(directory) / f"geometry{suffix}"
        path.write_text(configuration["geometry_xml"])
        geometry = Geometry(path)
    return Reconstructor(geometry, configuration["detector"], num_threads=num_threads,
                         **configuration["settings"])


def _check_unchanged(task: PointTask, info, reconstructor: Reconstructor) -> np.ndarray:
    """Return the depth grid; raise ``InputError`` if the input no longer matches ``task``."""
    depth_um = reconstructor._depth_grid(info.image_geometry)
    found = _input_description(info, depth_um, reconstructor)
    for name, value in found.items():
        expected = getattr(task, name)
        if value != expected:
            raise InputError(
                f"input {task.source} changed after the point was prepared: "
                f"{name} is {value}, expected {expected}"
            )
    return depth_um


def _check_point_file(path: Path, task: PointTask) -> None:
    """Check a closed point file against its task without reading pixels."""
    with PointReader(path) as point:
        found = (point.point_id, point.manifest_index, point.shape, point.dtype,
                 tuple(float(value) for value in point.depth_um[[0, -1]]))
        expected = (task.point_id, task.index, (task.n_depths, *task.image_shape),
                    PIXEL_DTYPES[task.pixel_type], task.depth_bounds_um)
        if found != expected:
            raise InvalidScanFile(f"{path} holds {found}, expected {expected}")
        expected_values = {
            **{name: task.settings[name] for name in RECORDED_SETTINGS},
            "normalization": task.settings["normalization"] or None,
            "detector": task.detector,
            "geometry_path": task.geometry_path,
            "geometry_xml": task.geometry_xml,
            "source_path": task.source,
            "raw_slices": task.raw_slices,
        }
        values = {**point.settings, "geometry_xml": point.geometry_xml(),
                  "source_path": point.source_path, "raw_slices": point.raw_slices}
        for name, expected_value in expected_values.items():
            if not _metadata_equal(values[name], expected_value):
                raise InvalidScanFile(f"{path}: {name} disagrees with the prepared task")
        for name in ("scan_number", "sample_position", "energy_kev"):
            expected_value = getattr(task, name)
            if name == "sample_position":
                expected_value = [np.nan if value is None else value for value in expected_value]
            if not _metadata_equal(getattr(point, name), expected_value):
                raise InvalidScanFile(f"{path}: {name} disagrees with the prepared task")


def _output_failure(task: PointTask, action: str, error: BaseException) -> ReconstructionError:
    return ReconstructionError(
        f"point {task.point_id!r}: {action} {task.output} failed: {error}"
    )


def _execute(task: PointTask, num_threads: int | None) -> PointOutcome:
    started = perf_counter()
    final = Path(task.output)
    partial = partial_path(final)

    def failed(error) -> PointOutcome:
        return PointOutcome(task.index, task.point_id, "failed", str(error),
                            perf_counter() - started)

    if final.exists():
        raise ReconstructionError(f"point {task.point_id!r}: {final} already exists")
    reconstructor = _reconstructor(task, num_threads)
    with ExitStack() as stack:
        # Input failures affect this point only.
        try:
            if not Path(task.source).is_file():
                raise InputError(f"input file does not exist: {task.source}")
            source = stack.enter_context(h5py.File(task.source, "r"))
            source_stat = os.stat(task.source)
            _check_raw_input(source)
            info = reconstructor._scan_info(source)
            depth_um = _check_unchanged(task, info, reconstructor)
            first_raw = source["entry1/data/data"][reconstructor.frame_skip]
            metadata = stack.enter_context(h5py.File(io.BytesIO(), "w"))
            _copy_metadata(source, metadata)
        except EXPECTED_POINT_ERRORS as error:
            return failed(error)

        # Output failures stop the run. Clean up only this worker's private file.
        try:
            final.parent.mkdir(parents=True, exist_ok=True)
            file = h5py.File(partial, "w-")
        except Exception as error:
            raise _output_failure(task, "creating the private file of", error) from error
        published = False
        try:
            try:
                write_point_metadata(
                    file, task, info=info, depth_um=depth_um, output_type=task.pixel_type,
                    first_raw=first_raw, source_metadata=metadata, source_stat=source_stat,
                    num_threads=reconstructor.num_threads,
                )
            except Exception as error:
                raise _output_failure(task, "writing", error) from error
            sink = PointSink(file, reconstructor.num_threads)
            try:
                result = reconstructor._run_file(source, info, sink, False)
                error = None if result.success else result.error
            except EXPECTED_POINT_ERRORS as caught:
                error = caught
            if sink.error is not None:
                raise _output_failure(task, "writing", sink.error) from sink.error
            if error is not None:
                return failed(error)
            try:
                file.close()
                _check_point_file(partial, task)
                publish_file(partial, final)
            except Exception as error:
                raise _output_failure(task, "publishing", error) from error
            published = True
        finally:
            file.close()
            if not published:
                try:
                    partial.unlink(missing_ok=True)
                except OSError:
                    pass
    return PointOutcome(task.index, task.point_id, "complete", seconds=perf_counter() - started,
                        output=os.fspath(final))


def reconstruct_point(task, output=None, *, num_threads: int | None = None,
                      **parameters) -> PointOutcome:
    """Reconstruct one point in the calling process and publish its point file.

    Call it with a :class:`PointTask` from :func:`prepare_scan`, or with an
    input file, an output path, and the parameters of a standalone point. No
    worker process is started.

    Parameters
    ----------
    task : PointTask, pathlib.Path, or str
        A prepared task, or the input 34-ID-E multi-image HDF5 file of a
        standalone point.
    output : pathlib.Path, str, or None
        Point file to write for a standalone point. It must not exist; its
        directory is created when needed. Omit it with a task, which carries
        its own output path.
    num_threads : int or None
        Positive OpenMP thread count. The default is ``None``, which estimates
        the physical cores.
    **parameters
        For a standalone point only: ``geometry``, ``detector``, optional
        ``point_id`` (default: the input stem) and ``compression``, and the
        :class:`Reconstructor` keyword arguments, including the required
        ``depth_range``.

    Returns
    -------
    PointOutcome
        ``"complete"`` with the published path, or ``"failed"`` with the error
        of an expected input or reconstruction failure. A failed point
        publishes nothing. Pixels are not returned; read them with
        :class:`PointReader`.

    Raises
    ------
    InputError
        If the arguments are inconsistent or the parameters of a standalone
        point are invalid, or its input cannot be read.
    ReconstructionError
        If the point file cannot be created, written, validated, or published,
        including when the output or another worker's private file already
        exists. The private file is removed; a published file is never
        replaced.

    Notes
    -----
    The point file is written to ``<output>.partial`` beside the final path,
    closed, validated, and published. The input is refused as a failed point
    if its output description, selected raw-slice range, or acquisition
    metadata differ from the task. Raw pixel content is not verified.
    """
    if isinstance(task, PointTask):
        if output is not None or parameters:
            raise InputError("pass a PointTask alone; its output and parameters are fixed")
    else:
        if output is None:
            raise InputError("output is required for a standalone point")
        task = _point_task(task, output, **parameters)
    return _execute(task, num_threads)


def _point_ids(point_ids, sources: list[Path]) -> list[str]:
    if point_ids is None:
        return [source.stem for source in sources]
    point_ids = list(point_ids)
    if len(point_ids) != len(sources):
        raise InputError("point_ids must have one entry for each path")
    if any(not isinstance(value, str) or not value for value in point_ids) or (
        len(set(point_ids)) != len(point_ids)
    ):
        raise InputError("point_ids must be unique, non-empty strings")
    return point_ids


def _point_paths(sources: list[Path]) -> list[str]:
    """Map each input to its point-file path; reject unsafe or colliding names."""
    paths = []
    claimed: dict[str, Path] = {}
    for source in sources:
        reason = layout.unsafe_stem(source.stem)
        if reason is not None:
            raise InputError(
                f"input {source} cannot name a point file because {reason}; "
                f"link or copy it to a file with a plain name"
            )
        path = layout.point_path(source.stem)
        earlier = claimed.setdefault(path.casefold(), source)
        if earlier is not source:
            raise InputError(
                f"inputs {earlier} and {source} would both write {path} (point files are "
                f"named after the input stem, ignoring case); reconstruct them into "
                f"separate output directories"
            )
        paths.append(path)
    return paths


def _check_destination(directory: Path) -> None:
    if not directory.exists():
        return
    if not directory.is_dir():
        raise FileExistsError(f"{directory} exists and is not a directory")
    if any(directory.iterdir()):
        raise FileExistsError(
            f"{directory} is not empty; choose a new output directory or remove the old output"
        )


def _empty_catalog(n_points: int) -> dict:
    """Return catalog columns filled with each column's fill value, or zero."""
    catalog = {}
    for name, spec in layout.POINT_COLUMNS.items():
        if h5py.check_string_dtype(spec.dtype) is not None:
            catalog[name] = [""] * n_points
        else:
            fill = 0 if spec.fill is None else spec.fill
            catalog[name] = np.full((n_points, *spec.shape[1:]), fill, dtype=spec.dtype)
    return catalog


def _set_point_row(catalog, index, source, point_id, point_path, description=None, error=""):
    """Populate one row using the same rules for batch, admission, and restore."""
    catalog["/points/point_ids"][index] = point_id
    catalog["/points/paths"][index] = point_path
    catalog["/points/source_paths"][index] = os.fspath(source)
    if description is None:
        catalog["/points/statuses"][index] = int(PointStatus.FAILED)
        catalog["/points/errors"][index] = error
        return
    catalog["/points/scan_numbers"][index] = _missing(description["scan_number"], -1)
    catalog["/points/sample_positions"][index] = [
        np.nan if value is None else value for value in description["sample_position"]
    ]
    catalog["/points/energies_kev"][index] = _missing(description["energy_kev"], np.nan)
    catalog["/points/shapes"][index] = (description["n_depths"], *description["image_shape"])
    catalog["/points/depth_bounds"][index] = description["depth_bounds_um"]
    catalog["/points/dtypes"][index] = description["pixel_type"]


def _catalog_request(configuration, version=None):
    return {
        "native_version": native_version() if version is None else version,
        "parameters": parameter_values(configuration["detector"], configuration["settings"]),
        "geometry_path": configuration["geometry_path"],
        "geometry_xml": configuration["geometry_xml"],
    }


class PreparedScan:
    """Track point outcomes and publish the scan catalog.

    Created by :func:`prepare_scan`. The coordinator is the only writer of
    ``scan.h5``: it holds the catalog in memory and batches changes into
    snapshots at most once every five seconds. Workers publish their own point files. Preparation and finalization publish immediately.
    Use it as a context manager in the process that prepared it; it cannot be
    sent to another process. Send its tasks instead.

    An external scheduler uses it in this order: :meth:`record_dispatch` when
    a task is handed to a worker, :meth:`record` with the worker's outcome,
    and :meth:`finish` after recording an outcome for every dispatched task.
    Assign each task once. If a worker is lost, record a failed outcome for
    its task; the coordinator does not retry it.

    Attributes
    ----------
    directory : pathlib.Path
        Absolute scan directory.
    path : pathlib.Path
        ``scan.h5`` in :attr:`directory`.
    tasks : tuple of PointTask
        Tasks of the points whose inputs passed inspection, in manifest order.
        Points that failed inspection are recorded in the catalog and have no
        task. Successful :meth:`add` calls append tasks; an earlier tuple remains
        an earlier snapshot. Treat tasks and their settings as read-only.
    """

    def __init__(self, directory: Path, tasks: tuple[PointTask, ...], request: dict,
                 catalog: dict, configuration: dict) -> None:
        self.directory = directory
        self.path = directory / layout.SCAN_FILENAME
        self.tasks = tasks
        self._request = request
        self._configuration = deepcopy(configuration)
        self._inspector = None
        self._catalog = catalog
        # Keep independent settings dictionaries: exported tasks can be edited
        # or rebuilt by an external scheduler, but cannot change this request.
        self._by_index = {task.index: replace(task) for task in tasks}
        self._recorded: dict[int, PointOutcome] = {}
        self._created = datetime.now(timezone.utc).isoformat()
        self._run_status = RunStatus.RUNNING
        self._published = False
        self._dirty = True
        self._last_snapshot = 0.0
        self._restoring = False

    def __reduce__(self):
        raise TypeError("a PreparedScan belongs to the process that prepared it; send its tasks")

    @property
    def outcomes(self) -> tuple[PointOutcome, ...]:
        """Current statuses in catalog order, without opening files.

        Includes pending/writing tasks and batch inspection failures. The tuple
        and its frozen outcomes are snapshots; later changes do not alter them.
        """
        return self._outcomes()

    def add(self, path, *, point_id=None) -> PointTask:
        """Inspect a ready input and append its immutable task without publishing.

        Uses the settings and geometry captured by :func:`prepare_scan`. The
        task's index is registration order, not an acquisition point number.
        A failed inspection or duplicate identity/source/output raises and
        leaves the registry unchanged. Input inspection reads metadata only;
        the caller establishes that acquisition is complete and immutable.

        Persist :meth:`export_state` before dispatch or publication. Call
        :meth:`snapshot` to publish pending changes. No work is scheduled here.

        Raises
        ------
        InputError
            The scan ended, the identity/source/output repeats or is invalid,
            or the input metadata is invalid.
        OSError
            The ready input cannot be read.
        """
        self._require_running()
        source = Path(os.path.abspath(path))
        point_id = _point_ids(None if point_id is None else [point_id], [source])[0]
        sources = [Path(value) for value in self._catalog["/points/source_paths"]]
        if os.fspath(source) in self._catalog["/points/source_paths"]:
            raise InputError(f"input {source} is already registered")
        if point_id in self._catalog["/points/point_ids"]:
            raise InputError(f"point ID {point_id!r} is already registered")
        point_path = _point_paths([*sources, source])[-1]
        output = self.directory / point_path
        if output.exists() or partial_path(output).exists():
            raise FileExistsError(f"point output or private file already exists: {output}")
        if self._inspector is None:
            self._inspector = _configured_reconstructor(self._configuration)
        description = _inspect(source, self._inspector)
        index = len(sources)
        task = PointTask(point_id=point_id, index=index, source=os.fspath(source),
                         output=os.fspath(output),
                         **self._configuration, **description)
        # Construct a candidate before committing any public task/row changes.
        catalog = _empty_catalog(index + 1)
        for name, values in self._catalog.items():
            catalog[name][:index] = values
        _set_point_row(catalog, index, source, point_id, point_path, description)
        self._catalog = catalog
        self.tasks = (*self.tasks, task)
        self._by_index[index] = replace(task)
        self._dirty = True
        return task

    def export_state(self) -> dict:
        """Return detached, versioned, strict-JSON-compatible control state.

        The caller owns atomic persistence and writer exclusion. Save newly
        registered tasks before dispatch; persist worker receipts before
        :meth:`record`. This does not write a checkpoint or publish a catalog.
        State contains frozen geometry/settings and metadata, never pixels or
        native handles. Round-trip it without editing its internal structure.
        """
        from ._scan_state import export_state

        return export_state(self)

    @classmethod
    def restore(cls, state, *, outcomes=()) -> "PreparedScan":
        """Restore an unfinished scan and replay durable receipts without writing.

        State is from :meth:`export_state`; outcomes are saved PointOutcome
        objects. Exact receipt replays are allowed, conflicting/unknown outcomes
        are rejected. Completed point files are validated without raw input
        or original geometry access. A published terminal state cannot regress.

        The directory/mount paths and package/native version tags must match.
        The caller reconciles queue/worker ownership before redispatch. Orphan
        outputs and private partial files are preserved for explicit recovery.
        This does not dispatch, clean up, publish, or provide a writer lock.

        Raises
        ------
        InputError
            State, versions, identities, receipt history, or existing catalog
            disagree, or the run has already ended.
        InvalidScanFile, OSError
            A catalog or recorded complete point is invalid or unavailable.
        """
        from ._scan_state import restore_state

        return restore_state(state, outcomes, cls)

    def __enter__(self) -> "PreparedScan":
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        if exc_type is None:
            self.close()
            return
        # The exception that ended the block is the one to report, even when
        # recording the failed run also fails.
        try:
            self.close()
        except Exception:
            _LOG.warning("could not record the failed run in %s", self.path, exc_info=True)

    def _publish(self) -> None:
        write_catalog(self.path, created=self._created, run_status=self._run_status,
                      points=self._catalog, replace=self._published, **self._request)
        self._published = True
        self._dirty = False
        self._last_snapshot = perf_counter()

    def snapshot(self, *, force: bool = False) -> bool:
        """Publish pending changes when five seconds have elapsed since publication.

        Dispatch and outcome recording call this automatically. An external
        scheduler should also call it in its polling loop while waiting for
        workers, so pending changes become visible during long-running points.
        No background thread writes the catalog. ``reconstruct_scan`` handles
        this polling itself.

        Set ``force=True`` to publish pending changes immediately. An unchanged
        catalog is never rewritten. Return whether a snapshot was published.
        Publication errors propagate; the previous snapshot remains available
        and pending changes remain in memory.
        """
        if self._restoring or not self._dirty:
            return False
        if not force and perf_counter() - self._last_snapshot < _SNAPSHOT_SECONDS:
            return False
        self._publish()
        return True

    def _status(self, index: int) -> PointStatus:
        return PointStatus(int(self._catalog["/points/statuses"][index]))

    def _set_status(self, index: int, status: PointStatus, error: str = "") -> None:
        self._catalog["/points/statuses"][index] = int(status)
        self._catalog["/points/errors"][index] = error
        self._dirty = True

    def _require_running(self) -> None:
        if self._run_status != RunStatus.RUNNING:
            raise InputError(f"the scan is {self._run_status.name.lower()}; it records no outcomes")

    def _task(self, index, point_id) -> PointTask:
        task = self._by_index.get(index)
        if task is None or task.point_id != point_id:
            raise InputError(f"point {point_id!r} at index {index} matches no task of this scan")
        return task

    def record_dispatch(self, task: PointTask) -> None:
        """Record that ``task`` was handed to a worker, publishing a snapshot if due.

        The point's status becomes ``"writing"`` until its outcome is
        recorded.

        Raises
        ------
        InputError
            If the task is not one of this scan's tasks, was already
            dispatched or recorded, or the run is no longer running.
        OSError
            If the catalog cannot be published.
        """
        if self._task(task.index, task.point_id) != task:
            raise InputError(f"task of point {task.point_id!r} differs from the prepared task")
        self._require_running()
        status = self._status(task.index)
        if status != PointStatus.PENDING:
            raise InputError(f"point {task.point_id!r} is already {status.name.lower()}")
        self._set_status(task.index, PointStatus.WRITING)
        self.snapshot()

    def _withdraw_dispatch(self, task: PointTask) -> None:
        """Return a dispatched task that never started to ``pending``."""
        self._set_status(task.index, PointStatus.PENDING)

    def record(self, outcome: PointOutcome) -> None:
        """Record the outcome of one task, publishing a snapshot if due.

        A complete outcome is accepted only when its point file is published at
        the task's output path and records the task's identity, output
        description, scientific settings, geometry, source path, input
        selection, and acquisition metadata. Pixel values are not read.
        Outcomes can be recorded in any order.

        Raises
        ------
        InputError
            If the outcome matches no task of this scan, the point was already
            recorded, the run is no longer running, or the outcome is neither
            complete nor failed with an error.
        InvalidScanFile
            If the point file of a complete outcome is invalid or disagrees
            with its task.
        OSError
            If the point file cannot be opened or the catalog cannot be
            published. The previous catalog snapshot is kept.
        """
        if not isinstance(outcome, PointOutcome):
            raise InputError("record requires a PointOutcome")
        task = self._task(outcome.index, outcome.point_id)
        self._require_running()
        status = self._status(task.index)
        if status not in (PointStatus.PENDING, PointStatus.WRITING):
            raise InputError(f"point {task.point_id!r} is already recorded as {status.name.lower()}")
        if (type(outcome.index) is not int
                or isinstance(outcome.seconds, (bool, np.bool_))
                or not isinstance(outcome.seconds, Real) or not math.isfinite(outcome.seconds)
                or outcome.seconds < 0):
            raise InputError("a point outcome requires an integer index and a finite nonnegative duration")
        if ((outcome.status == "complete" and outcome.error is not None)
                or (outcome.status == "failed" and (outcome.output is not None
                    or not isinstance(outcome.error, str)))):
            raise InputError("only a complete outcome or a failed outcome with an error and no output is recorded")
        if outcome.status == "complete":
            if outcome.output != task.output:
                raise InputError(
                    f"point {task.point_id!r} was published at {outcome.output}, "
                    f"expected {task.output}"
                )
            _check_point_file(Path(task.output), task)
            self._set_status(task.index, PointStatus.COMPLETE)
        elif outcome.status == "failed" and outcome.error:
            self._set_status(task.index, PointStatus.FAILED, outcome.error)
        else:
            raise InputError("only a complete outcome or a failed outcome with an error is recorded")
        self._recorded[task.index] = outcome
        self.snapshot()

    def _outcomes(self) -> tuple[PointOutcome, ...]:
        """Return one outcome per manifest entry, in manifest order, as recorded so far.

        A point without a recorded outcome reports its catalog status, such as
        ``"pending"`` or ``"writing"``.
        """
        statuses = self._catalog["/points/statuses"]
        ids = self._catalog["/points/point_ids"]
        return tuple(
            self._recorded.get(index) or PointOutcome(
                index, ids[index], PointStatus(int(statuses[index])).name.lower(),
                self._catalog["/points/errors"][index] or None,
            )
            for index in range(len(statuses))
        )

    def finish(self, *, cancelled: bool = False) -> "ScanResult":
        """Record the end of the run, publish the final catalog, and return the result.

        Parameters
        ----------
        cancelled : bool
            ``True`` when the run stopped before every task was dispatched.
            Points that were never dispatched become ``"unattempted"``.

        Raises
        ------
        InputError
            If a dispatched task has no recorded outcome; if points were never
            dispatched and ``cancelled`` is false; or if the run already ended.
        OSError
            If the catalog cannot be published.
        """
        self._require_running()
        statuses = self._catalog["/points/statuses"]
        if not len(statuses) and not cancelled:
            raise InputError("an empty scan cannot finish successfully; cancel it explicitly")
        unsettled = int((statuses == PointStatus.WRITING).sum())
        if unsettled:
            raise InputError(f"{unsettled} dispatched point(s) have no recorded outcome")
        pending = statuses == PointStatus.PENDING
        if pending.any() and not cancelled:
            raise InputError(
                f"{int(pending.sum())} point(s) were never dispatched; finish with "
                f"cancelled=True to record them as unattempted"
            )
        statuses[pending] = PointStatus.UNATTEMPTED
        self._run_status = RunStatus.CANCELLED if cancelled else RunStatus.FINISHED
        self._dirty = True
        self._publish()
        return ScanResult(self.path, cancelled, self._outcomes())

    def close(self) -> None:
        """Mark an unfinished run as failed and publish its final catalog.

        Dispatched points without an outcome become ``"interrupted"`` and
        points never dispatched become ``"unattempted"``. Safe to call more
        than once; it does nothing after :meth:`finish`. Leaving the context
        never reports a successful run.
        """
        if self._run_status != RunStatus.RUNNING:
            return
        statuses = self._catalog["/points/statuses"]
        statuses[statuses == PointStatus.WRITING] = PointStatus.INTERRUPTED
        statuses[statuses == PointStatus.PENDING] = PointStatus.UNATTEMPTED
        self._run_status = RunStatus.FAILED
        self._dirty = True
        self._publish()


def prepare_scan(
    paths: Sequence[str | os.PathLike],
    directory: str | os.PathLike,
    *,
    geometry,
    detector: int,
    point_ids: Sequence[str] | None = None,
    compression: str | None = None,
    **reconstructor_kwargs,
) -> PreparedScan:
    """Inspect scan inputs, publish the initial catalog, and return a coordinator.

    Parameters
    ----------
    paths
        Input 34-ID-E multi-image HDF5 point files, in manifest order. Each is
        made absolute against the current directory. An empty sequence prepares
        a running scan whose inputs can be admitted through PreparedScan.add.
    directory : pathlib.Path or str
        Scan directory. It must not exist or must be empty; it receives
        ``scan.h5`` and ``points/<input stem>.h5`` for each input.
    geometry : Geometry, pathlib.Path, or str
        Parsed geometry or path to a geometry XML file with a wire section.
        Its text is embedded in the catalog and in every task.
    detector : int
        Active physical detector slot in ``geometry``.
    point_ids
        Sequence of unique, non-empty strings corresponding to ``paths``, or
        ``None``, which uses the input stems. IDs are independent of the point
        filenames.
    compression : str or None
        ``"gzip"`` compresses the stored frames losslessly; the default is
        ``None``.
    **reconstructor_kwargs
        Keyword arguments for :class:`Reconstructor`, including the required
        ``depth_range``, except ``num_threads``.

    Returns
    -------
    PreparedScan
        The coordinator of the scan. ``scan.h5`` already lists every input;
        a point whose input cannot be read is recorded as failed and has no
        task.

    Raises
    ------
    InputError
        If the shared configuration, ``point_ids``, or
        ``compression`` is invalid; ``num_threads`` is passed; or an input
        stem is unsafe or two inputs would write the same point file.
    FileExistsError
        If ``directory`` exists and is not an empty directory.
    OSError
        If the scan directory or catalog cannot be written.

    Notes
    -----
    All shared settings and destinations are checked before writing. Input
    inspection reads metadata without loading frames.
    """
    sources = [Path(os.path.abspath(path)) for path in paths]
    point_ids = _point_ids(point_ids, sources)
    _check_compression(compression)
    reconstructor, shared = _shared_configuration(geometry, detector, reconstructor_kwargs)
    point_paths = _point_paths(sources)
    directory = Path(os.path.abspath(directory))
    _check_destination(directory)

    # Initialize columns with their fill values and pending status. Failed
    # input inspection leaves the shape at zero.
    catalog = _empty_catalog(len(sources))
    tasks = []
    for index, (source, point_id, point_path) in enumerate(zip(sources, point_ids, point_paths)):
        try:
            row = _inspect(source, reconstructor)
        except EXPECTED_POINT_ERRORS as error:
            _set_point_row(catalog, index, source, point_id, point_path, error=str(error))
            continue
        tasks.append(PointTask(
            point_id=point_id, index=index, source=os.fspath(source),
            output=os.fspath(directory / point_path), compression=compression,
            **shared, **row,
        ))
        _set_point_row(catalog, index, source, point_id, point_path, row)

    configuration = {**shared, "compression": compression}
    request = _catalog_request(configuration)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / layout.POINT_DIRECTORY).mkdir()
    scan = PreparedScan(directory, tuple(tasks), request, catalog, configuration)
    scan._inspector = reconstructor
    scan._publish()
    return scan


# How often the coordinator polls ``should_stop`` while points are running.
_POLL_SECONDS = 0.2


def _init_worker() -> None:
    # The coordinator decides what an interrupt means. A terminal Ctrl-C
    # reaches every process in the foreground group, so workers ignore it and
    # finish their point unless the coordinator terminates them.
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _run_task(task: PointTask, num_threads: int) -> PointOutcome:
    return _execute(task, num_threads)


def _terminate(pool: ProcessPoolExecutor) -> None:
    """Stop the workers of ``pool`` now, abandoning their points."""
    # ProcessPoolExecutor exposes no public way to stop running calls before
    # Python 3.14; its process table is the only handle on the workers.
    for process in list(getattr(pool, "_processes", {}).values()):
        process.terminate()
    pool.shutdown(wait=False, cancel_futures=True)


class _Interrupts:
    """Turn the first SIGINT into a stop request while a scan runs.

    The handler is installed only in the main thread and only when SIGINT has
    Python's default handler, as in a script or a running notebook cell, and
    the previous handler is restored on exit. A second SIGINT raises
    :class:`KeyboardInterrupt`. A handler installed by the caller, such as a
    portal or batch-job adapter, is left in place.
    """

    def __init__(self) -> None:
        self.requested = False
        self._previous = None

    def __enter__(self) -> "_Interrupts":
        if (threading.current_thread() is threading.main_thread()
                and signal.getsignal(signal.SIGINT) is signal.default_int_handler):
            self._previous = signal.signal(signal.SIGINT, self._handle)
        return self

    def _handle(self, signum, frame) -> None:
        if self.requested:
            raise KeyboardInterrupt
        self.requested = True

    def submit(self, pool: ProcessPoolExecutor, *args):
        """Submit to ``pool`` so that a worker it spawns ignores SIGINT from its start.

        A spawned interpreter keeps an inherited ignored SIGINT, so a Ctrl-C
        that arrives while the worker imports cannot kill it before
        ``_init_worker`` runs. A SIGINT during this call is dropped.
        """
        if self._previous is None:
            return pool.submit(*args)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            return pool.submit(*args)
        finally:
            signal.signal(signal.SIGINT, self._handle)

    def __exit__(self, *exc_info) -> None:
        if self._previous is not None:
            signal.signal(signal.SIGINT, self._previous)


def _schedule(scan: PreparedScan, workers: int, num_threads: int, progress, should_stop) -> bool:
    """Run every task of ``scan`` in worker processes; return whether the run was stopped.

    At most ``workers`` tasks are submitted at a time, and outcomes are
    recorded as they arrive. A stop request or a first interrupt stops
    admission, withdraws submitted tasks that have not started, and lets
    running points finish. An error of a point's output stops admission and
    is raised after the running points finish. A lost worker, a second
    interrupt, or any coordinator error abandons the running points.
    """
    waiting = deque(scan.tasks)
    running: dict = {}
    stopped = False
    fatal = None

    def stop() -> None:
        nonlocal stopped
        stopped = True
        for future, task in list(running.items()):
            if future.cancel():
                del running[future]
                scan._withdraw_dispatch(task)

    pool = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                               mp_context=multiprocessing.get_context("spawn"))
    try:
        with _Interrupts() as interrupts:
            while running or (waiting and not stopped and fatal is None):
                if not stopped and interrupts.requested:
                    stop()
                    _LOG.warning("Stopping the scan: %d running point(s) will finish; "
                                 "interrupt again to abandon them.", len(running))
                if not stopped and should_stop is not None and should_stop():
                    stop()
                while waiting and len(running) < workers and not stopped and fatal is None:
                    task = waiting.popleft()
                    scan.record_dispatch(task)
                    running[interrupts.submit(pool, _run_task, task, num_threads)] = task
                scan.snapshot()
                done, _ = wait(running, timeout=_POLL_SECONDS, return_when=FIRST_COMPLETED)
                for future in done:
                    task = running.pop(future)
                    try:
                        outcome = future.result()
                    except BrokenProcessPool:
                        raise
                    except Exception as error:
                        # The point's output failed. Its task stays dispatched
                        # and is recorded as interrupted when the run fails.
                        if fatal is None:
                            fatal = error
                            stop()
                        continue
                    scan.record(outcome)
                    if progress is not None:
                        progress(outcome)
    except BrokenProcessPool as error:
        _terminate(pool)
        raise ReconstructionError(
            f"a worker process ended unexpectedly; {len(running)} running point(s) "
            f"were interrupted"
        ) from error
    except BaseException:
        _terminate(pool)
        raise
    pool.shutdown(wait=True)
    if fatal is not None:
        raise fatal
    return stopped


def reconstruct_scan(
    paths: Sequence[str | os.PathLike],
    directory: str | os.PathLike,
    *,
    geometry,
    detector: int,
    point_ids: Sequence[str] | None = None,
    compression: str | None = None,
    workers: int = 1,
    threads_per_worker: int | None = None,
    progress: Callable[[PointOutcome], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    **reconstructor_kwargs,
) -> ScanResult:
    """Reconstruct wire-scan points into a scan directory with local worker processes.

    The scan is prepared with :func:`prepare_scan`, each point is reconstructed
    by :func:`reconstruct_point` in one of ``workers`` spawned processes, and
    the coordinator records outcomes in memory as they arrive. Pending catalog
    changes are published at five-second intervals and immediately at the end
    of the run. Completed point files can be read while the scan runs. Pixels
    stay in the worker that computed them.

    Parameters
    ----------
    paths, directory, geometry, detector, point_ids, compression
        As for :func:`prepare_scan`.
    workers : int
        Number of worker processes, at least 1. Each worker reconstructs one
        point at a time and holds that point's stripe buffers, so memory
        demand grows with ``workers``.
    threads_per_worker : int or None
        OpenMP threads of each worker. The default is ``None``, which divides
        the estimated physical cores among the workers. Storage conversion can
        overlap reconstruction, so ``workers * threads_per_worker`` is not a
        strict limit on runnable threads.
    progress
        Callable or ``None``. It is called in the calling process with the
        :class:`PointOutcome` of each point when it is recorded, including
        points whose input failed inspection.
    should_stop
        Callable or ``None``, polled in the calling process between outcomes.
        When it returns ``True``, no further point starts, points that have
        not started are withdrawn, and running points finish.
    **reconstructor_kwargs
        :class:`Reconstructor` keyword arguments, including the required
        ``depth_range``, except ``num_threads``.

    Returns
    -------
    ScanResult
        ``scan.h5`` and one outcome per point in manifest order. An expected
        failure of one point's input or reconstruction is recorded for that
        point and does not stop the run. A stopped run is ``cancelled`` and
        its remaining points are ``"unattempted"``. Check every outcome.

    Raises
    ------
    InputError
        If the shared configuration, ``workers``, ``threads_per_worker``, or a
        callback is invalid, or as :func:`prepare_scan` raises.
    FileExistsError
        If ``directory`` exists and is not empty.
    ReconstructionError
        If a point file cannot be written or published, or a worker process
        ends unexpectedly. No new point starts; completed point files are
        kept, and the catalog records the run as failed.
    OSError
        If the catalog cannot be published. The last valid snapshot is kept.

    Notes
    -----
    A first interrupt (Ctrl-C, or a notebook kernel interrupt) acts like
    ``should_stop``: the run reports that running points are finishing, and
    returns a cancelled result once they have. A second interrupt terminates
    the workers and raises :class:`KeyboardInterrupt`; their points are
    recorded as interrupted and their private files may remain. In the main
    thread, Python's default SIGINT handler is temporarily replaced and then
    restored. A handler installed by the caller is left in place.
    """
    for name, value in (("workers", workers), ("threads_per_worker", threads_per_worker)):
        if value is None and name == "threads_per_worker":
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise InputError(f"{name} must be a positive integer; received {value!r}")
    if progress is not None and not callable(progress):
        raise InputError("progress must be callable or None")
    if should_stop is not None and not callable(should_stop):
        raise InputError("should_stop must be callable or None")
    if not len(paths):
        raise InputError("paths must name at least one input file")
    num_threads = threads_per_worker or max(1, physical_core_count() // workers)
    with prepare_scan(paths, directory, geometry=geometry, detector=detector, point_ids=point_ids,
                      compression=compression, **reconstructor_kwargs) as scan:
        if progress is not None:
            for outcome in scan._outcomes():
                if outcome.status == "failed":
                    progress(outcome)
        cancelled = _schedule(scan, min(workers, max(1, len(scan.tasks))), num_threads,
                              progress, should_stop)
        return scan.finish(cancelled=cancelled)
