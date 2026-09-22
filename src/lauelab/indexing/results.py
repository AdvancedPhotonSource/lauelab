# Copyright 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Streaming HDF5 output for in-process indexing results."""

from __future__ import annotations

from dataclasses import dataclass, fields
from numbers import Integral
from pathlib import Path
from typing import Hashable, Iterable

import h5py
import numpy as np

from lauelab._hdf5 import check_format_version, write_root_attributes
from lauelab._results_layout import (
    DATASETS, FORMAT, FRAME_IDS_STRING_SPEC, SUPPORTED_VERSIONS, VERSION,
    set_attributes, write_crystal, write_dataset,
)

from .crystal import Crystal
from .errors import InvalidResultsFile
from .indexer import FrameResult, IndexParams, PeakParams

_METADATA_STRINGS = {
    "titles": "title",
    "sample_names": "sample_name",
    "user_names": "user_name",
    "beamlines": "beamline",
    "dates_exposed": "date_exposed",
    "ccd_shutters": "ccd_shutter",
    "mono_modes": "mono_mode",
}


class ResultsWriter:
    """Write indexing results incrementally to one HDF5 file.

    Notes
    -----
    Every :meth:`append` checks the result type and the frame identity
    before touching the file. If a write then raises, for any reason
    including malformed result content, the file is left with a partially
    appended frame and the writer is marked failed: ``failed`` becomes `True`, ``error``
    holds the exception, and further appends are refused. There is no
    transactional recovery; the file is not a valid results file and the run
    must be rewritten from the start. :func:`validate_results_file` detects
    such a file.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        crystal: Crystal | None,
        geometry,
        peak_params: PeakParams,
        index_params: IndexParams,
        detector_index: int,
        detector_id: str,
        cosmic_filter: bool,
        overwrite: bool = False,
        compression=None,
    ):
        self.path = Path(path)
        self.crystal = crystal
        self.geometry = geometry
        self.peak_params = peak_params
        self.index_params = index_params
        self.detector_index = detector_index
        self.detector_id = detector_id
        self.cosmic_filter = cosmic_filter
        self.compression = compression
        self._mode = "w" if overwrite else "x"
        self._file = None
        self._count = 0
        self._frame_id_kind = None
        self._frame_ids = set()
        self.error: Exception | None = None

    @property
    def failed(self) -> bool:
        """Whether a write raised, leaving the file unusable."""
        return self.error is not None

    @property
    def count(self) -> int:
        """Frames appended so far."""
        return self._count

    def __enter__(self) -> "ResultsWriter":
        self._file = h5py.File(self.path, self._mode)
        try:
            self._initialize()
        except Exception:
            self._file.close()
            self._file = None
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._file is not None:
            if exc_type is None and not self.failed and "/frames/frame_ids" not in self._file:
                self._create_resizable("/frames/frame_ids", DATASETS["/frames/frame_ids"])
            self._file.close()
            self._file = None

    def _initialize(self) -> None:
        target = self._file
        write_root_attributes(target, format_name=FORMAT, version=VERSION)
        write_crystal(target, self.crystal)
        self._write_geometry(target)
        run = target.create_group("run")
        run.attrs["program"] = "liblaue"
        run.attrs["detector_index"] = self.detector_index
        run.attrs["detector_id"] = self.detector_id
        run.attrs["cosmic_filter"] = self.cosmic_filter
        for params in (self.peak_params, self.index_params):
            for item in fields(params):
                value = getattr(params, item.name)
                run.attrs[item.name] = np.nan if value is None else value

        for group in ("frames", "peaks", "patterns", "assignments"):
            target.create_group(group)
        for path, spec in DATASETS.items():
            if spec.resizable and path != "/frames/frame_ids":
                dataset = self._create_resizable(path, spec)
                if path.endswith("_offsets"):
                    dataset.resize((1,))
                    dataset[0] = 0

    def _write_geometry(self, target) -> None:
        group = target.create_group("geometry")
        path = Path(self.geometry.path)
        group.attrs["path"] = str(path)
        try:
            text = path.read_text()
        except (OSError, UnicodeError):
            return
        write_dataset(target, "/geometry/xml", text)

    def _create_resizable(self, path, spec):
        dataset = self._file.create_dataset(
            path,
            shape=(0,) + spec.shape,
            maxshape=(None,) + spec.shape,
            chunks=(spec.chunk_rows,) + spec.shape,
            dtype=spec.dtype,
            compression=self.compression,
        )
        set_attributes(dataset, spec)
        return dataset

    def _append(self, path, values) -> None:
        dataset = self._file[path]
        values = np.asarray(values, dtype=dataset.dtype)
        count = 1 if values.ndim == len(dataset.shape) - 1 else len(values)
        start = len(dataset)
        dataset.resize((start + count,) + dataset.shape[1:])
        dataset[start:] = values

    def _check_frame_id(self, frame_id):
        if isinstance(frame_id, (bool, np.bool_)) or not isinstance(frame_id, (str, Integral)):
            raise TypeError("frame IDs must be strings or integers")
        kind = "string" if isinstance(frame_id, str) else "integer"
        if self._frame_id_kind is not None and kind != self._frame_id_kind:
            raise TypeError("frame IDs must not mix strings and integers")
        frame_id = frame_id if kind == "string" else int(frame_id)
        if frame_id in self._frame_ids:
            raise ValueError("frame IDs must be unique")
        return kind, frame_id

    def _append_frame_id(self, kind, frame_id) -> None:
        if self._frame_id_kind is None:
            self._frame_id_kind = kind
            spec = FRAME_IDS_STRING_SPEC if kind == "string" else DATASETS["/frames/frame_ids"]
            self._create_resizable("/frames/frame_ids", spec)
        self._append("/frames/frame_ids", frame_id)
        self._frame_ids.add(frame_id)

    def append(self, result: FrameResult, frame_id=None) -> None:
        """Append one frame result and its ragged peaks and patterns.

        Parameters
        ----------
        result
            The frame result to store.
        frame_id
            Unique string or integer identity; defaults to the zero-based
            append position. All identities in one file share one kind.

        Raises
        ------
        RuntimeError
            If the writer is not open or has failed earlier.
        TypeError, ValueError
            If ``result`` is not a ``FrameResult`` or ``frame_id`` is not a
            unique identity of the file's kind. These are raised before
            anything is written, so the writer remains usable.
        Exception
            Any error raised while writing, including one caused by
            malformed result content such as a sample position of the wrong
            length, marks the writer failed and propagates.
        """
        if self._file is None:
            raise RuntimeError("ResultsWriter must be used as a context manager")
        if self.failed:
            raise RuntimeError(
                f"ResultsWriter failed earlier ({self.error!r}); {self.path} is not a valid "
                "results file and must be rewritten"
            )
        if not isinstance(result, FrameResult):
            raise TypeError("result must be a FrameResult")
        kind, frame_id = self._check_frame_id(self._count if frame_id is None else frame_id)
        try:
            self._write_frame(result, kind, frame_id)
        except Exception as error:
            self.error = error
            raise
        self._count += 1

    def _write_frame(self, result: FrameResult, kind, frame_id) -> None:
        self._append_frame_id(kind, frame_id)
        metadata = result.metadata
        position = metadata.get("sample_position", (np.nan, np.nan, np.nan))
        frame_values = {
            "sample_positions": position,
            "depths": np.nan if result.depth is None else result.depth,
            "scan_numbers": metadata.get("scan_number", -1),
            "energies_kev": metadata.get("energy_kev", np.nan),
            "detector_ids": metadata.get("detector_id") or "",
            "input_images": result.input_image or "",
            "source_point_ids": "" if result.source is None else result.source.point_id,
            "source_depth_indices": -1 if result.source is None else result.source.depth_index,
            "image_shapes": result.image_shape,
            "roi_starts": result.start,
            "roi_groups": result.group,
            "n_peaks": result.n_peaks,
            "n_patterns": result.n_patterns,
            "threshold_used": result.threshold_used,
            "threshold_ratio": result.threshold_ratio,
            "total_sum": result.total_sum,
            "sum_above_threshold": result.sum_above_threshold,
            "num_above_threshold": result.num_above_threshold,
            "peak_minwidth": result.peak_minwidth,
            "peak_maxwidth": result.peak_maxwidth,
            "peak_max_cent_to_fit": result.peak_max_cent_to_fit,
            "peak_boxsize": result.peak_boxsize,
            "peaksearch_seconds": result.peaksearch_seconds,
            "indexing_seconds": result.indexing_seconds,
        }
        frame_values.update({name: metadata.get(key, "") or "" for name, key in _METADATA_STRINGS.items()})
        for name in ("exposure_seconds", "hutch_temperature", "sample_distance", "beam_bad", "light_on"):
            value = metadata.get(name)
            missing = -1 if name in ("beam_bad", "light_on") else np.nan
            frame_values[name] = missing if value is None else value
        for name, value in frame_values.items():
            self._append(f"/frames/{name}", value)

        for name in result.peaks.dtype.names or ():
            self._append(f"/peaks/{name}", result.peaks[name])
        for pattern_rank, pattern in enumerate(result.patterns):
            for name, value in (
                ("rank", pattern_rank),
                ("reciprocal", pattern.reciprocal),
                ("goodness", pattern.goodness),
                ("rms_error_deg", pattern.rms_error_deg),
                ("n_indexed", pattern.n_indexed),
            ):
                self._append(f"/patterns/{name}", value)
            for name, value in (
                ("peak_index", pattern.pk_index),
                ("hkl", pattern.hkl),
                ("error_deg", pattern.err_deg),
                ("energy_kev", pattern.energy_kev),
                ("pred_intens", pattern.pred_intens),
            ):
                self._append(f"/assignments/{name}", value)
            self._append("/patterns/assignment_offsets", len(self._file["/assignments/peak_index"]))
        self._append("/frames/peak_offsets", len(self._file["/peaks/fit_x"]))
        self._append("/frames/pattern_offsets", len(self._file["/patterns/rank"]))


def write_results(indexer, results: Iterable[FrameResult], path, *, frame_ids=None, overwrite=False) -> None:
    """Write an iterable of results using configuration from an indexer."""
    ids = iter(frame_ids) if frame_ids is not None else None
    with indexer.results_writer(path, overwrite=overwrite) as writer:
        count = 0
        for count, result in enumerate(results, start=1):
            if ids is None:
                frame_id = None
            else:
                try:
                    frame_id = next(ids)
                except StopIteration as error:
                    raise ValueError("frame_ids has fewer values than results") from error
            writer.append(result, frame_id)
        if ids is not None:
            try:
                next(ids)
            except StopIteration:
                pass
            else:
                raise ValueError(f"frame_ids must contain {count} values")


@dataclass(frozen=True)
class ResultsFileSummary:
    """Counts and provenance of a validated indexing-results file.

    Parameters
    ----------
    path
        The validated file.
    version
        Layout version.
    n_frames, n_peaks, n_patterns, n_assignments
        Row counts of the four record groups.
    frame_ids
        Frame identities in file order, as strings or integers.
    has_crystal
        Whether the file carries a crystal group.
    has_geometry_text
        Whether the geometry XML text is embedded.
    created, lauelab_version
        Root attributes written by the producer.
    source
        The XML document a converted file came from, or `None`.
    """

    path: Path
    version: int
    n_frames: int
    n_peaks: int
    n_patterns: int
    n_assignments: int
    frame_ids: tuple
    has_crystal: bool
    has_geometry_text: bool
    created: str
    lauelab_version: str
    source: str | None = None


def _attr_text(attrs, name, default=""):
    value = attrs.get(name, default)
    return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)


def validate_results_file(
    path, *, frame_ids: Iterable[Hashable] | None = None, n_frames: int | None = None
) -> ResultsFileSummary:
    """Check the structure of a closed indexing-results file with bounded reads.

    Parameters
    ----------
    path
        Results file to check. It must be closed by its writer.
    frame_ids
        Optional expected identities in order, such as a run manifest. The
        file's ``frames/frame_ids`` must equal this sequence exactly.
    n_frames
        Optional expected frame count.

    Returns
    -------
    ResultsFileSummary
        Counts and provenance read during the check.

    Raises
    ------
    OSError
        If the file cannot be opened as HDF5.
    InvalidResultsFile
        If the format marker or version is wrong; a group or dataset is
        missing or has the wrong dtype or trailing shape; per-frame,
        per-peak, per-pattern, or per-assignment datasets disagree in length;
        an offsets dataset does not start at zero, increase monotonically, and
        end at its group's row count; ``n_peaks``, ``n_patterns``, or
        ``n_indexed`` disagree with the offsets; pattern ranks do not restart
        at zero within each frame; an assignment refers outside its owning frame;
        frame identities repeat; or the file does
        not match ``frame_ids`` or ``n_frames``.

    Notes
    -----
    The check reads dataset shapes and dtypes, the per-frame and per-pattern
    bookkeeping arrays (identities, counts, ranks, offsets), and root
    attributes. Assignment peak indices are checked in bounded chunks. Peak
    values, reciprocal lattices, and other assignment arrays are not read. It
    establishes that the file is complete and self-consistent, not that the
    science in it is right; a valid file can describe an incomplete run.
    """
    path = Path(path)

    def fail(message):
        raise InvalidResultsFile(f"{path}: {message}")

    with h5py.File(path, "r") as source:
        try:
            version = check_format_version(
                source, format_name=FORMAT, supported_versions=SUPPORTED_VERSIONS
            )
        except ValueError as error:
            fail(str(error))
        for group in ("run", "geometry", "frames", "peaks", "patterns", "assignments"):
            if group not in source or not isinstance(source[group], h5py.Group):
                fail(f"missing group {group!r}")

        def dataset(name, spec=None):
            if name not in source or not isinstance(source[name], h5py.Dataset):
                fail(f"missing dataset {name!r}")
            data = source[name]
            if spec is not None:
                if spec.dtype.kind == "O":
                    if h5py.check_string_dtype(data.dtype) is None:
                        fail(f"dataset {name!r} must hold strings, not {data.dtype}")
                elif data.dtype != spec.dtype:
                    fail(f"dataset {name!r} has dtype {data.dtype}, expected {spec.dtype}")
                if data.shape[1:] != spec.shape:
                    fail(f"dataset {name!r} has shape {data.shape}, expected trailing {spec.shape}")
            return data

        ids_data = dataset("/frames/frame_ids")
        if ids_data.ndim != 1:
            fail("dataset '/frames/frame_ids' must be one-dimensional")
        if h5py.check_string_dtype(ids_data.dtype) is not None:
            ids = tuple(ids_data.asstr()[...])
        elif ids_data.dtype == DATASETS["/frames/frame_ids"].dtype:
            ids = tuple(int(value) for value in ids_data[...])
        else:
            fail(f"dataset '/frames/frame_ids' has dtype {ids_data.dtype}, expected int32 or string")
        count = len(ids)
        if len(set(ids)) != count:
            fail("frame identities repeat")

        source_fields = ("/frames/source_point_ids", "/frames/source_depth_indices")
        if (source_fields[0] in source) != (source_fields[1] in source):
            fail("source_point_ids and source_depth_indices must both be present or both absent")

        lengths = {}
        for name, spec in DATASETS.items():
            if not spec.resizable or name == "/frames/frame_ids":
                continue
            if spec.optional and name not in source:
                continue
            data = dataset(name, spec)
            lengths[name] = len(data)
        n_peaks = lengths["/peaks/fit_x"]
        n_patterns = lengths["/patterns/rank"]
        n_assignments = lengths["/assignments/peak_index"]
        for name, length in lengths.items():
            group = name.split("/")[1]
            if name.endswith("_offsets"):
                owners = count if group == "frames" else n_patterns
                expected = owners + 1
            else:
                expected = {"frames": count, "peaks": n_peaks, "patterns": n_patterns,
                            "assignments": n_assignments}[group]
            if length != expected:
                fail(f"dataset {name!r} has {length} rows, expected {expected}")

        if source_fields[0] in source:
            for first in range(0, count, 4096):
                selection = slice(first, first + 4096)
                point_ids = source[source_fields[0]].asstr()[selection]
                depths = source[source_fields[1]][selection]
                paths = source["/frames/input_images"].asstr()[selection]
                for point_id, depth, path in zip(point_ids, depths, paths):
                    if (point_id and (depth < 0 or not path)) or (not point_id and depth != -1):
                        fail("scan source needs a non-empty path and point ID with a nonnegative "
                             "depth index; an absent source uses an empty point ID and index -1")

        def offsets(name, total):
            values = source[name][...]
            if len(values) == 0 or values[0] != 0 or values[-1] != total or np.any(np.diff(values) < 0):
                fail(f"{name!r} does not partition {total} rows")
            return values

        peak_offsets = offsets("/frames/peak_offsets", n_peaks)
        pattern_offsets = offsets("/frames/pattern_offsets", n_patterns)
        assignment_offsets = offsets("/patterns/assignment_offsets", n_assignments)
        if not np.array_equal(source["/frames/n_peaks"][...], np.diff(peak_offsets)):
            fail("'/frames/n_peaks' disagrees with '/frames/peak_offsets'")
        if not np.array_equal(source["/frames/n_patterns"][...], np.diff(pattern_offsets)):
            fail("'/frames/n_patterns' disagrees with '/frames/pattern_offsets'")
        if not np.array_equal(source["/patterns/n_indexed"][...], np.diff(assignment_offsets)):
            fail("'/patterns/n_indexed' disagrees with '/patterns/assignment_offsets'")
        expected_rank = np.arange(n_patterns) - np.repeat(pattern_offsets[:-1], np.diff(pattern_offsets))
        if not np.array_equal(source["/patterns/rank"][...], expected_rank):
            fail("'/patterns/rank' does not restart at zero within each frame")

        # Assignment indices are frame-local, not global peak row numbers.
        # Bound temporary ownership/index arrays independently of file size.
        assignment_indices = source["/assignments/peak_index"]
        for start in range(0, n_assignments, 65536):
            stop = min(start + 65536, n_assignments)
            indices = assignment_indices[start:stop]
            pattern_rows = np.searchsorted(assignment_offsets, np.arange(start, stop), side="right") - 1
            frame_rows = np.searchsorted(pattern_offsets, pattern_rows, side="right") - 1
            limits = peak_offsets[frame_rows + 1] - peak_offsets[frame_rows]
            invalid = (indices < 0) | (indices >= limits)
            if np.any(invalid):
                local = int(np.flatnonzero(invalid)[0])
                fail(
                    f"assignment {start + local} has peak index {indices[local]} outside "
                    f"frame {ids[frame_rows[local]]!r}'s {limits[local]} peaks"
                )

        has_crystal = "crystal" in source
        if has_crystal:
            lattice = dataset("/crystal/lattice_parameters")
            if lattice.shape != (6,):
                fail(f"dataset '/crystal/lattice_parameters' has shape {lattice.shape}, expected (6,)")
            atom_lengths = set()
            for name, spec in DATASETS.items():
                if name.startswith("/crystal/atom_"):
                    atom_lengths.add(len(dataset(name, spec)))
            if len(atom_lengths) != 1:
                fail("crystal atom datasets disagree in length")
        has_geometry_text = "/geometry/xml" in source

        if n_frames is not None and count != n_frames:
            fail(f"file has {count} frames, expected {n_frames}")
        if frame_ids is not None:
            expected_ids = tuple(frame_ids)
            if len(expected_ids) != count:
                fail(f"file has {count} frames, manifest has {len(expected_ids)}")
            for position, (actual, expected) in enumerate(zip(ids, expected_ids)):
                if actual != expected:
                    fail(f"frame {position} has identity {actual!r}, manifest has {expected!r}")

        attrs = source.attrs
        return ResultsFileSummary(
            path=path,
            version=version,
            n_frames=count,
            n_peaks=n_peaks,
            n_patterns=n_patterns,
            n_assignments=n_assignments,
            frame_ids=ids,
            has_crystal=has_crystal,
            has_geometry_text=has_geometry_text,
            created=_attr_text(attrs, "created"),
            lauelab_version=_attr_text(attrs, "lauelab_version"),
            source=_attr_text(attrs, "source") if "source" in attrs else None,
        )
