# Copyright 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Streaming HDF5 output for in-process indexing results."""

from __future__ import annotations

from dataclasses import fields
from numbers import Integral
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np

from lauelab._hdf5 import write_root_attributes
from lauelab._results_layout import (
    DATASETS, FORMAT, FRAME_IDS_STRING_SPEC, VERSION,
    set_attributes, write_crystal, write_dataset,
)

from .crystal import Crystal
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
    """Write indexing results incrementally to one HDF5 file."""

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
            if exc_type is None and "/frames/frame_ids" not in self._file:
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

    def _append_frame_id(self, frame_id) -> None:
        if isinstance(frame_id, (bool, np.bool_)) or not isinstance(frame_id, (str, Integral)):
            raise TypeError("frame IDs must be strings or integers")
        kind = "string" if isinstance(frame_id, str) else "integer"
        if self._frame_id_kind is not None and kind != self._frame_id_kind:
            raise TypeError("frame IDs must not mix strings and integers")
        frame_id = frame_id if kind == "string" else int(frame_id)
        if frame_id in self._frame_ids:
            raise ValueError("frame IDs must be unique")
        self._frame_ids.add(frame_id)
        if self._frame_id_kind is None:
            self._frame_id_kind = kind
            spec = FRAME_IDS_STRING_SPEC if kind == "string" else DATASETS["/frames/frame_ids"]
            self._create_resizable("/frames/frame_ids", spec)
        self._append("/frames/frame_ids", frame_id)

    def append(self, result: FrameResult, frame_id=None) -> None:
        """Append one frame result and its ragged peaks and patterns."""
        if self._file is None:
            raise RuntimeError("ResultsWriter must be used as a context manager")
        if not isinstance(result, FrameResult):
            raise TypeError("result must be a FrameResult")
        frame_id = self._count if frame_id is None else frame_id
        self._append_frame_id(frame_id)

        metadata = result.metadata
        position = metadata.get("sample_position", (np.nan, np.nan, np.nan))
        frame_values = {
            "sample_positions": position,
            "depths": np.nan if result.depth is None else result.depth,
            "scan_numbers": metadata.get("scan_number", -1),
            "energies_kev": metadata.get("energy_kev", np.nan),
            "detector_ids": metadata.get("detector_id") or "",
            "input_images": result.input_image or "",
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
        self._count += 1


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
