# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Strict strain preparation over existing indexing translations."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from numbers import Integral
from pathlib import Path

import numpy as np

from lauelab.indexing import (
    DetectorGeometry,
    FrameResult,
    Geometry,
    InputError,
    validate_results_file,
)
from ._tensor import _matrix, _real


def _json(value):
    def convert(v):
        if isinstance(v, dict):
            return {str(k): convert(x) for k, x in v.items()}
        if isinstance(v, (list, tuple, np.ndarray)):
            return [convert(x) for x in v]
        if isinstance(v, np.generic):
            return convert(v.item())
        if isinstance(v, float) and not np.isfinite(v):
            return None
        if isinstance(v, Path):
            return str(v)
        return v

    return json.dumps(convert(value), allow_nan=False, sort_keys=True)


def _readonly(value, dtype=np.float64):
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


def _provenance(text):
    def reject(value):
        raise InputError(f"nonfinite JSON provenance value {value}")

    try:
        value = json.loads(text, parse_constant=reject)
    except (ValueError, TypeError) as error:
        raise InputError(f"invalid JSON provenance: {error}") from error
    if not isinstance(value, dict):
        raise InputError("provenance must be a JSON object")
    return value


def _index(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < 0
    ):
        raise InputError(f"{name} must be a nonnegative integer")
    return int(value)


@dataclass(frozen=True)
class StrainReflections:
    """Owned aligned measurements prepared for one independent pattern fit.

    Parameters
    ----------
    qhat, hkl : numpy.ndarray
        Measured directions and integer Miller indices, both shape ``(n, 3)``.
        Directions are normalized. Zero HKLs and coincident harmonics error.
    initial_reciprocal : numpy.ndarray or None
        Starting reciprocal rows, shape ``(3, 3)``, in 1/nm including 2π.
        None is permitted for invalid-input records, but cannot be fitted.
    xy : numpy.ndarray or None
        Zero-based frame centroids ``(x, y)``, shape ``(n, 2)``. None gives
        all-NaN pixels for direction-only data.
    peak_indices : numpy.ndarray or None
        Unique original measured-peak IDs, shape ``(n,)``. Defaults to row
        indices; these IDs are never renumbered after preparation/subsetting.
    frame_id : str or int or None
        Source identity; integer and string identities remain distinct.
    pattern_index : int
        Nonnegative source pattern rank, default zero.
    provenance_json : str
        JSON object recording source and calibration context, default ``"{}"``.

    Notes
    -----
    Arrays are copied and marked read-only. ``provenance`` returns a copy.
    No fitted or predicted values occur here. Malformed arrays or identities
    raise InputError.
    """

    qhat: np.ndarray
    hkl: np.ndarray
    initial_reciprocal: np.ndarray | None
    xy: np.ndarray | None = None
    peak_indices: np.ndarray | None = None
    frame_id: str | int | None = None
    pattern_index: int = 0
    provenance_json: str = field(default="{}", repr=False)

    def __post_init__(self):
        q = _real(self.qhat, "qhat")
        h = _real(self.hkl, "hkl")
        if q.ndim != 2 or q.shape[1:] != (3,) or h.shape != q.shape:
            raise InputError("qhat and hkl must have matching shape (n,3)")
        if np.any(h != np.round(h)) or np.any(np.abs(h) > np.iinfo(np.int32).max):
            raise InputError("hkl must contain integer Miller indices in int32 range")
        h = h.astype(np.int64)
        if np.any(np.all(h == 0, axis=1)):
            raise InputError("hkl cannot be (0,0,0)")
        if len(h):
            primitive = h // np.gcd.reduce(np.abs(h), axis=1)[:, None]
            if len(np.unique(primitive, axis=0)) != len(h):
                raise InputError(
                    "coincident HKL harmonics cannot count as independent reflections"
                )
        scale = np.max(np.abs(q), axis=1)
        if np.any(scale <= 0):
            raise InputError("measured scattering directions must be nonzero")
        with np.errstate(over="ignore", under="ignore"):
            norm = np.linalg.norm(q, axis=1)
        if np.any(~np.isfinite(norm) | (np.abs(norm - 1) > 1e-15)):
            q = q / scale[:, None]
            q = q / np.linalg.norm(q, axis=1)[:, None]
        peaks = (
            np.arange(len(q))
            if self.peak_indices is None
            else np.asarray(self.peak_indices)
        )
        if (
            peaks.shape != (len(q),)
            or peaks.dtype.kind not in "iu"
            or np.any(peaks < 0)
            or np.any(peaks > np.iinfo(np.int64).max)
            or len(np.unique(peaks)) != len(peaks)
        ):
            raise InputError(
                "peak_indices must be unique nonnegative integers aligned with qhat"
            )
        xy = np.full((len(q), 2), np.nan) if self.xy is None else np.asarray(self.xy)
        if (
            xy.dtype.kind not in "iuf"
            or xy.shape != (len(q), 2)
            or np.isinf(xy).any()
            or np.any(np.isnan(xy).any(axis=1) != np.isnan(xy).all(axis=1))
        ):
            raise InputError(
                "xy must have shape (n,2), with NaNs only for unavailable pixels"
            )
        if self.frame_id is not None and (
            isinstance(self.frame_id, (bool, np.bool_))
            or not isinstance(self.frame_id, (str, Integral))
        ):
            raise InputError("frame_id must be a string, integer or None")
        b = self.initial_reciprocal
        if b is not None:
            b = _matrix(b, "initial_reciprocal")
            if b.shape != (3, 3):
                raise InputError("initial_reciprocal must have shape (3,3)")
            object.__setattr__(self, "initial_reciprocal", _readonly(b))
        for name, array, dtype in (
            ("qhat", q, float),
            ("hkl", h, np.int64),
            ("xy", xy, float),
            ("peak_indices", peaks, np.int64),
        ):
            object.__setattr__(self, name, _readonly(array, dtype))
        object.__setattr__(self, "pattern_index", _index(self.pattern_index, "pattern"))
        if isinstance(self.frame_id, Integral):
            object.__setattr__(self, "frame_id", int(self.frame_id))
        _provenance(self.provenance_json)

    @property
    def provenance(self):
        return json.loads(self.provenance_json)


class _Source:
    """Parse once per call/batch; keep errors before permissive XML truncation."""

    def __init__(self, source, geometry=None):
        self.errors = {}
        self.info = {"kind": "memory"}
        if isinstance(source, (str, Path)):
            from lauelab.visualization.results import load_results
            from lauelab.visualization.xml import _load_visualization_xml

            path = Path(source)
            self.info = {
                "kind": "xml" if path.suffix.lower() == ".xml" else "hdf5",
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
            }
            if self.info["kind"] == "xml":
                source = _load_visualization_xml(
                    path, geometry=geometry, pattern_errors=self.errors
                )
                self.info["centroid_storage"] = "parsed decimal float64"
            else:
                validate_results_file(path)
                source = load_results(path, geometry=geometry)
                self.info["centroid_storage"] = "float32 (indexing format v1)"
        self.source = source
        if isinstance(source, FrameResult):
            self.frame_ids = (None,)
            self.keys = [(None, i) for i in range(source.n_patterns)]
            return
        from lauelab.visualization.data import ResultSet, VisualizationDataset

        if isinstance(source, ResultSet):
            self.frame_ids = source.frame_ids
            self.frame_rows = {fid: i for i, fid in enumerate(source.frame_ids)}
            self.keys = [
                (fid, i)
                for fid, frame in zip(source.frame_ids, source.results)
                for i in range(frame.n_patterns)
            ]
        elif isinstance(source, VisualizationDataset):
            self.frame_ids = source.frame_ids
            self.keys = [
                (source.frame_ids[f], int(p))
                for f, p in zip(source.pattern_frame_indices, source.pattern_indices)
            ]
            if len(set(self.keys)) != len(self.keys):
                raise InputError("source frame/pattern identities must be unique")
            self.pattern_rows = {key: i for i, key in enumerate(self.keys)}
            from lauelab.visualization.tables import _peak_lookup

            self.peak_lookup = _peak_lookup(source)
            self.assignment_order = np.argsort(
                source.assignment_pattern_rows, kind="stable"
            )
            self.assignment_offsets = np.searchsorted(
                source.assignment_pattern_rows[self.assignment_order],
                np.arange(source.n_patterns + 1),
            )
        elif isinstance(source, StrainReflections):
            self.frame_ids = (source.frame_id,)
            self.keys = [(source.frame_id, source.pattern_index)]
        else:
            raise TypeError(
                "source must be FrameResult, ResultSet, VisualizationDataset, prepared reflections or an XML/HDF5 path"
            )

    def extract(self, frame_id, pattern):
        source = self.source
        if isinstance(source, FrameResult):
            if pattern >= source.n_patterns:
                raise InputError(f"pattern {pattern} does not exist")
            pat = source.patterns[pattern]
            metadata = source.metadata
            return (
                source.peaks,
                pat.pk_index,
                pat.hkl,
                pat.reciprocal,
                {
                    "start": source.start,
                    "group": source.group,
                    "depth": source.depth,
                    "detector_id": metadata.get("detector_id"),
                    "input_image": source.input_image,
                    "image_shape": source.image_shape,
                },
                None,
            )
        from lauelab.visualization.data import ResultSet

        if isinstance(source, ResultSet):
            if frame_id is None and len(source.results) == 1:
                frame_id = source.frame_ids[0]
            try:
                i = self.frame_rows[frame_id]
            except KeyError as error:
                raise InputError("select an existing frame_id explicitly") from error
            values = list(_Source(source.results[i]).extract(None, pattern))
            values[4]["frame_id"] = frame_id
            values[5] = source.geometry
            return tuple(values)
        ids = source.frame_ids
        if frame_id is None and len(ids) == 1:
            frame_id = ids[0]
        key = (frame_id, pattern)
        if key in self.errors:
            raise InputError(self.errors[key])
        if key not in self.pattern_rows:
            raise InputError(f"select an existing frame_id and pattern: {key!r}")
        row = self.pattern_rows[key]
        fi = int(source.pattern_frame_indices[row])
        assignments = self.assignment_order[
            self.assignment_offsets[row] : self.assignment_offsets[row + 1]
        ]
        peak_ids = source.assignment_peak_indices[assignments]
        if len(assignments) != source.pattern_n_indexed[row]:
            raise InputError(
                "declared assignment count disagrees with explicit associations"
            )
        try:
            from lauelab.visualization.tables import _peak_rows

            rows = _peak_rows(
                source, np.full(len(peak_ids), fi), peak_ids, lookup=self.peak_lookup
            )
        except KeyError as error:
            raise InputError("assignment refers to a missing measured peak") from error
        # Preserve the common translated model and its explicit association join.
        peaks = source.peaks[rows]
        return (
            peaks,
            np.arange(len(peaks)),
            source.assignment_hkl[assignments],
            source.pattern_reciprocals[row],
            {
                "start": source.starts[fi],
                "group": source.groups[fi],
                "depth": source.depths[fi],
                "detector_id": source.detector_ids[fi],
                "input_image": source.input_images[fi],
                "image_shape": source.image_shapes[fi],
                "source_peak_indices": peak_ids,
                "frame_id": frame_id,
            },
            source.geometry,
        )


def prepare_reflections(
    source,
    *,
    frame_id=None,
    pattern=0,
    geometry=None,
    detector_index=None,
    depth=None,
    peak_indices=None,
):
    """Prepare one indexed pattern using measured centroids and calibration.

    Parameters
    ----------
    source : object
        FrameResult, ResultSet, VisualizationDataset, prepared StrainReflections
        or an indexing XML/HDF5 path. Existing readers are reused directly.
    frame_id : str or int or None
        Required for a multi-frame source; a sole frame is selected by default.
    pattern : int
        Nonnegative pattern rank, default zero.
    geometry : lauelab.indexing.Geometry or str or pathlib.Path or None
        Effective calibration, overriding saved geometry when supplied.
    detector_index : int or None
        Physical geometry slot, not ordinal active-detector position. Must
        agree with recorded detector identity. Ambiguous selection errors.
    depth : float or None
        Sample depth in µm. Explicit zero also overrides stored depth.
    peak_indices : numpy.ndarray or None
        Unique original measured-peak IDs to select, retaining source
        assignment order. No automatic rejection or rematching occurs.

    Returns
    -------
    StrainReflections
        Owned float64 directions/centroids, integer HKLs and calibration
        provenance. An already prepared input is returned with its fixed
        context; conflicting selection/context overrides raise InputError.

    Notes
    -----
    Invalid selected inputs raise InputError, ValueError or TypeError. File
    I/O errors propagate. Raw directions can be passed to refine_deviatoric
    without geometry.
    """
    pattern = _index(pattern, "pattern")
    context = source if isinstance(source, _Source) else _Source(source, geometry)
    if isinstance(context.source, StrainReflections):
        data = context.source
        if pattern not in (0, data.pattern_index) or frame_id not in (
            None,
            data.frame_id,
        ):
            raise InputError(
                "prepared reflection identity conflicts with requested selection"
            )
        if any(x is not None for x in (geometry, detector_index, depth, peak_indices)):
            raise InputError(
                "prepared reflections already fix measurement context; prepare the original source to change it"
            )
        return data
    peaks, indices, hkl, initial, meta, stored_geometry = context.extract(
        frame_id, pattern
    )
    indices = np.asarray(indices)
    hkl = np.asarray(hkl)
    if (
        indices.dtype.kind not in "iu"
        or indices.shape != (len(hkl),)
        or np.any(indices < 0)
        or np.any(indices >= len(peaks))
        or len(np.unique(indices)) != len(indices)
    ):
        raise InputError("invalid or repeated measured peak associations")
    original_ids = np.asarray(meta.get("source_peak_indices", indices))
    if peak_indices is not None:
        wanted = np.asarray(peak_indices)
        if not wanted.size:
            wanted = np.empty(0, dtype=np.int64)
        if (
            wanted.ndim != 1
            or wanted.dtype.kind not in "iu"
            or len(np.unique(wanted)) != len(wanted)
        ):
            raise InputError("peak_indices must be unique source peak IDs")
        if not np.isin(wanted, original_ids).all():
            raise InputError("selection includes a peak not assigned to this pattern")
        take = np.isin(original_ids, wanted)
        indices, hkl, original_ids = indices[take], hkl[take], original_ids[take]
    xy = np.column_stack([peaks["fit_x"][indices], peaks["fit_y"][indices]])
    calibration = geometry if geometry is not None else stored_geometry
    if calibration is None:
        raise InputError(
            "centroid-based strain preparation requires calibrated geometry"
        )
    calibration = (
        calibration if isinstance(calibration, Geometry) else Geometry(calibration)
    )
    detector_id = meta.get("detector_id")
    if detector_index is None:
        if detector_id:
            detector_index = calibration.find_detector(detector_id)
        else:
            active = [i for i in range(3) if _active(calibration, i)]
            if len(active) != 1:
                raise InputError("select a physical detector_index explicitly")
            detector_index = active[0]
    detector_index = _index(detector_index, "detector_index")
    detector = calibration.detector(detector_index)
    if detector_id and detector.detector_id != detector_id:
        raise InputError("detector selection conflicts with measured frame metadata")
    effective_depth = depth if depth is not None else meta.get("depth")
    if depth is not None and (not np.isscalar(depth) or not np.isfinite(depth)):
        raise InputError("depth override must be finite")
    if effective_depth is not None and not np.isfinite(effective_depth):
        effective_depth = None
    start, group = tuple(meta["start"]), tuple(meta["group"])
    qhat = calibration.pixels_to_q(
        xy,
        detector_index=detector_index,
        start=start,
        group=group,
        depth=effective_depth,
    )
    provenance = {
        "source": context.info,
        "selection": meta,
        "directions": "measured centroids and effective geometry",
        "start": start,
        "group": group,
        "depth_um": effective_depth,
        "detector_index": detector_index,
        "detector_id": detector.detector_id,
        "detector_calibration": {
            name: getattr(detector, name)
            for name in (
                "nx",
                "ny",
                "size_x",
                "size_y",
                "detector_id",
                "translation",
                "rotation_vector",
                "rotation",
            )
        },
        "geometry_xml": calibration._source_xml,
        "geometry_path": str(calibration.path),
        "centroid_sha256": hashlib.sha256(
            np.ascontiguousarray(xy).tobytes()
        ).hexdigest(),
    }
    return StrainReflections(
        qhat,
        hkl,
        initial,
        xy,
        original_ids,
        meta.get("frame_id", frame_id),
        pattern,
        _json(provenance),
    )


def _active(geometry, index):
    try:
        geometry.detector(index)
        return True
    except ValueError:
        return False


def _pixel_predictions(data, reciprocal):
    info = data.provenance
    if "detector_calibration" not in info:
        return np.full((len(data.hkl), 2), np.nan)
    detector = DetectorGeometry(**info["detector_calibration"])
    full = detector.q_to_pixel(data.hkl @ reciprocal, depth=info["depth_um"])
    group = np.asarray(info["group"])
    return (full - np.asarray(info["start"]) - (group - 1) / 2) / group
