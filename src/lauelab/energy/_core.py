# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Single-image monochromatic reduction, derived from EnergyScansN.ipf."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from importlib.metadata import version
import json
from numbers import Integral, Real
from os import PathLike
from pathlib import Path
from typing import Mapping
import xml.etree.ElementTree as ET

import numpy as np

from lauelab._frame import read_h5_frame, read_scan_frame, roi_to_detector_pixels
from lauelab.indexing import FrameMetadata, Geometry, InputError, ScanFrame

HC_KEV_NM = 1.239841857
DEFAULT_MAX_BINS = 1_000_000
_BLOCK_PIXELS = 65_536
_FLOAT_FIELDS = ("q_nm_inv", "intensity_mean", "intensity_sum")
_COUNT_FIELDS = ("pixel_count", "nonfinite_pixel_count")


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.floating):
        converted = float(value)
        if np.isfinite(value) and (not np.isfinite(converted) or (converted == 0 and value != 0)):
            return {"value": str(value), "dtype": value.dtype.str}
        return _json_value(converted)
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise InputError("provenance keys must be strings")
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return {"nonfinite": "NaN" if np.isnan(value) else ("Infinity" if value > 0 else "-Infinity")}
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, PathLike):
        return str(value)
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise InputError(f"unsupported provenance value of type {type(value).__name__}")


def _json_text(value):
    return json.dumps(_json_value(value), sort_keys=True, allow_nan=False, ensure_ascii=False)


def _intensity_units(provenance):
    mode = provenance.get("normalization", {}).get("mode", "none")
    units = provenance.get("effective_metadata", {}).get("intensity_units")
    if mode == "i0" or units in (None, "unspecified"):
        return "unspecified"
    if not isinstance(units, str) or not units:
        raise InputError("intensity_units must be a nonempty string when supplied")
    return f"{units}/s" if mode == "exposure" else units


def _count(value, name, minimum=2):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise InputError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _number(value, name, *, positive=False):
    array = np.asarray(value)
    if array.size != 1:
        raise InputError(f"{name} must be scalar or length one; supply an explicit scalar override")
    value = array.reshape(-1)[0]
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise InputError(f"{name} must be a finite real number")
    try:
        value = float(value)
    except (OverflowError, ValueError) as error:
        raise InputError(f"{name} is not representable in float64") from error
    if not np.isfinite(value) or (positive and value <= 0):
        raise InputError(f"{name} must be finite" + (" and positive" if positive else ""))
    return value


def _mode(value):
    if value is None:
        return ""
    values = np.asarray(value)
    if values.size != 1:
        raise InputError("mono_mode must describe one image; supply an explicit mode override")
    value = values.reshape(-1)[0]
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    return value.strip().casefold() if isinstance(value, str) else ""


def _has_white_mode(metadata):
    # A declaration can resolve ambiguous mode metadata, but must not erase
    # any recorded white-beam value, including one in a copied scan vector.
    return any(_mode(value).startswith("white") for value in np.asarray(metadata.get("mono_mode")).flat)


def _round_igor(values):
    """Round half ties away from zero, including the negative lower boundary."""
    values = np.asarray(values, dtype=np.float64)
    magnitude = np.abs(values)
    whole = np.floor(magnitude)
    return np.copysign(whole + (magnitude-whole >= 0.5), values)


def _validate_values(q, means, sums, counts, bad):
    """Validate profile values (also used on bounded HDF5 slices)."""
    if not np.isfinite(q).all() or (q < 0).any() or (np.diff(q) <= 0).any():
        raise InputError("Q centers must be finite, nonnegative and strictly increasing")
    if (counts < 0).any() or (bad < 0).any() or (bad > counts).any():
        raise InputError("pixel counts must satisfy 0 <= nonfinite_pixel_count <= pixel_count")
    empty, invalid = counts == 0, bad > 0
    valid = ~(empty | invalid)
    if not np.isnan(means[empty | invalid]).all() or not np.isnan(sums[invalid]).all():
        raise InputError("empty or contaminated bins must have NaN means; contaminated sums must be NaN")
    if (sums[empty] != 0).any():
        raise InputError("empty bins must have zero intensity sums")
    if not np.isfinite(means[valid]).all() or not np.isfinite(sums[valid]).all():
        raise InputError("occupied uncontaminated bins must have finite sums and means")
    if not np.allclose(means[valid], sums[valid] / counts[valid], rtol=16*np.finfo(float).eps, atol=0):
        raise InputError("mean intensity must equal intensity_sum / pixel_count")


@dataclass(frozen=True, init=False)
class QDistribution:
    """Owned numerical profile and a snapshot of its scientific provenance.

    Parameters
    ----------
    q_nm_inv, intensity_mean, intensity_sum : numpy.ndarray
        Float64 Q centers in 1/nm, mean intensity, and corrected intensity sums,
        all shape ``(n,)`` with ``n >= 2``. Q centers are uniformly spaced.
        Empty means and contaminated means/sums are NaN; empty sums are zero.
    pixel_count, nonfinite_pixel_count : numpy.ndarray
        Nonnegative integer occupancy and nonfinite-intensity counts, shape
        ``(n,)``. Occupancy includes assigned nonfinite pixels.
    provenance : typing.Mapping
        Settings, identities and metadata. A snapshot is taken as JSON; the
        :attr:`provenance` property returns a fresh mapping on each access.

    Notes
    -----
    Arrays are copied and marked read-only. Negative intensities are valid.
    Counts are diagnostics, not statistical uncertainties. Units and correction
    mode are recorded by :func:`q_distribution` in the provenance.
    """

    q_nm_inv: np.ndarray
    intensity_mean: np.ndarray
    intensity_sum: np.ndarray
    pixel_count: np.ndarray
    nonfinite_pixel_count: np.ndarray
    _provenance_json: str = field(repr=False)

    def __init__(self, q_nm_inv, intensity_mean, intensity_sum, pixel_count,
                 nonfinite_pixel_count, provenance):
        arrays = {}
        for name, values in zip((*_FLOAT_FIELDS, *_COUNT_FIELDS),
                                (q_nm_inv, intensity_mean, intensity_sum, pixel_count, nonfinite_pixel_count)):
            source = np.asarray(values)
            if source.ndim != 1 or source.dtype.kind not in "iuf":
                raise InputError(f"{name} must be a one-dimensional real numeric array")
            if name in _COUNT_FIELDS and (
                source.dtype.kind not in "iu" or (source > np.iinfo(np.int64).max).any()
            ):
                raise InputError(f"{name} must contain int64-representable integers")
            with np.errstate(over="raise", invalid="ignore"):
                try:
                    arrays[name] = np.array(source, dtype=np.int64 if name in _COUNT_FIELDS else np.float64, copy=True)
                except FloatingPointError as error:
                    raise InputError(f"{name} overflows float64") from error
        n = len(arrays["q_nm_inv"])
        if n < 2 or any(values.shape != (n,) for values in arrays.values()):
            raise InputError("all profile arrays must have the same length >= 2")
        _validate_values(*(arrays[name] for name in (*_FLOAT_FIELDS, *_COUNT_FIELDS)))
        q = arrays["q_nm_inv"]
        expected = np.linspace(q[0], q[-1], n)
        if not np.allclose(q, expected, rtol=8*np.finfo(float).eps,
                           atol=8*np.finfo(float).eps*max(1.0, q[-1])):
            raise InputError("Q centers must be uniformly spaced")
        if not isinstance(provenance, Mapping):
            raise InputError("provenance must be a mapping")
        for name, values in arrays.items():
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        object.__setattr__(self, "_provenance_json", _json_text(provenance))

    @property
    def provenance(self) -> dict:
        """Return a fresh copy of settings, source identities and metadata."""
        return json.loads(self._provenance_json)

    @property
    def intensity_label(self) -> str:
        """Intensity-axis label identifying the scalar correction mode."""
        mode = self.provenance.get("normalization", {}).get("mode", "none")
        return {"none": "Intensity", "exposure": "Intensity / s", "i0": "Intensity / I₀"}[mode]

    @property
    def intensity_units(self) -> str:
        """Known raw/exposure units, or 'unspecified' for I0-relative intensity."""
        return _intensity_units(self.provenance)


def _metadata(value):
    try:
        return value.as_dict() if isinstance(value, FrameMetadata) else dict(value or {})
    except (TypeError, ValueError) as error:
        raise InputError("metadata must be FrameMetadata or a mapping") from error


def _source(frame):
    identity = {"kind": "array"}
    try:
        if isinstance(frame, ScanFrame):
            image, metadata, processing = read_scan_frame(frame, energy_metadata=True)
            identity = {"kind": "scan_frame", "path": str(Path(frame.path).resolve()),
                        "point_id": frame.point_id, "depth_index": frame.depth_index,
                        "stored_depth_um": processing["depth"]}
        elif isinstance(frame, (str, PathLike)):
            image, metadata, processing = read_h5_frame(frame, energy_metadata=True)
            identity = {"kind": "hdf5", "path": str(Path(frame).resolve())}
        else:
            image, metadata, processing = np.asarray(frame), {}, {}
    except (OSError, KeyError, ValueError, IndexError, TypeError) as error:
        raise InputError(f"cannot read frame {frame}: {error}") from error
    if image.ndim != 2 or image.dtype.kind not in "iuf" or not all(image.shape):
        raise InputError(f"frame must be a nonempty 2D real integer/floating image; got {image.shape}, {image.dtype}")
    return image, metadata, processing, identity


def _identity(image, identity):
    digest = hashlib.sha256()
    for block in np.nditer(image, flags=["external_loop", "buffered", "zerosize_ok"],
                           op_flags=["readonly"], order="C", buffersize=_BLOCK_PIXELS):
        digest.update(block.tobytes())
    return {**identity, "shape": list(image.shape), "dtype": image.dtype.str,
            "sha256": digest.hexdigest(), "hash_order": "C"}


def _pair(values, name, minimum):
    try:
        valid = len(values) == 2 and all(
            isinstance(v, Integral) and not isinstance(v, (bool, np.bool_)) and v >= minimum for v in values
        )
    except TypeError:
        valid = False
    if not valid:
        raise InputError(f"{name} must be two integers >= {minimum} in (x,y) order")
    return tuple(int(v) for v in values)


def _bounds(roi, shape):
    if roi is None:
        return 0, shape[0], 0, shape[1]
    try:
        if len(roi) != 4 or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) for v in roi):
            raise TypeError
        y0, y1, x0, x1 = map(int, roi)
    except (TypeError, ValueError) as error:
        raise InputError("roi must be four integers (y0,y1,x0,x1)") from error
    if not (0 <= y0 < y1 <= shape[0] and 0 <= x0 < x1 <= shape[1]):
        raise InputError(f"roi {roi} must select a nonempty region inside frame {shape}")
    return y0, y1, x0, x1


def _blocks(bounds, mask):
    y0, y1, x0, x1 = bounds
    width, size = x1-x0, (y1-y0)*(x1-x0)
    for first in range(0, size, _BLOCK_PIXELS):
        indices = np.arange(first, min(size, first+_BLOCK_PIXELS))
        ys, xs = y0+indices//width, x0+indices % width
        if mask is not None:
            kept = mask[ys, xs] == 0
            ys, xs = ys[kept], xs[kept]
        if len(ys):
            yield ys, xs


def _q(pixels, detector, depth, energy):
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            rays = detector.pixel_to_lab(pixels) - np.array([0.0, 0.0, depth])
            rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
            rays[..., 2] -= 1
            q = (2*np.pi*energy/HC_KEV_NM)*np.linalg.norm(rays, axis=-1)
    except FloatingPointError as error:
        raise InputError("geometry/energy produced an invalid or overflowing scattering vector") from error
    if not np.isfinite(q).all():
        raise InputError("geometry/energy produced nonfinite Q coordinates")
    return q


def _detector(geometry, detector_index, detector_id, metadata_id):
    if detector_index is not None:
        detector_index = _count(detector_index, "detector_index", 0)
    ids = [value for value in (detector_id, metadata_id) if value is not None]
    if any(not isinstance(value, str) or not value for value in ids):
        raise InputError("detector_id must be a nonempty string")
    if len(set(ids)) > 1:
        raise InputError("explicit detector_id conflicts with frame metadata")
    if ids:
        matched = geometry.find_detector(ids[0])
        if matched < 0:
            raise InputError(f"detector_id {ids[0]!r} is absent from the geometry")
        if detector_index is not None and matched != detector_index:
            raise InputError("detector_index conflicts with detector_id")
        detector_index = matched
    if detector_index is None:
        if geometry.detector_count != 1:
            raise InputError("multiple detectors: provide detector_index or detector_id")
        # The supported LaueGo geometry format has physical slots 0, 1, 2.
        for slot in range(3):
            try:
                geometry.detector(slot)
            except ValueError:
                continue
            detector_index = slot
            break
    try:
        return detector_index, geometry.detector(detector_index)
    except (TypeError, ValueError, OverflowError) as error:
        raise InputError(f"detector_index {detector_index!r} is not an active detector slot") from error


def _geometry(geometry):
    xml = getattr(geometry, "_source_xml", None)
    if not isinstance(geometry, Geometry):
        try:
            path = Path(geometry)
            geometry = Geometry(path)
            xml = geometry._source_xml
        except (OSError, TypeError, ValueError) as error:
            raise InputError(f"cannot read geometry: {error}") from error
    return geometry, xml


def _normalization(mode, metadata):
    mode = "none" if mode is None else mode
    if not isinstance(mode, str) or mode not in ("none", "exposure", "i0"):
        raise InputError("normalization must be 'none', 'exposure', or 'i0'")
    settings = {"mode": mode, "factor": 1.0, "i0_gain_assumed": False}
    if mode == "none":
        return settings
    exposure = _number(metadata.get("exposure_seconds"), "exposure_seconds", positive=True)
    settings["exposure_seconds"] = exposure
    factor = 1.0/exposure
    if mode == "i0":
        counts = _number(metadata.get("i0_counts"), "i0_counts", positive=True)
        duration = _number(metadata.get("scaler_count_time_seconds"), "scaler_count_time_seconds", positive=True)
        assumed = metadata.get("i0_gain") is None
        gain = 1e9 if assumed else _number(metadata["i0_gain"], "i0_gain", positive=True)
        factor = 1e5/exposure * (duration/counts) * (1e9/gain)
        settings.update(i0_counts=counts, scaler_count_time_seconds=duration, i0_gain=gain,
                        i0_gain_base=1e9, intensity_scale=1e5, i0_gain_assumed=assumed)
    if not np.isfinite(factor) or factor <= 0:
        raise InputError("normalization factor must be positive and representable in float64")
    settings["factor"] = factor
    return settings


def q_distribution(
    frame, *, geometry, energy_kev=None, metadata=None, monochromatic=None,
    monochromatic_reason=None, detector_index=None, detector_id=None,
    start=(0, 0), group=(1, 1), depth=None, roi=None, mask=None,
    dark=None, dark_metadata=None, normalization="i0",
    q_first=None, q_last=None, q_points=None, max_bins=DEFAULT_MAX_BINS,
) -> QDistribution:
    """Reduce one monochromatic image to mean intensity versus Q.

    Parameters
    ----------
    frame : numpy.ndarray or str or pathlib.Path or lauelab.indexing.ScanFrame
        One 2D real image. HDF5 uses the supported 34-ID-E image layout;
        ScanFrame selects exactly one stored reconstructed depth image.
    geometry : lauelab.indexing.Geometry or str or pathlib.Path
        Calibrated detector geometry in the 34-ID-E conventions.
    energy_kev : float or None
        Positive photon energy, overriding metadata and file energy.
    metadata, dark_metadata : lauelab.indexing.FrameMetadata or typing.Mapping or None
        Explicit metadata overlays file values. Monitor data must be scalar
        or length one; longer vectors require a scalar override.
    monochromatic : bool or None
        Explicit declaration when mode metadata is insufficient. Recorded
        white-beam metadata requires True and a nonempty monochromatic_reason.
    monochromatic_reason : str or None
        Recorded reason for overriding white-beam metadata.
    detector_index, detector_id : int or str or None
        Physical geometry slot or exact ID. Metadata ID or a sole active
        detector can select automatically; conflicting selections error.
    start, group : tuple of int
        Full-detector (x,y) origin and grouping. Stored values take precedence
        over call arguments.
    depth : float or None
        Sample depth in um; explicit values, including zero, override the file.
    roi : tuple of int or None
        Half-open stored-image (y0,y1,x0,x1); None uses the full frame.
    mask : numpy.ndarray or None
        Full-frame exclusion mask; nonzero excludes before arithmetic.
    dark : numpy.ndarray or str or pathlib.Path or lauelab.indexing.ScanFrame or None
        Direct subtraction in matching stored coordinates, units and exposure.
        No automatic scaling/resampling; primary metadata supplies normalization.
    normalization : str or None
        'i0' (default): Igor exposure/I0 correction, including 1e5 scale.
        'exposure': divide by exposure only. 'none' or None: no scalar correction.
    q_first, q_last : float or None
        First/last Q sample centers in 1/nm, not edges. Supply together with
        q_points, or omit all three for Igor-style automatic spacing.
    q_points : int or None
        Number of uniform Q centers, at least two.
    max_bins : int
        Allocation guard, default 1,000,000 centers. Raise explicitly for larger
        profiles. Neither automatic coarsening nor an image-size limit is used.

    Returns
    -------
    QDistribution
        Owned/read-only Q, corrected sums/means, occupancy/nonfinite counts and
        provenance. Empty means and contaminated means/sums are NaN.

    Raises
    ------
    InputError
        Invalid source, selection, beam qualification, correction metadata,
        geometry, grid or arithmetic overflow. No selected in-range pixels.

    Notes
    -----
    Uses uniform centers and nearest-center rounding, half ties away from zero.
    All-zero and negative intensities are valid. Unmasked NaN/Inf contaminates
    its bin only. Counts are not statistical errors. No files are written.
    Distortion/flat-field/solid-angle/polarization/Lorentz corrections, background
    estimation and peak fitting are not performed. See the energy guide for
    input conventions, corrections, precision limits and validation evidence.
    """
    limit = _count(max_bins, "max_bins")
    image, file_metadata, processing, source_identity = _source(frame)
    supplied = _metadata(metadata)
    effective = {**file_metadata, **supplied}
    units = effective.get("intensity_units")
    if units is not None and (not isinstance(units, str) or not units):
        raise InputError("intensity_units must be a nonempty string when supplied")
    energy = _number(energy_kev if energy_kev is not None else effective.get("energy_kev"), "energy_kev", positive=True)
    if monochromatic is not None and not isinstance(monochromatic, (bool, np.bool_)):
        raise InputError("monochromatic must be True, False or None")
    if monochromatic is not None:
        monochromatic = bool(monochromatic)
    recorded_white = any(_has_white_mode(values) for values in (file_metadata, supplied))
    if monochromatic is False or (monochromatic is None and _mode(effective.get("mono_mode")) != "monochromatic"):
        raise InputError("monochromatic illumination must be established by mono_mode or an explicit declaration")
    if recorded_white and not (monochromatic is True and isinstance(monochromatic_reason, str) and monochromatic_reason.strip()):
        raise InputError("white-beam metadata requires monochromatic=True and a recorded monochromatic_reason")
    effective["energy_kev"] = energy
    requested = {"start": start, "group": group, "depth": depth, "energy_kev": energy_kev}
    start = _pair(processing.get("start", start), "start", 0)
    group = _pair(processing.get("group", group), "group", 1)
    depth = _number(processing.get("depth", 0.0) if depth is None else depth, "depth")
    geometry, xml = _geometry(geometry)
    slot, detector = _detector(geometry, detector_index, detector_id, effective.get("detector_id"))
    # Python integer arithmetic avoids wrapping an invalid huge start/group
    # into an apparently legal NumPy endpoint before checking detector bounds.
    end = (start[0]+image.shape[1]*group[0]-1, start[1]+image.shape[0]*group[1]-1)
    if end[0] >= detector.nx or end[1] >= detector.ny:
        raise InputError("stored frame start/group/shape exceeds detector bounds")
    bounds = _bounds(roi, image.shape)
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != image.shape or mask.dtype.kind not in "biuf":
            raise InputError("mask must be a full-frame real numeric exclusion array; no broadcasting/resizing")
    normalization_info = _normalization(normalization, effective)
    dark_image, dark_info = None, None
    if dark is not None:
        dark_image, dark_file_meta, dark_processing, dark_identity = _source(dark)
        dark_effective = {**dark_file_meta, **_metadata(dark_metadata)}
        if dark_image.shape != image.shape:
            raise InputError("dark must match the full stored frame shape")
        for name, value in (("start", start), ("group", group)):
            if name in dark_processing and tuple(dark_processing[name]) != value:
                raise InputError(f"dark {name} does not match the primary frame")
        if dark_effective.get("detector_id") not in (None, detector.detector_id):
            raise InputError("dark detector_id does not match the primary detector")
        for name in ("exposure_seconds", "intensity_units"):
            left, right = effective.get(name), dark_effective.get(name)
            if left is not None and right is not None:
                if name == "exposure_seconds":
                    left, right = _number(left, name, positive=True), _number(right, f"dark {name}", positive=True)
                if left != right:
                    raise InputError(f"dark {name} does not match the primary frame; no automatic rescaling")
        dark_info = {**_identity(dark_image, dark_identity), "file_metadata": dark_file_meta,
                     "effective_metadata": dark_effective, "processing": dark_processing}
    elif dark_metadata is not None:
        raise InputError("dark_metadata requires a dark image")

    automatic = q_first is None and q_last is None and q_points is None
    step_est = None
    if automatic:
        low, high, included = np.inf, -np.inf, 0
        for ys, xs in _blocks(bounds, mask):
            q = _q(roi_to_detector_pixels(np.column_stack((xs, ys)), start, group), detector, depth, energy)
            low, high, included = min(low, float(q.min())), max(high, float(q.max())), included+len(q)
        if not included:
            raise InputError("ROI has no unmasked pixels")
        y0, y1, x0, x1 = bounds
        # Round the midpoint to a stored pixel before mapping its group center.
        midpoint = _round_igor([(x0+x1-1)/2, (y0+y1-1)/2])
        probes = roi_to_detector_pixels(np.array([midpoint, midpoint+1]), start, group)
        probe_q = _q(probes, detector, depth, energy)
        step_est = abs(float(probe_q[1]-probe_q[0]))
        if high <= low or not np.isfinite(step_est) or step_est <= 0:
            raise InputError("automatic Q span/spacing is degenerate; supply q_first, q_last and q_points")
        ratio = (high-low)/step_est
        if not np.isfinite(ratio) or ratio >= limit:
            raise InputError(f"automatic Q grid exceeds max_bins={limit}; supply a grid or explicitly raise max_bins")
        n = int(_round_igor(ratio)) + 1
        q_first, q_last = low, high
    else:
        if any(v is None for v in (q_first, q_last, q_points)):
            raise InputError("supply q_first, q_last and q_points together")
        q_first = _number(q_first, "q_first")
        q_last = _number(q_last, "q_last")
        n = _count(q_points, "q_points")
    if n < 2 or q_first < 0 or q_last <= q_first:
        raise InputError("Q grid needs >=2 centers and 0 <= q_first < q_last; supply an explicit grid")
    if n > limit:
        raise InputError(f"Q grid has {n} centers, exceeding max_bins={limit}; explicitly raise the limit")
    step = (q_last-q_first)/(n-1)
    if q_first+step <= q_first or q_last-step >= q_last:
        raise InputError("Q spacing is not representable in float64")
    centers = np.linspace(q_first, q_last, n)
    sums, counts, bad = np.zeros(n), np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    selected, outside = 0, 0
    try:
        with np.errstate(over="raise", invalid="ignore", divide="raise"):
            for ys, xs in _blocks(bounds, mask):
                q = _q(roi_to_detector_pixels(np.column_stack((xs, ys)), start, group), detector, depth, energy)
                # Compare in Q coordinates before division to avoid overflowing
                # a floating bin index for an explicit, very narrow distant grid.
                nearby = (q >= q_first-step) & (q <= q_last+step)
                rounded = _round_igor((q[nearby]-q_first)/step)
                kept = (rounded >= 0) & (rounded < n)
                positions = np.flatnonzero(nearby)[kept]
                selected += len(q)
                outside += len(q)-len(positions)
                indices = rounded[kept].astype(np.int64)
                values = image[ys[positions], xs[positions]].astype(np.float64)
                if dark_image is not None:
                    values -= dark_image[ys[positions], xs[positions]].astype(np.float64)
                invalid = ~np.isfinite(values)
                block_sum = np.bincount(indices, weights=np.where(invalid, 0, values), minlength=n)
                if not np.isfinite(block_sum).all():
                    raise InputError("finite intensity sum overflows float64")
                sums += block_sum
                counts += np.bincount(indices, minlength=n)
                bad += np.bincount(indices[invalid], minlength=n)
            sums *= normalization_info["factor"]
    except FloatingPointError as error:
        raise InputError("finite intensity arithmetic overflows float64") from error
    if not selected:
        raise InputError("ROI has no unmasked pixels")
    if not counts.any():
        raise InputError("no selected pixels map to the requested Q grid")
    sums[bad > 0] = np.nan
    means = np.full(n, np.nan)
    np.divide(sums, counts, out=means, where=counts > 0)
    maps = []
    if xml is not None:
        try:
            maps = [node.text for node in ET.fromstring(xml).findall(".//{*}distortionMap") if node.text]
        except ET.ParseError:
            pass  # Native Geometry also supports the tagged text format.
    calibration = {name: getattr(detector, name) for name in (
        "nx", "ny", "size_x", "size_y", "detector_id", "translation", "rotation_vector", "rotation")}
    provenance = {
        "algorithm": "EnergyScansN-single-image-v1", "lauelab_version": version("lauelab"),
        "hc_kev_nm": HC_KEV_NM, "arithmetic_dtype": "float64",
        "source": _identity(image, source_identity), "file_metadata": file_metadata,
        "supplied_metadata": supplied, "effective_metadata": effective,
        "requested": requested, "file_processing": processing,
        "beam": {"monochromatic": True, "declaration": monochromatic, "reason": monochromatic_reason},
        "geometry": {"path": str(geometry.path), "xml": xml, "detector_index": slot,
                     "calibration": calibration, "length_units": "um",
                     "unapplied_distortion_maps": maps,
                     "xml_unavailable_reason": "non-XML geometry; numerical calibration retained" if xml is None else None},
        "selection": {"start": start, "group": group, "depth_um": depth, "roi": bounds},
        "mask": None if mask is None else _identity(mask, {"kind": "exclusion_mask"}),
        "dark": dark_info, "normalization": normalization_info,
        "grid": {"automatic": automatic, "first_nm_inv": q_first, "last_nm_inv": q_last,
                 "points": n, "step_nm_inv": step, "estimated_step_nm_inv": step_est,
                 "rounding": "half-away-from-zero", "max_bins": limit},
        "diagnostics": {"roi_pixel_count": (bounds[1]-bounds[0])*(bounds[3]-bounds[2]),
                        "masked_pixel_count": (bounds[1]-bounds[0])*(bounds[3]-bounds[2])-selected,
                        "assigned_pixel_count": selected-outside, "outside_grid_pixel_count": outside,
                        "nonfinite_pixel_count": int(bad.sum())},
        "corrections": {"dark_subtraction": dark is not None, "distortion": False,
                        "flat_field": False, "solid_angle": False, "polarization": False,
                        "lorentz": False, "background_estimation": False},
        "validation_evidence": "analytic and source-derived Python tests; no Igor-runtime or measured-data validation",
    }
    return QDistribution(centers, means, sums, counts, bad, provenance)
