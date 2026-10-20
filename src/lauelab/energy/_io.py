# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Validated, explicitly published scientific profile artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile

import h5py
import numpy as np

from lauelab._hdf5 import UTF8, check_format_version, set_units, write_root_attributes
from lauelab._publish import publish_file
from lauelab.indexing.errors import InputError

from . import _layout as layout
from ._core import (
    DEFAULT_MAX_BINS, QDistribution, _bounds, _count, _intensity_units,
    _normalization, _number, _pair, _validate_values,
)


def _settings(provenance, provenance_json):
    """Materialize the layout from the immutable provenance snapshot."""
    p = provenance
    try:
        meta, selection, norm = p["effective_metadata"], p["selection"], p["normalization"]
        geometry = p["geometry"]
        cal = geometry["calibration"]
        return {
            "parameters/energy_kev": meta["energy_kev"],
            "parameters/depth_um": selection["depth_um"],
            "parameters/start": selection["start"], "parameters/group": selection["group"],
            "parameters/roi": selection["roi"], "parameters/hc_kev_nm": p["hc_kev_nm"],
            "parameters/normalization_factor": norm["factor"],
            "parameters/normalization": norm["mode"],
            **{f"parameters/{name}": norm.get(name, np.nan) for name in (
                "exposure_seconds", "i0_counts", "scaler_count_time_seconds", "i0_gain")},
            "geometry/pixels": [cal["nx"], cal["ny"]],
            "geometry/size_um": [cal["size_x"], cal["size_y"]],
            "geometry/translation_um": cal["translation"],
            "geometry/rotation_vector_rad": cal["rotation_vector"], "geometry/rotation": cal["rotation"],
            "geometry/detector_id": cal["detector_id"], "geometry/detector_index": geometry["detector_index"],
            "geometry/xml": geometry["xml"] or "", "provenance/json": provenance_json,
        }
    except (KeyError, TypeError) as error:
        raise InputError("saving requires complete provenance from q_distribution") from error


def _validate_selection_geometry(provenance):
    """Check saved scientific settings independently of duplicated datasets."""
    try:
        selection, geometry = provenance["selection"], provenance["geometry"]
        calibration = geometry["calibration"]
        shape = provenance["source"]["shape"]
        if len(shape) != 2:
            raise InputError("source shape must be (ny,nx)")
        shape = tuple(_count(value, "source shape", 1) for value in shape)
        start = _pair(selection["start"], "start", 0)
        group = _pair(selection["group"], "group", 1)
        pixels = _pair([calibration["nx"], calibration["ny"]], "detector dimensions", 1)
        if any(value > np.iinfo(np.int64).max for value in (*shape, *start, *group, *pixels)):
            raise InputError("source, selection and detector dimensions must fit int64")
        end = (start[0]+shape[1]*group[0]-1, start[1]+shape[0]*group[1]-1)
        if any(last >= size for last, size in zip(end, pixels)):
            raise InputError("stored frame start/group/shape exceeds detector bounds")
        if selection["roi"] is None:
            raise InputError("saved roi must contain four integer bounds")
        y0, y1, x0, x1 = _bounds(selection["roi"], shape)
        if provenance["diagnostics"]["roi_pixel_count"] != (y1-y0)*(x1-x0):
            raise InputError("ROI pixel count disagrees with selection")
        for name in ("size_x", "size_y"):
            _number(calibration[name], f"detector {name}", positive=True)
        for name, expected_shape in (("translation", (3,)), ("rotation_vector", (3,)), ("rotation", (3, 3))):
            values = np.asarray(calibration[name])
            if values.shape != expected_shape or values.dtype.kind not in "iuf" or not np.isfinite(values).all():
                raise InputError(f"detector {name} must be a finite real array with shape {expected_shape}")
        if _count(geometry["detector_index"], "detector_index", 0) > 2:
            raise InputError("detector_index must be a supported physical slot (0, 1 or 2)")
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise InputError(f"invalid selection/calibration provenance: {error}") from error


def _text(dataset):
    value = dataset[()]
    return value.decode("utf-8") if isinstance(value, bytes) else value


def _reject_constant(value):
    raise ValueError(f"nonstandard JSON number {value}")


def _units(units, mode, provenance=None):
    if units == "by_intensity":
        return _intensity_units(provenance or {"normalization": {"mode": mode}})
    if units != "by_normalization":
        return units
    try:
        return {"none": None, "exposure": "1/s", "i0": "unspecified"}[mode]
    except (KeyError, TypeError) as error:
        raise InputError("invalid normalization mode") from error


def _dataset(source, path, dtype, shape, units):
    if path not in source or not isinstance(source[path], h5py.Dataset):
        raise InputError(f"missing required dataset /{path}")
    data = source[path]
    correct_type = (h5py.check_string_dtype(data.dtype) is not None) if dtype == "text" else data.dtype == np.dtype(dtype)
    if data.shape != shape or not correct_type:
        raise InputError(f"/{path} must have dtype {dtype} and shape {shape}")
    actual = data.attrs.get("units")
    if isinstance(actual, bytes):
        actual = actual.decode("utf-8")
    if actual != units:
        raise InputError(f"/{path} units must be {units!r}, got {actual!r}")
    return data


def _validate(source, max_bins):
    try:
        check_format_version(source, format_name=layout.FORMAT, supported_versions=(layout.VERSION,))
    except ValueError as error:
        raise InputError(str(error)) from error
    for name in ("created", "lauelab_version"):
        if not isinstance(source.attrs.get(name), (str, bytes)) or not source.attrs[name]:
            raise InputError(f"missing or invalid root attribute {name}")
    path = "profile/q_nm_inv"
    if path not in source or not isinstance(source[path], h5py.Dataset) or source[path].ndim != 1:
        raise InputError("missing one-dimensional /profile/q_nm_inv")
    n = source[path].shape[0]
    if not 2 <= n <= max_bins:
        raise InputError(f"profile count {n} must be between 2 and max_bins={max_bins}")
    mode = _text(_dataset(source, "parameters/normalization", "text", (), None))
    for path, (dtype, shape, units) in layout.SETTINGS.items():
        _dataset(source, path, dtype, shape, _units(units, mode))
    try:
        provenance_json = _text(source["provenance/json"])
        provenance = json.loads(provenance_json, parse_constant=_reject_constant)
        if not isinstance(provenance, dict) or provenance["normalization"]["mode"] not in ("none", "exposure", "i0"):
            raise ValueError("invalid normalization mode")
        expected = _settings(provenance, provenance_json)
        _validate_selection_geometry(provenance)
        _number(provenance["effective_metadata"]["energy_kev"], "energy_kev", positive=True)
        _number(provenance["hc_kev_nm"], "hc_kev_nm", positive=True)
        _number(provenance["selection"]["depth_um"], "depth_um")
        normalized = _normalization(provenance["normalization"]["mode"], provenance["effective_metadata"])
        if normalized != provenance["normalization"]:
            raise ValueError("normalization disagrees with effective metadata")
    except (KeyError, TypeError, ValueError) as error:
        raise InputError(f"invalid scientific provenance: {error}") from error
    arrays = {name: _dataset(source, f"profile/{name}", dtype, (n,), _units(units, mode, provenance))
              for name, (dtype, units) in layout.PROFILE.items()}
    for path, value in expected.items():
        dtype = layout.SETTINGS[path][0]
        actual = _text(source[path]) if dtype == "text" else source[path][()]
        if dtype == "text":
            equal = actual == value
        else:
            equal = np.array_equal(actual, np.asarray(value), equal_nan=True)
        if not equal:
            raise InputError(f"/{path} disagrees with scientific provenance")
    first, last = float(arrays["q_nm_inv"][0]), float(arrays["q_nm_inv"][-1])
    if not np.isfinite([first, last]).all() or first < 0 or first >= last:
        raise InputError("Q endpoints must be finite and 0 <= first < last")
    step = (last-first)/(n-1)
    previous = -np.inf
    total, nonfinite_total = 0, 0
    for start in range(0, n, 65_536):
        stop = min(n, start+65_536)
        chunk = {name: dataset[start:stop] for name, dataset in arrays.items()}
        _validate_values(*(chunk[name] for name in layout.PROFILE))
        q = chunk["q_nm_inv"]
        if q[0] <= previous or not np.allclose(
            q, first+np.arange(start, stop)*step, rtol=8*np.finfo(float).eps,
            atol=8*np.finfo(float).eps*max(1.0, last),
        ):
            raise InputError("Q centers must be uniformly spaced and increasing")
        previous = q[-1]
        total += sum(map(int, chunk["pixel_count"]))
        nonfinite_total += sum(map(int, chunk["nonfinite_pixel_count"]))
    try:
        diagnostics = provenance["diagnostics"]
        grid = provenance["grid"]
        for name in ("roi_pixel_count", "masked_pixel_count", "assigned_pixel_count", "outside_grid_pixel_count", "nonfinite_pixel_count"):
            _count(diagnostics[name], name, 0)
        if (total != diagnostics["assigned_pixel_count"] or nonfinite_total != diagnostics["nonfinite_pixel_count"]
            or not total or grid["points"] != n or grid["first_nm_inv"] != first or grid["last_nm_inv"] != last
            or grid["step_nm_inv"] != step
            or diagnostics["roi_pixel_count"] != diagnostics["masked_pixel_count"]+total+diagnostics["outside_grid_pixel_count"]):
            raise ValueError("profile totals/endpoints disagree with provenance")
    except (KeyError, TypeError, ValueError) as error:
        raise InputError(f"invalid profile diagnostics: {error}") from error
    return provenance


def validate_q_distribution_file(path, *, max_bins=DEFAULT_MAX_BINS) -> None:
    """Validate format, units, settings and scientific arrays with bounded reads.

    The allocation guard is checked before reading profile arrays. Unknown
    fields are ignored. Validation failures raise InputError; opening failures
    raise OSError. The validator does not open source image files.
    """
    limit = _count(max_bins, "max_bins")
    with h5py.File(path, "r") as source:
        _validate(source, limit)


def _publish(path, overwrite, write, validate):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".partial", dir=path.parent, delete=False) as handle:
        partial = Path(handle.name)
    try:
        write(partial)
        validate(partial)
        return publish_file(partial, path, overwrite=overwrite)
    finally:
        partial.unlink(missing_ok=True)


def save_q_distribution(result: QDistribution, path, *, overwrite=False) -> Path:
    """Save a standalone versioned HDF5 profile and its complete provenance.

    Files contain profile arrays, dimensioned settings, geometry and provenance,
    not source/dark/mask pixels. A closed, validated temporary file is published
    in the destination directory. Existing files require overwrite=True.
    """
    if not isinstance(result, QDistribution):
        raise TypeError("result must be a QDistribution")
    provenance = result.provenance
    settings = _settings(provenance, result._provenance_json)
    _validate_selection_geometry(provenance)
    mode = provenance["normalization"]["mode"]

    def write(partial):
        with h5py.File(partial, "w") as target:
            write_root_attributes(target, format_name=layout.FORMAT, version=layout.VERSION)
            for name, (dtype, units) in layout.PROFILE.items():
                data = set_units(target.create_dataset(f"profile/{name}", data=getattr(result, name), dtype=dtype), _units(units, mode, provenance))
                if name.startswith("intensity_"):
                    data.attrs["normalization"] = mode
            for path, (dtype, shape, units) in layout.SETTINGS.items():
                set_units(target.create_dataset(path, data=settings[path], dtype=UTF8 if dtype == "text" else dtype), _units(units, mode))

    return _publish(path, overwrite, write,
                    lambda path: validate_q_distribution_file(path, max_bins=max(DEFAULT_MAX_BINS, len(result.q_nm_inv))))


def load_q_distribution(path, *, max_bins=DEFAULT_MAX_BINS) -> QDistribution:
    """Load a validated standalone profile, with owned/read-only arrays.

    No original frame or geometry file is opened. max_bins is checked before
    loading arrays; explicitly raise it to read larger profiles.
    """
    limit = _count(max_bins, "max_bins")
    with h5py.File(path, "r") as source:
        provenance = _validate(source, limit)
        return QDistribution(*(source[f"profile/{name}"][...] for name in layout.PROFILE), provenance)


def export_q_distribution_csv(result: QDistribution, path, *, overwrite=False) -> Path:
    """Export five numeric columns with a comment header for units/corrections.

    Skip lines starting with '#' when reading the CSV. NaN denotes empty or
    contaminated means. Use counts to distinguish these cases. Save as HDF5
    for complete provenance. An existing destination requires overwrite=True.
    """
    if not isinstance(result, QDistribution):
        raise TypeError("result must be a QDistribution")
    p = result.provenance

    def write(partial):
        with open(partial, "w", newline="", encoding="utf-8") as stream:
            stream.write(f"# {layout.FORMAT} version {layout.VERSION}\n")
            stream.write(f"# Q: 1/nm; intensity: {json.dumps(result.intensity_units)}; counts: dimensionless; NaN: unavailable\n")
            stream.write(f"# normalization: {json.dumps(p.get('normalization', {}), sort_keys=True)}\n")
            writer = csv.writer(stream)
            writer.writerow(layout.PROFILE)
            writer.writerows(zip(*(getattr(result, name) for name in layout.PROFILE)))

    def validate(partial):
        # Parse in bounded rows: verify complete output and valid numeric fields
        # before the destination becomes visible.
        with open(partial, newline="", encoding="utf-8") as stream:
            rows = csv.reader(line for line in stream if not line.startswith("#"))
            if next(rows) != list(layout.PROFILE):
                raise InputError("invalid exported CSV header")
            count = 0
            for row in rows:
                if len(row) != 5:
                    raise InputError("incomplete exported CSV row")
                for value in row[:3]:
                    float(value)
                for value in row[3:]:
                    int(value)
                count += 1
            if count != len(result.q_nm_inv):
                raise InputError("incomplete exported CSV profile")

    return _publish(path, overwrite, write, validate)
