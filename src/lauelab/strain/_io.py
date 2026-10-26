# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Explicit validated HDF5 persistence and two CSV exchange tables."""

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
from ._inputs import StrainReflections
from ._models import StrainResult


def _results(results):
    values = (results,) if isinstance(results, StrainResult) else tuple(results)
    if any(not isinstance(value, StrainResult) for value in values):
        raise TypeError("results must contain StrainResult objects")
    return values


def _destination(path, values, overwrite):
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(path)
    for value in values:
        source = value.measurements.provenance.get("source", {}).get("path")
        if source and Path(source).resolve() == path.resolve():
            raise InputError(
                "write strain results to a separate destination from indexing input"
            )
    return path


def _temporary(path):
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.name}.", suffix=".partial", dir=path.parent, delete=False
    ) as handle:
        return Path(handle.name)


def _write_array(group, name, data, units, dtype="<f8"):
    dataset = group.create_dataset(name, data=data, dtype=dtype)
    if isinstance(units, tuple):
        dataset.attrs["component_units"] = np.array(units, dtype=UTF8)
        return dataset
    return set_units(dataset, units)


def save_strain_results(results, path, *, overwrite=False):
    """Save one result or an ordered collection, including failed fits.

    Parameters
    ----------
    results : StrainResult or list of StrainResult
        A result or an ordered iterable of results, including failed fits.
    path : str or pathlib.Path
        Separate HDF5 destination. Its parent directory must exist.
    overwrite : bool
        Replace an existing destination only when True, default False.

    Returns
    -------
    pathlib.Path
        Published destination.

    Notes
    -----
    Scientific arrays use float64; identities/HKL use integer or JSON string
    encodings without loss. Inputs, calibration snapshots, settings, status,
    frames and conventions are retained. The temporary file is validated before
    atomic publication. Existing destinations raise FileExistsError by default.
    Source indexing files cannot be overwritten by this writer.
    """
    values = _results(results)
    path = _destination(path, values, overwrite)
    temporary = _temporary(path)
    try:
        with h5py.File(temporary, "w") as target:
            write_root_attributes(
                target, format_name=layout.FORMAT, version=layout.VERSION
            )
            target.attrs["strain_measure"] = "deviatoric Biot stretch"
            target.attrs["scale_convention"] = "reference volume"
            target.attrs["shear_convention"] = "tensor"
            target.attrs["count"] = len(values)
            collection = target.create_group("results")
            for i, value in enumerate(values):
                group = collection.create_group(str(i))
                group.create_dataset(
                    "status",
                    data=layout.STATUS_CODES[value.status],
                    dtype=layout.STATUS_TYPE,
                )
                for name in layout.TEXT:
                    group.create_dataset(name, data=getattr(value, name), dtype=UTF8)
                for name in layout.DIAGNOSTICS:
                    dtype = "<f8" if name == "condition_number" else "<i8"
                    group.create_dataset(name, data=getattr(value, name), dtype=dtype)
                for name, (_, units) in layout.ARRAYS.items():
                    _write_array(group, name, getattr(value, name), units)
                data = value.measurements
                measured = group.create_group("measurements")
                for name, units, dtype in (
                    ("qhat", None, "<f8"),
                    ("hkl", None, "<i8"),
                    ("xy", "pixel", "<f8"),
                    ("peak_indices", None, "<i8"),
                ):
                    _write_array(measured, name, getattr(data, name), units, dtype)
                measured.create_dataset(
                    "frame_id_json", data=json.dumps(data.frame_id), dtype=UTF8
                )
                measured.create_dataset(
                    "provenance_json", data=data.provenance_json, dtype=UTF8
                )
                measured.create_dataset(
                    "pattern_index", data=data.pattern_index, dtype="<i8"
                )
                measured.attrs["has_initial"] = data.initial_reciprocal is not None
                _write_array(
                    measured,
                    "initial_reciprocal",
                    data.initial_reciprocal
                    if data.initial_reciprocal is not None
                    else np.full((3, 3), np.nan),
                    "1/nm",
                )
        load_strain_results(temporary)
        publish_file(temporary, path, overwrite=overwrite)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _text(group, name):
    data = group[name]
    if data.shape != () or h5py.check_string_dtype(data.dtype) is None:
        raise InputError(f"{data.name} must be a scalar UTF-8 string")
    return data.asstr()[()]


def _array(group, name, shape, units, dtype="<f8"):
    data = group[name]
    if isinstance(units, tuple):
        valid_units = (
            data.attrs.get("units") is None
            and tuple(np.asarray(data.attrs.get("component_units", ())).astype(str))
            == units
        )
    else:
        valid_units = (
            data.attrs.get("units") == units and "component_units" not in data.attrs
        )
    if data.dtype != np.dtype(dtype) or data.shape != shape or not valid_units:
        raise InputError(f"{data.name} has invalid dtype, shape or units")
    return data[...]


def _scalar(group, name, dtype):
    data = group[name]
    if data.shape != () or data.dtype != np.dtype(dtype):
        raise InputError(f"{data.name} has invalid scalar dtype/shape")
    return data[()].item()


def _status(group):
    data = group["status"]
    if (
        data.shape != ()
        or data.dtype != layout.STATUS_TYPE
        or h5py.check_enum_dtype(data.dtype) != layout.STATUS_CODES
    ):
        raise InputError(f"{data.name} must use the strain-status enum")
    value = int(data[()])
    if value not in layout.STATUS_CODES.values():
        raise InputError(f"{data.name} has an unknown status code")
    return layout.STATUSES[value]


def load_strain_results(path):
    """Load and validate an ordered list of self-contained strain results.

    Parameters
    ----------
    path : str or pathlib.Path
        Standalone strain-results HDF5 file.

    Returns
    -------
    list of StrainResult
        Results in saved order, including failed fits. Arrays are owned and
        read-only exactly as with freshly computed results.

    Notes
    -----
    No experimental inputs or original geometry paths are reopened. Reject
    unsupported versions, inconsistent shapes/units/types and result tensors
    inconsistent with their saved lattice/reference/frame with InputError.
    File I/O errors propagate.
    """
    results = []
    try:
        with h5py.File(path, "r") as source:
            check_format_version(
                source, format_name=layout.FORMAT, supported_versions=(layout.VERSION,)
            )
            if (
                source.attrs.get("strain_measure") != "deviatoric Biot stretch"
                or source.attrs.get("scale_convention") != "reference volume"
                or source.attrs.get("shear_convention") != "tensor"
            ):
                raise InputError("unsupported strain convention metadata")
            count = source.attrs["count"]
            if (
                isinstance(count, (bool, np.bool_))
                or not isinstance(count, (int, np.integer))
                or count < 0
            ):
                raise InputError("result count must be a nonnegative integer")
            if set(source["results"]) != {str(i) for i in range(count)}:
                raise InputError("result count disagrees with stored groups")
            for i in range(count):
                group = source[f"results/{i}"]
                measured = group["measurements"]
                n = len(measured["hkl"])
                initial = _array(measured, "initial_reciprocal", (3, 3), "1/nm")
                has_initial = measured.attrs["has_initial"]
                if not isinstance(has_initial, (bool, np.bool_)):
                    raise InputError("has_initial must be boolean")
                if not has_initial:
                    if not np.isnan(initial).all():
                        raise InputError(
                            "unavailable initial lattice must contain NaNs"
                        )
                    initial = None
                data = StrainReflections(
                    _array(measured, "qhat", (n, 3), None),
                    _array(measured, "hkl", (n, 3), None, "<i8"),
                    initial,
                    _array(measured, "xy", (n, 2), "pixel"),
                    _array(measured, "peak_indices", (n,), None, "<i8"),
                    json.loads(_text(measured, "frame_id_json")),
                    _scalar(measured, "pattern_index", "<i8"),
                    _text(measured, "provenance_json"),
                )
                values = {name: _text(group, name) for name in layout.TEXT}
                values["status"] = _status(group)
                values.update(
                    {
                        name: _scalar(
                            group, name, "<f8" if name == "condition_number" else "<i8"
                        )
                        for name in layout.DIAGNOSTICS
                    }
                )
                for name, (shape, units) in layout.ARRAYS.items():
                    shape = tuple(n if s is None else s for s in shape)
                    values[name] = _array(group, name, shape, units)
                results.append(StrainResult(measurements=data, **values))
    except (KeyError, TypeError, ValueError) as error:
        raise InputError(f"invalid strain results file {path}: {error}") from error
    return results


def export_strain_csv(results, catalogue_path, reflections_path, *, overwrite=False):
    """Export separate fit-summary and reflection CSV tables.

    Parameters
    ----------
    results : StrainResult or list of StrainResult
        A result or ordered iterable, including failed fits.
    catalogue_path, reflections_path : str or pathlib.Path
        Distinct destinations, one row per fit and one row per reflection,
        respectively. Parent directories must exist.
    overwrite : bool
        Replace existing destinations only when True, default False.

    Returns
    -------
    tuple of pathlib.Path
        Published catalogue and reflection paths, in that order.

    Notes
    -----
    Column names state units. Each row records the fit frame, Biot measure,
    reference-volume normalization and tensor shear. Source frame IDs use JSON
    to preserve string versus integer identity. NaN means unavailable. Files
    are published independently after both are written; this is not a two-file
    transaction. HDF5 is the complete provenance-preserving artifact.
    """
    values = _results(results)
    paths = [
        _destination(p, values, overwrite) for p in (catalogue_path, reflections_path)
    ]
    if paths[0].resolve() == paths[1].resolve():
        raise InputError("catalogue and reflection CSV destinations must differ")
    temps = []
    try:
        for path in paths:
            temps.append(_temporary(path))
        common = [
            "result_index",
            "frame_id_json",
            "pattern_index",
            "status",
            "coordinates",
            "strain_measure",
            "scale_convention",
            "shear_convention",
        ]
        tensor_names = [f"epsilon_{i}{j}" for i in range(3) for j in range(3)]
        rotation_names = [f"rotation_{i}{j}" for i in range(3) for j in range(3)]
        reciprocal_names = [
            f"reciprocal_{i}{j}_nm_inv" for i in range(3) for j in range(3)
        ]
        with temps[0].open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                common
                + [
                    "message",
                    "fit_mask",
                    "n_reflections",
                    "n_parameters",
                    "nfev",
                    "rms_error_deg",
                    "rank",
                    "condition_number",
                    "von_mises_strain_igor",
                ]
                + tensor_names
                + rotation_names
                + reciprocal_names
            )
            for i, value in enumerate(values):
                row = [
                    i,
                    json.dumps(value.frame_id),
                    value.pattern_index,
                    value.status,
                    value.coordinates,
                    "Biot deviatoric",
                    "reference volume",
                    "tensor",
                ]
                writer.writerow(
                    row
                    + [
                        value.message,
                        value.fit_mask,
                        len(value.measurements.hkl),
                        value.n_parameters,
                        value.nfev,
                        value.rms_error_deg,
                        value.rank,
                        value.condition_number,
                        value.von_mises_strain,
                    ]
                    + value.tensor.ravel().tolist()
                    + value.rotation.ravel().tolist()
                    + value.reciprocal.ravel().tolist()
                )
        columns = [
            "peak_index",
            "h",
            "k",
            "l",
            "measured_x_px",
            "measured_y_px",
            "predicted_x_px",
            "predicted_y_px",
            "angular_residual_deg",
            "predicted_energy_kev",
        ]
        with temps[1].open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(common + columns)
            for i, value in enumerate(values):
                identity = [
                    i,
                    json.dumps(value.frame_id),
                    value.pattern_index,
                    value.status,
                    value.coordinates,
                    "Biot deviatoric",
                    "reference volume",
                    "tensor",
                ]
                data = value.measurements
                energies = value.predicted_energy_kev
                for j in range(len(data.hkl)):
                    writer.writerow(
                        identity
                        + [
                            data.peak_indices[j],
                            *data.hkl[j],
                            *data.xy[j],
                            *value.predicted_xy[j],
                            value.angular_residual_deg[j],
                            energies[j],
                        ]
                    )
        for temporary, path in zip(temps, paths):
            publish_file(temporary, path, overwrite=overwrite)
    finally:
        for temporary in temps:
            temporary.unlink(missing_ok=True)
    return tuple(paths)
