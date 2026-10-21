# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Versioned control-state round trips for externally scheduled scan coordinators.

Scientific output remains the catalog and closed point files. This module never
writes a checkpoint, publishes a catalog, inspects raw inputs, or schedules work.
"""

from dataclasses import asdict, fields
from datetime import datetime
from importlib.metadata import version as distribution_version
import json
import math
import os
from pathlib import Path
from time import perf_counter

import h5py
import numpy as np

from lauelab.indexing.errors import InputError
from . import _scan_layout as layout
from ._scan_reader import ScanReader, _metadata_equal
from ._scan_writer import native_version
from .scan import (
    PointOutcome, PointTask, SETTING_NAMES, _catalog_request,
    _configured_reconstructor, _empty_catalog, _point_ids, _point_paths,
    _set_point_row,
)
from ._writer import PIXEL_DTYPES

_FORMAT = "lauelab-prepared-scan-state"
_VERSION = 1


def _json_copy(value):
    def convert(item):
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"unsupported control-state value {type(item).__name__}")

    return json.loads(json.dumps(value, default=convert, allow_nan=False))


def export_state(scan):
    points = []
    for outcome in scan.outcomes:
        task = scan._by_index.get(outcome.index)
        points.append({
            "point_id": outcome.point_id,
            "source": scan._catalog["/points/source_paths"][outcome.index],
            "output": os.fspath(scan.directory / scan._catalog["/points/paths"][outcome.index]),
            "task": None if task is None else asdict(task),
            "outcome": asdict(outcome),
        })
    return _json_copy({
        "format": _FORMAT, "version": _VERSION,
        "lauelab_version": distribution_version("lauelab"),
        "native_version": scan._request["native_version"],
        "directory": os.fspath(scan.directory), "created": scan._created,
        "status": scan._run_status.name.lower(),
        "configuration": scan._configuration, "points": points,
    })


def _require(condition, message):
    if not condition:
        raise InputError(f"invalid prepared scan state: {message}")


def _keys(value, keys, label):
    _require(isinstance(value, dict) and set(value) == set(keys), f"invalid {label} fields")


def _path(value):
    _require(isinstance(value, str) and os.path.isabs(value)
             and os.path.normpath(value) == value, "paths must be normalized and absolute")
    return Path(value)


def _integer(value, minimum=0):
    return type(value) is int and value >= minimum


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _task(values, index, point, configuration):
    _keys(values, (f.name for f in fields(PointTask)), "task")
    _require(type(values["index"]) is int and values["index"] == index,
             "task index differs from registration order")
    for name in ("point_id", "source", "output"):
        _require(values[name] == point[name], f"task {name} disagrees with its registration")
    for name, value in configuration.items():
        _require(_metadata_equal(values[name], value), f"task {name} differs from frozen configuration")
    shape, depths, raw = values["image_shape"], values["depth_bounds_um"], values["raw_slices"]
    _require(isinstance(shape, list) and len(shape) == 2
             and all(_integer(n, 1) for n in shape), "invalid task image shape")
    _require(_integer(values["n_depths"], 1), "invalid depth count")
    _require(isinstance(depths, list) and len(depths) == 2
             and all(_finite(v) for v in depths) and depths[0] <= depths[1], "invalid depths")
    _require(type(values["pixel_type"]) is int and values["pixel_type"] in PIXEL_DTYPES,
             "invalid pixel type")
    _require(isinstance(raw, list) and len(raw) == 2 and all(_integer(n) for n in raw)
             and raw[0] == configuration["settings"]["frame_skip"] and raw[0] < raw[1],
             "invalid selected raw slices")
    number = values["scan_number"]
    _require(number is None or (_integer(number) and number <= np.iinfo(np.int32).max),
             "invalid scan number")
    position = values["sample_position"]
    _require(isinstance(position, list) and len(position) == 3
             and all(v is None or _finite(v) for v in position), "invalid sample position")
    _require(values["energy_kev"] is None or _finite(values["energy_kev"]), "invalid energy")
    return PointTask(**values)


def _outcome(values, index, point_id):
    _keys(values, (f.name for f in fields(PointOutcome)), "outcome")
    _require(type(values["index"]) is int and values["index"] == index
             and values["point_id"] == point_id, "outcome identity differs from registration")
    _require(_finite(values["seconds"]) and values["seconds"] >= 0, "invalid outcome duration")
    _require(values["status"] in {"pending", "writing", "complete", "failed"},
             "invalid status for an unfinished scan")
    status, error, output = values["status"], values["error"], values["output"]
    if status == "failed":
        _require(isinstance(error, str) and bool(error) and output is None, "invalid failure receipt")
    elif status == "complete":
        _require(error is None and isinstance(output, str), "invalid completion receipt")
    else:
        _require(error is None and output is None and values["seconds"] == 0,
                 "unsettled point contains a final receipt")
    return PointOutcome(**values)


def _check_catalog(scan):
    """An older view is valid; unrelated or newer terminal progress is not."""
    if not scan.path.exists():
        return
    with ScanReader(scan.path) as reader:
        _require(reader.run_status == "running", "existing catalog is terminal; read it as an artifact")
        _require(len(reader.points) <= len(scan.outcomes), "catalog contains unregistered points")
        with h5py.File(scan.path, "r") as source:
            _require(source.attrs["created"] == scan._created, "existing catalog has a different identity")
            for name in ("native_version", "lauelab_version"):
                expected = scan._request[name] if name == "native_version" else distribution_version("lauelab")
                _require(source.attrs[name] == expected, f"catalog {name} differs")
            _require(source["geometry"].attrs["path"] == scan._configuration["geometry_path"]
                     and source["geometry/xml"].asstr()[()] == scan._configuration["geometry_xml"],
                     "catalog geometry differs")
            for name, value in scan._request["parameters"].items():
                found = source[name].asstr()[()] if isinstance(value, str) else source[name][()]
                _require(_metadata_equal(found, value), f"catalog parameter {name} differs")
            count = len(reader.points)
            for name in layout.POINT_COLUMNS:
                if name in ("/points/statuses", "/points/errors"):
                    continue
                dataset = source[name]
                found = dataset.asstr()[:] if dataset.dtype.kind == "O" else dataset[:]
                _require(_metadata_equal(found, scan._catalog[name][:count]),
                         f"catalog point descriptions differ at {name}")
        for old, current in zip(reader.points, scan.outcomes):
            if old.status in {"complete", "failed", "interrupted", "unattempted"}:
                _require(old.status == current.status and old.error == (current.error or ""),
                         f"catalog has terminal progress absent from receipts for {old.point_id!r}")
            elif old.status == "writing" and current.status == "pending":
                scan._set_status(old.index, layout.PointStatus.WRITING)


def restore_state(state, outcomes, cls):
    try:
        state = _json_copy(state)
        return _restore(state, tuple(outcomes), cls)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise InputError(f"invalid prepared scan state: {error}") from error


def _restore(state, receipts, cls):
    _keys(state, ("format", "version", "lauelab_version", "native_version", "directory",
                  "created", "status", "configuration", "points"), "root")
    _require(state["format"] == _FORMAT and type(state["version"]) is int
             and state["version"] == _VERSION, "unsupported control-state format/version")
    _require(state["lauelab_version"] == distribution_version("lauelab")
             and state["native_version"] == native_version(), "package/native version tags differ")
    _require(state["status"] == "running", "only unfinished scans can be restored")
    _require(isinstance(state["created"], str)
             and datetime.fromisoformat(state["created"]).tzinfo is not None, "invalid creation time")
    directory = _path(state["directory"])
    _require(directory.is_dir() and (directory / layout.POINT_DIRECTORY).is_dir(),
             "scan directory/points directory is unavailable")
    configuration = state["configuration"]
    _keys(configuration, ("detector", "settings", "geometry_path", "geometry_xml", "compression"),
          "configuration")
    _require(type(configuration["detector"]) is int, "invalid detector slot")
    _require(isinstance(configuration["geometry_path"], str)
             and isinstance(configuration["geometry_xml"], str), "invalid geometry")
    _require(configuration["compression"] in (None, "gzip"), "invalid compression")
    _keys(configuration["settings"], SETTING_NAMES, "settings")
    inspector = _configured_reconstructor(configuration)
    normalized = {name: getattr(inspector, name) for name in SETTING_NAMES}
    _require(_json_copy(normalized) == configuration["settings"], "settings are not normalized")
    points = state["points"]
    _require(isinstance(points, list), "points must be a list")
    catalog = _empty_catalog(len(points))
    tasks, saved = [], []
    sources = []
    for index, point in enumerate(points):
        _keys(point, ("point_id", "source", "output", "task", "outcome"), "point")
        source, output = _path(point["source"]), _path(point["output"])
        sources.append(source)
        _require(output == directory / layout.point_path(source.stem), "point output path changed")
        outcome = _outcome(point["outcome"], index, point["point_id"])
        task = None if point["task"] is None else _task(point["task"], index, point, configuration)
        if task is None:
            _require(outcome.status == "failed" and outcome.seconds == 0,
                     "a point without a task must be an inspection failure")
        else:
            tasks.append(task)
        _set_point_row(catalog, index, source, point["point_id"], output.relative_to(directory).as_posix(),
                       None if task is None else asdict(task), outcome.error or "")
        saved.append(outcome)
    _point_ids([p["point_id"] for p in points], sources)
    _point_paths(sources)
    _require(len(set(sources)) == len(sources), "source paths repeat")
    scan = cls(directory, tuple(tasks), _catalog_request(configuration, state["native_version"]),
               catalog, configuration)
    scan._created = state["created"]
    scan._inspector = inspector
    scan._restoring = True  # record() must not publish while receipts are replayed.
    for outcome in saved:
        if outcome.index not in scan._by_index:
            continue
        if outcome.status in {"complete", "failed"}:
            scan.record(outcome)
        elif outcome.status == "writing":
            scan._set_status(outcome.index, layout.PointStatus.WRITING)
    for outcome in receipts:
        _require(isinstance(outcome, PointOutcome), "replayed receipts must be PointOutcome objects")
        values = _json_copy(asdict(outcome))
        _require(_integer(outcome.index) and outcome.index < len(points), "unknown receipt index")
        outcome = _outcome(values, outcome.index, points[outcome.index]["point_id"])
        _require(outcome.status in {"complete", "failed"}, "replayed receipt is not terminal")
        current = scan.outcomes[outcome.index]
        if current.status in {"complete", "failed"}:
            _require(current == outcome, "conflicting receipt replay")
        else:
            scan.record(outcome)
    _check_catalog(scan)
    for outcome in scan.outcomes:
        if outcome.status in {"pending", "writing"}:
            task = scan._by_index[outcome.index]
            _require(not Path(task.output).exists(),
                     f"unrecorded output for {task.point_id!r}; reconcile its receipt before restoring")
    scan._published = scan.path.exists()
    scan._restoring = False
    scan._last_snapshot = perf_counter()
    return scan
