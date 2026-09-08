# Copyright 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Indexing-results HDF5 adapter for visualization data."""

from __future__ import annotations

from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import h5py
import numpy as np

from lauelab._hdf5 import check_format_version, write_root_attributes
from lauelab._results_layout import (
    FORMAT, SUPPORTED_VERSIONS, VERSION, write_crystal, write_dataset,
)
from lauelab.analysis import lattice_params_to_reciprocal
from lauelab.indexing import Atom, Cell, Crystal, Geometry
from lauelab.indexing.indexer import PEAK_DTYPE

from .data import VisualizationDataset
from .xml import load_visualization_xml, orientations_from_reciprocals


def _text(value) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _strings(dataset) -> tuple[str | None, ...]:
    return tuple(value or None for value in dataset.asstr()[...])


def _load_crystal(source) -> Crystal | None:
    if "crystal" not in source:
        return None
    group = source["crystal"]
    parameters = group["lattice_parameters"][...]
    symbols = group["atom_symbols"].asstr()[...]
    labels = group["atom_labels"].asstr()[...]
    positions = group["atom_positions"][...]
    occupancies = group["atom_occupancies"][...]
    atoms = tuple(
        Atom(symbol, tuple(position), occupancy=float(occupancy), label=label or None)
        for symbol, label, position, occupancy in zip(
            symbols, labels, positions, occupancies, strict=True
        )
    )
    setting = _text(group.attrs.get("setting", "")) or None
    crystal_source = _text(group.attrs.get("source", "")) or None
    return Crystal(
        _text(group.attrs["name"]),
        int(group.attrs["space_group"]),
        Cell(*parameters, unit="nm"),
        atoms,
        source=crystal_source,
        setting=setting,
    )


def _load_geometry(source, path, geometry):
    if geometry is not None:
        return geometry if isinstance(geometry, Geometry) else Geometry(geometry)
    if "geometry" not in source:
        return None
    group = source["geometry"]
    if "xml" in group:
        temporary = tempfile.TemporaryDirectory(prefix="lauelab-geometry-")
        temporary_path = Path(temporary.name) / "geometry.xml"
        try:
            temporary_path.write_text(group["xml"].asstr()[()])
            loaded = Geometry(temporary_path)
        except Exception:
            temporary.cleanup()
            raise
        loaded._temporary_directory = temporary
        return loaded
    embedded_path = _text(group.attrs.get("path", ""))
    candidates = []
    if embedded_path:
        candidate = Path(embedded_path).expanduser()
        candidates.extend((candidate,) if candidate.is_absolute() else (candidate, path.parent / candidate))
    for candidate in candidates:
        if candidate.is_file():
            try:
                return Geometry(candidate)
            except (ImportError, OSError, ValueError):
                pass
    return None


def _owners(offsets: np.ndarray, expected_count: int, name: str) -> np.ndarray:
    if (
        offsets.ndim != 1
        or len(offsets) == 0
        or offsets[0] != 0
        or offsets[-1] != expected_count
        or np.any(np.diff(offsets) < 0)
    ):
        raise ValueError(f"invalid {name} offsets")
    return np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))


def _xml_frame_values(path):
    names = {
        "titles": "title",
        "sample_names": "sampleName",
        "user_names": "userName",
        "beamlines": "beamline",
        "dates_exposed": "dateExposed",
        "ccd_shutters": "CCDshutter",
        "mono_modes": "monoMode",
    }
    values = {name: [] for name in names}
    numeric = {name: [] for name in (
        "threshold_used", "threshold_ratio", "total_sum", "sum_above_threshold",
        "num_above_threshold", "peak_minwidth", "peak_maxwidth",
        "peak_max_cent_to_fit", "peak_boxsize", "peaksearch_seconds", "indexing_seconds",
        "exposure_seconds", "beam_bad", "light_on", "hutch_temperature", "sample_distance",
    )}
    geometry_path = ""
    run_values = {}
    for _, step in ET.iterparse(path, events=("end",)):
        if step.tag != "step":
            continue
        detector = step.find("detector")
        peaks = detector.find("peaksXY") if detector is not None else None
        indexing = step.find("indexing")
        _collect_run_values(run_values, detector, peaks, indexing)
        if not geometry_path:
            geometry_path = (detector.findtext("geoFile") or "").strip() if detector is not None else ""
        for output_name, xml_name in names.items():
            node = step.find(xml_name)
            values[output_name].append(node.text.strip() if node is not None and node.text else "")
        numeric["threshold_used"].append(_attribute_float(peaks, "threshold"))
        numeric["threshold_ratio"].append(_attribute_float(peaks, "thresholdRatio"))
        numeric["total_sum"].append(_child_float(detector, "totalSum"))
        numeric["sum_above_threshold"].append(_child_float(detector, "sumAboveThreshold"))
        numeric["num_above_threshold"].append(_child_float(detector, "numAboveThreshold", 0))
        numeric["peak_minwidth"].append(_attribute_float(peaks, "minwidth"))
        numeric["peak_maxwidth"].append(_attribute_float(peaks, "maxwidth"))
        numeric["peak_max_cent_to_fit"].append(_attribute_float(peaks, "maxCentToFit"))
        numeric["peak_boxsize"].append(_attribute_float(peaks, "boxsize", 0))
        numeric["peaksearch_seconds"].append(_attribute_float(peaks, "executionTime"))
        numeric["indexing_seconds"].append(_attribute_float(indexing, "executionTime"))
        numeric["exposure_seconds"].append(_child_float(detector, "exposure"))
        numeric["beam_bad"].append(_child_float(step, "beamBad", -1))
        numeric["light_on"].append(_child_float(step, "lightOn", -1))
        numeric["hutch_temperature"].append(_child_float(step, "hutchTemperature"))
        numeric["sample_distance"].append(_child_float(step, "sampleDistance"))
        step.clear()
    values.update(numeric)
    return values, geometry_path, run_values


def _collect_run_values(values, detector, peaks, indexing):
    # Effective thresholds and fit widths already live on each frame. Only
    # attributes describing the producer configuration belong on /run.
    for element, fields in (
        (peaks, (
            ("peakProgram", "peak_program", str),
            ("maxRfactor", "max_rfactor", float),
            ("NpeakMax", "max_peaks", int),
            ("max_number", "max_peaks", int),
            ("minSeparation", "min_separation", int),
            ("min_separation", "min_separation", int),
            ("peakShape", "peak_shape", str),
            ("maskFile", "mask_file", str),
        )),
        (indexing, (
            ("indexProgram", "program", str),
            ("keVmaxCalc", "kev_max_calc", float),
            ("keVmaxTest", "kev_max_test", float),
            ("angleTolerance", "angle_tolerance_deg", float),
            ("cone", "cone_deg", float),
            ("hklPrefer", "hkl_prefer", lambda text: tuple(int(v) for v in text.split())),
        )),
    ):
        if element is None:
            continue
        for xml_name, name, parse in fields:
            text = element.get(xml_name)
            if text is not None and text.strip().lower() not in ("", "none", "nan"):
                _record_run_value(values, name, parse(text.strip()))
    if detector is not None:
        text = detector.findtext("cosmicFilter")
        if text and text.strip().lower() not in ("", "none", "nan"):
            flags = {"true": True, "false": False, "1": True, "0": False}
            if text.strip().lower() not in flags:
                raise ValueError(f"invalid XML cosmicFilter {text!r}")
            _record_run_value(values, "cosmic_filter", flags[text.strip().lower()])


def _record_run_value(values, name, value):
    if name in values and values[name] != value:
        raise ValueError(f"XML contains conflicting run parameter {name!r}")
    values[name] = value


def _attribute_float(element, name, default=np.nan):
    if element is None:
        return default
    try:
        return float(element.get(name, default))
    except (TypeError, ValueError):
        return default


def _child_float(element, name, default=np.nan):
    if element is None:
        return default
    child = element.find(name)
    try:
        return float(child.text) if child is not None and child.text else default
    except ValueError:
        return default


def convert_xml(xml_path, output_path=None, *, geometry=None, overwrite=False) -> Path:
    """Convert a LaueGo ``AllSteps`` XML document to a results HDF5 file."""
    xml_path = Path(xml_path)
    output_path = xml_path.with_suffix(".h5") if output_path is None else Path(output_path)
    if not overwrite and output_path.exists():
        raise FileExistsError(output_path)
    dataset = load_visualization_xml(xml_path, geometry=geometry)
    frame_values, embedded_geometry_path, run_values = _xml_frame_values(xml_path)
    mode = "w" if overwrite else "x"
    with h5py.File(output_path, mode) as target:
        write_root_attributes(target, format_name=FORMAT, version=VERSION, source=str(xml_path))
        write_crystal(target, dataset.crystal)
        geometry_group = target.create_group("geometry")
        resolved_geometry = dataset.geometry
        if resolved_geometry is not None:
            geometry_path = Path(resolved_geometry.path)
        else:
            geometry_path = Path(embedded_geometry_path) if embedded_geometry_path else None
        geometry_group.attrs["path"] = "" if geometry_path is None else str(geometry_path)
        if resolved_geometry is not None:
            try:
                geometry_text = Path(resolved_geometry.path).read_text()
            except (OSError, UnicodeError):
                pass
            else:
                write_dataset(target, "/geometry/xml", geometry_text)

        run = target.create_group("run")
        run.attrs.update(run_values)
        frame_count = dataset.n_frames
        frame_data = {
            "frame_ids": dataset.frame_ids,
            "sample_positions": dataset.sample_positions,
            "depths": dataset.depths,
            "scan_numbers": np.where(np.isnan(dataset.scan_numbers), -1, dataset.scan_numbers),
            "energies_kev": dataset.energies_kev,
            "detector_ids": [value or "" for value in dataset.detector_ids],
            "input_images": [value or "" for value in dataset.input_images],
            "image_shapes": dataset.image_shapes,
            "roi_starts": dataset.starts,
            "roi_groups": dataset.groups,
            "n_peaks": dataset.frame_n_peaks,
            "n_patterns": np.bincount(dataset.pattern_frame_indices, minlength=frame_count),
        }
        frame_data.update(frame_values)
        peak_counts = np.bincount(dataset.peak_frame_indices, minlength=frame_count)
        pattern_counts = np.bincount(dataset.pattern_frame_indices, minlength=frame_count)
        frame_data["peak_offsets"] = np.r_[0, np.cumsum(peak_counts)]
        frame_data["pattern_offsets"] = np.r_[0, np.cumsum(pattern_counts)]
        for name, data in frame_data.items():
            write_dataset(target, f"/frames/{name}", data)

        for name in dataset.peaks.dtype.names:
            write_dataset(target, f"/peaks/{name}", dataset.peaks[name])
        pattern_data = {
            "rank": dataset.pattern_indices,
            "reciprocal": dataset.pattern_reciprocals,
            "goodness": dataset.pattern_goodness,
            "rms_error_deg": dataset.pattern_rms_error_deg,
            "n_indexed": dataset.pattern_n_indexed,
            "assignment_offsets": np.r_[0, np.cumsum(np.bincount(
                dataset.assignment_pattern_rows, minlength=dataset.n_patterns
            ))],
        }
        for name, data in pattern_data.items():
            write_dataset(target, f"/patterns/{name}", data)
        assignment_data = {
            "peak_index": dataset.assignment_peak_indices,
            "hkl": dataset.assignment_hkl,
            "error_deg": dataset.assignment_error_deg,
            "energy_kev": dataset.assignment_energy_kev,
            "pred_intens": dataset.assignment_predicted_intensity,
        }
        for name, data in assignment_data.items():
            write_dataset(target, f"/assignments/{name}", data)
    return output_path


def load_results(path, *, geometry=None, frame_ids=None) -> VisualizationDataset:
    """Eagerly load a lauelab indexing-results HDF5 file."""
    path = Path(path)
    with h5py.File(path, "r") as source:
        check_format_version(
            source, format_name=FORMAT, supported_versions=SUPPORTED_VERSIONS
        )
        stored_ids = source["frames/frame_ids"]
        if h5py.check_string_dtype(stored_ids.dtype) is not None:
            ids = tuple(stored_ids.asstr()[...])
        else:
            ids = tuple(stored_ids[...].tolist())
        if frame_ids is not None:
            ids = tuple(frame_ids)
            if len(ids) != len(stored_ids):
                raise ValueError(f"frame_ids must contain {len(stored_ids)} values")

        crystal = _load_crystal(source)
        reciprocals = source["patterns/reciprocal"][...]
        reference = None
        if crystal is not None:
            cell = crystal.cell
            reference = lattice_params_to_reciprocal(
                cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma,
                space_group=crystal.space_group,
            )
        rotations = orientations_from_reciprocals(reciprocals, reference)

        peak_offsets = source["frames/peak_offsets"][...]
        pattern_offsets = source["frames/pattern_offsets"][...]
        assignment_offsets = source["patterns/assignment_offsets"][...]
        if len(peak_offsets) != len(ids) + 1 or len(pattern_offsets) != len(ids) + 1:
            raise ValueError("frame offsets do not align with frame_ids")
        peaks = np.empty(len(source["peaks/fit_x"]), dtype=PEAK_DTYPE)
        for name in peaks.dtype.names:
            peaks[name] = source[f"peaks/{name}"][...]

        scan_numbers = source["frames/scan_numbers"][...].astype(float)
        scan_numbers[scan_numbers == -1] = np.nan
        return VisualizationDataset(
            frame_ids=ids,
            sample_positions=source["frames/sample_positions"][...],
            depths=source["frames/depths"][...],
            frame_n_peaks=source["frames/n_peaks"][...],
            scan_numbers=scan_numbers,
            energies_kev=source["frames/energies_kev"][...],
            detector_ids=_strings(source["frames/detector_ids"]),
            image_shapes=source["frames/image_shapes"][...],
            starts=source["frames/roi_starts"][...],
            groups=source["frames/roi_groups"][...],
            input_images=_strings(source["frames/input_images"]),
            images=(None,) * len(ids),
            peak_frame_indices=_owners(peak_offsets, len(peaks), "peak"),
            peak_indices=np.arange(len(peaks)) - np.repeat(peak_offsets[:-1], np.diff(peak_offsets)),
            peaks=peaks,
            pattern_frame_indices=_owners(pattern_offsets, len(reciprocals), "pattern"),
            pattern_indices=source["patterns/rank"][...],
            pattern_rotations=rotations,
            pattern_reciprocals=reciprocals,
            pattern_goodness=source["patterns/goodness"][...],
            pattern_rms_error_deg=source["patterns/rms_error_deg"][...],
            pattern_n_indexed=source["patterns/n_indexed"][...],
            assignment_pattern_rows=_owners(
                assignment_offsets,
                len(source["assignments/peak_index"]),
                "assignment",
            ),
            assignment_peak_indices=source["assignments/peak_index"][...],
            assignment_hkl=source["assignments/hkl"][...],
            assignment_error_deg=source["assignments/error_deg"][...],
            assignment_energy_kev=source["assignments/energy_kev"][...],
            assignment_predicted_intensity=source["assignments/pred_intens"][...],
            crystal=crystal,
            geometry=_load_geometry(source, path, geometry),
        )
