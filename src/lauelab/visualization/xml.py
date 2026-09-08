# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""LaueGo indexing XML adapter for visualization data."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from lauelab.analysis import lattice_params_to_reciprocal, reciprocal_to_orientation
from lauelab.indexing import Atom, Cell, Crystal, Geometry
from lauelab.indexing.indexer import PEAK_DTYPE

from .data import VisualizationDataset, _validate_frame_ids


def _text(parent, name):
    node = parent.find(name) if parent is not None else None
    return node.text.strip() if node is not None and node.text else None


def _number(parent, name, default=np.nan):
    value = _text(parent, name)
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


def _array(parent, names, *, dtype=float):
    for name in names:
        value = _text(parent, name)
        if value is not None:
            return np.fromstring(value, sep=" ", dtype=dtype)
    return np.empty(0, dtype=dtype)


def _attribute_number(element, name, default=np.nan):
    try:
        return float(element.get(name, default))
    except (TypeError, ValueError):
        return default


def _integer(value, *, path, step, field):
    if not np.isfinite(value) or value != int(value):
        raise ValueError(f"invalid integer in {path}, step {step}, field {field}: {value!r}")
    return int(value)


def _integer_array(parent, name, *, path, step, field):
    values = _array(parent, (name,))
    if not np.isfinite(values).all() or np.any(values != values.astype(int)):
        raise ValueError(f"invalid integer in {path}, step {step}, field {field}")
    return values.astype(int)


def orientations_from_reciprocals(reciprocals, reference):
    """Return orientation matrices for finite reciprocal lattices, else NaN.

    One batched inverse of the reference serves every lattice. A singular
    reference leaves every orientation NaN, as the per-lattice form did.
    """
    reciprocals = np.asarray(reciprocals, dtype=float).reshape((-1, 3, 3))
    rotations = np.full(reciprocals.shape, np.nan)
    if reference is None:
        return rotations
    finite = np.isfinite(reciprocals).all(axis=(1, 2))
    if np.any(finite):
        try:
            rotations[finite] = reciprocal_to_orientation(reciprocals[finite], reference)
        except np.linalg.LinAlgError:
            pass
    return rotations


def _load_crystal(steps):
    for step in steps:
        node = step.find("indexing/xtl")
        if node is None:
            continue
        space_group = _text(node, "SpaceGroup")
        parameters = _array(node, ("latticeParameters",))
        if space_group is None or len(parameters) != 6:
            continue
        unit = node.find("latticeParameters").get("unit", "nm")
        atoms = []
        for atom in node.findall("atom"):
            position = np.fromstring(atom.text or "", sep=" ")
            if len(position) == 3 and atom.get("symbol"):
                try:
                    occupancy = float(atom.get("occupancy", "1"))
                except ValueError:
                    occupancy = 1.0
                atoms.append(Atom(
                    atom.get("symbol"),
                    tuple(position),
                    occupancy=occupancy,
                    label=atom.get("label"),
                ))
        number, separator, setting = space_group.partition(":")
        try:
            space_group_number = int(number)
        except ValueError:
            continue
        if separator and setting.upper() not in {"H", "R"}:
            continue
        try:
            return Crystal(
                _text(node, "structureDesc") or "LaueGo XML crystal",
                space_group_number,
                Cell(*parameters, unit=unit),
                tuple(atoms),
                source=_text(node, "xtlFile"),
                setting=setting.upper() if setting else None,
            )
        except (TypeError, ValueError):
            continue
    return None


def _resolve_geometry(xml_path, geometry, steps):
    if geometry is not None:
        return geometry if isinstance(geometry, Geometry) else Geometry(geometry)
    candidates = []
    for step in steps:
        path = _text(step.find("detector"), "geoFile")
        if path and path not in candidates:
            candidates.append(path)
    xml_directory = Path(xml_path).resolve().parent
    for candidate in candidates:
        path = Path(candidate).expanduser()
        paths = (path,) if path.is_absolute() else (path, xml_directory / path)
        for resolved in paths:
            if resolved.is_file():
                try:
                    return Geometry(resolved)
                except (ImportError, OSError, ValueError):
                    pass
    return None


def load_visualization_xml(path, *, geometry=None, frame_ids=None):
    """Load a LaueGo ``AllSteps`` indexing XML document.

    Geometry is optional. An explicit geometry object or path takes precedence;
    otherwise readable paths embedded in the XML are tried. Failure to resolve
    embedded geometry does not prevent loading non-detector visualizations.
    """
    path = Path(path)
    root = ET.parse(path).getroot()
    steps = root.findall("step")
    if not steps:
        raise ValueError(f"no <step> elements found in {path}")
    ids = tuple(range(len(steps))) if frame_ids is None else tuple(frame_ids)
    ids = _validate_frame_ids(ids, len(steps))
    crystal = _load_crystal(steps)
    reference_recip = None
    if crystal is not None:
        cell = crystal.cell.in_angstrom
        reference_recip = lattice_params_to_reciprocal(
            cell.a / 10.0,
            cell.b / 10.0,
            cell.c / 10.0,
            cell.alpha,
            cell.beta,
            cell.gamma,
            space_group=crystal.space_group,
        )

    positions = np.full((len(steps), 3), np.nan)
    depths = np.full(len(steps), np.nan)
    scan_numbers = np.full(len(steps), np.nan)
    energies = np.full(len(steps), np.nan)
    image_shapes = np.zeros((len(steps), 2), dtype=int)
    starts = np.zeros((len(steps), 2), dtype=int)
    groups = np.ones((len(steps), 2), dtype=int)
    frame_n_peaks = np.zeros(len(steps), dtype=int)
    detector_ids = []
    input_images = []
    peak_frames = []
    peak_indices = []
    peak_arrays = []
    pattern_frames = []
    pattern_indices = []
    reciprocals = []
    goodness = []
    rms_errors = []
    pattern_counts = []
    assignment_patterns = []
    assignment_peaks = []
    assignment_hkl = []
    assignment_errors = []
    assignment_energies = []
    assignment_intensities = []

    for frame_index, step in enumerate(steps):
        positions[frame_index] = [_number(step, name) for name in ("Xsample", "Ysample", "Zsample")]
        depths[frame_index] = _number(step, "depth")
        scan_numbers[frame_index] = _number(step, "scanNum")
        energies[frame_index] = _number(step, "energy")
        detector = step.find("detector")
        detector_ids.append(_text(detector, "detectorID"))
        input_images.append(_text(detector, "inputImage"))
        image_shapes[frame_index] = [
            _integer(_number(detector, name, 0), path=path, step=frame_index, field=name)
            for name in ("Ny", "Nx")
        ]
        roi = detector.find("ROI") if detector is not None else None
        if roi is not None:
            starts[frame_index] = [
                _integer(_attribute_number(roi, name, 0), path=path, step=frame_index, field=f"ROI.{name}")
                for name in ("startx", "starty")
            ]
            groups[frame_index] = [
                _integer(_attribute_number(roi, name, 1), path=path, step=frame_index, field=f"ROI.{name}")
                for name in ("groupx", "groupy")
            ]

        peaks_node = detector.find("peaksXY") if detector is not None else None
        x = _array(peaks_node, ("fitX", "Xpixel"))
        y = _array(peaks_node, ("fitY", "Ypixel"))
        lengths = [len(value) for value in (x, y) if len(value)]
        declared_count = (
            _integer(
                _attribute_number(peaks_node, "Npeaks", 0),
                path=path,
                step=frame_index,
                field="peaksXY.Npeaks",
            )
            if peaks_node is not None else 0
        )
        peak_count = declared_count if declared_count > 0 else (max(lengths) if lengths else 0)
        frame_n_peaks[frame_index] = peak_count
        peaks = np.full(peak_count, np.nan, dtype=PEAK_DTYPE)
        fields = {
            "fit_x": x,
            "fit_y": y,
            "intens": _array(peaks_node, ("Intens", "intens")),
            "integral": _array(peaks_node, ("Integral", "integral")),
            "hwhm_x": _array(peaks_node, ("hwhmX",)),
            "hwhm_y": _array(peaks_node, ("hwhmY",)),
            "tilt": _array(peaks_node, ("tilt",)),
            "chisq": _array(peaks_node, ("chisq",)),
            "background": _array(peaks_node, ("background",)),
        }
        for name, values in fields.items():
            peaks[name][:min(peak_count, len(values))] = values[:peak_count]
        q = [_array(peaks_node, (name,)) for name in ("Qx", "Qy", "Qz")]
        if all(len(value) for value in q):
            q_count = min(peak_count, *(len(value) for value in q))
            peaks["qhat"][:q_count] = np.column_stack([value[:q_count] for value in q])
        peak_frames.extend([frame_index] * peak_count)
        peak_indices.extend(range(peak_count))
        peak_arrays.append(peaks)

        indexing = step.find("indexing")
        for rank, pattern in enumerate(indexing.findall("pattern") if indexing is not None else ()):
            reciprocal_node = pattern.find("recip_lattice")
            vectors = [_array(reciprocal_node, (name,)) for name in ("astar", "bstar", "cstar")]
            reciprocal = np.asarray(vectors) if all(len(value) == 3 for value in vectors) else np.full((3, 3), np.nan)
            pattern_row = len(pattern_indices)
            pattern_frames.append(frame_index)
            pattern_indices.append(_integer(
                _attribute_number(pattern, "num", rank),
                path=path,
                step=frame_index,
                field=f"pattern[{rank}].num",
            ))
            reciprocals.append(reciprocal)
            goodness.append(_attribute_number(pattern, "goodness"))
            rms_errors.append(_attribute_number(pattern, "rms_error"))

            hkl_node = pattern.find("hkl_s")
            h, k, l = (
                _integer_array(
                    hkl_node,
                    name,
                    path=path,
                    step=frame_index,
                    field=f"pattern[{rank}].hkl_s.{name}",
                )
                for name in ("h", "k", "l")
            )
            peak_refs = _integer_array(
                hkl_node,
                "PkIndex",
                path=path,
                step=frame_index,
                field=f"pattern[{rank}].hkl_s.PkIndex",
            )
            errors = _array(hkl_node, ("err_deg",))
            assignment_energy = _array(hkl_node, ("energy_kev",))
            predicted_intensity = _array(hkl_node, ("pred_intens",))
            count = min(len(h), len(k), len(l), len(peak_refs))
            pattern_counts.append(_integer(
                _attribute_number(pattern, "Nindexed", count),
                path=path,
                step=frame_index,
                field=f"pattern[{rank}].Nindexed",
            ))
            if count:
                valid = (peak_refs[:count] >= 0) & (peak_refs[:count] < peak_count)
                count_valid = int(np.count_nonzero(valid))
                assignment_patterns.extend([pattern_row] * count_valid)
                assignment_peaks.extend(peak_refs[:count][valid])
                assignment_hkl.extend(np.column_stack([h[:count], k[:count], l[:count]])[valid])
                for target, values in (
                    (assignment_errors, errors),
                    (assignment_energies, assignment_energy),
                    (assignment_intensities, predicted_intensity),
                ):
                    aligned = np.full(count, np.nan)
                    aligned[:min(count, len(values))] = values[:count]
                    target.extend(aligned[valid])

    peaks = np.concatenate(peak_arrays) if peak_arrays else np.empty(0, dtype=PEAK_DTYPE)
    reciprocals = np.asarray(reciprocals, dtype=float).reshape((-1, 3, 3))
    return VisualizationDataset(
        frame_ids=ids,
        sample_positions=positions,
        depths=depths,
        frame_n_peaks=frame_n_peaks,
        scan_numbers=scan_numbers,
        energies_kev=energies,
        detector_ids=tuple(detector_ids),
        image_shapes=image_shapes,
        starts=starts,
        groups=groups,
        input_images=tuple(input_images),
        images=(None,) * len(steps),
        peak_frame_indices=np.asarray(peak_frames),
        peak_indices=np.asarray(peak_indices),
        peaks=peaks,
        pattern_frame_indices=np.asarray(pattern_frames),
        pattern_indices=np.asarray(pattern_indices),
        pattern_rotations=orientations_from_reciprocals(reciprocals, reference_recip),
        pattern_reciprocals=reciprocals,
        pattern_goodness=np.asarray(goodness),
        pattern_rms_error_deg=np.asarray(rms_errors),
        pattern_n_indexed=np.asarray(pattern_counts),
        assignment_pattern_rows=np.asarray(assignment_patterns),
        assignment_peak_indices=np.asarray(assignment_peaks),
        assignment_hkl=np.asarray(assignment_hkl, dtype=int).reshape((-1, 3)),
        assignment_error_deg=np.asarray(assignment_errors),
        assignment_energy_kev=np.asarray(assignment_energies),
        assignment_predicted_intensity=np.asarray(assignment_intensities),
        crystal=crystal,
        geometry=_resolve_geometry(path, geometry, steps),
    )
