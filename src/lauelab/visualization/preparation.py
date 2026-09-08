# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Backend-neutral preparation of maps, pole figures, and detector views."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Callable, Literal

import numpy as np

from lauelab.analysis import (
    SurfaceFrame,
    closest_pole_colors,
    cubic_hkl_family,
    cubic_ipf_colors,
    misorientation_matrix,
    orientation_to_rodrigues,
    crystal_direction,
    pole_color_radius,
    pole_figure_points,
    rodrigues_colors,
    simulate_reflections,
    symmetry_operations,
    symmetry_reduce_orientation,
)
from lauelab.indexing._frame import detector_to_roi_pixels, read_h5_frame

from .data import DataScope, FrameId, ResultSet, VisualizationDataset, _readonly
from .options import AXIS_OPTIONS, COLOR_MODES, POLE_COLOR_MODES


_AXIS_VALUES = tuple(choice.value for choice in AXIS_OPTIONS)
_COLOR_VALUES = tuple(choice.value for choice in COLOR_MODES)
_POLE_COLOR_VALUES = tuple(choice.value for choice in POLE_COLOR_MODES)

Values = np.ndarray | Callable[[VisualizationDataset], np.ndarray]
Alignment = Literal["frame", "pattern", "selected"]


@dataclass(frozen=True)
class Axis:
    """Custom map coordinate values and display metadata."""

    values: Values
    label: str
    unit: str | None = None
    alignment: Alignment = "frame"

    def __post_init__(self):
        if not self.label:
            raise ValueError("axis label cannot be empty")
        if self.alignment not in ("frame", "pattern", "selected"):
            raise ValueError("axis alignment must be 'frame', 'pattern', or 'selected'")


@dataclass(frozen=True)
class ScalarColor:
    """Scalar values and rendering metadata for a spatial map."""

    values: str | Values
    label: str | None = None
    palette: str = "Viridis"
    limits: tuple[float, float] | None = None
    alignment: Alignment = "pattern"

    def __post_init__(self):
        if self.alignment not in ("frame", "pattern", "selected"):
            raise ValueError("color alignment must be 'frame', 'pattern', or 'selected'")
        if self.limits is not None:
            if len(self.limits) != 2 or not np.isfinite(self.limits).all() or self.limits[0] >= self.limits[1]:
                raise ValueError("color limits must be two finite increasing values")


@dataclass(frozen=True)
class MapData:
    """Prepared records for a two- or three-dimensional spatial map."""

    coordinates: np.ndarray
    axis_labels: tuple[str, ...]
    frame_ids: tuple[FrameId, ...]
    pattern_indices: np.ndarray
    colors: np.ndarray
    color_kind: Literal["scalar", "rgb"]
    color_label: str
    palette: str | None = None
    color_limits: tuple[float, float] | None = None
    indexed: np.ndarray | None = None
    spatial_axes: bool = True

    def __post_init__(self):
        count = len(self.frame_ids)
        if self.color_kind not in ("scalar", "rgb"):
            raise ValueError("color_kind must be 'scalar' or 'rgb'")
        coordinates = _readonly(self.coordinates, dtype=float, name="coordinates")
        if coordinates.ndim != 2 or coordinates.shape[0] != count or coordinates.shape[1] not in (2, 3):
            raise ValueError("coordinates must have shape (n, 2) or (n, 3)")
        if len(self.axis_labels) != coordinates.shape[1]:
            raise ValueError("axis_labels must align with coordinate columns")
        patterns = _readonly(self.pattern_indices, dtype=int, shape=(count,), name="pattern_indices")
        colors = _readonly(self.colors, name="colors")
        expected = (count,) if self.color_kind == "scalar" else (count, 3)
        if colors.shape != expected:
            raise ValueError(f"colors must have shape {expected}")
        indexed = np.ones(count, dtype=bool) if self.indexed is None else self.indexed
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "pattern_indices", patterns)
        object.__setattr__(self, "colors", colors)
        object.__setattr__(self, "indexed", _readonly(indexed, dtype=bool, shape=(count,), name="indexed"))
        object.__setattr__(self, "frame_ids", tuple(self.frame_ids))
        object.__setattr__(self, "axis_labels", tuple(self.axis_labels))


@dataclass(frozen=True)
class PoleFigureData:
    """Prepared stereographic pole positions and pattern identities."""

    points: np.ndarray
    frame_ids: tuple[FrameId, ...]
    pattern_indices: np.ndarray
    colors: np.ndarray
    hkl: tuple[int, int, int]
    color_kind: Literal["rgb", "uniform"]
    color_radius: float | None = None
    center: tuple[float, float] = (0.0, 0.0)

    def __post_init__(self):
        count = len(self.frame_ids)
        if self.color_kind not in ("rgb", "uniform"):
            raise ValueError("color_kind must be 'rgb' or 'uniform'")
        if len(self.hkl) != 3 or tuple(self.hkl) == (0, 0, 0):
            raise ValueError("hkl must contain three integers and cannot be (0, 0, 0)")
        if len(self.center) != 2 or not np.isfinite(self.center).all():
            raise ValueError("center must contain two finite values")
        object.__setattr__(self, "points", _readonly(self.points, dtype=float, shape=(count, 2), name="points"))
        object.__setattr__(self, "pattern_indices", _readonly(self.pattern_indices, dtype=int, shape=(count,), name="pattern_indices"))
        colors = _readonly(self.colors, dtype=float, name="colors")
        if colors.shape not in ((3,), (count, 3)):
            raise ValueError("colors must have shape (3,) or (n, 3)")
        object.__setattr__(self, "colors", colors)
        object.__setattr__(self, "frame_ids", tuple(self.frame_ids))


@dataclass(frozen=True)
class DetectorPatternData:
    """Prepared indexed reflections for one pattern."""

    pattern_index: int
    hkl: np.ndarray
    predicted_xy: np.ndarray
    measured_peak_indices: np.ndarray

    def __post_init__(self):
        count = len(self.measured_peak_indices)
        object.__setattr__(self, "hkl", _readonly(self.hkl, dtype=int, shape=(count, 3), name="hkl"))
        object.__setattr__(self, "predicted_xy", _readonly(self.predicted_xy, dtype=float, shape=(count, 2), name="predicted_xy"))
        object.__setattr__(self, "measured_peak_indices", _readonly(self.measured_peak_indices, dtype=int, shape=(count,), name="measured_peak_indices"))


@dataclass(frozen=True)
class DetectorSimulationData:
    """Prepared missing simulated reflections for one indexed pattern.

    Parameters
    ----------
    pattern_index
        Nonnegative frame-local pattern rank.
    hkl
        Integer Miller indices with shape ``(n, 3)``.
    predicted_xy
        Zero-based frame pixel ``(x, y)`` coordinates with shape ``(n, 2)``.
        The coordinates include the frame ROI and detector-pixel grouping.
    energy_kev
        Photon energies in keV with shape ``(n,)``.
    relative_intensity
        Uncalibrated relative intensities with shape ``(n,)``.

    Notes
    -----
    Construction copies, validates, and marks every array read-only. A row at
    the same index across all arrays describes one simulated reflection.
    """

    pattern_index: int
    hkl: np.ndarray
    predicted_xy: np.ndarray
    energy_kev: np.ndarray
    relative_intensity: np.ndarray

    def __post_init__(self):
        if (
            isinstance(self.pattern_index, (bool, np.bool_))
            or not isinstance(self.pattern_index, (int, np.integer))
            or self.pattern_index < 0
        ):
            raise ValueError("pattern_index must be a nonnegative integer")
        hkl_input = np.asarray(self.hkl)
        if hkl_input.dtype.kind not in "iu":
            raise TypeError("hkl must have an integer dtype")
        if hkl_input.ndim != 2 or hkl_input.shape[1:] != (3,):
            raise ValueError("hkl must have shape (n, 3)")
        count = len(hkl_input)
        hkl = _readonly(hkl_input, dtype=int, shape=(count, 3), name="hkl")
        predicted = _readonly(
            self.predicted_xy,
            dtype=float,
            shape=(count, 2),
            name="predicted_xy",
        )
        energy = _readonly(
            self.energy_kev, dtype=float, shape=(count,), name="energy_kev"
        )
        intensity = _readonly(
            self.relative_intensity,
            dtype=float,
            shape=(count,),
            name="relative_intensity",
        )
        for name, array in (
            ("predicted_xy", predicted),
            ("energy_kev", energy),
            ("relative_intensity", intensity),
        ):
            if not np.isfinite(array).all():
                raise ValueError(f"{name} must contain only finite values")
        object.__setattr__(self, "pattern_index", int(self.pattern_index))
        object.__setattr__(self, "hkl", hkl)
        object.__setattr__(self, "predicted_xy", predicted)
        object.__setattr__(self, "energy_kev", energy)
        object.__setattr__(self, "relative_intensity", intensity)


@dataclass(frozen=True)
class DetectorViewData:
    """Prepared detector image, peaks, and indexed-reflection overlays."""

    frame_id: FrameId
    detector_id: str
    extent: tuple[float, float]
    measured_xy: np.ndarray
    measured_peak_indices: np.ndarray
    measured_intensity: np.ndarray
    measured_indexed: np.ndarray
    patterns: tuple[DetectorPatternData, ...]
    image: np.ndarray | None = None
    simulations: tuple[DetectorSimulationData, ...] = ()

    def __post_init__(self):
        count = len(self.measured_xy)
        object.__setattr__(self, "measured_xy", _readonly(self.measured_xy, dtype=float, shape=(count, 2), name="measured_xy"))
        object.__setattr__(self, "measured_peak_indices", _readonly(self.measured_peak_indices, dtype=int, shape=(count,), name="measured_peak_indices"))
        object.__setattr__(self, "measured_intensity", _readonly(self.measured_intensity, dtype=float, shape=(count,), name="measured_intensity"))
        object.__setattr__(self, "measured_indexed", _readonly(self.measured_indexed, dtype=bool, shape=(count,), name="measured_indexed"))
        object.__setattr__(self, "patterns", tuple(self.patterns))
        simulations = tuple(self.simulations)
        if any(not isinstance(value, DetectorSimulationData) for value in simulations):
            raise TypeError("simulations must contain DetectorSimulationData values")
        object.__setattr__(self, "simulations", simulations)
        if self.image is not None:
            image = _readonly(self.image, name="image")
            if image.ndim != 2:
                raise ValueError("detector image must be two-dimensional")
            object.__setattr__(self, "image", image)


def _dataset(source):
    if isinstance(source, ResultSet):
        return source.to_visualization()
    if isinstance(source, VisualizationDataset):
        return source
    raise TypeError("source must be a ResultSet or VisualizationDataset")


def _pattern_rows(dataset, scope):
    return np.flatnonzero((scope or DataScope()).pattern_mask(dataset))


def _frame_axis(dataset, name):
    positions = dataset.sample_positions
    labels = dict((choice.value, choice.label) for choice in AXIS_OPTIONS)
    if name not in labels:
        raise ValueError(f"unknown axis {name!r}; choose from {_AXIS_VALUES}")
    if name in "XYZ" and len(name) == 1:
        values = positions[:, "XYZ".index(name)]
    elif name == "depth":
        values = dataset.depths
    elif name in ("H", "F"):
        sign = 1 if name == "H" else -1
        values = (sign * positions[:, 1] + positions[:, 2]) / np.sqrt(2.0)
    else:
        lab = -positions.copy()
        lab[:, 2] += dataset.depths
        if name in ("Xlab", "Ylab", "Zlab"):
            values = lab[:, ("Xlab", "Ylab", "Zlab").index(name)]
        else:
            sign = 1 if name == "Hlab" else -1
            values = (sign * lab[:, 1] + lab[:, 2]) / np.sqrt(2.0)
    return values, labels[name]


def _aligned_values(values, dataset, rows, alignment, name):
    values = values(dataset) if callable(values) else values
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} values must be one-dimensional")
    if alignment == "frame":
        if len(array) != dataset.n_frames:
            raise ValueError(f"{name} values must contain one value per frame")
        return array[dataset.pattern_frame_indices[rows]]
    if alignment == "pattern":
        if len(array) != dataset.n_patterns:
            raise ValueError(f"{name} values must contain one value per pattern")
        return array[rows]
    if len(array) != len(rows):
        raise ValueError(f"{name} values must contain one value per selected pattern")
    return array


def _resolve_axis(axis, dataset, rows):
    if isinstance(axis, str):
        values, label = _frame_axis(dataset, axis)
        return values[dataset.pattern_frame_indices[rows]], label
    if not isinstance(axis, Axis):
        raise TypeError("axes must contain names or Axis objects")
    values = _aligned_values(axis.values, dataset, rows, axis.alignment, "axis")
    label = f"{axis.label} ({axis.unit})" if axis.unit else axis.label
    return values, label


def _finite_rows(rotations, function):
    """Apply ``function`` to the finite ``(n, 3, 3)`` rows; other rows give NaN vectors."""
    rotations = np.asarray(rotations, dtype=float).reshape((-1, 3, 3))
    vectors = np.full((len(rotations), 3), np.nan)
    finite = np.isfinite(rotations).all(axis=(1, 2))
    if np.any(finite):
        vectors[finite] = function(rotations[finite])
    return vectors


def _crystal_directions(rotations, normal):
    rotations = np.asarray(rotations, dtype=float).reshape((-1, 3, 3))
    directions = np.full((len(rotations), 3), np.nan)
    finite = np.isfinite(rotations).all(axis=(1, 2))
    if not np.any(finite):
        return directions
    try:
        directions[finite] = crystal_direction(rotations[finite], normal)
    except np.linalg.LinAlgError:
        # A singular matrix somewhere in the stack: keep NaN for those rows
        # and solve the rest one at a time.
        for index in np.flatnonzero(finite):
            try:
                directions[index] = crystal_direction(rotations[index], normal)
            except np.linalg.LinAlgError:
                pass
    return directions


def _map_colors(
    color,
    dataset,
    rows,
    surface,
    misorientation_reference,
    pole_hkl,
    pole_center,
    pole_color_radius_deg,
):
    scalar_fields = {
        "n_indexed": (dataset.pattern_n_indexed, "Indexed peaks"),
        "goodness": (dataset.pattern_goodness, "Goodness"),
        "rms_error": (dataset.pattern_rms_error_deg, "RMS error (deg)"),
        "n_patterns": (
            np.bincount(dataset.pattern_frame_indices, minlength=dataset.n_frames),
            "Patterns",
        ),
    }
    if not isinstance(color, (str, ScalarColor)):
        color = ScalarColor(color)
    if isinstance(color, ScalarColor):
        if isinstance(color.values, str):
            if color.values not in scalar_fields:
                raise ValueError(f"unknown scalar color {color.values!r}; choose from {tuple(scalar_fields)}")
            values, default_label = scalar_fields[color.values]
            alignment = "frame" if color.values == "n_patterns" else "pattern"
            values = _aligned_values(values, dataset, rows, alignment, "color")
        else:
            values = _aligned_values(color.values, dataset, rows, color.alignment, "color")
            default_label = "Value"
        return values.astype(float), "scalar", color.label or default_label, color.palette, color.limits
    if color in scalar_fields:
        values, label = scalar_fields[color]
        alignment = "frame" if color == "n_patterns" else "pattern"
        return _aligned_values(values, dataset, rows, alignment, "color").astype(float), "scalar", label, "Viridis", None
    rotations = dataset.pattern_rotations[rows]
    if color == "cubic_ipf":
        if dataset.crystal is None:
            raise ValueError("crystal context is required for cubic IPF coloring")
        if dataset.crystal.crystal_system != "cubic":
            raise ValueError("cubic IPF coloring requires a cubic crystal")
        directions = _crystal_directions(rotations, surface.normal)
        return cubic_ipf_colors(directions), "rgb", "Cubic IPF", None, None
    if color == "rodrigues":
        operations = (
            symmetry_operations(dataset.crystal.space_group)
            if dataset.crystal and dataset.crystal.crystal_system in ("cubic", "hexagonal")
            else None
        )
        vectors = _finite_rows(
            rotations,
            lambda finite: orientation_to_rodrigues(
                symmetry_reduce_orientation(finite, operations=operations)
            ),
        )
        return rodrigues_colors(vectors), "rgb", "Rodrigues RGB", None, None
    if color == "misorientation":
        if misorientation_reference is None:
            raise ValueError("misorientation_reference is required for misorientation coloring")
        try:
            frame_id, pattern_index = misorientation_reference
            reference_row = next(
                row
                for row in range(dataset.n_patterns)
                if dataset.frame_ids[dataset.pattern_frame_indices[row]] == frame_id
                and dataset.pattern_indices[row] == pattern_index
            )
        except (TypeError, ValueError, StopIteration) as error:
            raise ValueError("misorientation_reference must identify a pattern") from error
        reference = dataset.pattern_rotations[reference_row]
        operations = (
            symmetry_operations(dataset.crystal.space_group)
            if dataset.crystal and dataset.crystal.crystal_system in ("cubic", "hexagonal")
            else None
        )
        vectors = _finite_rows(
            rotations,
            lambda finite: orientation_to_rodrigues(
                misorientation_matrix(finite, reference, operations=operations)
            ),
        )
        return rodrigues_colors(vectors), "rgb", "Misorientation", None, None
    if color == "pole_hsv":
        if dataset.crystal is None or dataset.crystal.crystal_system != "cubic":
            raise ValueError("cubic crystal context is required for pole HSV coloring")
        points, local_rows = pole_figure_points(
            dataset.pattern_reciprocals[rows], cubic_hkl_family(pole_hkl), surface=surface
        )
        finite = np.isfinite(points).all(axis=1)
        points, local_rows = points[finite], local_rows[finite]
        radius = pole_color_radius(pole_center, pole_color_radius_deg)
        colors = closest_pole_colors(
            points, local_rows, len(rows), center=pole_center, radius=radius
        )
        return colors, "rgb", "Pole Figure HSV", None, None
    raise ValueError(f"unknown color mode {color!r}; choose from {_COLOR_VALUES}")


def prepare_map(
    source,
    *,
    axes=("X", "Y"),
    color="n_indexed",
    scope=None,
    surface=None,
    misorientation_reference=None,
    pole_hkl=(1, 0, 0),
    pole_center=(0.0, 0.0),
    pole_color_radius_deg=22.5,
):
    """Prepare a two- or three-dimensional spatial map.

    ``scope=None`` uses ``DataScope(patterns="best", min_indexed=3)``. This
    differs from detector-view preparation, whose ``patterns`` argument
    defaults to ``"all"``.
    """
    dataset = _dataset(source)
    if len(axes) not in (2, 3):
        raise ValueError("axes must contain two or three entries")
    rows = _pattern_rows(dataset, scope)
    surface = SurfaceFrame.aps_34ide(surface or "normal") if isinstance(surface, (str, type(None))) else surface
    if not isinstance(surface, SurfaceFrame):
        raise TypeError("surface must be a SurfaceFrame, a preset name, or None")
    resolved = [_resolve_axis(axis, dataset, rows) for axis in axes]
    coordinates = np.column_stack([item[0] for item in resolved]) if rows.size else np.empty((0, len(axes)))
    if len(coordinates) and not np.isfinite(coordinates).all():
        invalid = ~np.isfinite(coordinates).all(axis=1)
        frame_ids = tuple(dict.fromkeys(
            dataset.frame_ids[index]
            for index in dataset.pattern_frame_indices[rows[invalid]]
        ))
        shown = frame_ids[:5]
        suffix = ", ..." if len(frame_ids) > len(shown) else ""
        names = tuple(axis if isinstance(axis, str) else axis.label for axis in axes)
        raise ValueError(
            f"map axes {names} contain missing or non-finite coordinates for frames "
            f"{shown}{suffix}"
        )
    colors, kind, label, palette, limits = _map_colors(
        color,
        dataset,
        rows,
        surface,
        misorientation_reference,
        pole_hkl,
        pole_center,
        pole_color_radius_deg,
    )
    frame_ids = tuple(dataset.frame_ids[index] for index in dataset.pattern_frame_indices[rows])
    indexed = np.isfinite(dataset.pattern_rotations[rows]).all(axis=(1, 2))
    return MapData(
        coordinates,
        tuple(item[1] for item in resolved),
        frame_ids,
        dataset.pattern_indices[rows],
        colors,
        kind,
        label,
        palette,
        limits,
        indexed,
        all(isinstance(axis, str) for axis in axes),
    )


def prepare_pole_figure(
    source,
    *,
    hkl=(1, 0, 0),
    scope=None,
    surface=None,
    color="hsv_position",
    pole_center=(0.0, 0.0),
    pole_color_radius_deg=22.5,
):
    """Prepare stereographic pole positions for selected patterns.

    ``scope=None`` uses ``DataScope(patterns="best", min_indexed=3)``. This
    differs from detector-view preparation, whose ``patterns`` argument
    defaults to ``"all"``.
    """
    dataset = _dataset(source)
    if dataset.crystal is None:
        raise ValueError("crystal context is required for a cubic pole figure")
    if dataset.crystal.crystal_system != "cubic":
        raise ValueError("cubic HKL families require a cubic crystal")
    rows = _pattern_rows(dataset, scope)
    surface = SurfaceFrame.aps_34ide(surface or "normal") if isinstance(surface, (str, type(None))) else surface
    if not isinstance(surface, SurfaceFrame):
        raise TypeError("surface must be a SurfaceFrame, a preset name, or None")
    points, local_rows = pole_figure_points(
        dataset.pattern_reciprocals[rows], cubic_hkl_family(hkl), surface=surface
    )
    finite = np.isfinite(points).all(axis=1)
    points, local_rows = points[finite], local_rows[finite]
    selected_rows = rows[local_rows]
    if color == "hsv_position":
        radius = pole_color_radius(pole_center, pole_color_radius_deg)
        pattern_colors = closest_pole_colors(
            points, local_rows, len(rows), center=pole_center, radius=radius
        )
        colors = pattern_colors[local_rows]
        color_kind = "rgb"
    elif color == "ipf":
        rotations = dataset.pattern_rotations[rows]
        colors = cubic_ipf_colors(_crystal_directions(rotations, surface.normal))[local_rows]
        radius = None
        color_kind = "rgb"
    elif color == "uniform":
        colors = np.array([214 / 255, 20 / 255, 0.0])
        radius = None
        color_kind = "uniform"
    else:
        raise ValueError(f"unknown pole color mode {color!r}; choose from {_POLE_COLOR_VALUES}")
    return PoleFigureData(
        points=points,
        frame_ids=tuple(dataset.frame_ids[index] for index in dataset.pattern_frame_indices[selected_rows]),
        pattern_indices=dataset.pattern_indices[selected_rows],
        colors=colors,
        hkl=tuple(int(value) for value in hkl),
        color_kind=color_kind,
        color_radius=radius,
        center=tuple(float(value) for value in pole_center),
    )


def _frame_index(dataset, frame_id):
    try:
        return dataset.frame_ids.index(frame_id)
    except ValueError as error:
        raise KeyError(f"unknown frame_id {frame_id!r}") from error


def _load_image(image):
    if isinstance(image, np.ndarray):
        return image
    path = Path(image)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    return read_h5_frame(path)[0]


def prepare_detector_view(
    source,
    *,
    frame_id,
    patterns="all",
    image=None,
    detector_index=None,
    simulation_energy_range_kev=None,
):
    """Prepare measured, indexed, and optional simulated detector overlays.

    ``patterns="all"`` includes every indexed pattern in the selected frame.
    Map and pole-figure preparation instead use the default ``DataScope``,
    which selects the best pattern with at least three assignments.
    """
    dataset = _dataset(source)
    frame_index = _frame_index(dataset, frame_id)
    if dataset.geometry is None:
        raise ValueError("detector geometry is required to prepare a detector view")
    if detector_index is None:
        detector_id = dataset.detector_ids[frame_index]
        if detector_id:
            detector_index = dataset.geometry.find_detector(detector_id)
            if detector_index < 0:
                raise ValueError(f"detector {detector_id!r} is not present in the geometry")
        else:
            detector_index = 0
    detector = dataset.geometry.detector(detector_index)
    if simulation_energy_range_kev is not None and dataset.crystal is None:
        raise ValueError("crystal context is required to simulate detector reflections")

    peak_rows = np.flatnonzero(dataset.peak_frame_indices == frame_index)
    measured = np.column_stack([
        dataset.peaks["fit_x"][peak_rows], dataset.peaks["fit_y"][peak_rows]
    ])
    intensity = dataset.peaks["intens"][peak_rows]
    pattern_rows = np.flatnonzero(dataset.pattern_frame_indices == frame_index)
    if patterns == "best" and len(pattern_rows):
        pattern_rows = pattern_rows[[np.argmin(dataset.pattern_indices[pattern_rows])]]
    elif patterns != "all":
        if isinstance(patterns, str):
            detail = "; 'all_frames' is available only through DataScope" if patterns == "all_frames" else ""
            raise ValueError(f"patterns must be 'best', 'all', or a sequence of ranks{detail}")
        requested = tuple(patterns)
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, Integral)
            or value < 0
            for value in requested
        ):
            raise ValueError("pattern ranks must be nonnegative integers")
        requested = tuple(int(value) for value in requested)
        pattern_rows = pattern_rows[np.isin(dataset.pattern_indices[pattern_rows], requested)]

    indexed_mask = np.zeros(len(peak_rows), dtype=bool)
    prepared_patterns = []
    prepared_simulations = []
    depth = None if np.isnan(dataset.depths[frame_index]) else dataset.depths[frame_index]
    start = dataset.starts[frame_index]
    group = dataset.groups[frame_index]
    for pattern_row in pattern_rows:
        assignment_rows = np.flatnonzero(dataset.assignment_pattern_rows == pattern_row)
        peak_indices = dataset.assignment_peak_indices[assignment_rows]
        indexed_mask[peak_indices] = True
        q = dataset.assignment_hkl[assignment_rows] @ dataset.pattern_reciprocals[pattern_row]
        full_xy = detector.q_to_pixel(q, depth=depth)
        roi_xy = detector_to_roi_pixels(full_xy, start, group)
        prepared_patterns.append(DetectorPatternData(
            int(dataset.pattern_indices[pattern_row]),
            dataset.assignment_hkl[assignment_rows],
            roi_xy,
            peak_indices,
        ))
        if simulation_energy_range_kev is not None:
            simulated = simulate_reflections(
                dataset.crystal,
                dataset.pattern_reciprocals[pattern_row],
                detector,
                energy_range_kev=simulation_energy_range_kev,
                depth=0.0 if depth is None else float(depth),
            ).missing_from(dataset.assignment_hkl[assignment_rows])
            simulated_xy = detector_to_roi_pixels(simulated.detector_xy, start, group)
            width, height = dataset.image_shapes[frame_index][::-1]
            valid = (
                np.isfinite(simulated_xy).all(axis=1)
                & (simulated_xy[:, 0] >= 0)
                & (simulated_xy[:, 0] <= width - 1)
                & (simulated_xy[:, 1] >= 0)
                & (simulated_xy[:, 1] <= height - 1)
            )
            prepared_simulations.append(DetectorSimulationData(
                pattern_index=int(dataset.pattern_indices[pattern_row]),
                hkl=simulated.hkl[valid],
                predicted_xy=simulated_xy[valid],
                energy_kev=simulated.energy_kev[valid],
                relative_intensity=simulated.relative_intensity[valid],
            ))

    if image is True:
        retained = dataset.images[frame_index]
        source_path = dataset.input_images[frame_index]
        if retained is not None:
            image_data = retained
        elif source_path:
            image_data = _load_image(source_path)
        else:
            raise ValueError(f"frame {frame_id!r} has no retained image or source image path")
    elif image is None or image is False:
        image_data = None
    else:
        image_data = _load_image(image)
    if image_data is not None and np.asarray(image_data).ndim != 2:
        raise ValueError("detector image must be two-dimensional")

    extent = tuple((dataset.image_shapes[frame_index][::-1]).astype(float))
    return DetectorViewData(
        frame_id=frame_id,
        detector_id=detector.detector_id,
        extent=extent,
        measured_xy=measured,
        measured_peak_indices=dataset.peak_indices[peak_rows],
        measured_intensity=intensity,
        measured_indexed=indexed_mask,
        patterns=tuple(prepared_patterns),
        image=image_data,
        simulations=tuple(prepared_simulations),
    )
