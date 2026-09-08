# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Plotly renderers for prepared Laue visualization data."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

import numpy as np
import plotly.graph_objects as go

from .preparation import (
    DetectorViewData,
    MapData,
    PoleFigureData,
    prepare_detector_view,
    prepare_map,
    prepare_pole_figure,
)

_BACKGROUND = "rgb(245, 245, 245)"


class _PreparationDefault:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return repr(self.value)


_AXES_DEFAULT = _PreparationDefault(("X", "Y"))
_COLOR_DEFAULT = _PreparationDefault("n_indexed")
_NONE_DEFAULT = _PreparationDefault(None)
_HKL_DEFAULT = _PreparationDefault((1, 0, 0))
_CENTER_DEFAULT = _PreparationDefault((0.0, 0.0))
_RADIUS_DEFAULT = _PreparationDefault(22.5)
_PATTERNS_DEFAULT = _PreparationDefault("all")


def _prepared_data(source, data_type, function_name, preparation):
    supplied = tuple(
        name for name, value in preparation
        if not isinstance(value, _PreparationDefault)
    )
    if isinstance(source, data_type):
        if supplied:
            raise TypeError(
                f"{function_name} preparation keywords are invalid for prepared source: "
                f"{', '.join(supplied)}"
            )
        return source
    return None


def _value_or_default(value):
    return value.value if isinstance(value, _PreparationDefault) else value


_MISSING_COLOR = "#969696"


def _rgb(values):
    """Return one ``#rrggbb`` string per row; non-finite rows become gray.

    Hex strings are shorter than ``rgb(r,g,b)`` and cheaper for Plotly to
    validate, which matters for traces with 10^5 per-point colors.
    """
    array = np.asarray(values, dtype=float)
    if array.shape == (3,):
        array = array.reshape(1, 3)
    finite = np.isfinite(array).all(axis=1)
    quantized = np.zeros(array.shape, dtype=int)
    quantized[finite] = np.clip(np.rint(array[finite] * 255), 0, 255).astype(int)
    colors = [f"#{red:02x}{green:02x}{blue:02x}" for red, green, blue in quantized.tolist()]
    for index in np.flatnonzero(~finite):
        colors[index] = _MISSING_COLOR
    return colors


def _customdata(frame_ids, pattern_indices=None, peak_indices=None, *, array=True):
    """Return ``[frame_id, pattern_index, peak_index]`` identity rows.

    With integer frame IDs and ``array=True`` the rows form one floating-point
    array (``float32`` when every value is below 2**24, else ``float64``),
    which Plotly serializes as binary and the browser reads as numbers; a
    missing identity is ``NaN`` there and ``None`` in the list form. String
    frame IDs always use the list form.
    """
    count = len(frame_ids)
    if any(isinstance(value, Integral) and abs(int(value)) > 2**53 - 1 for value in frame_ids):
        raise ValueError("Plotly integer frame IDs must be within ±(2**53 - 1); use string frame IDs for larger values")
    patterns = [None] * count if pattern_indices is None else pattern_indices
    peaks = [None] * count if peak_indices is None else peak_indices
    numeric = array and count and all(
        isinstance(value, Integral) and not isinstance(value, (bool, np.bool_))
        for value in frame_ids
    )
    if numeric:
        rows = np.full((count, 3), np.nan)
        rows[:, 0] = frame_ids
        if pattern_indices is not None:
            rows[:, 1] = pattern_indices
        if peak_indices is not None:
            rows[:, 2] = peak_indices
        # float32 is exact for integers below 2**24 and halves the payload.
        if np.nanmax(np.abs(rows)) < 2**24:
            rows = rows.astype(np.float32)
        return rows
    return [
        [frame_id, None if pattern is None else int(pattern), None if peak is None else int(peak)]
        for frame_id, pattern, peak in zip(frame_ids, patterns, peaks, strict=True)
    ]


def _apply_updates(figure, roles, allowed_roles, trace_update, layout_update):
    if trace_update is not None:
        if not isinstance(trace_update, Mapping):
            raise TypeError("trace_update must be a mapping or None")
        unknown = set(trace_update) - set(allowed_roles)
        if unknown:
            raise ValueError(
                f"unknown trace roles {tuple(sorted(unknown))}; choose from {tuple(allowed_roles)}"
            )
        for role, update in trace_update.items():
            if not isinstance(update, Mapping):
                raise TypeError(f"trace update for role {role!r} must be a mapping")
            for index in roles.get(role, ()):
                figure.data[index].update(update)
    if layout_update is not None:
        if not isinstance(layout_update, Mapping):
            raise TypeError("layout_update must be a mapping or None")
        figure.update_layout(layout_update)
    return figure


def _add_empty_annotation(figure, text):
    figure.add_annotation(
        text=text,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"color": "rgb(100,100,100)", "size": 14},
    )


def plot_map(
    source,
    *,
    axes=_AXES_DEFAULT,
    color=_COLOR_DEFAULT,
    scope=_NONE_DEFAULT,
    surface=_NONE_DEFAULT,
    misorientation_reference=_NONE_DEFAULT,
    pole_hkl=_HKL_DEFAULT,
    pole_center=_CENTER_DEFAULT,
    pole_color_radius_deg=_RADIUS_DEFAULT,
    marker_size=10,
    layout_update=None,
    trace_update=None,
):
    """Render a two- or three-dimensional spatial map with Plotly.

    ``source`` can be a :class:`MapData`, :class:`ResultSet`, or
    :class:`VisualizationDataset`. Semantic trace roles are ``"data"`` and,
    for orientation-color maps only, ``"unindexed"``; scalar colors render as
    a single ``"data"`` trace.
    """
    preparation = (
        ("axes", axes),
        ("color", color),
        ("scope", scope),
        ("surface", surface),
        ("misorientation_reference", misorientation_reference),
        ("pole_hkl", pole_hkl),
        ("pole_center", pole_center),
        ("pole_color_radius_deg", pole_color_radius_deg),
    )
    data = _prepared_data(source, MapData, "plot_map", preparation)
    if data is None:
        data = prepare_map(
            source,
            **{name: _value_or_default(value) for name, value in preparation},
        )
    if isinstance(marker_size, bool) or not np.isfinite(marker_size) or marker_size <= 0:
        raise ValueError("marker_size must be positive and finite")

    figure = go.Figure()
    roles = {"data": [], "unindexed": []}
    dimensions = data.coordinates.shape[1]
    masks = (
        (("data", np.ones(len(data.frame_ids), dtype=bool)),)
        if data.color_kind == "scalar"
        else (("data", data.indexed), ("unindexed", ~data.indexed))
    )
    for role, mask in masks:
        if not np.any(mask):
            continue
        marker = {"size": marker_size, "line": {"width": 0}}
        if role == "unindexed":
            marker["color"] = "rgb(150,150,150)"
        elif data.color_kind == "rgb":
            marker["color"] = _rgb(data.colors[mask])
        else:
            marker.update({
                "color": data.colors[mask],
                "colorscale": data.palette,
                "colorbar": {"title": {"text": data.color_label}},
                "showscale": True,
            })
            if data.color_limits is not None:
                marker["cmin"], marker["cmax"] = data.color_limits
        customdata = _customdata(
            [data.frame_ids[index] for index in np.flatnonzero(mask)],
            data.pattern_indices[mask],
        )
        common = {
            "mode": "markers",
            "name": "Data" if role == "data" else "Unindexed",
            "marker": marker,
            "customdata": customdata,
            "meta": {"role": role},
            "uid": f"map-{role}",
            "hovertemplate": (
                "frame: %{customdata[0]}<br>pattern: %{customdata[1]}<br>"
                + (f"{data.color_label}: %{{marker.color:.4g}}<br>" if data.color_kind == "scalar" and role == "data" else "")
                + "<extra></extra>"
            ),
        }
        coordinates = data.coordinates[mask]
        trace = (
            go.Scattergl(x=coordinates[:, 0], y=coordinates[:, 1], **common)
            if dimensions == 2
            else go.Scatter3d(
                x=coordinates[:, 0],
                y=coordinates[:, 1],
                z=coordinates[:, 2],
                **common,
            )
        )
        figure.add_trace(trace)
        roles[role].append(len(figure.data) - 1)

    if dimensions == 2:
        yaxis = {"title": {"text": data.axis_labels[1]}}
        if data.spatial_axes:
            yaxis.update({"scaleanchor": "x", "scaleratio": 1})
        figure.update_layout(
            xaxis={"title": {"text": data.axis_labels[0]}},
            yaxis=yaxis,
            plot_bgcolor=_BACKGROUND,
            margin={"l": 60, "r": 30, "t": 35, "b": 55},
            dragmode="lasso",
            uirevision="map-2d",
        )
    else:
        figure.update_layout(
            scene={
                "xaxis": {"title": {"text": data.axis_labels[0]}},
                "yaxis": {"title": {"text": data.axis_labels[1]}},
                "zaxis": {"title": {"text": data.axis_labels[2]}},
                "aspectmode": "data",
                "bgcolor": _BACKGROUND,
            },
            margin={"l": 0, "r": 0, "t": 35, "b": 0},
            uirevision="map-3d",
        )
    if not len(data.coordinates):
        _add_empty_annotation(figure, "No patterns match the selected scope.")
    return _apply_updates(
        figure, roles, ("data", "unindexed"), trace_update, layout_update
    )


def plot_pole_figure(
    source,
    *,
    hkl=_HKL_DEFAULT,
    scope=_NONE_DEFAULT,
    surface=_NONE_DEFAULT,
    color=_PreparationDefault("hsv_position"),
    pole_center=_CENTER_DEFAULT,
    pole_color_radius_deg=_RADIUS_DEFAULT,
    marker_size=7,
    hover_point_limit=100_000,
    layout_update=None,
    trace_update=None,
):
    """Render an upper-hemisphere stereographic pole figure with Plotly.

    Semantic trace roles are ``"data"``, ``"boundary"``, and ``"reference"``.
    """
    preparation = (
        ("hkl", hkl),
        ("scope", scope),
        ("surface", surface),
        ("color", color),
        ("pole_center", pole_center),
        ("pole_color_radius_deg", pole_color_radius_deg),
    )
    data = _prepared_data(source, PoleFigureData, "plot_pole_figure", preparation)
    if data is None:
        data = prepare_pole_figure(
            source,
            **{name: _value_or_default(value) for name, value in preparation},
        )
    if isinstance(marker_size, bool) or not np.isfinite(marker_size) or marker_size <= 0:
        raise ValueError("marker_size must be positive and finite")
    if hover_point_limit is not None and (
        isinstance(hover_point_limit, bool)
        or not isinstance(hover_point_limit, int)
        or hover_point_limit < 0
    ):
        raise ValueError("hover_point_limit must be a nonnegative integer or None")

    figure = go.Figure()
    roles = {"data": [], "boundary": [], "reference": []}
    hover_disabled = hover_point_limit is not None and len(data.points) > hover_point_limit
    if len(data.points):
        marker_color = _rgb(data.colors)
        if data.color_kind == "uniform":
            marker_color = marker_color[0]
        figure.add_trace(go.Scattergl(
            x=data.points[:, 0],
            y=data.points[:, 1],
            mode="markers",
            marker={"size": marker_size, "color": marker_color, "line": {"width": 0}},
            customdata=_customdata(data.frame_ids, data.pattern_indices),
            name=f"{{{''.join(str(value) for value in data.hkl)}}}",
            hoverinfo="skip" if hover_disabled else None,
            hovertemplate=None if hover_disabled else (
                "frame: %{customdata[0]}<br>pattern: %{customdata[1]}<br>"
                "x: %{x:.4f}<br>y: %{y:.4f}<extra></extra>"
            ),
            meta={"role": "data"},
            uid="pole-data",
        ))
        roles["data"].append(len(figure.data) - 1)

    theta = np.linspace(0, 2 * np.pi, 241)
    figure.add_trace(go.Scattergl(
        x=np.cos(theta),
        y=np.sin(theta),
        mode="lines",
        line={"color": "black", "width": 1},
        showlegend=False,
        hoverinfo="skip",
        meta={"role": "boundary"},
        uid="pole-boundary",
    ))
    roles["boundary"].append(len(figure.data) - 1)

    if data.color_radius is not None:
        figure.add_trace(go.Scattergl(
            x=data.center[0] + data.color_radius * np.cos(theta),
            y=data.center[1] + data.color_radius * np.sin(theta),
            mode="lines",
            line={"color": "black", "width": 1, "dash": "dot"},
            showlegend=False,
            hoverinfo="skip",
            meta={"role": "reference"},
            uid="pole-reference-radius",
        ))
        roles["reference"].append(len(figure.data) - 1)
    figure.add_trace(go.Scattergl(
        x=[data.center[0]],
        y=[data.center[1]],
        mode="markers",
        marker={"symbol": "cross", "size": 8, "color": "black"},
        showlegend=False,
        hoverinfo="skip",
        meta={"role": "reference"},
        uid="pole-reference-center",
    ))
    roles["reference"].append(len(figure.data) - 1)

    figure.update_layout(
        xaxis={"range": [-1.1, 1.1], "scaleanchor": "y", "scaleratio": 1, "showgrid": False, "zeroline": False},
        yaxis={"range": [-1.1, 1.1], "showgrid": False, "zeroline": False},
        plot_bgcolor=_BACKGROUND,
        margin={"l": 40, "r": 40, "t": 40, "b": 40},
        dragmode="lasso",
        uirevision="pole-figure",
    )
    if not len(data.points):
        _add_empty_annotation(figure, "No poles match the selected scope.")
    elif hover_disabled:
        figure.add_annotation(
            text=f"Point hover disabled above {hover_point_limit:,} poles.",
            x=0.01,
            y=0.99,
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="top",
            showarrow=False,
        )
    return _apply_updates(
        figure, roles, ("data", "boundary", "reference"), trace_update, layout_update
    )


def plot_detector_view(
    source,
    *,
    frame_id=_NONE_DEFAULT,
    patterns=_PATTERNS_DEFAULT,
    image=_NONE_DEFAULT,
    detector_index=_NONE_DEFAULT,
    simulation_energy_range_kev=_NONE_DEFAULT,
    show_detected=True,
    show_indexed=True,
    show_simulated=True,
    show_hkl_labels=False,
    marker_size=8,
    image_colorscale="Gray",
    image_limits=None,
    image_opacity=1.0,
    layout_update=None,
    trace_update=None,
):
    """Render a detector image and reflection overlays with Plotly.

    Semantic trace roles are ``"image"``, ``"boundary"``, ``"detected"``,
    ``"indexed"``, and ``"simulated"``.
    """
    preparation = (
        ("frame_id", frame_id),
        ("patterns", patterns),
        ("image", image),
        ("detector_index", detector_index),
        ("simulation_energy_range_kev", simulation_energy_range_kev),
    )
    data = _prepared_data(source, DetectorViewData, "plot_detector_view", preparation)
    if data is None:
        frame_id = _value_or_default(frame_id)
        if frame_id is None:
            raise TypeError("frame_id is required when source is not DetectorViewData")
        data = prepare_detector_view(
            source,
            **{name: _value_or_default(value) for name, value in preparation},
        )
    if isinstance(marker_size, bool) or not np.isfinite(marker_size) or marker_size <= 0:
        raise ValueError("marker_size must be positive and finite")
    if not np.isfinite(image_opacity) or not 0 <= image_opacity <= 1:
        raise ValueError("image_opacity must be between 0 and 1")
    if image_limits is not None and (
        len(image_limits) != 2
        or not np.isfinite(image_limits).all()
        or image_limits[0] >= image_limits[1]
    ):
        raise ValueError("image_limits must contain two finite increasing values")

    figure = go.Figure()
    roles = {
        "image": [],
        "boundary": [],
        "detected": [],
        "indexed": [],
        "simulated": [],
    }
    if data.image is not None:
        heatmap = {
            "z": data.image,
            "colorscale": image_colorscale,
            "opacity": image_opacity,
            "colorbar": {"title": {"text": "Intensity"}},
            "name": "Detector image",
            "hovertemplate": "x: %{x}<br>y: %{y}<br>I: %{z:.4g}<extra></extra>",
            "meta": {"role": "image"},
            "uid": "detector-image",
        }
        if image_limits is not None:
            heatmap["zmin"], heatmap["zmax"] = image_limits
        figure.add_trace(go.Heatmap(**heatmap))
        roles["image"].append(len(figure.data) - 1)

    width, height = data.extent
    x_min, x_max = -0.5, width - 0.5
    y_min, y_max = -0.5, height - 0.5
    figure.add_trace(go.Scatter(
        x=[x_min, x_max, x_max, x_min, x_min],
        y=[y_min, y_min, y_max, y_max, y_min],
        mode="lines",
        line={"color": "rgb(120,120,120)", "width": 1},
        showlegend=False,
        hoverinfo="skip",
        meta={"role": "boundary"},
        uid="detector-boundary",
    ))
    roles["boundary"].append(len(figure.data) - 1)

    if show_detected and len(data.measured_xy):
        figure.add_trace(go.Scattergl(
            x=data.measured_xy[:, 0],
            y=data.measured_xy[:, 1],
            mode="markers",
            name=f"Detected ({len(data.measured_xy)})",
            marker={
                "size": marker_size,
                "symbol": "square-open",
                "color": "rgb(0,180,210)",
                "line": {"width": 1.5, "color": "rgb(0,180,210)"},
            },
            customdata=_customdata(
                [data.frame_id] * len(data.measured_xy),
                peak_indices=data.measured_peak_indices,
                array=False,
            ),
            hovertemplate=(
                "frame: %{customdata[0]}<br>peak: %{customdata[2]}<br>"
                "x: %{x:.2f} px<br>y: %{y:.2f} px<extra></extra>"
            ),
            meta={"role": "detected"},
            uid="detector-detected",
        ))
        roles["detected"].append(len(figure.data) - 1)

    if show_indexed:
        for pattern in data.patterns:
            finite = np.isfinite(pattern.predicted_xy).all(axis=1)
            if not np.any(finite):
                continue
            positions = pattern.predicted_xy[finite]
            hkl = pattern.hkl[finite]
            peaks = pattern.measured_peak_indices[finite]
            on_detector = (
                (positions[:, 0] >= 0)
                & (positions[:, 0] <= width - 1)
                & (positions[:, 1] >= 0)
                & (positions[:, 1] <= height - 1)
            )
            for mask, suffix, visible in (
                (on_detector, "on-detector", True),
                (~on_detector, "off-detector", "legendonly"),
            ):
                if not np.any(mask):
                    continue
                selected = positions[mask]
                selected_hkl = hkl[mask]
                stable = _customdata(
                    [data.frame_id] * len(selected),
                    [pattern.pattern_index] * len(selected),
                    peaks[mask],
                    array=False,
                )
                customdata = [
                    identity + [int(value) for value in reflection]
                    for identity, reflection in zip(stable, selected_hkl, strict=True)
                ]
                labels = [
                    f"({value[0]} {value[1]} {value[2]})" for value in selected_hkl
                ]
                figure.add_trace(go.Scattergl(
                    x=selected[:, 0],
                    y=selected[:, 1],
                    mode="markers+text" if show_hkl_labels else "markers",
                    text=labels if show_hkl_labels else None,
                    textposition="top right",
                    visible=visible,
                    name=f"Pattern {pattern.pattern_index} ({suffix})",
                    marker={
                        "size": marker_size + 4,
                        "symbol": "x-thin-open",
                        "color": "rgb(220,50,47)",
                        "line": {"width": 1.5, "color": "rgb(220,50,47)"},
                    },
                    customdata=customdata,
                    hovertemplate=(
                        "frame: %{customdata[0]}<br>pattern: %{customdata[1]}<br>"
                        "peak: %{customdata[2]}<br>hkl: (%{customdata[3]} "
                        "%{customdata[4]} %{customdata[5]})<br>"
                        "x: %{x:.2f} px<br>y: %{y:.2f} px<extra></extra>"
                    ),
                    meta={"role": "indexed"},
                    uid=f"detector-indexed-{pattern.pattern_index}-{suffix}",
                ))
                roles["indexed"].append(len(figure.data) - 1)

    if show_simulated:
        for simulation in data.simulations:
            if not len(simulation.predicted_xy):
                continue
            stable = _customdata(
                [data.frame_id] * len(simulation.predicted_xy),
                [simulation.pattern_index] * len(simulation.predicted_xy),
                array=False,
            )
            customdata = [
                identity
                + [int(value) for value in reflection]
                + [float(energy), float(intensity)]
                for identity, reflection, energy, intensity in zip(
                    stable,
                    simulation.hkl,
                    simulation.energy_kev,
                    simulation.relative_intensity,
                    strict=True,
                )
            ]
            labels = [
                f"({value[0]} {value[1]} {value[2]})"
                for value in simulation.hkl
            ]
            figure.add_trace(go.Scattergl(
                x=simulation.predicted_xy[:, 0],
                y=simulation.predicted_xy[:, 1],
                mode="markers+text" if show_hkl_labels else "markers",
                text=labels if show_hkl_labels else None,
                textposition="top right",
                name=f"Pattern {simulation.pattern_index} simulated missing",
                marker={
                    "size": marker_size + 4,
                    "symbol": "triangle-up-open",
                    "color": "rgb(128,0,128)",
                    "line": {"width": 1.5, "color": "rgb(128,0,128)"},
                },
                customdata=customdata,
                hovertemplate=(
                    "frame: %{customdata[0]}<br>pattern: %{customdata[1]}<br>"
                    "hkl: (%{customdata[3]} %{customdata[4]} %{customdata[5]})<br>"
                    "energy: %{customdata[6]:.4g} keV<br>"
                    "relative intensity: %{customdata[7]:.4g}<br>"
                    "x: %{x:.2f} px<br>y: %{y:.2f} px<extra></extra>"
                ),
                meta={"role": "simulated"},
                uid=f"detector-simulated-{simulation.pattern_index}",
            ))
            roles["simulated"].append(len(figure.data) - 1)

    figure.update_layout(
        xaxis={
            "title": {"text": "X pixel"},
            "range": [x_min, x_max],
            "scaleanchor": "y",
            "scaleratio": 1,
            "constrain": "domain",
            "showgrid": False,
            "zeroline": False,
        },
        yaxis={
            "title": {"text": "Y pixel"},
            "range": [y_max, y_min],
            "constrain": "domain",
            "showgrid": False,
            "zeroline": False,
        },
        plot_bgcolor="white",
        margin={"l": 55, "r": 30, "t": 40, "b": 50},
        uirevision=f"detector-{width:g}x{height:g}",
    )
    has_indexed = any(len(pattern.hkl) for pattern in data.patterns)
    has_simulated = any(len(simulation.hkl) for simulation in data.simulations)
    if not len(data.measured_xy) and not has_indexed and not has_simulated:
        message = "This frame has no detected peaks or indexed patterns."
        if data.simulations:
            message += " Simulation found no missing reflections."
        _add_empty_annotation(figure, message)
    return _apply_updates(
        figure,
        roles,
        ("image", "boundary", "detected", "indexed", "simulated"),
        trace_update,
        layout_update,
    )
