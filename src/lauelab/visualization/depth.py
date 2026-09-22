# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Plotly figures for inspecting one reconstructed point through depth.

The builders take prepared data from :mod:`lauelab.reconstruct.inspection`
and display choices only. Changing an axis, a colour, or a name never reads
the stack again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import plotly.graph_objects as go

from lauelab.reconstruct.inspection import (
    Bounds, DepthTrace, ReferenceImage, _bounds, bounds_center,
)

from .options import DEPTH_AXIS_OPTIONS
from .rendering import _add_empty_annotation, _apply_updates

_AXIS_VALUES = tuple(choice.value for choice in DEPTH_AXIS_OPTIONS)
_AXIS_LABELS = {choice.value: choice.label for choice in DEPTH_AXIS_OPTIONS}
DEFAULT_ROI_COLORS = (
    "rgb(230,90,60)", "rgb(60,140,230)", "rgb(60,180,90)", "rgb(220,160,40)",
    "rgb(150,90,210)", "rgb(40,180,190)", "rgb(200,70,150)", "rgb(120,120,120)",
)


@dataclass(frozen=True)
class RoiOverlay:
    """A named, coloured square to draw on a reference image.

    Attributes
    ----------
    name : str
        Caller-chosen identity, shown in hover text and the legend.
    bounds : tuple of int
        Half-open ``(y0, y1, x0, x1)`` in stored-image pixels, as
        :func:`~lauelab.reconstruct.inspection.square_bounds` returns.
    color : str
        Any Plotly colour string. The caller keeps colours consistent between
        the overlay and the ROI's trace.
    """

    name: str
    bounds: Bounds
    color: str

    def __post_init__(self):
        if not self.name:
            raise ValueError("ROI name cannot be empty")
        object.__setattr__(self, "bounds", _bounds(self.bounds))
        if not isinstance(self.color, str) or not self.color:
            raise ValueError("ROI color must be a non-empty string")


def _axis_values(trace: DepthTrace, axis: str) -> np.ndarray:
    if axis not in _AXIS_VALUES:
        raise ValueError(f"axis must be one of {_AXIS_VALUES}; received {axis!r}")
    return trace.depth_um if axis == "depth" else trace.depth_index


def plot_reference_image(
    reference: ReferenceImage,
    rois: Sequence[RoiOverlay] = (),
    *,
    colorscale: str = "Gray",
    limits: tuple[float, float] | None = None,
    layout_update: Mapping | None = None,
    trace_update: Mapping | None = None,
) -> go.Figure:
    """Render a reference image with square ROI overlays.

    The image is drawn in stored-image pixel coordinates with the origin at
    the upper left, x increasing to the right, y increasing downward, and
    equal pixel scales. Each square is drawn on its pixel edges, so a
    1-pixel ROI encloses exactly one pixel. Semantic trace roles are
    ``"image"`` and ``"roi"``.

    Parameters
    ----------
    reference
        A :class:`~lauelab.reconstruct.inspection.ReferenceImage`.
    rois
        Squares to draw, in order. Every square must lie inside the image.
    colorscale : str
        Plotly colour scale for the image.
    limits : tuple of float or None
        ``(zmin, zmax)`` of the colour scale; the default spans the data.
    layout_update, trace_update
        Plotly layout update, and per-role trace updates keyed by role.
    """
    if not isinstance(reference, ReferenceImage):
        raise TypeError("reference must be a ReferenceImage")
    rows, columns = reference.shape
    if limits is not None and (
        len(limits) != 2 or not np.isfinite(limits).all() or limits[0] >= limits[1]
    ):
        raise ValueError("limits must contain two finite increasing values")
    overlays = tuple(rois)
    for roi in overlays:
        if not isinstance(roi, RoiOverlay):
            raise TypeError("rois must contain RoiOverlay values")
        y0, y1, x0, x1 = roi.bounds
        if y1 > rows or x1 > columns:
            raise ValueError(f"ROI {roi.name!r} lies outside the {rows} by {columns} image")

    figure = go.Figure()
    roles = {"image": [], "roi": []}
    heatmap = {
        "z": reference.image,
        "colorscale": colorscale,
        "colorbar": {"title": {"text": "Intensity"}},
        "name": reference.label,
        "hovertemplate": "x: %{x}<br>y: %{y}<br>I: %{z:.4g}<extra></extra>",
        "meta": {"role": "image", "kind": reference.kind, "values": reference.values},
        "uid": f"reference-{reference.kind}",
    }
    if limits is not None:
        heatmap["zmin"], heatmap["zmax"] = limits
    figure.add_trace(go.Heatmap(**heatmap))
    roles["image"].append(0)

    for roi in overlays:
        y0, y1, x0, x1 = roi.bounds
        left, right, top, bottom = x0 - 0.5, x1 - 0.5, y0 - 0.5, y1 - 0.5
        center_x, center_y = bounds_center(roi.bounds)
        size = x1 - x0
        figure.add_trace(go.Scatter(
            x=[left, right, right, left, left],
            y=[top, top, bottom, bottom, top],
            mode="lines",
            name=roi.name,
            line={"color": roi.color, "width": 2},
            hovertemplate=(
                f"{roi.name}<br>centre: ({center_x:g}, {center_y:g})<br>"
                f"size: {size} by {y1 - y0}<extra></extra>"
            ),
            meta={"role": "roi", "name": roi.name},
            uid=f"roi-{roi.name}",
        ))
        roles["roi"].append(len(figure.data) - 1)

    figure.update_layout(
        title={"text": f"{reference.label} ({reference.point_id})"},
        xaxis={
            "title": {"text": "X pixel"},
            "range": [-0.5, columns - 0.5],
            "scaleanchor": "y",
            "scaleratio": 1,
            "constrain": "domain",
            "showgrid": False,
            "zeroline": False,
        },
        yaxis={
            "title": {"text": "Y pixel"},
            "range": [rows - 0.5, -0.5],
            "constrain": "domain",
            "showgrid": False,
            "zeroline": False,
        },
        plot_bgcolor="white",
        margin={"l": 55, "r": 30, "t": 40, "b": 50},
        # Zoom survives a change of reference image or ROI set and resets
        # only when the pixel grid changes.
        uirevision=f"reference-{columns}x{rows}",
    )
    return _apply_updates(figure, roles, ("image", "roi"), trace_update, layout_update)


def plot_depth_trace(
    trace: DepthTrace,
    *,
    axis: str = "depth",
    log: bool = False,
    selected_index: int | None = None,
    color: str = "rgb(40,40,40)",
    layout_update: Mapping | None = None,
    trace_update: Mapping | None = None,
) -> go.Figure:
    """Render one intensity trace through depth.

    Semantic trace roles are ``"trace"`` and ``"selected"``.

    Parameters
    ----------
    trace
        A :class:`~lauelab.reconstruct.inspection.DepthTrace`.
    axis : str
        ``"depth"`` for physical depth in µm or ``"index"`` for the
        zero-based depth index; see ``DEPTH_AXIS_OPTIONS``.
    log : bool
        Logarithmic intensity axis. Samples that are zero or negative are
        omitted, leaving gaps in the plotted line. An annotation reports
        the number of omitted samples.
    selected_index : int or None
        Depth index to mark with a vertical line, for a depth browser.
    color : str
        Line colour.
    """
    if not isinstance(trace, DepthTrace):
        raise TypeError("trace must be a DepthTrace")
    x = _axis_values(trace, axis)
    figure = go.Figure()
    roles = {"trace": [], "selected": []}
    kept = trace.log_samples() if log else None
    values = np.where(trace.values > 0, trace.values, np.nan) if log else trace.values
    figure.add_trace(go.Scatter(
        x=x, y=values, mode="lines+markers", name=trace.label, connectgaps=False,
        line={"color": color}, marker={"size": 5, "color": color},
        customdata=trace.depth_index,
        hovertemplate=(
            "depth: %{x:.4g}<br>index: %{customdata}<br>I: %{y:.6g}<extra></extra>"
            if axis == "depth" else
            "index: %{x}<br>I: %{y:.6g}<extra></extra>"
        ),
        meta={"role": "trace", "point_id": trace.point_id},
        uid="depth-trace",
    ))
    roles["trace"].append(0)
    if selected_index is not None:
        if not 0 <= int(selected_index) < len(x):
            raise ValueError(f"selected_index {selected_index} is outside the trace")
        figure.add_vline(x=float(x[int(selected_index)]), line={"color": "rgb(200,60,60)", "dash": "dot"})
    if kept is not None and kept.n_omitted:
        _add_empty_annotation(
            figure, f"{kept.n_omitted} of {len(x)} samples are not positive and are "
            "omitted from the logarithmic axis",
        )
    figure.update_layout(
        title={"text": f"{trace.label} ({trace.point_id})"},
        xaxis={"title": {"text": _AXIS_LABELS[axis]}},
        yaxis={"title": {"text": "Stored intensity"}, "type": "log" if log else "linear"},
        margin={"l": 60, "r": 30, "t": 40, "b": 50},
        uirevision=f"depth-trace-{axis}",
    )
    return _apply_updates(figure, roles, ("trace", "selected"), trace_update, layout_update)


def plot_roi_traces(
    traces: Mapping[str, DepthTrace],
    *,
    colors: Mapping[str, str] | None = None,
    axis: str = "depth",
    normalized: bool = False,
    log: bool = False,
    layout_update: Mapping | None = None,
    trace_update: Mapping | None = None,
) -> go.Figure:
    """Render one trace per ROI through depth.

    Semantic trace role is ``"roi"``. Every trace's ``meta["name"]`` and
    ``uid`` carry its ROI name, so selection and colour follow identity, not
    position.

    Parameters
    ----------
    traces
        ROI name to trace, in display order, as
        :func:`~lauelab.reconstruct.inspection.roi_traces` returns. An empty
        mapping renders an annotated empty figure.
    colors
        ROI name to colour. Unnamed ROIs cycle through ``DEFAULT_ROI_COLORS``.
    axis : str
        ``"depth"`` or ``"index"``.
    normalized : bool
        Divide each trace by its own positive maximum. A trace whose maximum
        is not positive is left out, and an annotation says why.
    log : bool
        Logarithmic intensity axis; nonpositive samples are omitted and
        counted in an annotation.
    """
    if not isinstance(traces, Mapping):
        raise TypeError("traces must map ROI names to DepthTrace values")
    colors = dict(colors or {})
    figure = go.Figure()
    roles = {"roi": []}
    notes = []
    omitted = 0
    total = 0
    for position, (name, trace) in enumerate(traces.items()):
        if not isinstance(trace, DepthTrace):
            raise TypeError(f"trace {name!r} must be a DepthTrace")
        x = _axis_values(trace, axis)
        values = trace.values
        if normalized:
            result = trace.normalized()
            if not result.available:
                notes.append(result.reason)
                continue
            values = result.values
        if log:
            kept = np.flatnonzero(values > 0)
            omitted += len(x) - len(kept)
            total += len(x)
            values = np.where(values > 0, values, np.nan)
        color = colors.get(name, DEFAULT_ROI_COLORS[position % len(DEFAULT_ROI_COLORS)])
        figure.add_trace(go.Scatter(
            x=x, y=values, mode="lines+markers", name=name, connectgaps=False,
            line={"color": color}, marker={"size": 5, "color": color},
            customdata=trace.depth_index,
            hovertemplate=(
                f"{name}<br>depth: %{{x:.4g}}<br>index: %{{customdata}}<br>I: %{{y:.6g}}<extra></extra>"
                if axis == "depth" else
                f"{name}<br>index: %{{x}}<br>I: %{{y:.6g}}<extra></extra>"
            ),
            meta={"role": "roi", "name": name, "point_id": trace.point_id},
            uid=f"roi-trace-{name}",
        ))
        roles["roi"].append(len(figure.data) - 1)
    if not traces:
        _add_empty_annotation(figure, "No ROIs selected")
    if log and omitted:
        notes.append(
            f"{omitted} of {total} samples are not positive and are omitted from the "
            "logarithmic axis"
        )
    if notes:
        _add_empty_annotation(figure, "<br>".join(notes))
    figure.update_layout(
        xaxis={"title": {"text": _AXIS_LABELS[axis] if axis in _AXIS_LABELS else axis}},
        yaxis={
            "title": {"text": "Normalized intensity" if normalized else "Stored intensity"},
            "type": "log" if log else "linear",
        },
        margin={"l": 60, "r": 30, "t": 40, "b": 50},
        uirevision=f"roi-traces-{axis}",
    )
    return _apply_updates(figure, roles, ("roi",), trace_update, layout_update)
