# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Stateless conversion of Plotly events to scientific identities."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .data import FrameId


@dataclass(frozen=True)
class PlotlySelection:
    """Stable identities extracted from a Plotly click or selection event."""

    frame_ids: tuple[FrameId, ...] = ()
    pattern_ids: tuple[tuple[FrameId, int], ...] = ()
    peak_ids: tuple[tuple[FrameId, int], ...] = ()
    reflection_ids: tuple[tuple[FrameId, int, int, int, int], ...] = ()


def _from_numeric(value):
    """Map numeric-array identities back to their list-form values."""
    if isinstance(value, (float, np.floating)) and not isinstance(value, (bool, np.bool_)):
        if np.isnan(value):
            return None
        if float(value).is_integer():
            return int(value)
    return value


def selection_from_plotly(event_data) -> PlotlySelection:
    """Extract stable identities from Plotly ``clickData`` or ``selectedData``.

    The first three ``customdata`` values must be ``frame_id``,
    ``pattern_index``, and ``peak_index``. A missing pattern or peak value is
    represented by `None`, or by ``NaN`` when the trace stored its identities
    as a numeric array. Integral floating-point identities from such an array
    are accepted as integers. Duplicate identities are removed in event order.
    """
    if event_data is None:
        return PlotlySelection()
    if not isinstance(event_data, dict):
        raise TypeError("event_data must be a mapping or None")
    points = event_data.get("points", ())
    if points is None:
        return PlotlySelection()
    if not isinstance(points, (list, tuple)):
        raise ValueError("event_data['points'] must be a sequence")

    frames = []
    patterns = []
    peaks = []
    reflections = []
    seen_frames = set()
    seen_patterns = set()
    seen_peaks = set()
    seen_reflections = set()
    for point in points:
        if not isinstance(point, dict):
            raise ValueError("each Plotly event point must be a mapping")
        customdata = point.get("customdata")
        if customdata is None:
            continue
        if not isinstance(customdata, (list, tuple)) or len(customdata) < 3:
            raise ValueError("point customdata must contain frame, pattern, and peak identity")
        frame_id, pattern_index, peak_index = (
            _from_numeric(value) for value in customdata[:3]
        )
        if isinstance(frame_id, (bool, np.bool_)) or not isinstance(frame_id, (str, Integral)):
            raise ValueError("customdata frame IDs must be strings or integers")
        frame_id = int(frame_id) if isinstance(frame_id, Integral) else frame_id
        if frame_id not in seen_frames:
            seen_frames.add(frame_id)
            frames.append(frame_id)
        if pattern_index is not None:
            if isinstance(pattern_index, (bool, np.bool_)) or not isinstance(pattern_index, Integral):
                raise ValueError("customdata pattern indices must be integers or None")
            pattern_index = int(pattern_index)
            identity = (frame_id, pattern_index)
            if identity not in seen_patterns:
                seen_patterns.add(identity)
                patterns.append(identity)
        if peak_index is not None:
            if isinstance(peak_index, (bool, np.bool_)) or not isinstance(peak_index, Integral):
                raise ValueError("customdata peak indices must be integers or None")
            peak_index = int(peak_index)
            identity = (frame_id, peak_index)
            if identity not in seen_peaks:
                seen_peaks.add(identity)
                peaks.append(identity)
        if peak_index is None and pattern_index is not None and len(customdata) >= 6:
            reflection = customdata[3:6]
            if any(
                isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral)
                for value in reflection
            ):
                raise ValueError("customdata Miller indices must be integers")
            identity = (frame_id, pattern_index, *(int(value) for value in reflection))
            if identity not in seen_reflections:
                seen_reflections.add(identity)
                reflections.append(identity)
    return PlotlySelection(
        tuple(frames), tuple(patterns), tuple(peaks), tuple(reflections)
    )
