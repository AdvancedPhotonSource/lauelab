# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Igor-style mean-intensity-versus-Q figures."""

from __future__ import annotations

from html import escape
from pathlib import Path
from typing import Mapping

import numpy as np
import plotly.graph_objects as go

from lauelab.energy import QDistribution
from .rendering import _apply_updates


def plot_q_distribution(
    result: QDistribution, *, title: str | None = None,
    layout_update: Mapping | None = None, trace_update: Mapping | None = None,
) -> go.Figure:
    """Plot mean intensity versus Q using Igor graph defaults.

    The axes are linear, Q is in 1/nm, the trace has blue points and a dashed
    connecting line, ticks point inward, and the default y minimum is zero.
    NaN bins leave gaps. Negative values remain in the numerical result; use
    layout_update or modify the returned figure to change the displayed range.

    No files are opened or written. Input is a computed or loaded QDistribution.
    Semantic trace role is 'profile'; layout_update and per-role trace_update
    follow the other lauelab figure builders. No fits or error bars are added.
    """
    if not isinstance(result, QDistribution):
        raise TypeError("result must be a QDistribution")
    p = result.provenance
    meta = p.get("effective_metadata", {})
    figure = go.Figure(go.Scatter(
        x=result.q_nm_inv, y=result.intensity_mean, mode="lines+markers",
        name="Mean intensity", line={"color": "rgb(64,64,255)", "dash": "dash"},
        marker={"color": "rgb(64,64,255)", "size": 5, "symbol": "circle"},
        connectgaps=False, customdata=np.column_stack((result.pixel_count, result.nonfinite_pixel_count)),
        hovertemplate="Q: %{x:.6g} nm⁻¹<br>I: %{y:.6g}<br>pixels: %{customdata[0]}<br>nonfinite: %{customdata[1]}<extra></extra>",
        meta={"role": "profile"}, uid="q-distribution",
    ))
    finite = result.intensity_mean[np.isfinite(result.intensity_mean)]
    peak = float(finite.max()) if finite.size else 0.0
    maximum = 1.0 if peak <= 0 else (
        peak*1.05 if peak <= np.finfo(float).max/1.05 else peak
    )
    axes = dict(type="linear", ticks="inside", mirror="ticks", showline=True,
                linecolor="black", tickcolor="black", linewidth=1,
                showgrid=False, zeroline=False,
                minor={"ticks": "inside", "tickcolor": "black", "showgrid": False})
    figure.update_layout(
        title={"text": escape(title if title is not None else str(meta.get("title") or "Monochromatic I(Q)"))},
        xaxis={**axes, "title": {"text": "Q (nm⁻¹)"}},
        yaxis={**axes, "title": {"text": result.intensity_label}, "range": [0, maximum]},
        plot_bgcolor="white", showlegend=False, margin={"l": 70, "r": 25, "t": 65, "b": 55},
        uirevision="q-distribution",
    )
    source = p.get("source", {})
    details = [Path(source["path"]).name] if source.get("path") else []
    if meta.get("date_exposed"):
        details.append(str(meta["date_exposed"]))
    if details:
        figure.add_annotation(x=1, y=1, xref="paper", yref="paper", xanchor="right", yanchor="top",
                              text="<br>".join(map(escape, details)), showarrow=False)
    return _apply_updates(figure, {"profile": [0]}, ("profile",), trace_update, layout_update)
