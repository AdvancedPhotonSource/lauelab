# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Data models shared by Laue visualization and tabular APIs."""

from .data import DataScope, ResultSet, VisualizationDataset
from .interaction import PlotlySelection, selection_from_plotly
from .depth import (
    DEFAULT_ROI_COLORS,
    RoiOverlay,
    plot_depth_trace,
    plot_reference_image,
    plot_roi_traces,
)
from .options import (
    AXIS_OPTIONS,
    COLOR_MODES,
    DEPTH_AXIS_OPTIONS,
    INTENSITY_OPTIONS,
    PALETTE_OPTIONS,
    POLE_COLOR_MODES,
    REFERENCE_OPTIONS,
    SURFACE_PRESETS,
    Choice,
)
from .preparation import (
    Axis,
    DetectorPatternData,
    DetectorSimulationData,
    DetectorViewData,
    MapData,
    NO_PATTERN,
    PoleFigureData,
    ScalarColor,
    prepare_detector_view,
    prepare_map,
    prepare_pole_figure,
)
from .rendering import plot_detector_view, plot_map, plot_pole_figure
from .results import convert_xml, load_results
from .tables import Table, assignment_table, indexed_peak_table, pattern_table, peak_table
from .xml import load_visualization_xml

__all__ = [
    "AXIS_OPTIONS",
    "COLOR_MODES",
    "DEFAULT_ROI_COLORS",
    "DEPTH_AXIS_OPTIONS",
    "INTENSITY_OPTIONS",
    "PALETTE_OPTIONS",
    "POLE_COLOR_MODES",
    "SURFACE_PRESETS",
    "Axis",
    "Choice",
    "DataScope",
    "DetectorPatternData",
    "DetectorSimulationData",
    "DetectorViewData",
    "MapData",
    "NO_PATTERN",
    "PlotlySelection",
    "PoleFigureData",
    "REFERENCE_OPTIONS",
    "ResultSet",
    "RoiOverlay",
    "ScalarColor",
    "Table",
    "VisualizationDataset",
    "assignment_table",
    "convert_xml",
    "indexed_peak_table",
    "load_results",
    "load_visualization_xml",
    "pattern_table",
    "peak_table",
    "plot_depth_trace",
    "plot_detector_view",
    "plot_map",
    "plot_pole_figure",
    "plot_reference_image",
    "plot_roi_traces",
    "prepare_detector_view",
    "prepare_map",
    "prepare_pole_figure",
    "selection_from_plotly",
]
