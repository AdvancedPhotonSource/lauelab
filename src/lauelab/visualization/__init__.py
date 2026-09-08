# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Data models shared by Laue visualization and tabular APIs."""

from .data import DataScope, ResultSet, VisualizationDataset
from .interaction import PlotlySelection, selection_from_plotly
from .options import (
    AXIS_OPTIONS,
    COLOR_MODES,
    PALETTE_OPTIONS,
    POLE_COLOR_MODES,
    SURFACE_PRESETS,
    Choice,
)
from .preparation import (
    Axis,
    DetectorPatternData,
    DetectorSimulationData,
    DetectorViewData,
    MapData,
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
    "PlotlySelection",
    "PoleFigureData",
    "ResultSet",
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
    "plot_detector_view",
    "plot_map",
    "plot_pole_figure",
    "prepare_detector_view",
    "prepare_map",
    "prepare_pole_figure",
    "selection_from_plotly",
]
