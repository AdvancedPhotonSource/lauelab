# Visualization

The visualization API provides maps, pole figures, detector views, and tables, along with Plotly selection handling. Pass a `ResultSet` or `VisualizationDataset` to preparation and plotting functions.

See [Visualization data](../guides/visualization.md) for workflows, coordinate conventions, and examples.

## Data and selection

```{eval-rst}
.. currentmodule:: lauelab.visualization

.. autoclass:: DataScope
   :members: pattern_mask

.. autoclass:: ResultSet
   :members: from_indexer, to_visualization

.. autoclass:: VisualizationDataset
   :members: n_frames, n_patterns, n_assignments, pattern_ids

.. autofunction:: load_results

.. autofunction:: convert_xml

.. autofunction:: load_visualization_xml
```

## Prepared views

```{eval-rst}
.. currentmodule:: lauelab.visualization

.. autoclass:: Axis

.. autoclass:: ScalarColor

.. autoclass:: MapData
   :members: has_pattern

.. autodata:: NO_PATTERN

.. autoclass:: PoleFigureData

.. autoclass:: DetectorPatternData

.. autoclass:: DetectorSimulationData

.. autoclass:: DetectorViewData

.. autofunction:: prepare_map

.. autofunction:: prepare_pole_figure

.. autofunction:: prepare_detector_view
```

Prepared models copy their arrays and mark them read-only. Their frame and pattern identity fields remain aligned with the prepared rows.

## Plotly rendering

```{eval-rst}
.. currentmodule:: lauelab.visualization

.. autofunction:: plot_map

.. autofunction:: plot_pole_figure

.. autofunction:: plot_detector_view
```

Each renderer accepts a prepared model or normalized input. `trace_update` changes traces by the roles documented in the [visualization guide](../guides/visualization.md). `layout_update` applies after package defaults.

## Plotly selection

```{eval-rst}
.. currentmodule:: lauelab.visualization

.. autoclass:: PlotlySelection

.. autofunction:: selection_from_plotly
```

## Tables

```{eval-rst}
.. currentmodule:: lauelab.visualization

.. autoclass:: Table
   :members: to_dataframe

.. autofunction:: peak_table

.. autofunction:: pattern_table

.. autofunction:: assignment_table

.. autofunction:: indexed_peak_table
```

## Depth inspection

These builders render the prepared data of `lauelab.reconstruct.inspection`; see [Inspect a reconstructed point](../guides/depth-inspection.md). Trace roles are `"image"` and `"roi"` for the reference image, `"trace"` and `"selected"` for one depth trace, and `"roi"` for ROI traces.

```{eval-rst}
.. currentmodule:: lauelab.visualization

.. autoclass:: RoiOverlay
   :members:

.. autofunction:: plot_reference_image

.. autofunction:: plot_depth_trace

.. autofunction:: plot_roi_traces

.. autodata:: DEFAULT_ROI_COLORS
```

## Built-in choices

`AXIS_OPTIONS`, `COLOR_MODES`, `POLE_COLOR_MODES`, `SURFACE_PRESETS`, `PALETTE_OPTIONS`, `DEPTH_AXIS_OPTIONS`, `INTENSITY_OPTIONS`, and `REFERENCE_OPTIONS` are immutable tuples of {class}`~lauelab.visualization.Choice` objects. Their values match the corresponding Laue Portal controls.

```{eval-rst}
.. currentmodule:: lauelab.visualization

.. autoclass:: Choice

.. autodata:: AXIS_OPTIONS

.. autodata:: COLOR_MODES

.. autodata:: POLE_COLOR_MODES

.. autodata:: SURFACE_PRESETS

.. autodata:: PALETTE_OPTIONS

.. autodata:: DEPTH_AXIS_OPTIONS

.. autodata:: INTENSITY_OPTIONS

.. autodata:: REFERENCE_OPTIONS
```

## Surface frames

```{eval-rst}
.. currentmodule:: lauelab.analysis

.. autoclass:: SurfaceFrame
   :members: from_vectors, aps_34ide
```
