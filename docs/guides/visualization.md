# Visualization data

The visualization API normalizes indexing output before it prepares a specific view. This separation lets the same data support Plotly figures, package tables, and custom plotting code.

```text
FrameResult sequence -> ResultSet -> VisualizationDataset
results file --------------------> VisualizationDataset
LaueGo XML ----------------------> VisualizationDataset

VisualizationDataset -> prepare_*() -> immutable NumPy data
VisualizationDataset -> plot_*()    -> Plotly Figure
VisualizationDataset -> *_table()   -> Table
```

Preparation returns immutable NumPy arrays. Plot functions accept either normalized input or prepared data and return ordinary `plotly.graph_objects.Figure` objects.

## Prepare modern results

Use {class}`~lauelab.visualization.ResultSet` to attach stable frame IDs and the shared crystal and geometry to a sequence of {class}`~lauelab.indexing.FrameResult` objects.

```python
from lauelab.indexing import Indexer
from lauelab.visualization import ResultSet, prepare_map

indexer = Indexer("geometry.xml", "crystal.xml")
results = indexer.index_many(frames)
result_set = ResultSet.from_indexer(
    indexer,
    results,
    frame_ids=frame_ids,
)

map_data = prepare_map(result_set, axes=("X", "H"), color="goodness")
```

The `X`, `Y`, and `Z` map axes read `sample_position` from each result's metadata. Supply that metadata during indexing:

```python
result = indexer.index(
    frame,
    metadata={"sample_position": (x_um, y_um, z_um)},
)
```

`ResultSet.from_indexer()` copies the indexer's crystal and geometry references. Crystal context is required for cubic pole figures and inverse pole figure colors. Geometry is required for detector back-projection.

Call `result_set.to_visualization()` when you need the normalized arrays. This conversion copies result arrays into one read-only columnar snapshot. It also copies retained images. Later changes to a `FrameResult` do not change the snapshot. Preparation and table functions also accept `ResultSet` directly and perform this conversion for each call.

## Load a results file

{func}`~lauelab.visualization.load_results` reads a results file written by `Indexer.write_results()` into the same normalized model:

```python
from lauelab.visualization import load_results, prepare_map

dataset = load_results("results.h5")
map_data = prepare_map(dataset, axes=("X", "H"), color="n_indexed")
```

The file stores the crystal and the geometry, so maps, pole figures, tables, and detector views need no other input. Pass `geometry` to use a different geometry file and `frame_ids` to replace the recorded identifiers. Plotly figures require integer frame IDs within the browser's safe range, `±(2**53 - 1)`; larger IDs raise `ValueError` when plotting, so supply string IDs instead. See [Results files](results-file.md) for the file contents and XML conversion.

## Load LaueGo XML

{func}`~lauelab.visualization.load_visualization_xml` reads an `AllSteps` indexing XML document into the same normalized model. Loading parses every element, so a large document takes tens of seconds; convert it once with {func}`~lauelab.visualization.convert_xml` when it will be loaded more than once.

```python
from lauelab.visualization import load_visualization_xml, prepare_map

dataset = load_visualization_xml("indexed-scan.xml")
map_data = prepare_map(dataset, axes=("X", "H"), color="n_indexed")
```

Pass a geometry file when the XML does not contain a readable geometry path:

```python
dataset = load_visualization_xml(
    "indexed-scan.xml",
    geometry="geometry.xml",
    frame_ids=frame_ids,
)
```

An explicit geometry takes precedence over paths recorded in the XML. Missing geometry does not prevent maps, pole figures, or tables. It causes an error only when detector preparation needs back-projection.

The loader preserves declared peak rows when optional XML columns are absent. Missing values become `NaN`. It reconstructs rotation matrices only when the XML contains enough crystal and reciprocal-lattice information.

## Select patterns

{class}`~lauelab.visualization.DataScope` applies the same pattern selection to maps, pole figures, and tables. The default is:

```python
DataScope(patterns="best", min_indexed=3)
```

This selects the lowest pattern rank in each frame and requires at least three assignments. Select all patterns or explicit ranks when needed:

```python
from lauelab.visualization import DataScope

all_patterns = DataScope(patterns="all", min_indexed=3)
all_frames = DataScope(patterns="all_frames")
selected_ranks = DataScope(patterns=(0, 2), min_indexed=3)
detected_threshold = DataScope(min_indexed=3, min_detected=5)
```

`patterns="all_frames"` lets `peak_table` include detected peaks from frames with no indexed patterns. Pattern filtering applies only to frames where patterns exist. An empty selection is valid. Prepared arrays keep their documented dimensionality, and tables keep their columns.

Stable IDs do not depend on row order. A pattern uses `(frame_id, pattern_index)`, and a peak uses `(frame_id, peak_index)`.

## Prepare a map

{func}`~lauelab.visualization.prepare_map` accepts two or three axes. Built-in axes include motor positions, the 34-ID-E `H` and `F` transforms, depth, and laboratory-coordinate variants. Inspect {data}`~lauelab.visualization.AXIS_OPTIONS` for the implemented names.

```python
map_data = prepare_map(
    result_set,
    axes=("X", "H", "depth"),
    color="rms_error",
    scope=DataScope(patterns="all", min_indexed=3),
)
```

`map_data.coordinates` has shape `(n, 2)` or `(n, 3)`. Scalar colors have shape `(n,)`. IPF and Rodrigues colors have shape `(n, 3)` with RGB values between 0 and 1. The Plotly renderers pass RGB colors as `#rrggbb` strings, one per point.

Use {class}`~lauelab.visualization.Axis` and {class}`~lauelab.visualization.ScalarColor` for aligned custom values:

```python
import numpy as np

from lauelab.visualization import Axis, ScalarColor

map_data = prepare_map(
    dataset,
    axes=(
        Axis(load_newtons, label="Load", unit="N", alignment="frame"),
        Axis(
            lambda data: np.arange(data.n_patterns),
            label="Pattern order",
            alignment="pattern",
        ),
    ),
    color=ScalarColor(
        strain,
        label="Strain",
        palette="Plasma",
        alignment="pattern",
    ),
    scope=DataScope(patterns="all", min_indexed=0),
)
```

An alignment of `"frame"` requires one value per frame. `"pattern"` requires one value per normalized pattern. `"selected"` requires one value per pattern left by `scope`. A callable receives the complete {class}`~lauelab.visualization.VisualizationDataset`.

Named scalar colors are `"n_indexed"`, `"goodness"`, `"rms_error"`, and `"n_patterns"`. Orientation-map colors use the same names as Laue Portal: `"cubic_ipf"`, `"rodrigues"`, `"misorientation"`, and `"pole_hsv"`. Inspect {data}`~lauelab.visualization.COLOR_MODES` for the complete map-color list.

Cubic IPF and pole HSV coloring require a cubic crystal. Misorientation coloring also requires `misorientation_reference=(frame_id, pattern_index)`. Use `pole_hkl`, `pole_center`, and `pole_color_radius_deg` to configure pole HSV coloring.

## Create Plotly figures

The three Plotly functions accept normalized input and call the matching preparation function:

```python
from lauelab.visualization import (
    plot_detector_view,
    plot_map,
    plot_pole_figure,
)

map_figure = plot_map(
    result_set,
    axes=("X", "H"),
    color="goodness",
    marker_size=10,
)
pole_figure = plot_pole_figure(result_set, hkl=(1, 1, 0))
detector_figure = plot_detector_view(
    result_set,
    frame_id="scan-42-point-7",
    image=True,
)
```

You can also prepare once and render later:

```python
map_data = prepare_map(result_set, axes=("X", "H"), color="goodness")
map_figure = plot_map(map_data)
```

Valid empty selections return a figure with an explanatory annotation. Pole figures retain their unit boundary. Detector views retain their detector boundary.

### Customize Plotly output

Use `trace_update` to change traces by semantic role. Use `layout_update` for final layout changes:

```python
pole_figure = plot_pole_figure(
    result_set,
    layout_update={"template": "plotly_dark"},
    trace_update={
        "data": {"marker": {"size": 8}},
        "boundary": {"line": {"color": "white"}},
    },
)
```

The renderer applies these mappings after its defaults. An unknown role raises `ValueError`.

| Renderer | Trace roles |
|---|---|
| `plot_map` | `data`, `unindexed` |
| `plot_pole_figure` | `data`, `boundary`, `reference` |
| `plot_detector_view` | `image`, `boundary`, `detected`, `indexed`, `simulated` |

A role can update several traces. For example, the `indexed` role updates every selected detector pattern, including its on-detector and off-detector traces.

The returned figure remains a normal Plotly figure. You can call `update_layout()`, `update_traces()`, or add traces after rendering.

### Read Plotly selections

Map, pole, and detector traces store stable identities in the first three `customdata` values:

```text
[frame_id, pattern_index, peak_index]
```

A value is `None` when the trace does not represent that identity type. Map and pole-figure traces with integer frame IDs store the rows as one floating-point array, where a missing value is `NaN` in the figure and `null` in browser events; detector traces and string frame IDs use a list per point. {func}`~lauelab.visualization.selection_from_plotly` accepts both forms from Plotly `clickData` or `selectedData`:

```python
selection = selection_from_plotly(event_data)
print(selection.frame_ids)
print(selection.pattern_ids)
print(selection.peak_ids)
print(selection.reflection_ids)
```

The helper removes duplicate identities in event order. Simulated points add stable `(frame_id, pattern_index, h, k, l)` values to `reflection_ids`. It does not store selection state or depend on Dash.

## Plot prepared data with Matplotlib

The package does not provide a Matplotlib renderer. The prepared arrays are sufficient for a custom plot:

```python
import matplotlib.pyplot as plt

from lauelab.visualization import prepare_map

map_data = prepare_map(result_set, axes=("X", "H"), color="goodness")
figure, axes = plt.subplots()
points = axes.scatter(
    map_data.coordinates[:, 0],
    map_data.coordinates[:, 1],
    c=map_data.colors,
    cmap=map_data.palette,
)
axes.set_xlabel(map_data.axis_labels[0])
axes.set_ylabel(map_data.axis_labels[1])
figure.colorbar(points, ax=axes, label=map_data.color_label)
```

For `color_kind == "rgb"`, pass `c=map_data.colors` and omit `cmap` and the scalar color bar.

## Prepare a pole figure

{func}`~lauelab.visualization.prepare_pole_figure` generates the cubic symmetry family for `hkl` and projects upper-hemisphere poles stereographically:

```python
from lauelab.visualization import prepare_pole_figure

pole_data = prepare_pole_figure(
    result_set,
    hkl=(1, 1, 0),
    surface="normal",
    color="hsv_position",
    pole_center=(0.0, 0.0),
    pole_color_radius_deg=22.5,
)
```

`pole_data.points` has shape `(n, 2)`. One pattern can produce several rows, so its stable identity can occur more than once. `pole_center` and `pole_color_radius_deg` use the same names as pole HSV coloring in `prepare_map()`. Inspect {data}`~lauelab.visualization.POLE_COLOR_MODES` for the available colors: `"hsv_position"`, `"ipf"`, and `"uniform"`.

HKL-family generation and IPF colors currently support cubic crystals only. The function rejects other crystal systems instead of applying cubic symmetry to them.

## Prepare a detector view

{func}`~lauelab.visualization.prepare_detector_view` returns measured peaks and one indexed-reflection layer per selected pattern:

```python
from lauelab.visualization import prepare_detector_view

detector_data = prepare_detector_view(
    result_set,
    frame_id="scan-42-point-7",
    patterns="best",
    image=True,
)
```

`image=True` uses a retained modern image. For XML data, it reads the recorded input path. Image loading is opt-in. You can also pass a two-dimensional NumPy array, a `.npy` path, or a supported HDF5 path.

Measured and predicted positions use frame pixel `(x, y)` coordinates. `measured_xy` contains every detected peak. Each item in `patterns` contains `predicted_xy`, `hkl`, and the corresponding frame-local peak indices. Back-projection applies the frame's region origin and grouping so predicted positions align with the supplied frame.

Pass `simulation_energy_range_kev=(low, high)` to add predicted missing reflections. Simulation is opt-in. See [Add simulation to a detector view](detector-simulation.md) for coordinate conversion, prepared-data reuse, Plotly traces, and reflection selection.

## Work with tables

The table functions return immutable, named NumPy columns. In Jupyter, the last `Table` value in a cell renders as an HTML table. Call `.to_dataframe()` for pandas operations.

| Function | One row per record | Stable identity columns |
|---|---|---|
| {func}`~lauelab.visualization.peak_table` | Detected peak | `frame_id`, `peak_index` |
| {func}`~lauelab.visualization.pattern_table` | Indexed pattern | `frame_id`, `pattern_index` |
| {func}`~lauelab.visualization.assignment_table` | Pattern-to-peak assignment | `frame_id`, `pattern_index`, `peak_index` |
| {func}`~lauelab.visualization.indexed_peak_table` | Assignment joined to peak and pattern values | `frame_id`, `pattern_index`, `peak_index` |

```python
from lauelab.visualization import indexed_peak_table

table = indexed_peak_table(dataset, scope=all_patterns)
dataframe = table.to_dataframe()
selected = dataframe.query("energy_kev > 12 and goodness > 100")
```

The package does not add a second query language. Use pandas to filter, sort, group, or join table data. A DataFrame does not share writable storage with its source table.

## Coordinate and matrix conventions

Keep these spaces separate when you combine prepared data with other software:

| Data | Convention |
|---|---|
| NumPy image | Shape `(ny, nx)` and access `image[y, x]` |
| Frame pixel | Zero-based `(x, y)` in the supplied frame |
| Full-detector pixel | Frame coordinates transformed by `start` and `group` |
| Sample position and depth | Micrometres in the recorded acquisition coordinates |
| Detector size and translation | Micrometres |
| Detector rotation vector | Axis-angle vector in radians |
| Scattering vector | Components in the 34-ID-E laboratory convention |
| `Pattern.rotation` | Canonical modern orientation matrix |
| `Pattern.reciprocal` | Rows follow the reciprocal-matrix convention used by the indexer |

For grouped data, the frame-to-detector conversion maps a frame coordinate to the center of its full-detector pixel group. Detector slots are physical geometry slots and can be sparse.

The physical names, positive directions, and handedness of the laboratory axes still require beamline review. Do not infer those meanings from `Xlab`, `Ylab`, or `Zlab` alone.

## APS 34-ID-E surface presets

Map IPF colors and pole figures accept `"normal"`, `"X"`, `"H"`, `"Y"`, `"Z"`, and `"F"`. These names are APS 34-ID-E acquisition conventions, not general crystallographic names.

Use {class}`~lauelab.analysis.SurfaceFrame` when a named preset does not match the sample:

```python
from lauelab.analysis import SurfaceFrame

surface = SurfaceFrame.from_vectors(
    tilt=(1, 0, 0),
    roll=(0, 1, 0),
    normal=(0, 0, 1),
    name="sample surface",
)
pole_data = prepare_pole_figure(result_set, surface=surface)
```

The vectors must form a finite, orthonormal, right-handed frame with `tilt x roll = normal`.

## Errors and missing context

Preparation checks only the context required by the requested view:

- A spatial map raises `ValueError` when a selected coordinate is missing or non-finite.
- A cubic pole figure or cubic IPF color raises `ValueError` without cubic crystal context.
- A detector view raises `ValueError` without geometry.
- `image=True` raises `ValueError` when no retained image or source path is available.
- An unknown frame ID raises `KeyError`.
- Invalid names, alignments, shapes, and surface frames raise before plotting.

Missing coordinates do not prevent tables or detector views. Missing geometry does not prevent maps or pole figures.
