# Inspect a reconstructed point

A reconstructed point is a stack of frames, one per sample depth. To examine how a feature varies with depth, choose a reference image, place square regions of interest (ROIs) on it, and plot the intensity of each region through depth. The inspection functions support both Jupyter notebooks and the Laue Portal.

Intensity traces use the **stored** pixels, after conversion to the output pixel type. Signed output types preserve negative values, so a trace can cross zero. [Reconstruction scan format](../development/reconstruction-scan-format.md) defines stored, computed, and raw values.

The examples inspect a point reconstructed with {func}`~lauelab.reconstruct.reconstruct_scan`; see [Reconstruct a wire scan](reconstruction.md). Replace `run/scan.h5` and the point ID with your own. The same functions accept a {class}`~lauelab.reconstruct.PointReader` opened directly on a point file.

## Choose a reference image

Each point embeds three images with the shape of one frame:

| Kind | Values | Meaning |
| --- | --- | --- |
| `"sum_reconstructed"` | stored | Sum of the stored frames through depth; the default |
| `"first_raw"` | raw | The first scan frame the reconstruction read |
| `"sum_raw"` | raw | Sum of the scan frames, before filtering and normalization |

{func}`~lauelab.reconstruct.inspection.reference_image` returns one of them as a {class}`~lauelab.reconstruct.inspection.ReferenceImage`, containing the image array, its kind, and whether it contains raw or stored values.

```python
from lauelab.reconstruct import ScanReader
from lauelab.reconstruct.inspection import reference_image

with ScanReader("run/scan.h5") as scan:
    point = scan.point("scan12_p1")
    background = reference_image(point, "sum_reconstructed")

assert background.values == "stored"
rows, columns = background.shape
```

Use the reference image to locate features and place ROIs. Traces are calculated from the reconstructed stack, independently of the reference image selected.

## Place square ROIs

An ROI is a square of `N × N` stored-image pixels. {func}`~lauelab.reconstruct.inspection.square_bounds` places it from a size and a click position and returns half-open bounds `(y0, y1, x0, x1)`, which select `frame[y0:y1, x0:x1]`. Positions are zero-based stored-image pixel coordinates `(x, y)`; an integer is a pixel centre.

```python
from lauelab.reconstruct.inspection import bounds_center, square_bounds

spot = square_bounds(5, 64.2, 63.8, (rows, columns))
edge = square_bounds(4, 52.0, 70.0, (rows, columns))

assert spot == (62, 67, 62, 67)
assert bounds_center(spot) == (64.0, 64.0)
assert bounds_center(edge) == (51.5, 69.5)
```

The centre snaps to the nearest integer coordinate for odd sizes or half-integer coordinate for even sizes, with ties resolved toward the lower coordinate. The entire square must fit inside the image; otherwise `square_bounds` raises {class}`~lauelab.indexing.InputError`. Each call creates new bounds, so existing ROIs retain their size.

## Trace intensity through depth

{func}`~lauelab.reconstruct.inspection.depth_trace` sums the stored pixels of a region at every depth and returns a {class}`~lauelab.reconstruct.inspection.DepthTrace`. With no bounds it returns the full-frame trace, which is the reduction embedded in the file and reads no pixels. {func}`~lauelab.reconstruct.inspection.roi_traces` does the same for several named regions.

```python
from lauelab.reconstruct.inspection import depth_trace, roi_traces

with ScanReader("run/scan.h5") as scan:
    point = scan.point("scan12_p1")
    whole = depth_trace(point)
    traces = roi_traces(point, {"spot": spot, "edge": edge})

peak_depth_um = whole.depth_um[whole.values.argmax()]
spot_peak_um = traces["spot"].depth_um[traces["spot"].values.argmax()]
```

ROI reductions read one depth block at a time, using at most 64 MiB of pixel data by default. Set `max_bytes` on `depth_trace` or `roi_traces` to choose another budget; it must hold at least one plane of the ROI. The small returned traces and HDF5 caches are additional. Multiple ROIs are reduced in sequence.

`values` has shape `(n_depths,)`. Integer pixels are summed exactly in `numpy.int64`; floating-point pixels are summed in `numpy.float64`. For the horizontal axis, use `depth_um` for physical depth in µm or `depth_index` for the zero-based frame index.

The trace provides two display transformations:

- `trace.normalized()` divides by the trace's own maximum and preserves signs. When the maximum is zero or negative, `values` is `None` and `reason` says why. The division preserves zero and the sign of each sample.
- `trace.log_samples()` returns the indices of positive samples for a logarithmic axis and counts the zero or negative samples omitted.

```python
normalized = traces["spot"].normalized()
assert normalized.available and normalized.values.max() == 1.0

samples = traces["edge"].log_samples()
n_shown, n_omitted = len(samples.kept), samples.n_omitted
```

## Plot

The figure builders in `lauelab.visualization` take prepared data and display choices. Axis, colour, and normalization changes operate on the prepared arrays without further file reads.

```python
from lauelab.visualization import (
    RoiOverlay, plot_depth_trace, plot_reference_image, plot_roi_traces,
)

overlays = [RoiOverlay("spot", spot, "rgb(230,90,60)"), RoiOverlay("edge", edge, "rgb(60,140,230)")]
image_figure = plot_reference_image(background, overlays)
whole_figure = plot_depth_trace(whole, axis="depth", selected_index=int(whole.values.argmax()))
roi_figure = plot_roi_traces(
    traces, colors={"spot": "rgb(230,90,60)", "edge": "rgb(60,140,230)"}, normalized=True,
)
```

{func}`~lauelab.visualization.plot_reference_image` places the origin at the upper left, with x increasing to the right and y downward. Both axes use the same pixel scale, and ROI outlines follow pixel edges.

{func}`~lauelab.visualization.plot_depth_trace` plots intensity against physical depth in µm or depth index, with an optional marker at the selected depth. {func}`~lauelab.visualization.plot_roi_traces` plots one trace per ROI. If normalization is unavailable, the plot omits that trace and annotates the reason. On a logarithmic axis, it omits nonpositive samples and reports their count. Omitted samples leave gaps in the plotted lines. Supply matching names and colours to identify each ROI across the image and trace plots.

Every builder accepts `layout_update` and per-role `trace_update`; the roles are `"image"` and `"roi"` for the image, `"trace"` and `"selected"` for the single trace, and `"roi"` for the ROI traces. {data}`~lauelab.visualization.DEPTH_AXIS_OPTIONS`, {data}`~lauelab.visualization.INTENSITY_OPTIONS`, and {data}`~lauelab.visualization.REFERENCE_OPTIONS` list the choices for controls.

## Inspect an in-memory result

{class}`~lauelab.reconstruct.inspection.ArrayPoint` gives a `(depth, y, x)` array the same interface, so a result of {meth}`Reconstructor.reconstruct_array() <lauelab.reconstruct.Reconstructor.reconstruct_array>` or of `Reconstructor.reconstruct(..., return_images=True)` can be inspected without writing a file. The array is taken as the stored data, so a `numpy.float64` stack keeps float64 values. It provides the `"sum_reconstructed"` reference; requests for raw reference images raise `InputError`.

```python
from lauelab.reconstruct.inspection import ArrayPoint

point = ArrayPoint(result.images, result.depth_um, "in-memory")
trace = depth_trace(point, spot)
```

## Inspect existing per-depth files

{class}`~lauelab.reconstruct.PerDepthReader` supplies the same frame, region, trace, and reconstructed-reference operations for the individual HDF5 files written by `Reconstructor`, `reconstruct`, or `export_per_depth`. Give it the files for one point, excluding the summary text file. It orders them by embedded physical depth and checks that their shapes, dtypes, detector metadata and normalization agree. It reads stored values without rescaling them again.

```python
from pathlib import Path
from lauelab.reconstruct import PerDepthReader
from lauelab.reconstruct.inspection import depth_trace, reference_image

paths = Path("run/export").glob("scan12_p1_*.h5")
with PerDepthReader(paths, point_id="scan12_p1") as point:
    whole = depth_trace(point)
    background = reference_image(point, "sum_reconstructed")
    trace = depth_trace(point, (60, 65, 50, 55))
```

These files have no embedded full-frame reduction, so `depth_trace(point)` reads one frame at a time. The reconstructed reference also reads one frame at a time. Raw references are unavailable and raise `InputError`; retain the point file if you need those backgrounds. The reader opens at most one file at a time and closes it after each operation. Exiting the context prevents further reads; returned arrays remain usable.

## Do your own analysis

For custom analysis, use the NumPy arrays in the prepared objects or read a selection directly with {class}`~lauelab.reconstruct.PointReader`. `frame()` reads one depth, and `region()` reads a rectangular region over the requested depth range.

```python
import numpy as np

with ScanReader("run/scan.h5") as scan:
    point = scan.point("scan12_p1")
    y0, y1, x0, x1 = spot
    pixels = point.region(spot)                 # (n_depths, 5, 5) in the stored dtype
    brightest = point.frame(int(whole.values.argmax()))

profile = pixels.reshape(len(pixels), -1).max(axis=1)
```
