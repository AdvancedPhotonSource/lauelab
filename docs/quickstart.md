# Quickstart

This example processes synthetic nickel data included in the repository for API demonstrations and regression tests. Run each block from the repository root after installing the package.

## Prepare the inputs

Use the included HDF5 frame, 34-ID-E detector geometry, and nickel crystal description:

```python
from pathlib import Path

from lauelab.indexing import load_crystal, load_geometry

root = Path.cwd()
frame = root / "tests/data/synthetic/frames/synthetic_ni_two_grains.h5"
geometry = load_geometry(
    root / "tests/data/geo/geoN_2022-03-29_14-15-05.xml"
)
crystal = load_crystal(root / "tests/config/Ni.xml")
```

The frame is a two-dimensional `numpy.uint16` image with shape `(ny, nx)`. Peak coordinates use zero-based `(x, y)` order. The geometry and crystal files match this synthetic frame.

## Index one frame

Use {func}`~lauelab.indexing.index_frame` for a single frame:

```python
from lauelab.indexing import index_frame

result = index_frame(frame, geometry=geometry, crystal=crystal)
print(result)
print(f"indexed assignments: {result.n_indexed}")
```

The result contains detected peaks, two synthetic crystal orientations, frame statistics, and the retained image. Pass `keep_image=False` if later work does not need the image. A valid call can return no crystal patterns. In that case, `result.indexed` is `False`, while peak data and frame statistics remain available.

## Reuse an indexer

Construct {class}`~lauelab.indexing.Indexer` once when frames share geometry, crystal, detector selection, and processing parameters:

```python
from lauelab.indexing import Indexer

indexer = Indexer(geometry, crystal)
frames = sorted((root / "tests/data/synthetic/frames").glob("*.h5"))
results = indexer.index_many(frames)

assert len(results) == 4
assert all(item.image is None for item in results)
```

`index_many()` processes frames sequentially, preserves their order, and does not retain images by default.

## Inspect the result

`result.peaks` is a structured NumPy array. Each row contains a fitted `(x, y)` position, fit measurements, and a three-component `qhat` vector.

```python
import numpy as np

xy = np.column_stack((result.peaks["fit_x"], result.peaks["fit_y"]))
qhat = result.peaks["qhat"]

print(xy.shape)    # (n_peaks, 2)
print(qhat.shape)  # (n_peaks, 3)
```

Each item in `result.patterns` describes one indexed orientation. Its `pk_index` array contains zero-based indices into `result.peaks`.

```python
for pattern in result.patterns:
    assigned_peaks = result.peaks[pattern.pk_index]
    print(pattern.euler_deg, assigned_peaks.shape[0])
```

See [Results](guides/results.md) before interpreting the complete peak and pattern schemas.

## Plot the detector result

Attach the indexer's crystal and geometry to the result, then plot the detector image and indexed reflections:

```python
from lauelab.visualization import ResultSet, plot_detector_view

result_set = ResultSet(
    results,
    crystal=crystal,
    geometry=geometry,
)
figure = plot_detector_view(
    result_set,
    frame_id=3,
    image=True,
    patterns="best",
)
figure.show()
```

Add `simulation_energy_range_kev=(6.0, 30.0)` to predict missing reflections between 6 and 30 keV. Simulation requires an indexed pattern and is not run unless you supply an energy range.

See [Visualization data](guides/visualization.md) for maps, pole figures, detector views, and tables. See [Detector-view simulation](guides/detector-simulation.md) for simulated overlays and their coordinate conventions.

## Write XML

XML output is explicit in the in-process API. Writing XML does not create peak, pixel-to-q, or indexing text files.

```python
results[3].write_xml("indexed-frame.xml")
```

Use XML to exchange results with 34-ID-E tools that read this format. Use `FrameResult` directly for analysis in Python.

## Handle errors

Catch invalid user or data input separately from native numerical failures:

```python
from lauelab.indexing import IndexingError, InputError

try:
    checked_result = indexer.index(frames[0])
except InputError as error:
    print(f"Check the frame and configuration: {error}")
except IndexingError as error:
    print(f"The native indexing stage failed: {error}")
except MemoryError as error:
    print(f"Indexing could not allocate memory: {error}")
```

Do not retry unchanged invalid input. See [Error handling](guides/error-handling.md) for batch strategies and useful diagnostic context.

## Next steps

- [Select and inspect detector geometry](guides/geometry.md)
- [Load or construct a crystal](guides/crystals.md)
- [Prepare frame data and metadata](guides/frame-input.md)
- [Configure processing parameters](guides/parameters.md)
- [Interpret results](guides/results.md)
- [Process batches](guides/batch-indexing.md)
- [Reconstruct a wire scan](guides/reconstruction.md)
- [Visualize indexing results](guides/visualization.md)
- [Simulate detector reflections](guides/simulation.md)
- [Read the API reference](reference/index.md)
