# Run the indexing stages separately

`Indexer.index` runs three stages for each frame: peak search, pixel-to-q conversion, and crystal indexing. You can also run each stage separately to compare settings or investigate an unexpected result without repeating the full pipeline. Once you have chosen the settings, use them with an `Indexer` to process a scan.

| Stage | Call | Inputs | LaueGo program |
| --- | --- | --- | --- |
| Peak search | {func}`~lauelab.indexing.peak_search` | Image, optional mask | `peaksearch` |
| Pixel-to-q conversion | {meth}`Geometry.pixels_to_q <lauelab.indexing.Geometry.pixels_to_q>` | Peak positions, geometry, frame region, depth | `pix2qs` |
| Crystal indexing | {func}`~lauelab.indexing.index_orientations` | Scattering vectors, crystal description | `euler` |

Each call uses the same native code as `Indexer.index`. Given the same inputs, settings, and frame region, the separate calls return the same peaks, scattering vectors, and patterns as `Indexer.index`. They differ in how they handle peak positions outside the detector; see [Constraints](#indexing-stage-constraints).

## Run the three stages

This example compares peak-search thresholds for the synthetic two-grain Ni frame, then runs pixel-to-q conversion and crystal indexing. The image is a full-detector `numpy.uint16` array with shape `(2048, 2048)`. The defaults for `pixels_to_q` therefore apply: origin `(0, 0)`, no pixel grouping, and no depth correction.

```python
import h5py

from lauelab.indexing import index_orientations, load_crystal, load_geometry, peak_search

geometry = load_geometry("tests/data/geo/geoN_2022-03-29_14-15-05.xml")
crystal = load_crystal("tests/config/Ni.xml")
with h5py.File("tests/data/synthetic/frames/synthetic_ni_two_grains.h5") as source:
    image = source["entry1/data/data"][...]

settings = dict(boxsize=18, max_rfactor=0.5, min_separation=20, threshold=None, max_peaks=200)
for ratio in (3.0, 4.0, 6.0):
    found = peak_search(image, threshold_ratio=ratio, **settings)
    print(ratio, found.threshold_used, found.n_peaks)

found = peak_search(image, threshold_ratio=4.0, **settings)
qhat = geometry.pixels_to_q(found.peaks)
patterns = index_orientations(qhat, crystal, kev_max_calc=17.2, angle_tolerance_deg=0.1)
```

`peak_search` and `index_orientations` accept keyword arguments with the same names, defaults, and units as the fields of {class}`~lauelab.indexing.PeakParams` and {class}`~lauelab.indexing.IndexParams`, respectively.

`peak_search` returns a {class}`~lauelab.indexing.PeakSearch` object. Its structured array, `found.peaks`, contains the same fields as `FrameResult.peaks` except `qhat`. Peak positions are zero-based frame pixel coordinates `(x, y)`, where `image[y, x]` accesses the pixel at `(x, y)`.

`pixels_to_q` returns unit scattering vectors in the 34-ID-E laboratory frame as a `numpy.float64` array with shape `(n, 3)`. `index_orientations` returns a tuple of {class}`~lauelab.indexing.Pattern` objects. Here, `found.n_peaks` is 48 and `patterns` contains two patterns.

## Index a subset of peaks

`Pattern.pk_index` contains zero-based row indices into the `qhat` array that is passed to `index_orientations`. To index only some peaks, select their rows:

```python
import numpy as np

bright = found.peaks["intens"] > np.median(found.peaks["intens"])
bright_patterns = index_orientations(
    qhat[bright], crystal, kev_max_calc=17.2, angle_tolerance_deg=0.1
)
```

In this example, `bright_patterns[0].pk_index` contains row indices into `qhat[bright]`, not into `qhat`.

Row order matters because `index_orientations` uses at most the first `max_data` rows (250 by default). `peak_search` sorts peaks by the maximum pixel value in the blob used for each fit, largest first. This usually matches decreasing `intens`, but the two orders can differ. Preserving the peak-search order when computing `qhat` gives priority to peaks from the brightest blobs.

If `qhat` has fewer than two rows, `index_orientations` returns an empty tuple.

## Reuse the settings with an indexer

`PeakSearch.params` contains the validated peak-search settings, including defaults. Use it to configure an `Indexer` for a scan:

```python
from lauelab.indexing import Indexer, IndexParams

indexer = Indexer(
    geometry,
    crystal,
    peak_params=found.params,
    index_params=IndexParams(kev_max_calc=17.2, angle_tolerance_deg=0.1),
)
```

To run a stage with settings from an existing indexer, convert its parameter object to keyword arguments with {func}`dataclasses.asdict`:

```python
from dataclasses import asdict

again = peak_search(image, **asdict(indexer.peak_params))
```

## Frame region and depth

`pixels_to_q` uses `start` and `group` to transform frame pixel coordinates to full-detector pixel coordinates. For a cropped or binned frame, pass its origin and grouping; each coordinate maps to the center of the corresponding full-detector pixel group. For a reconstructed frame, also pass its depth in µm along the incident beam relative to the Si origin of the geometry file. You can obtain these values and the retained image from a {class}`~lauelab.indexing.FrameResult`:

```python
result = indexer.index("tests/data/synthetic/frames/synthetic_ni_two_grains.h5")
found = peak_search(result.image, **asdict(indexer.peak_params))
qhat = geometry.pixels_to_q(
    found.peaks,
    detector_index=indexer.detector_index,
    start=result.start,
    group=result.group,
    depth=result.depth,
)
```

`pixels_to_q` uses detector slot 0 unless you pass `detector_index`.

```{warning}
`detector_index` identifies a physical detector slot in the geometry file, not a position among the active detectors. Slots can be sparse. Selecting another detector’s slot can produce incorrect scattering vectors without raising an error. Use the slot of the detector that recorded the frame, such as `indexer.detector_index`. See [Geometry](geometry.md) for details.
```

(indexing-stage-constraints)=
## Constraints

- The stage calls return arrays and patterns, not `FrameResult` objects. The tables, plots, and result writers accept only `FrameResult` objects. To store or plot a result, index the frame with an `Indexer` that uses the chosen settings.
- `Geometry.pixels_to_q` raises `ValueError` if a fitted peak position falls outside the detector. `Indexer.index` does not check these bounds and can return a result for the same frame.
- `peak_search` accepts arrays only. To search a frame that is stored in a file, use `FrameResult.image` or read the array with `h5py`.

See the [Processing reference](../reference/processing.md) for complete signatures and field descriptions.
