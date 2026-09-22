# Reconstruct a wire scan

A wire-scan point is a stack of detector frames recorded while a wire moves across the diffracted beams at one sample position. Reconstruction resolves that stack into one detector image for each sample depth. Depth is in µm along the incident beam, measured from the Si origin in the geometry file. [Wire-scan reconstruction](../concepts/wire-scan-reconstruction.md) explains the model; this page covers the decisions needed to run it.

## Choose the inputs

Reconstruction needs a geometry file with a complete wire section, a detector slot, and a depth grid. Geometry files without a wire section load for indexing but raise {class}`~lauelab.indexing.InputError` here.

```{warning}
`detector` is a physical slot in the geometry file, not an ordinal position among active detectors. See [Geometry](geometry.md) before selecting a slot.
```

The constructor options control the depth grid, the wire edge, and the normalizations.

| Option | Default | Units | Effect |
| --- | --- | --- | --- |
| `depth_range` | required | µm | Inclusive `(start, end)`. Endpoints are rounded to multiples of `resolution`. Equal endpoints give one depth. |
| `resolution` | `1.0` | µm | Spacing between reconstructed depths. |
| `wire_edge` | `"leading"` | | `"leading"`, `"trailing"`, or `"both"`. |
| `percent_brightest` | `100.0` | % | Share of intensity-map pixels to reconstruct. Pixels below 1 count are always skipped. |
| `normalization` | `None` | | HDF5 vector below `entry1` that scales each frame. File input only. |
| `norm_exponent` | `None` | | Exponent normalization from the intensity map. |
| `norm_threshold` | `None` | counts | Threshold for exponent normalization. `None` derives it from the intensity map. |
| `cosmic_filter` | `False` | | Remove single-frame spikes before differencing. |
| `output_pixel_type` | `None` | | Pixel type code for written files. |
| `num_threads` | `None` | | OpenMP threads per call. `None` estimates physical cores. |
| `rows_per_stripe` | `None` | rows | Rows per stripe. `None` uses at most 256, fewer to respect the memory limit. |
| `memory_limit_mb` | `8192` | MiB | Limit on live stripe buffers, including scan-output conversion. |

See the [reference](../reference/reconstruction.md) for accepted ranges and the [algorithm page](../concepts/algorithms/depth-reconstruction.md) for how each option enters the calculation.

## Reconstruct one point

Create the {class}`~lauelab.reconstruct.Reconstructor` once and reuse it for every point with the same detector layout and options. The repository ships a geometry file but no wire-scan file, so replace `point_1.h5` with your own point. The test suite generates a synthetic point from `tests/data/reconstruction/generate_reference.py`.

```python
from pathlib import Path

from lauelab.reconstruct import Reconstructor

reconstructor = Reconstructor(
    Path("tests/data/geo/geoN_2022-03-29_14-15-05.xml"),
    detector=0,
    depth_range=(-25.0, 25.0),
    resolution=1.0,
    num_threads=4,
)
result = reconstructor.reconstruct("point_1.h5", return_images=True)

assert result.success, result.error
peak_depth_um = result.depth_um[result.depth_intensity.argmax()]
brightest_image = result.images[result.depth_intensity.argmax()]
```

`depth_um` has shape `(n_depths,)`. `depth_intensity` holds the sum of each reconstructed image and has the same shape. `images` has shape `(n_depths, rows, columns)` and dtype `numpy.float64`. It is present only with `return_images=True`; a full 2048 by 2048 detector at 121 depths needs about 4 GiB. `timings` lists read, compute, and write seconds for each stripe.

## Write depth files

Pass `output_base` to write one HDF5 file per depth and a summary text file. It is a filename prefix, not a directory.

```python
result = reconstructor.reconstruct("point_1.h5", output_base="run/point_1_")
```

This writes `run/point_1_0.h5`, `run/point_1_1.h5`, one file for each further depth, and `run/point_1_summary.txt`. The directory is created when needed, and existing files with those names are overwritten. Each file copies the source metadata except the image data and wire positions.

`output_pixel_type` selects the stored dtype. By default, file output keeps the input dtype when it has a code, `wire_edge="both"` selects `numpy.int32`, and other dtypes fall back to `numpy.float64`. HDF5 performs the float-to-integer conversion, including saturation. When `norm_exponent` is set and an integer type is written, the written images are multiplied by the factor stored in `entry1/microDiffraction/norm_rescale`. The `images` and `depth_intensity` fields keep the unscaled values.

## Reconstruct arrays

{meth}`~lauelab.reconstruct.Reconstructor.reconstruct_array` accepts frames and wire positions that are already aligned:

- `images` with shape `(N, rows, columns)`. `numpy.uint16` is used as is. Any other numeric dtype is converted to `numpy.float64`.
- `wire_xyz` with shape `(N + 1, 3)` in the acquisition coordinate system. No file-format offset is applied.
- `positioner`: `"none"`, `"pm500"`, or `"alio"`.
- `scale` with shape `(N,)`: dimensionless per-frame factors, the array equivalent of `normalization`.
- `intensity_map` with shape `(rows, columns)`, defaulting to the first frame.

The default {class}`~lauelab.reconstruct.ImageGeometry` describes an unbinned full frame whose detector size equals the array size. Pass an explicit `ImageGeometry` for a binned image or a detector ROI. Its `start` and `group` are zero-based unbinned pixels. The array path writes no files and always returns `numpy.float64` images, so `normalization` and `output_pixel_type` do not apply.

## How a point file is read

A 34-ID-E multi-image file stores a bookkeeping frame at slice 0, which is skipped. Slice 1 is both the intensity map and the first scan frame. The last stored slice is never differenced. An `N`-slice file therefore reconstructs scan frames from slices 1 to `N - 2`, and needs at least 5 slices.

Stored wire vectors include acquisition bookkeeping entries. Scan frame `f` pairs with stored wire entries `f + 2` and `f + 3`. This offset applies to file input only.

The positioner correction comes from the file's `file_time` attribute. Files before May 2006 get no correction, files before October 2009 get the PM500 correction, and later files get the Alio identity correction. Matching the executable, a `file_time` written with the ISO `T` separator is not parsed and receives no correction.

`normalization` reads `entry1/<tag>` and divides by 102 for `mA` and 88100 for `cnt3`. A missing or short vector raises `InputError`. The executable silently skips normalization in that case, so a file that reconstructs with the executable can still fail here.

## Control memory and threads

The kernel processes rows in stripes. With `rows_per_stripe=None`, the stripe is at most 256 rows and is reduced until the live stripe buffers fit in `memory_limit_mb`:

```{math}
2 \times \text{input bytes} + 2 \times \text{computed output bytes} + \text{sink buffers} \le \text{memory\_limit\_mb} \times 2^{20}
```

Input stripes use 2 bytes per pixel per frame for `numpy.uint16` and 8 bytes for every other dtype. Computed output uses 8 bytes per pixel per depth. For scan output, sink buffers include the converted stored stripe, one 8-byte sum per depth, and one 8-byte sum per stripe pixel. Written stripes are released before the next allocation. An explicit `rows_per_stripe` must also fit the limit. If even one row cannot fit, reconstruction raises `InputError` before processing; `reconstruct_scan` records that point as failed.

`memory_limit_mb` covers the stripe buffers. Total process memory also includes allocations for frame-sized masks, normalization maps and reference accumulators, retained result images, native per-thread scratch, and HDF5 caches. Integer stored sums are exact; floating-point totals can differ in their last bits when the stripe size changes.

`num_threads` sets the OpenMP threads for each call. The default estimates physical cores from Linux SMT topology and otherwise uses `os.cpu_count()`.

## Reconstruct many points

{func}`~lauelab.reconstruct.reconstruct_points` processes point files in worker processes. Each worker creates one reconstructor and uses `threads_per_worker` OpenMP threads. Workers start with the spawn method so they do not inherit an initialized OpenMP runtime.

```python
from lauelab.reconstruct import reconstruct_points

paths = ["point_1.h5", "point_2.h5", "point_3.h5"]
results = reconstruct_points(
    paths,
    "run",
    geometry="tests/data/geo/geoN_2022-03-29_14-15-05.xml",
    detector=0,
    depth_range=(-25.0, 25.0),
    workers=2,
    threads_per_worker=8,
)
failed = [path for path, result in zip(paths, results) if not result.success]
```

Output files use `<output_dir>/<stem>_` as their prefix. Pass reconstruction options as keyword arguments, and set threads with `threads_per_worker` rather than `num_threads`. Keep `workers × threads_per_worker` within the physical core count; oversubscription slows every worker.

An expected input, memory, or I/O failure in one point returns `success=False` for that point and does not stop the batch. Check every result.

## Reconstruct a scan into one file

{func}`~lauelab.reconstruct.reconstruct_scan` stores the reconstructed points together in one HDF5 file for browsing and indexing. Each point includes its frames, physical depths, detector metadata, reference images, and intensity totals at each depth.

```python
from lauelab.reconstruct import reconstruct_scan

paths = ["point_1.h5", "point_2.h5", "point_3.h5"]
result = reconstruct_scan(
    paths,
    "run/scan.h5",
    geometry="tests/data/geo/geoN_2022-03-29_14-15-05.xml",
    detector=0,
    point_ids=["scan12_p1", "scan12_p2", "scan12_p3"],
    depth_range=(-25.0, 25.0),
    progress=lambda outcome: print(outcome.point_id, outcome.status),
)
failed = [outcome for outcome in result.outcomes if outcome.status != "complete"]
```

Use `point_ids` to assign a unique, non-empty identifier to each input point. By default, the IDs are `"0"`, `"1"`, and so on, following the order of `paths`. These IDs are used to select points when reading the output, including points whose input files have the same name.

Points are reconstructed one after another, and each point uses all `num_threads` OpenMP threads. Image buffers are reused between points; catalog metadata grows with the number of points. Frames are stored uncompressed by default. Set `compression="gzip"` for lossless compression. In a full-size synthetic test, compression increased run time by about 40%; the reduction in file size depends on the data. Compare both settings on a representative point.

`progress` receives a {class}`~lauelab.reconstruct.PointOutcome` when each point ends. `should_stop` is polled before each point: a `True` return stops the run between points and records the remaining points as `unattempted`.

The run writes to `run/scan.h5.partial` and renames the file to `run/scan.h5` after closing and validating it. The published file includes the status of every point, including failed or unattempted points. An existing `run/scan.h5` raises `FileExistsError` before any point is read unless you pass `overwrite=True`.

An input that cannot be read, or a reconstruction that fails, marks that point `failed` with its message and does not stop the run. A failure of the output file itself raises {class}`~lauelab.indexing.ReconstructionError`; the `.partial` file that remains is not a valid result.

### Read the file

{class}`~lauelab.reconstruct.ScanReader` lists the points and reads frames without loading a whole point.

```python
from lauelab.reconstruct import ScanReader

with ScanReader("run/scan.h5") as scan:
    for entry in scan.points:
        print(entry.point_id, entry.status, entry.shape, entry.error)

    point = scan.point("scan12_p1")
    depth_index = int(point.depth_intensity().argmax())
    frame = point.frame(depth_index)
    depth_um = point.depth_um[depth_index]
    spot = point.region((60, 70, 50, 54))
```

`scan.points` reads only the catalog. `scan.point()` raises {class}`~lauelab.indexing.InputError` for a point that is not `complete`, because its image data may be incomplete. `frame()` returns one `(rows, columns)` image in the stored dtype. `region()` takes half-open bounds `(y0, y1, x0, x1)` in stored-image pixels and returns shape `(n_depths, y1 - y0, x1 - x0)`; bounds outside the image raise `InputError`. Use `iter_blocks(max_bytes)` to read successive depth blocks within a pixel-data budget. Returned arrays belong to the caller and remain usable after the file closes.

`depth_index` selects a frame by its zero-based position in the stack. Its physical depth in µm is `point.depth_um[depth_index]`.

`point.depth_intensity()` returns the sum of each stored frame, after conversion to the output pixel type. `point.computed_depth_intensity()` is the unscaled `numpy.float64` total that `ReconstructionResult.depth_intensity` reports. The two differ whenever storage changed a pixel: an unsigned type stores 0 for every negative pixel, and exponent normalization rescales integer output. `point.reference("first_raw")`, `"sum_raw"`, and `"sum_reconstructed"` return the embedded reference images. [Reconstruction scan format](../development/reconstruction-scan-format.md) defines these values exactly.

{func}`~lauelab.reconstruct.validate_scan_file` checks the structure of a file with bounded reads and reports how many points are complete. Files from cancelled or partly failed runs can pass structural validation. Check the point statuses to determine which results are available.

### Index frames from the file

A {class}`~lauelab.indexing.ScanFrame` identifies a stored frame using the scan file path, point ID, and depth index. {meth}`Indexer.index() <lauelab.indexing.Indexer.index>`, `index_many()`, and {meth}`Indexer.iter_index() <lauelab.indexing.Indexer.iter_index>` accept it wherever they accept a frame path.

```python
from lauelab.indexing import Indexer
from lauelab.reconstruct import ScanFrame, ScanReader

indexer = Indexer(
    "tests/data/geo/geoN_2022-03-29_14-15-05.xml", "tests/config/Ni.xml"
)
with ScanReader("run/scan.h5") as scan:
    point = scan.point("scan12_p1")
    brightest = int(point.depth_intensity().argmax())
    indices = range(max(0, brightest - 2), min(point.shape[0], brightest + 3))
frames = [ScanFrame("run/scan.h5", "scan12_p1", index) for index in indices]
results = indexer.index_many(frames)
indexer.write_results(results, "run/indexed.h5")
```

Indexing reads exactly the selected frame, in its stored dtype, together with the point's physical depth, detector identifier, frame origin, and binning, and the scan number, sample position, and energy of the input. The stored depth is used unless you pass `depth`, and an explicit `depth`, including `0.0`, overrides it. A frame of a point that is not `complete` raises {class}`~lauelab.indexing.InputError`, or in `iter_index` becomes a failed outcome for that input.

Each result retains this reference in `FrameResult.source`. Saving and reopening the indexing results preserves the scan path, point ID, and depth index, so the detector view can display the frame used for indexing. See [Frame input](frame-input.md) for the values a scan frame supplies.

### Export per-depth files

{func}`~lauelab.reconstruct.export_per_depth` exports a complete point to the per-depth HDF5 layout described in "Write depth files" above.

```python
from lauelab.reconstruct import export_per_depth

paths = export_per_depth("run/scan.h5", "scan12_p1", "run/export/scan12_p1_")
```

The export copies the stored pixels, dtype, physical depth, detector metadata, and normalization provenance. The resulting files support the same reading and indexing operations as per-depth output from `Reconstructor.reconstruct()`. The summary file reports `$executionTime` as zero and `$rows_at_one_time` as the frame height because the export did no reconstruction. A failed export leaves the scan file untouched.

## Handle failures

Invalid arguments, geometry, file metadata, and array shapes raise `InputError` before any work starts. A native allocation failure during setup raises `MemoryError`. After the first stripe starts, an expected failure returns a result with `success=False`, the message in `error`, and `last_completed_stripe` marking the last stripe fully written to every output file. See [Error handling](error-handling.md).

## Cross-check with the executable

{func}`~lauelab.reconstruct.reconstruct` runs the `reconstructN_cpu` program in a subprocess and returns the same result type. The in-process path reproduces its output bit for bit on the regression references. Use the executable as an independent check when validating a new acquisition configuration.
