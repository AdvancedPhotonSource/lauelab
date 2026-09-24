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
| `num_threads` | `None` | | OpenMP threads per call. `None` estimates physical cores. Scans use `threads_per_worker` instead. |
| `rows_per_stripe` | `None` | rows | Rows per stripe. `None` uses at most 256, fewer to respect the memory limit. |
| `memory_limit_mb` | `8192` | MiB | Limit on live stripe buffers of one process, including point-file conversion. |

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

Input stripes use 2 bytes per pixel per frame for `numpy.uint16` and 8 bytes for every other dtype. Computed output uses 8 bytes per pixel per depth. For point-file output, sink buffers include the converted stored stripe, one 8-byte sum per depth, and one 8-byte sum per stripe pixel. Written stripes are released before the next allocation. An explicit `rows_per_stripe` must also fit the limit. If even one row cannot fit, reconstruction raises `InputError` before processing; `reconstruct_point` and `reconstruct_scan` record that point as failed.

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

## Reconstruct a scan

{func}`~lauelab.reconstruct.reconstruct_scan` writes one HDF5 file per point under `points/`, plus a shared `scan.h5` catalog. The catalog lists all requested points, their status, and their acquisition metadata.

```text
run/
    scan.h5
    points/
        point_1.h5
        point_2.h5
        point_3.h5
```

```python
from lauelab.reconstruct import reconstruct_scan

paths = ["point_1.h5", "point_2.h5", "point_3.h5"]
result = reconstruct_scan(
    paths,
    "run",
    geometry="tests/data/geo/geoN_2022-03-29_14-15-05.xml",
    detector=0,
    point_ids=["scan12_p1", "scan12_p2", "scan12_p3"],
    depth_range=(-25.0, 25.0),
    workers=2,
    threads_per_worker=4,
    progress=lambda outcome: print(outcome.point_id, outcome.status),
)
failed = [outcome for outcome in result.outcomes if outcome.status != "complete"]
```

The output directory must not exist or must be empty. Output filenames use the input stem: `point_1.h5` writes `run/points/point_1.h5`. Two inputs whose stems are equal, or differ only in letter case, are rejected before any file is written; reconstruct them into separate directories. Use `point_ids` to supply unique, non-empty identifiers for selecting points from the scan. These IDs are also saved in the point files. If omitted, they default to the input stems.

Choose `workers` and `threads_per_worker` for the cores you have. Each worker is a separate process that reconstructs one point at a time with `threads_per_worker` OpenMP threads. `threads_per_worker=None` divides the estimated physical cores among the workers. Memory demand grows with `workers`, because each worker holds its own stripe buffers and frame-sized references; `memory_limit_mb` applies to each worker separately. Storage conversion can run while a worker computes the next stripe, so `workers × threads_per_worker` is not a strict limit on busy threads. Measure a representative point on your nodes before choosing a configuration for a large scan.

Frames are stored uncompressed by default. Set `compression="gzip"` for lossless compression. In a full-size synthetic test, compression increased run time by about 40%; the reduction in file size depends on the data. Compare both settings on a representative point.

`progress` receives a {class}`~lauelab.reconstruct.PointOutcome` in the calling process when each point is recorded, starting with any input that could not be read. `result.outcomes` holds one outcome per input, in input order.

The coordinator keeps live progress in memory and batches catalog changes into five-second snapshots. The initial and final catalog are written immediately. After a hard termination, completed point files remain independently usable, although the last catalog snapshot may not list all of them as complete. This provides a recent record of progress, without automatic resume or a power-loss durability guarantee.

### Stop a scan

`should_stop` is polled in the calling process while points run. When it returns `True`, no further point starts and the running points finish. The first interrupt, such as Ctrl-C in a terminal or **Interrupt** in Jupyter, does the same and logs that running points are finishing. The result is then `cancelled`, and points that never started are `unattempted`. A second interrupt terminates the workers and raises `KeyboardInterrupt`; their points are recorded as `interrupted`.

### Failures

An input that cannot be read, or a reconstruction that fails, marks that point `failed` with its message and does not stop the scan. If writing or publishing a point file fails, the scan waits for running points to finish and then raises {class}`~lauelab.indexing.ReconstructionError`. An unexpected worker exit also raises `ReconstructionError`. In both cases completed point files are kept, and `scan.h5` records the run as `failed`. A failure to write `scan.h5` itself raises `OSError` and leaves the last complete version of the catalog in place.

### Read the scan

Point files retain acquisition metadata under `/entry1` and store reconstructed images at `/entry1/data/data`, with shape `(n_depths, ny, nx)`. NeXus attributes identify the images and their physical depth axis. Reconstruction settings, reference images, and intensity sums are under `/entry1/reconstruction`. The [format specification](../development/reconstruction-scan-format.md) lists all paths and attributes.

{class}`~lauelab.reconstruct.ScanReader` lists the points from `scan.h5` and opens the point files you ask for. Completed points can be read while the scan is still running. A reader shows the catalog as it was when the reader was created; create a new reader to see later progress.

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

`scan.points` reads only `scan.h5`. `scan.point()` opens only that point's file, and raises {class}`~lauelab.indexing.InputError` for a point that is not `complete`. `frame()` returns one `(ny, nx)` image in the stored dtype; `frame[y, x]` is the pixel at `(x, y)`. `region()` takes half-open bounds `(y0, y1, x0, x1)` in stored-image pixels and returns shape `(n_depths, y1 - y0, x1 - x0)`; bounds outside the image raise `InputError`. Use `iter_blocks(max_bytes)` to read successive depth blocks within a pixel-data budget. Returned arrays belong to the caller and remain usable after the file closes.

`depth_index` selects a frame by its zero-based position in the stack. Its physical depth in µm is `point.depth_um[depth_index]`.

`point.depth_intensity()` returns the sum of each stored frame, after conversion to the output pixel type. `point.computed_depth_intensity()` is the unscaled `numpy.float64` total that `ReconstructionResult.depth_intensity` reports. The two differ whenever storage changed a pixel: an unsigned type stores 0 for every negative pixel, and exponent normalization rescales integer output. `point.reference("first_raw")`, `"sum_raw"`, and `"sum_reconstructed"` return the embedded reference images. [Reconstruction scan format](../development/reconstruction-scan-format.md) defines these values exactly.

Use {class}`~lauelab.reconstruct.PointReader` to open a point directly, including a copy moved elsewhere. All the metadata needed to read it is included in the file; the catalog, raw input, and original geometry file are unnecessary:

```python
from lauelab.reconstruct import PointReader

with PointReader("run/points/point_1.h5") as point:
    print(point.point_id, point.shape, point.scan_number, point.sample_position)
    print(point.settings["depth_range"], point.settings["wire_edge"])
```

{func}`~lauelab.reconstruct.validate_scan_file` checks the structure of `scan.h5` and counts points by status. The catalogs of cancelled or partly failed runs can pass structural validation. Check the point statuses to determine which results are available.

### Reconstruct one point file

{func}`~lauelab.reconstruct.reconstruct_point` reconstructs a single point in the calling process. The output uses the same layout and validation as the point files in a scan:

```python
from lauelab.reconstruct import reconstruct_point

outcome = reconstruct_point(
    "point_1.h5",
    "single/point_1.h5",
    geometry="tests/data/geo/geoN_2022-03-29_14-15-05.xml",
    detector=0,
    depth_range=(-25.0, 25.0),
    num_threads=4,
)
assert outcome.status == "complete", outcome.error
```

The output must not exist; its directory is created when needed. Invalid parameters or unreadable input can raise an exception during preparation. Once prepared, an input or reconstruction failure returns a `failed` outcome without publishing output. Failure to write the output raises `ReconstructionError`.

### Schedule points yourself

Use `prepare_scan` when you want an MPI application or batch scheduler to distribute the work. {func}`~lauelab.reconstruct.prepare_scan` checks the inputs, publishes the first `scan.h5`, and returns the coordinator with one {class}`~lauelab.reconstruct.PointTask` per readable input. A task carries the request and the input metadata checked during preparation. It can be pickled or converted to JSON with `dataclasses.asdict` and rebuilt with `PointTask(**values)`. Each worker calls `reconstruct_point(task)` and returns the small outcome; pixels never leave the worker. Only the coordinator writes `scan.h5`.

Keep prepared tasks unchanged: the coordinator checks completed files against its original request. A worker rejects inputs whose selected frame range or acquisition metadata changed after preparation.

```python
from lauelab.reconstruct import prepare_scan, reconstruct_point

with prepare_scan(
    ["point_1.h5", "point_2.h5", "point_3.h5"],
    "scheduled",
    geometry="tests/data/geo/geoN_2022-03-29_14-15-05.xml",
    detector=0,
    depth_range=(-25.0, 25.0),
) as scan:
    for task in scan.tasks:
        scan.record_dispatch(task)
        # Send the task to a worker; the worker runs the next line.
        outcome = reconstruct_point(task, num_threads=4)
        scan.record(outcome)
    result = scan.finish()
```

Record each dispatch before the worker starts and each outcome when it returns, in any order. These calls publish a snapshot when the five-second interval has elapsed. For an asynchronous scheduler, also call `scan.snapshot()` in the coordinator's polling loop while workers run; no background thread publishes snapshots. Call `scan.snapshot(force=True)` when you need pending changes on disk immediately.

Assign each task once. If a worker is lost, record a failed outcome for its task, such as `PointOutcome(task.index, task.point_id, "failed", "worker lost")`; the coordinator does not retry.

`finish()` requires an outcome for every dispatched task. Pass `cancelled=True` to finish before every task was dispatched. Leaving the `with` block without `finish()` records the run as `failed`. [Reconstruction scan format](../development/reconstruction-scan-format.md) describes these steps and the catalog statuses.

### Index frames from a scan

A {class}`~lauelab.indexing.ScanFrame` identifies a stored frame by the point file, the point ID, and the depth index. {meth}`ScanReader.point_path() <lauelab.reconstruct.ScanReader.point_path>` resolves a catalog point ID to its file. {meth}`Indexer.index() <lauelab.indexing.Indexer.index>`, `index_many()`, and {meth}`Indexer.iter_index() <lauelab.indexing.Indexer.iter_index>` accept a `ScanFrame` wherever they accept a frame path.

```python
from lauelab.indexing import Indexer
from lauelab.reconstruct import ScanFrame, ScanReader

indexer = Indexer(
    "tests/data/geo/geoN_2022-03-29_14-15-05.xml", "tests/config/Ni.xml"
)
with ScanReader("run/scan.h5") as scan:
    path = scan.point_path("scan12_p1")
    point = scan.point("scan12_p1")
    brightest = int(point.depth_intensity().argmax())
    indices = range(max(0, brightest - 2), min(point.shape[0], brightest + 3))
frames = [ScanFrame(path, "scan12_p1", index) for index in indices]
results = indexer.index_many(frames)
indexer.write_results(results, "run/indexed.h5")
```

Indexing opens only the point file. It reads exactly the selected frame, in its stored dtype, together with the point's physical depth, detector identifier, frame origin, and binning, and the scan number, sample position, and energy of the input. The stored depth is used unless you pass `depth`, and an explicit `depth`, including `0.0`, overrides it. A missing file, a file that records another point ID, or a depth index outside the stack raises {class}`~lauelab.indexing.InputError`, or in `iter_index` becomes a failed outcome for that input.

Each result retains this reference in `FrameResult.source`. Saving and reopening the indexing results preserves the point file path, point ID, and depth index, so the detector view can display the frame used for indexing. See [Frame input](frame-input.md) for the values a scan frame supplies.

### Export per-depth files

{func}`~lauelab.reconstruct.export_per_depth` exports a point file to the per-depth HDF5 layout described in "Write depth files" above.

```python
from lauelab.reconstruct import export_per_depth

paths = export_per_depth("run/points/point_1.h5", "run/export/scan12_p1_")
```

The export copies the stored pixels, dtype, physical depth, detector metadata, and normalization provenance. The resulting files support the same reading and indexing operations as per-depth output from `Reconstructor.reconstruct()`. The summary file reports `$executionTime` as zero because export does not run reconstruction, and `$rows_at_one_time` as the stripe height used during reconstruction. A failed export leaves the point file unchanged.

## Handle failures

Invalid arguments, geometry, file metadata, and array shapes raise `InputError` before any work starts. A native allocation failure during setup raises `MemoryError`. After the first stripe starts, an expected failure returns a result with `success=False`, the message in `error`, and `last_completed_stripe` marking the last stripe fully written to every output file. See [Error handling](error-handling.md).

## Cross-check with the executable

{func}`~lauelab.reconstruct.reconstruct` runs the `reconstructN_cpu` program in a subprocess and returns the same result type. The in-process path reproduces its output bit for bit on the regression references. Use the executable as an independent check when validating a new acquisition configuration.
