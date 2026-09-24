# Reconstruction scan format

Reconstructing a scan produces one HDF5 file per point and a shared catalog, `scan.h5`, in the output directory. The catalog lists the requested points, their acquisition metadata, and their outcomes. Under `points/`, each completed file contains the reconstructed depth stack and everything needed to inspect, index, or export it independently: physical depths, reference images, settings, geometry, and acquisition metadata.

This page specifies the two file layouts, scan preparation, pixel values, frame selection, and square regions of interest (ROIs). For examples, see [Reconstruct a wire scan](../guides/reconstruction.md).

{func}`~lauelab.reconstruct.prepare_scan` checks the inputs and publishes the first catalog. {func}`~lauelab.reconstruct.reconstruct_point` writes one point file, and {func}`~lauelab.reconstruct.reconstruct_scan` runs the full workflow using local worker processes. {class}`~lauelab.reconstruct.ScanReader` reads a catalog, {class}`~lauelab.reconstruct.PointReader` reads a point file, and {func}`~lauelab.reconstruct.validate_scan_file` checks a catalog. Writers, readers, and validators share the layout definitions in `lauelab/reconstruct/_scan_layout.py`. {class}`~lauelab.reconstruct.Reconstructor`, {func}`~lauelab.reconstruct.reconstruct_points`, {func}`~lauelab.reconstruct.reconstruct`, and their per-depth output do not change.

The ROI rules on this page are implemented by `lauelab.reconstruct.inspection` and rendered by the depth-inspection builders in `lauelab.visualization`; the fixtures under `tests/data/reconstruction_contract/` pin them so that the Laue Portal is built against the same definition.

Both files follow the [HDF5 file conventions](hdf5-conventions.md). The catalog's root `format` attribute is `lauelab-reconstruction-scan` and its `version` is 2. A point file's `format` is `lauelab-reconstruction-point` and its `version` is 2. Point version 1 is no longer supported.

## Scan directory

```text
<scan directory>/
    scan.h5
    points/
        Twin2_wire_1.h5
        Twin2_wire_2.h5
```

For an input named `Twin2_wire_1.h5`, reconstruction writes `points/Twin2_wire_1.h5`. The input stem must be nonempty, must not start with `.`, and must not contain path separators or control characters; other names accepted by the file system are allowed.

Inputs with the same stem, including names that differ only in letter case, would produce conflicting output paths. Preparation rejects these inputs before writing any files. This also applies to an input listed twice. Reconstruct inputs with conflicting names into separate scan directories.

The scan directory must not exist or must be empty. A nonempty directory is rejected so that a new scan never mixes with old output; choose another directory or remove the old output deliberately.

The catalog records each point file as a relative path with `/` separators, such as `points/Twin2_wire_1.h5`, resolved against the directory that contains `scan.h5`. Moving the whole directory keeps the scan readable. Individual point files can also be copied and used independently. Both layouts use ordinary datasets, without HDF5 external links or virtual datasets.

## NeXus structure

Point files use the [NeXus base classes](https://manual.nexusformat.org/examples/python/simple_example_write1/index.html) to identify the reconstructed data and its axes. The catalog remains a lauelab HDF5 table. This format does not claim conformance to a specialized NeXus application definition, and copied acquisition metadata is not automatically reclassified.

The root has `default="entry1"`. `/entry1` has `NX_class="NXentry"` and `default="data"`. `/entry1/data` has `NX_class="NXdata"`, `signal="data"`, and `axes=["depth", ".", "."]`. Its `depth` child is an internal hard link to `/entry1/depth`; the other two dimensions use implicit image indices. Physical depth remains in µm along the incident beam from the geometry's Si origin. Image indexing remains `data[depth_index, y, x]`.

`/entry1/reconstruction` has `NX_class="NXprocess"`. Its `point`, `settings`, `execution`, `geometry`, `acquisition`, `detector`, and `normalization` groups have `NX_class="NXparameters"`.

The same process group contains five `NXdata` groups, each with `signal="data"`:

| Group under `/entry1/reconstruction` | Axes | Data shape |
| --- | --- | --- |
| `first_raw` | `[".", "."]` | `(rows, columns)` |
| `sum_raw` | `[".", "."]` | `(rows, columns)` |
| `sum_reconstructed` | `[".", "."]` | `(rows, columns)` |
| `stored_depth_intensity` | `["depth"]` | `(n_depths,)` |
| `computed_depth_intensity` | `["depth"]` | `(n_depths,)` |

Each intensity group has a `depth` hard link to `/entry1/depth`. These links refer to the same dataset inside the file, add no copy of the depth values, and remain valid when the file is moved. Readers validate the NeXus attributes and depth links along with dataset shapes, dtypes, units, and value conventions.

## Identities

Frames are selected by point and depth index. Catalog entries follow the input order, which is frozen when the scan is prepared.

| Identity | Type | Meaning |
| --- | --- | --- |
| Point ID | string | Unique, non-empty identifier of a point within its scan. Defaults to the input stem. It is independent of the point filename. |
| Manifest index | zero-based integer | Position of the point in the input order. Every catalog dataset uses this order. |
| Point path | string | Location of the point file relative to the scan directory. It is assigned when the scan is prepared, whatever the point's outcome. |
| Depth index | zero-based integer | Position of a frame in the point's `(n_depths, ny, nx)` stack. |
| Physical depth | `numpy.float64`, µm | Depth along the incident beam from the Si origin of the geometry file. It is `depth_um[depth index]`. |

The point ID and original manifest index are saved in the point file and remain available if you copy it elsewhere. Standalone points have no manifest index. Physical depths are stored in `/entry1/depth`. For example, depth index 0 corresponds to -25 µm when the reconstructed depth grid starts at -25 µm.

A **frame reference**, {class}`~lauelab.indexing.ScanFrame`, selects one stored frame by the path of its point file and a depth index. The reference can be serialized and passed to indexing workers, which open the file locally. Indexing results preserve the point identity alongside the depth index.

## Preparation and tasks

{func}`~lauelab.reconstruct.prepare_scan` takes the input files in manifest order, the scan directory, the geometry, and the reconstruction settings. Before writing, it validates the settings, detector slot, point IDs, output filenames, and destination. It then reads each input's metadata without loading frames and creates one {class}`~lauelab.reconstruct.PointTask` for each point whose input can be read. Input paths are made absolute against the current directory; symbolic links are not resolved.

The first catalog is published before any point is computed. It lists every input. An input that cannot be read, or whose metadata is invalid, is recorded as a `failed` point with its error and has no task; it does not stop the other points.

Each task includes the absolute input and output paths, point identity, detector slot, validated reconstruction settings, and complete geometry XML, together with the frame shape, depth count and bounds, pixel type, selected raw-slice range, scan number, sample position, and incident energy found during inspection. Missing acquisition values are `None` in the task and use the schema's missing values in HDF5. Tasks contain only strings, numbers, and plain containers, so they can be pickled or converted to JSON.

The executing process opens its own input and builds its own native state. It checks the output description, raw-slice range, and acquisition values against the prepared task before creating an output file. A disagreement produces a failed point. These checks read metadata; they do not verify raw pixel content. Treat tasks as read-only. The coordinator keeps an independent copy of the prepared request and rejects changed tasks or completed files that disagree with it.

Exactly one coordinator, the {class}`~lauelab.reconstruct.PreparedScan` returned by `prepare_scan`, writes `scan.h5`. Workers write only the point files of their tasks. A scheduler other than {func}`~lauelab.reconstruct.reconstruct_scan` uses the coordinator in this order:

1. Call {meth}`~lauelab.reconstruct.PreparedScan.record_dispatch` when handing a task to a worker. This sets its status to `writing`.
2. The worker calls {func}`~lauelab.reconstruct.reconstruct_point` with the task and returns the small {class}`~lauelab.reconstruct.PointOutcome`. Pixels stay in the worker.
3. Call {meth}`~lauelab.reconstruct.PreparedScan.record` as outcomes arrive, in any order.
4. Call {meth}`~lauelab.reconstruct.PreparedScan.finish` after every dispatched task has an outcome. Pass `cancelled=True` if some tasks were never dispatched; they become `unattempted`.

The scheduler assigns each task once and records a failed outcome for a task whose worker was lost; the coordinator provides no retry. Leaving the coordinator's context without `finish` records the run as `failed`, dispatched points without an outcome as `interrupted`, and the other points as `unattempted`. A standalone point, reconstructed without a scan, is prepared with the same rules and written by the same writer; its manifest index is missing.

## Value semantics

Pixel values are described at three stages of processing:

| Kind | Dtype | Meaning |
| --- | --- | --- |
| Raw | dtype of the input frames | Detector counts as acquired, before the cosmic-ray filter and before any normalization. |
| Computed | `numpy.float64` | Output of the reconstruction kernel. These are `ReconstructionResult.images`, and their per-depth totals are `ReconstructionResult.depth_intensity`. |
| Stored | the point's output pixel type | The pixels in the file. Inspection, ROI traces, and downstream indexing use these values. |

Before writing, the library multiplies computed intensities by `rescale` and converts them to the output pixel type. `rescale` is 1 unless exponent normalization writes an integer type; it is recorded as `/entry1/reconstruction/normalization/rescale` and is applied once. For a floating-point type, the conversion equals a NumPy cast. For an integer type, the conversion truncates toward zero and then saturates at the limits of the type. Positive and negative infinity saturate. NaN stores 0 in an integer type.

The native library converts pixels and computes reductions in one pass. For finite values and infinities, conversion matches the HDF5 conversion used by per-depth output. Integer conversion of NaN depends on the HDF5 destination type and can differ from the scan format's defined value of zero.

```{warning}
Reconstructed intensities can be negative with any `wire_edge` setting. An unsigned output type stores 0 for each negative pixel, so its stored totals are larger than the computed totals. For the recorded leading-edge reference, the `numpy.uint16` frames sum to 573801 and the computed frames sum to 512684.9.
```

### Reductions

Stored-pixel totals are calculated after conversion, using the same values you would read back from the file.

- Integer stored pixels accumulate in `numpy.int64` and are exact. A sum of `n` values of at most 4 bytes cannot overflow while `n` is below 2³¹, which exceeds the pixel count of a 4096 by 4096 frame.
- Floating-point stored pixels accumulate in `numpy.float64`. The summation order is not specified. Reconstructed pixels are signed, so a total can be much smaller than the pixels that produce it, and a tolerance relative to the total is not meaningful. Compare a reduction with a recomputed one at an absolute tolerance of 10⁻¹² times the sum of the absolute pixel values in the reduction.

`/entry1/reconstruction/computed_depth_intensity/data` contains the totals before storage conversion, as reported in the LaueGo-compatible summary. `/entry1/reconstruction/stored_depth_intensity/data` contains the stored-pixel totals used for inspection. Conversion and `rescale` can therefore change the totals.

### Reference images

Three reference images are saved with each point for inspection after the raw input becomes unavailable. Each has the same shape as a detector frame.

| Image | Values | Definition |
| --- | --- | --- |
| `/entry1/reconstruction/first_raw/data` | raw | The first scan frame the reader selects. |
| `/entry1/reconstruction/sum_raw/data` | raw | The sum of every selected scan frame. |
| `/entry1/reconstruction/sum_reconstructed/data` | stored | The sum of the stored frames through depth. |

`/entry1/reconstruction/acquisition/raw_slices` records the half-open range of stored input slices that was selected. For a 34-ID-E multi-image file with `n` stored slices the range is `[1, n - 1)`: slice 0 is bookkeeping, and the last slice is never differenced. See "How a point file is read" in [Reconstruct a wire scan](../guides/reconstruction.md). The raw stack itself is not copied.


## File layout

In the tables below, `n_points` is the number of points in the scan. The depth count (`n_depths`), frame dimensions (`rows`, `columns`), and output pixel type can vary between points. The following dtype rules apply separately to each point:

| Rule | Dtype |
| --- | --- |
| `stored` | The point's output pixel type. |
| `input` | The dtype of the point's raw frames. |
| `stored_accumulator` | `<i8` when `stored` is an integer type, otherwise `<f8`. |
| `raw_accumulator` | `<i8` when `input` is an integer type of at most 4 bytes, otherwise `<f8`. |

The **Missing** column gives the value stored when the quantity does not apply or was not available.

### Settings

The table below gives the catalog paths. Point files store the same fields under `/entry1/reconstruction`: for example, `/settings/detector` becomes `/entry1/reconstruction/settings/detector`, and `/geometry/xml` becomes `/entry1/reconstruction/geometry/xml`. The point-file `memory_limit_mb` dataset also has `units="MiB"`.

| Path | Dtype | Shape | Units | Missing | Meaning |
| --- | --- | --- | --- | --- | --- |
| `/settings/detector` | `<i4` | `()` | | | Physical detector slot in the geometry |
| `/settings/depth_range` | `<f8` | `(2,)` | um | | Requested inclusive depth range |
| `/settings/resolution` | `<f8` | `()` | um | | Spacing between depths |
| `/settings/wire_edge` | `string` | `()` | | | `leading`, `trailing`, or `both` |
| `/settings/percent_brightest` | `<f8` | `()` | | | Percentage of intensity-map pixels reconstructed |
| `/settings/normalization` | `string` | `()` | | empty | Name of the normalization vector |
| `/settings/norm_exponent` | `<f8` | `()` | | NaN | Exponent normalization |
| `/settings/norm_threshold` | `<f8` | `()` | | NaN | Requested threshold; each point records the value used |
| `/settings/cosmic_filter` | `u1` | `()` | | | 1 when the cosmic-ray filter ran |
| `/settings/output_pixel_type` | `<i4` | `()` | | -1 | Requested pixel type code; each point records the type used |
| `/settings/rows_per_stripe` | `<i4` | `()` | | -1 | Requested rows per stripe; -1 selects them automatically |
| `/settings/memory_limit_mb` | `<i4` | `()` | | | Stripe-buffer limit of one worker in MiB |
| `/geometry/path` | `string` | `()` | | empty | Geometry file path as given |
| `/geometry/xml` | `string` | `()` | | | Complete text of the geometry file, read once when the scan is prepared |

### Scan run

| Path | Dtype | Shape | Units | Missing | Meaning |
| --- | --- | --- | --- | --- | --- |
| `/run/status` | `u1` | `()` | | | Run status code |
| `/run/native_version` | `string` | `()` | | | Version of the native library that prepared the scan |

### Catalog

Each catalog dataset has one entry per point in manifest order. Listing a scan reads `scan.h5` only; no point file is opened.

| Path | Dtype | Shape | Units | Missing | Meaning |
| --- | --- | --- | --- | --- | --- |
| `/catalog/point_ids` | `string` | `(n_points,)` | | | Point ID |
| `/catalog/point_paths` | `string` | `(n_points,)` | | | Point file relative to the scan directory |
| `/catalog/source_paths` | `string` | `(n_points,)` | | | Absolute input path |
| `/catalog/source_sizes` | `<i8` | `(n_points,)` | | -1 | Input size in bytes |
| `/catalog/source_mtimes_ns` | `<i8` | `(n_points,)` | | -1 | Input modification time in ns since the epoch |
| `/catalog/raw_slices` | `<i8` | `(n_points, 2)` | | -1 | Half-open range of stored input slices that is reconstructed |
| `/catalog/scan_numbers` | `<i4` | `(n_points,)` | | -1 | Acquisition scan number |
| `/catalog/sample_positions` | `<f8` | `(n_points, 3)` | um | NaN | Sample `(x, y, z)` in the acquisition coordinate system |
| `/catalog/energies_kev` | `<f8` | `(n_points,)` | keV | NaN | Incident energy |
| `/catalog/image_shapes` | `<i4` | `(n_points, 2)` | | | Frame shape `(ny, nx)`; `(0, 0)` when the input could not be read |
| `/catalog/n_depths` | `<i4` | `(n_points,)` | | | Number of stored frames; 0 when the input could not be read |
| `/catalog/depth_bounds` | `<f8` | `(n_points, 2)` | um | NaN | First and last physical depth |
| `/catalog/pixel_types` | `<i4` | `(n_points,)` | | -1 | Output pixel type code |
| `/catalog/status` | `u1` | `(n_points,)` | | | Point status code |
| `/catalog/errors` | `string` | `(n_points,)` | | empty | Error text of a failed point |

Input paths, sizes, and modification times record provenance. They do not establish the identity of the input contents and do not support resuming a scan. A file-status call retrieves the size and time without reading the input: an unchanged file can acquire a new modification time, and changed contents can keep the same size and modification time.

### Point file

Each completed point file includes the datasets below and the settings listed above. The file becomes available at its final path only after writing and validation succeed.

| Path | Dtype | Shape | Units | Missing | Meaning |
| --- | --- | --- | --- | --- | --- |
| `/entry1/reconstruction/program` | `string` | `()` | | | Program name, `lauelab` |
| `/entry1/reconstruction/version` | `string` | `()` | | | Package version used for reconstruction |
| `/entry1/reconstruction/date` | `string` | `()` | | | UTC processing timestamp in ISO 8601 format |
| `/entry1/reconstruction/point/id` | `string` | `()` | | | Point ID |
| `/entry1/reconstruction/point/manifest_index` | `<i8` | `()` | | -1 | Manifest index in the scan; -1 for a standalone point |
| `/entry1/reconstruction/point/complete` | `u1` | `()` | | | 1 when every dataset is written; set last |
| `/entry1/reconstruction/point/native_version` | `string` | `()` | | | Version of the native library that reconstructed the point |
| `/entry1/reconstruction/execution/num_threads` | `<i4` | `()` | | | OpenMP threads used |
| `/entry1/reconstruction/execution/rows_per_stripe` | `<i4` | `()` | | | Rows per stripe used |
| `/entry1/reconstruction/acquisition/source_path` | `string` | `()` | | | Absolute input path |
| `/entry1/reconstruction/acquisition/source_size` | `<i8` | `()` | bytes | -1 | Input size in bytes |
| `/entry1/reconstruction/acquisition/source_mtime_ns` | `<i8` | `()` | | -1 | Input modification time in ns since the epoch |
| `/entry1/reconstruction/acquisition/scan_number` | `<i4` | `()` | | -1 | Acquisition scan number |
| `/entry1/reconstruction/acquisition/sample_position` | `<f8` | `(3,)` | um | NaN | Sample `(x, y, z)` in the acquisition coordinate system |
| `/entry1/reconstruction/acquisition/energy_kev` | `<f8` | `()` | keV | NaN | Incident energy |
| `/entry1/data/data` | `stored` | `(n_depths, rows, columns)` | | | Stored frames |
| `/entry1/depth` | `<f8` | `(n_depths,)` | um | | Physical depth of each frame |
| `/entry1/reconstruction/detector/id` | `string` | `()` | | empty | Detector identifier from the input |
| `/entry1/reconstruction/detector/size` | `<i4` | `(2,)` | pixel | | Full detector `(x, y)` size in unbinned pixels |
| `/entry1/reconstruction/detector/roi_start` | `<i4` | `(2,)` | pixel | | Zero-based `(x, y)` frame origin in unbinned pixels |
| `/entry1/reconstruction/detector/roi_group` | `<i4` | `(2,)` | | | `(x, y)` binning factors |
| `/entry1/reconstruction/normalization/threshold` | `<f8` | `()` | | NaN | Exponent-normalization threshold used |
| `/entry1/reconstruction/normalization/rescale` | `<f8` | `()` | | | Factor applied before conversion; 1 when none |
| `/entry1/reconstruction/first_raw/data` | `input` | `(rows, columns)` | | | First selected raw frame |
| `/entry1/reconstruction/sum_raw/data` | `raw_accumulator` | `(rows, columns)` | | | Sum of the selected raw frames |
| `/entry1/reconstruction/acquisition/raw_slices` | `<i8` | `(2,)` | | | Half-open range of selected input slices |
| `/entry1/reconstruction/sum_reconstructed/data` | `stored_accumulator` | `(rows, columns)` | | | Sum of the stored frames |
| `/entry1/reconstruction/stored_depth_intensity/data` | `stored_accumulator` | `(n_depths,)` | | | Stored total of each frame |
| `/entry1/reconstruction/computed_depth_intensity/data` | `<f8` | `(n_depths,)` | | | Computed total of each frame |

The `/entry1/reconstruction/execution` datasets record the thread count and stripe height used during reconstruction. Runs with identical results can use different execution settings.

Acquisition metadata retains its original paths. The reconstructed stack replaces `/entry1/data/data`; the depth vector replaces `/entry1/depth`. The writer omits `/entry1/wire` and replaces `/entry1/data/depth` with a link to the new depth vector. It replaces the data group's signal, axes, default, auxiliary-signal, and axis-index attributes so they cannot describe raw frames as reconstructed depths. Other input datasets and attributes are copied.

Root format/version attributes and NeXus entry/data annotations describe the new output. Inputs containing the reserved `/entry1/reconstruction` path are rejected. Per-depth export retains the acquisition metadata and writes one image and one physical depth per file, with NeXus annotations adjusted for a 2-D image.

A point file supports inspection, indexing, and per-depth export without its catalog, raw input, or geometry file. It does not contain the raw frames, so reconstructing the point again requires the input.

`/entry1/data/data` uses HDF5 chunks. Chunk shape and compression may change as measurements of frame reads, ROI reads through depth, and stripe writes inform storage choices.

## Status and completeness

| Code | Point status | Meaning |
| --- | --- | --- |
| 0 | `pending` | Not started. This is also the value of an entry that was never written. |
| 1 | `writing` | Dispatched for reconstruction; awaiting a recorded outcome. |
| 2 | `complete` | The point file is published. |
| 3 | `failed` | An expected input or reconstruction failure; `/catalog/errors` explains it. |
| 4 | `interrupted` | Dispatched, but the run ended before an outcome was recorded. |
| 5 | `unattempted` | The run ended before the point was dispatched. |

| Code | Run status | Meaning |
| --- | --- | --- |
| 0 | `running` | The coordinator is running the scan. |
| 1 | `finished` | Every point was attempted. Individual points can have failed. |
| 2 | `cancelled` | The run stopped on request after its active points drained. |
| 3 | `failed` | A shared failure stopped the run, or the coordinator was closed before the run was finished. |

The coordinator keeps live state in memory. Dispatches and outcomes update that state immediately; pending changes are published at most once every five seconds. Preparation, successful completion, cancellation, and failure publish immediately, without waiting for the interval. An unchanged catalog is not rewritten.

The local runner checks for a due snapshot while waiting for workers. An external scheduler should call {meth}`~lauelab.reconstruct.PreparedScan.snapshot` in its polling loop as well as recording dispatches and outcomes. The method returns quickly when nothing is due. `snapshot(force=True)` publishes pending changes immediately. There is no background writer, so a coordinator blocked outside these calls cannot publish on schedule.

The coordinator writes each snapshot to a private `.partial` file in the scan directory, closes and validates it, and moves it over `scan.h5`. The first snapshot is published before any point is computed and has run status `running`. Published catalogs can therefore have any run status. A `finished` or `cancelled` catalog holds only the point statuses `complete`, `failed`, `interrupted`, and `unattempted`. A catalog reader retains the snapshot it loaded; create a new reader to see later changes.

A worker writes each point file to a private `.partial` file beside its final path, closes it, validates it against its task, and publishes it without replacing another file. Published point files are complete and immutable. {func}`~lauelab.reconstruct.reconstruct_point` handles input and reconstruction failures differently from output failures:

| Failure | Examples | Result |
| --- | --- | --- |
| Input or reconstruction of the point | Missing or unreadable input, invalid metadata, an input changed since preparation, a stripe-buffer budget too small for one row, a native reconstruction error | A `failed` outcome with the error. Nothing is published, and other points continue. |
| Output of the point | The private file cannot be created or written, validation or publication fails, the final file or another worker's private file already exists | {class}`~lauelab.indexing.ReconstructionError` is raised. The worker removes its own private file and never replaces another file. |

When you call {meth}`~lauelab.reconstruct.PreparedScan.record` with a complete outcome, the coordinator validates the published file against the prepared task. It checks the point ID, manifest index, shape, depth bounds, pixel type, settings, embedded geometry, source path, selected raw-slice range, and acquisition values without reading pixels. Outcomes for unknown tasks and duplicate outcomes are rejected.

Point files and the catalog are published separately. If a run stops after a point file is published but before the next catalog snapshot, that file remains usable directly even though the catalog does not list it as complete. Readers use the catalog status; they do not search the directory for additional completed files. Periodic snapshots preserve recent progress after a process crash, but automatic resume and reconciliation are not supported. The format makes no power-loss durability guarantee.

Structural validation of a catalog checks required datasets, dtypes, shapes, catalog lengths, status codes, point IDs, and point paths, and the consistency of each catalog row. It reads `scan.h5` only and leaves it unchanged. A scan is complete when every point has `complete` status. Catalogs of cancelled or partly failed runs can therefore pass validation while listing incomplete points.

## Reader responsibilities

Point files, per-depth files, and in-memory arrays provide the same frame and region access methods for inspection and plotting.

{class}`~lauelab.reconstruct.ScanReader` loads a catalog snapshot on creation, then closes the catalog file. It lists point IDs, statuses, and metadata in manifest order without opening point files, and resolves each point path against the scan directory. Opening one point opens only that point's file and checks its identity, output shape and dtype, physical depth bounds, source path, and acquisition values against the catalog.

{class}`~lauelab.reconstruct.PointReader` owns the handle of one point file. Open it as a context manager in the process that uses it; it cannot be sent to another process. A point opened through a `ScanReader` closes when the caller closes it or closes that `ScanReader`. {class}`~lauelab.reconstruct.PerDepthReader` accesses existing per-depth files, and {class}`~lauelab.reconstruct.inspection.ArrayPoint` accesses arrays. `PerDepthReader` orders files by their embedded physical depths and opens at most one file at a time. Its totals and reconstructed reference are computed on demand from stored pixels; raw references are unavailable.

For each point, `frame()` reads one frame, `region()` reads a rectangular selection over a depth range, and `iter_blocks()` reads successive depth blocks within a caller-supplied byte budget. Reference-image requests raise `InputError` when the requested image is unavailable. Returned arrays belong to the caller.

ROI reductions read bounded depth blocks, with a default pixel budget of 64 MiB. The budget must hold one ROI plane. It excludes the small returned trace and HDF5 caches. Direct `region()` calls allocate the requested selection, so callers doing their own analysis must choose bounded depth slices for large regions.

## Square ROIs

Inspection ROIs are square selections in stored-image pixels, created during analysis and kept outside the point file. The point file's `/entry1/reconstruction/detector/roi_start` and `/entry1/reconstruction/detector/roi_group` describe the acquisition readout region and binning used by indexing.

ROI coordinates are zero-based frame pixel coordinates `(x, y)` of the stored image, and an integer coordinate is a pixel centre, so pixel `(x, y)` covers `x - 0.5` to `x + 0.5`. NumPy reads that pixel as `frame[y, x]`. Unbinned detector coordinates are not used.

An ROI is represented by integer half-open bounds `(y0, y1, x0, x1)`, which select `data[:, y0:y1, x0:x1]`. A square of size `N` has exactly `N × N` pixels. `N` is a positive integer.

A click at `(x, y)` places the square whose centre is the nearest legal centre. The legal centres of an odd size are the integers, and those of an even size are the half-integers. A click at equal distance from two legal centres selects the lower coordinate. On each axis this gives

```{math}
x_0 = \lceil x - N/2 \rceil, \qquad x_1 = x_0 + N,
```

and the displayed centre is {math}`x_0 + (N - 1)/2`, which is derived from the bounds. A placement is rejected when any bound lies outside the image or the click is not finite. Accepted bounds are used unchanged at every display scale.

An ROI trace is the sum of the stored pixels in the bounds at each depth, in the accumulator dtype of the point. The full-frame trace is `/entry1/reconstruction/stored_depth_intensity/data`. Both are plotted against physical depth in µm or against depth index.

A normalized trace is the trace divided by its own maximum. Signs are preserved. When the maximum is zero or negative, the normalized trace is unavailable and the reason is reported; the sum remains available. A logarithmic axis omits each sample that is zero or negative and reports how many it omitted. Omitted samples leave gaps in plotted lines. An empty ROI selection returns no traces.


## Compatibility

- Version 1 of `lauelab-reconstruction-scan` stored every point inside one file. That layout was retired before release. Readers and the validator reject it with an error that names the retired layout; they never interpret it as a catalog or a point file. Reconstruct such points again to write a scan directory.
- The existing reconstruction API keeps its return types, defaults, and per-depth output.
- The in-process and executable paths remain cross-checks, and the recorded references under `tests/data/reconstruction/` pin both. The stored frames of a point file must equal the per-depth frames for every recorded variant.
- {func}`~lauelab.reconstruct.export_per_depth` reads stored pixels from a point file and writes them unchanged with their dtype, their physical depth, the acquisition metadata, and the normalization provenance. It does not run reconstruction and does not apply `rescale` again. Its LaueGo-compatible summary reports `/entry1/reconstruction/computed_depth_intensity/data`. A failed export does not invalidate the point file.

## Evidence

`tests/test_reconstruction_contract.py` checks the conversion rule against HDF5, records what each reference variant contains, and verifies the hand-worked fixtures in `tests/data/reconstruction_contract/fixtures.py` with brute-force oracles. The fixtures supply the cases that no recorded reference contains: a zero frame, a frame with no positive pixel, saturation at both limits, and integer output with `rescale` above 1. `tests/test_scan_preparation.py` checks preparation, tasks, destinations, and the catalog validator. `tests/test_scan_layout.py` compares the tables on this page with the layout module.
