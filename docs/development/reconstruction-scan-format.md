# Reconstruction scan format

The `lauelab-reconstruction-scan` format stores a reconstruction run in one HDF5 file. A catalog lists the input points and their status. Each completed point contains reconstructed frames, physical depths, reference images, and acquisition and reconstruction metadata. This page defines the layout, pixel values, frame selection, and square regions of interest (ROIs).

{func}`~lauelab.reconstruct.reconstruct_scan` writes the format, {class}`~lauelab.reconstruct.ScanReader` reads it, and {func}`~lauelab.reconstruct.validate_scan_file` checks it; [Reconstruct a wire scan](../guides/reconstruction.md) shows their use. The layout module `lauelab/reconstruct/_scan_layout.py` is the single definition that all three follow. {class}`~lauelab.reconstruct.Reconstructor`, {func}`~lauelab.reconstruct.reconstruct_points`, {func}`~lauelab.reconstruct.reconstruct`, and their per-depth output do not change.

The ROI rules on this page are implemented by `lauelab.reconstruct.inspection` and rendered by the depth-inspection builders in `lauelab.visualization`; the fixtures under `tests/data/reconstruction_contract/` pin them so that the Laue Portal is built against the same definition.

The file follows the [HDF5 file conventions](hdf5-conventions.md). Its root `format` attribute is `lauelab-reconstruction-scan` and its `version` is 1.

## Identities

Frames are selected by point ID and depth index. Catalog entries follow the input manifest order.

| Identity | Type | Meaning |
| --- | --- | --- |
| Point ID | string | Unique, non-empty identifier assigned to a point for the duration of the run. Defaults to the manifest index in decimal. |
| Manifest index | zero-based integer | Position of the point in the input manifest, which is frozen before computation. Every catalog dataset uses this order. Determines the point group path: `/points/000000` for index 0. |
| Depth index | zero-based integer | Position of a frame in the point's `(depth, y, x)` stack. |
| Physical depth | `numpy.float64`, µm | Depth along the incident beam from the Si origin of the geometry file. It is `depth_um[depth index]`. |

Point IDs are stored as text values, allowing inputs with the same file stem to have distinct IDs. HDF5 group paths use manifest indices.

Physical depths are stored in `depth_um`. For example, depth index 0 corresponds to -25 µm when the reconstructed depth grid starts at -25 µm.

A **frame reference**, {class}`~lauelab.indexing.ScanFrame`, selects one stored frame by the path of the file, a point ID, and a depth index. The reference can be serialized and passed to indexing workers, which open the file locally. Indexing results preserve the path alongside `frames/source_point_ids` and `frames/source_depth_indices`.

## Value semantics

Pixel values are described at three stages of processing:

| Kind | Dtype | Meaning |
| --- | --- | --- |
| Raw | dtype of the input frames | Detector counts as acquired, before the cosmic-ray filter and before any normalization. |
| Computed | `numpy.float64` | Output of the reconstruction kernel. These are `ReconstructionResult.images`, and their per-depth totals are `ReconstructionResult.depth_intensity`. |
| Stored | the point's output pixel type | The pixels in the file. Inspection, ROI traces, and downstream indexing use these values. |

A stored pixel is the conversion of `computed × rescale` to the output pixel type. `rescale` is 1 unless exponent normalization writes an integer type; it is recorded as `normalization/rescale` and is applied once. For a floating-point type, the conversion equals a NumPy cast. For an integer type, the conversion truncates toward zero and then saturates at the limits of the type. Positive and negative infinity saturate. NaN stores 0 in an integer type.

The native library converts pixels and computes reductions in one pass. For finite values and infinities, conversion matches the HDF5 conversion used by per-depth output. Integer conversion of NaN depends on the HDF5 destination type and can differ from the scan format's defined value of zero.

```{warning}
Reconstructed intensities can be negative with any `wire_edge` setting. An unsigned output type stores 0 for each negative pixel, so its stored totals are larger than the computed totals. For the recorded leading-edge reference, the `numpy.uint16` frames sum to 573801 and the computed frames sum to 512684.9.
```

### Reductions

A reduction of stored pixels is computed after conversion, so that it equals the same reduction of the reopened file.

- Integer stored pixels accumulate in `numpy.int64` and are exact. A sum of `n` values of at most 4 bytes cannot overflow while `n` is below 2³¹, which exceeds the pixel count of a 4096 by 4096 frame.
- Floating-point stored pixels accumulate in `numpy.float64`. The summation order is not specified. Reconstructed pixels are signed, so a total can be much smaller than the pixels that produce it, and a tolerance relative to the total is not meaningful. Compare a reduction with a recomputed one at an absolute tolerance of 10⁻¹² times the sum of the absolute pixel values in the reduction.

`computed/depth_intensity` contains the totals before storage conversion, as reported in the LaueGo-compatible summary. `reductions/depth_intensity` contains the stored-pixel totals used for inspection. Conversion and `rescale` can therefore change the totals.

### Reference images

Each point embeds three images with the shape of one frame, so that a point can be inspected when its raw file is no longer available.

| Image | Values | Definition |
| --- | --- | --- |
| `reference/first_raw` | raw | The first scan frame the reader selects. |
| `reference/sum_raw` | raw | The sum of every selected scan frame. |
| `reference/sum_reconstructed` | stored | The sum of the stored frames through depth. |

`reference/raw_slices` records the half-open range of stored input slices that was selected. For a 34-ID-E multi-image file with `n` stored slices the range is `[1, n - 1)`: slice 0 is bookkeeping, and the last slice is never differenced. See "How a point file is read" in [Reconstruct a wire scan](../guides/reconstruction.md). The raw stack itself is not copied.

## File layout

Shapes use `n_points` for the manifest length. `n_depths`, `rows`, and `columns` belong to one point; points in one run can differ in all three and in output pixel type. A dtype given as a rule is resolved for each point:

| Rule | Dtype |
| --- | --- |
| `stored` | The point's output pixel type. |
| `input` | The dtype of the point's raw frames. |
| `stored_accumulator` | `<i8` when `stored` is an integer type, otherwise `<f8`. |
| `raw_accumulator` | `<i8` when `input` is an integer type of at most 4 bytes, otherwise `<f8`. |

The **Missing** column gives the value stored when the quantity does not apply or was not available.

### Run

| Path | Dtype | Shape | Units | Missing | Meaning |
| --- | --- | --- | --- | --- | --- |
| `/run/status` | `u1` | `()` | | | Run status code |
| `/run/native_version` | `string` | `()` | | | Version of the native reconstruction library |
| `/settings/detector` | `<i4` | `()` | | | Detector slot in the geometry |
| `/settings/depth_range` | `<f8` | `(2,)` | um | | Requested inclusive depth range |
| `/settings/resolution` | `<f8` | `()` | um | | Spacing between depths |
| `/settings/wire_edge` | `string` | `()` | | | `leading`, `trailing`, or `both` |
| `/settings/percent_brightest` | `<f8` | `()` | | | Percentage of intensity-map pixels reconstructed |
| `/settings/normalization` | `string` | `()` | | empty | Name of the normalization vector |
| `/settings/norm_exponent` | `<f8` | `()` | | NaN | Exponent normalization |
| `/settings/norm_threshold` | `<f8` | `()` | | NaN | Requested threshold; each point records the value used |
| `/settings/cosmic_filter` | `u1` | `()` | | | 1 when the cosmic-ray filter ran |
| `/settings/output_pixel_type` | `<i4` | `()` | | -1 | Requested pixel type code; each point records the type used |
| `/settings/memory_limit_mb` | `<i4` | `()` | | | Stripe-buffer limit of the run in MiB |
| `/geometry/path` | `string` | `()` | | empty | Geometry file path as given to the run |
| `/geometry/xml` | `string` | `()` | | | Complete text of the geometry file |

### Catalog

Each catalog dataset has one entry per point in manifest order. Listing a run reads the catalog only.

| Path | Dtype | Shape | Units | Missing | Meaning |
| --- | --- | --- | --- | --- | --- |
| `/catalog/point_ids` | `string` | `(n_points,)` | | | Point ID |
| `/catalog/source_paths` | `string` | `(n_points,)` | | | Input path as given |
| `/catalog/source_sizes` | `<i8` | `(n_points,)` | | -1 | Input size in bytes |
| `/catalog/source_mtimes_ns` | `<i8` | `(n_points,)` | | -1 | Input modification time in ns since the epoch |
| `/catalog/scan_numbers` | `<i4` | `(n_points,)` | | -1 | Acquisition scan number |
| `/catalog/sample_positions` | `<f8` | `(n_points, 3)` | um | NaN | Sample `(x, y, z)` in the acquisition coordinate system |
| `/catalog/energies_kev` | `<f8` | `(n_points,)` | keV | NaN | Incident energy |
| `/catalog/image_shapes` | `<i4` | `(n_points, 2)` | | | Frame shape `(rows, columns)` |
| `/catalog/n_depths` | `<i4` | `(n_points,)` | | | Number of stored frames |
| `/catalog/depth_bounds` | `<f8` | `(n_points, 2)` | um | NaN | First and last physical depth |
| `/catalog/pixel_types` | `<i4` | `(n_points,)` | | -1 | Output pixel type code used |
| `/catalog/status` | `u1` | `(n_points,)` | | | Point status code |
| `/catalog/errors` | `S1024` | `(n_points,)` | | empty | UTF-8 error text for a failed point |

Input sizes and modification times are recorded for possible future resume support; version 1 does not use them. A file-status call retrieves both values without reading the input contents. They provide a limited change check: an unchanged file can acquire a new modification time, while changed contents can retain the same size and modification time. Content verification would require a checksum.

### Point

These paths are relative to the point group. A point whose input could not be read when the manifest was frozen has a `failed` catalog entry with zero shape and no group; it does not stop the other points.

| Path | Dtype | Shape | Units | Missing | Meaning |
| --- | --- | --- | --- | --- | --- |
| `data` | `stored` | `(n_depths, rows, columns)` | | | Stored frames |
| `depth_um` | `<f8` | `(n_depths,)` | um | | Physical depth of each frame |
| `detector/id` | `string` | `()` | | empty | Detector identifier from the input |
| `detector/size` | `<i4` | `(2,)` | pixel | | Full detector `(x, y)` size in unbinned pixels |
| `detector/roi_start` | `<i4` | `(2,)` | pixel | | Zero-based `(x, y)` frame origin in unbinned pixels |
| `detector/roi_group` | `<i4` | `(2,)` | | | `(x, y)` binning factors |
| `normalization/threshold` | `<f8` | `()` | | NaN | Exponent-normalization threshold used |
| `normalization/rescale` | `<f8` | `()` | | | Factor applied before conversion; 1 when none |
| `reference/first_raw` | `input` | `(rows, columns)` | | | First selected raw frame |
| `reference/sum_raw` | `raw_accumulator` | `(rows, columns)` | | | Sum of the selected raw frames |
| `reference/raw_slices` | `<i8` | `(2,)` | | | Half-open range of selected input slices |
| `reference/sum_reconstructed` | `stored_accumulator` | `(rows, columns)` | | | Sum of the stored frames |
| `reductions/depth_intensity` | `stored_accumulator` | `(n_depths,)` | | | Stored total of each frame |
| `computed/depth_intensity` | `<f8` | `(n_depths,)` | | | Computed total of each frame |

The `source` group of a point holds a copy of the input file's objects without `entry1/data/data` and `entry1/wire`. That content follows the 34-ID-E layout, which `lauelab` does not define. Per-depth export copies it back unchanged.

`data` uses HDF5 chunks. Chunk shape and compression may change as measurements of frame reads, ROI reads through depth, and stripe writes inform storage choices. Every dataset of a point is created before computation starts, and no dataset that a writer changes later has a variable-length type; `/catalog/errors` is a fixed-width string for that reason.

## Status and completeness

| Code | Point status | Meaning |
| --- | --- | --- |
| 0 | `pending` | Not started. This is also the value of an entry that was never written. |
| 1 | `writing` | Reconstruction or writing is in progress. |
| 2 | `complete` | Frames, reference images, reductions, and metadata are written. |
| 3 | `failed` | An expected input or reconstruction failure; `/catalog/errors` explains it. |
| 4 | `interrupted` | Started, and the run ended before the point completed. |
| 5 | `unattempted` | The run ended before the point started. |

| Code | Run status | Meaning |
| --- | --- | --- |
| 0 | `running` | The writer has the file open. |
| 1 | `finished` | Every point was attempted. Individual points can have failed. |
| 2 | `cancelled` | The run stopped on request after its active points drained. |
| 3 | `failed` | A shared failure stopped the run. |

Pixel, reference-image, and reduction reads require `complete` status. Incomplete points may contain unwritten chunks whose fill values would be mistaken for reconstructed intensities. Completed points are immutable.

A producer writes to the `.partial` name, closes the file, validates it, and publishes it, as the conventions require. A published file has run status `finished` or `cancelled` and holds only the point statuses `complete`, `failed`, `interrupted`, and `unattempted`. A file with run status `running` or `failed` is never published.

Structural validation checks required datasets, dtypes, shapes, catalog lengths, and allowed status codes. A run is complete when every point has `complete` status. Files from cancelled or partly failed runs can therefore pass validation while containing incomplete points. Validation uses bounded reads and leaves the file unchanged.

## Reader responsibilities

Scan files, per-depth files, and in-memory arrays provide the same frame and region access methods for inspection and plotting.

{class}`~lauelab.reconstruct.PointReader` accesses the new scan format, {class}`~lauelab.reconstruct.PerDepthReader` accesses existing per-depth files, and {class}`~lauelab.reconstruct.inspection.ArrayPoint` accesses arrays. `PerDepthReader` orders files by their embedded physical depths and opens at most one file at a time. Its totals and reconstructed reference are computed on demand from stored pixels; raw references are unavailable.

Open file readers as context managers in the process that will use them. `ScanReader` lists point IDs and statuses in manifest order and exposes point metadata without loading pixels.

For each point, `frame()` reads one frame, `region()` reads a rectangular selection over a depth range, and `iter_blocks()` reads successive depth blocks within a caller-supplied byte budget. Reference-image requests raise `InputError` when the requested image is unavailable. Returned arrays belong to the caller.

ROI reductions read bounded depth blocks, with a default pixel budget of 64 MiB. The budget must hold one ROI plane. It excludes the small returned trace and HDF5 caches. Direct `region()` calls allocate the requested selection, so callers doing their own analysis must choose bounded depth slices for large regions.

## Square ROIs

Inspection ROIs are square selections in stored-image pixels, created during analysis and kept outside the scan file. The file's `detector/roi_start` and `detector/roi_group` describe the acquisition readout region and binning used by indexing.

ROI coordinates are zero-based frame pixel coordinates `(x, y)` of the stored image, and an integer coordinate is a pixel centre, so pixel `(x, y)` covers `x - 0.5` to `x + 0.5`. NumPy reads that pixel as `frame[y, x]`. Unbinned detector coordinates are not used.

An ROI is represented by integer half-open bounds `(y0, y1, x0, x1)`, which select `data[:, y0:y1, x0:x1]`. A square of size `N` has exactly `N × N` pixels. `N` is a positive integer.

A click at `(x, y)` places the square whose centre is the nearest legal centre. The legal centres of an odd size are the integers, and those of an even size are the half-integers. A click at equal distance from two legal centres selects the lower coordinate. On each axis this gives

```{math}
x_0 = \lceil x - N/2 \rceil, \qquad x_1 = x_0 + N,
```

and the displayed centre is {math}`x_0 + (N - 1)/2`, which is derived from the bounds. A placement is rejected when any bound lies outside the image or the click is not finite. Accepted bounds are used unchanged at every display scale.

An ROI trace is the sum of the stored pixels in the bounds at each depth, in the accumulator dtype of the point. The full-frame trace is `reductions/depth_intensity`. Both are plotted against physical depth in µm or against depth index.

A normalized trace is the trace divided by its own maximum. Signs are preserved. When the maximum is zero or negative, the normalized trace is unavailable and the reason is reported; the sum remains available. A logarithmic axis omits each sample that is zero or negative and reports how many it omitted. Omitted samples leave gaps in plotted lines. An empty ROI selection returns no traces.

## Compatibility

- The existing reconstruction API keeps its return types, defaults, and per-depth output. The new format is added beside it.
- The in-process and executable paths remain cross-checks, and the recorded references under `tests/data/reconstruction/` pin both. The stored frames of this format must equal the per-depth frames for every recorded variant.
- {func}`~lauelab.reconstruct.export_per_depth` reads stored pixels and writes them unchanged with their dtype, their physical depth, the `source` metadata, and the normalization provenance. It does not run reconstruction and does not apply `rescale` again. Its LaueGo-compatible summary reports `computed/depth_intensity`. A failed export does not invalidate the scan file.

## Evidence

`tests/test_reconstruction_contract.py` checks the conversion rule against HDF5, records what each reference variant contains, and verifies the hand-worked fixtures in `tests/data/reconstruction_contract/fixtures.py` with brute-force oracles. The fixtures supply the cases that no recorded reference contains: a zero frame, a frame with no positive pixel, saturation at both limits, and integer output with `rescale` above 1. `tests/test_scan_layout.py` compares the tables on this page with the layout module.
