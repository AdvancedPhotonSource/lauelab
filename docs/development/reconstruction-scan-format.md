# Reconstruction scan format

A reconstructed scan is a directory with one HDF5 file per point and a catalog, `scan.h5`. The catalog contains the list of requested points, their acquisition metadata, and their outcomes. Each completed file under `points/` contains the reconstructed depth stack and all the data needed to inspect, index, or export it without other files: physical depths, reference images, settings, geometry, and acquisition metadata.

This page specifies the two file layouts and the pixel values that they contain. [Reconstruction scan protocol](reconstruction-scan-protocol.md) describes scan preparation, coordination, point and run statuses, and the readers. [Inspect a reconstructed point](../guides/depth-inspection.md) defines square regions of interest (ROIs). For examples, see [Reconstruct a wire scan](../guides/reconstruction.md).

The writers, readers, and validators use the layout definitions in `lauelab/reconstruct/_scan_layout.py`.

Both files use the [HDF5 file conventions](hdf5-conventions.md), with the exception for point files stated there. The root `format` attribute of the catalog is `lauelab-reconstruction-scan`, and its `version` is 2. The root `format` attribute of a point file is `lauelab-reconstruction-point`, and its `version` is 2.

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

The scan directory must not exist or must be empty. A nonempty directory is rejected, so the output of two scans cannot mix. Choose another directory, or remove the existing output yourself.

`/points/paths` contains the path of each point file relative to the directory of `scan.h5`, with `/` separators, such as `points/Twin2_wire_1.h5`. The scan remains readable when the whole directory is moved. A point file can also be copied and used by itself.

`scan.h5` also contains one HDF5 external link for each complete point. `/point_files/<stem>` is a link to the root of `points/<stem>.h5` by its relative path. HDF5 resolves the path from the directory of `scan.h5`, so an HDF5 viewer such as HDFView can open every complete point from the catalog. The lauelab readers use `/points/paths` and do not follow the links. Neither file contains virtual datasets.

## Point file structure

A point file has the layout of the per-depth files that the reconstruction executable `reconstructN_cpu` writes, with all depths in one file. The writer copies the input file as the executable does. The root attributes, the groups and their attributes, and the datasets are copied with their paths and values, except `/entry1/wire` and the raw frames. The writer then adds these objects:

- `/entry1/data/data` contains the stored frames, shape `(n_depths, rows, columns)`, with the integer attribute `signal=1`. Image indexing is `data[depth_index, y, x]`.
- `/entry1/depth` contains the physical depth of each frame in µm along the incident beam from the Si origin of the geometry, shape `(n_depths,)`, with `units="micron"`. It replaces the `/entry1/depth` of the input.
- `/entry1/microDiffraction/cosmic_filter` is written, with the value 1, only when the cosmic-ray filter ran. `/entry1/microDiffraction/norm_exponent`, `norm_threshold`, and `norm_rescale` are written only when exponent normalization ran. These are the paths and types that the executable writes.
- `/entry1/reconstruction` is a group of datasets that contain the reconstruction settings that were used, the identity and provenance of the point, the reference images, and the depth totals.

The root also has the lauelab `format`, `version`, `lauelab_version`, and `created` attributes. The root attributes copied from the input, such as `file_name` and `file_time`, are attributes of the acquisition file. The `NX_class` attributes that the acquisition software wrote are copied with their groups. The writer adds no NeXus attributes.

The datasets added by lauelab have the form of the acquisition values: a single value has shape `(1,)`, text is fixed-length and UTF-8 encoded, and units have the spelling of the acquisition files, such as `micron`. A value that does not apply is not written. For example, a standalone point has no `manifest_index`, and a point reconstructed without a normalization vector has no `normalization`. `tests/test_point_files.py` compares a point file with the output of the executable for every recorded variant.

## Identities

A frame is selected by its point and its depth index. The initial catalog entries are in input order. For an incremental scan, each successful `PreparedScan.add` appends one entry in registration order. Existing indices and point files never change. Acquisition order and sample coordinates are independent of registration order; use the point IDs and acquisition metadata to place points in a view.

A running catalog can contain zero points. It lists only registered inputs, whose metadata has been inspected; expected future inputs belong to the caller's acquisition manifest. A finished catalog must contain at least one point. These rules use the same version-2 datasets and point-file layout as a batch scan.

| Identity | Type | Meaning |
| --- | --- | --- |
| Point ID | string | Unique, non-empty identifier of a point within its scan. Defaults to the input stem. It is independent of the point filename. |
| Manifest index | zero-based integer | Position of the point in registration order, initially the input order. Every dataset under `/points` uses this order. |
| Point path | string | Location of the point file relative to the scan directory. It is assigned when the point is registered, whatever the point's outcome. |
| Depth index | zero-based integer | Position of a frame in the point's `(n_depths, ny, nx)` stack. |
| Physical depth | `numpy.float64`, µm | Depth along the incident beam from the Si origin of the geometry file. It is `depth_um[depth index]`. |

The point ID and the manifest index are stored in the point file, so they are available when the file is copied elsewhere. Standalone points have no manifest index. Physical depths are stored in `/entry1/depth`. For example, depth index 0 corresponds to -25 µm when the reconstructed depth grid starts at -25 µm.

A **frame reference**, {class}`~lauelab.indexing.ScanFrame`, contains the path of a point file and a depth index, which identify one stored frame. A frame reference can be serialized and sent to indexing workers, which open the file locally. Indexing results contain the point ID and the depth index.

## Value semantics

Pixel values have three kinds, one for each stage of processing:

| Kind | Dtype | Meaning |
| --- | --- | --- |
| Raw | dtype of the input frames | Detector counts as acquired, before the cosmic-ray filter and before any normalization. |
| Computed | `numpy.float64` | Output of the reconstruction kernel. These are `ReconstructionResult.images`, and their per-depth totals are `ReconstructionResult.depth_intensity`. |
| Stored | the point's output pixel type | The pixels in the file. Inspection, ROI traces, and downstream indexing use these values. |

Before writing, the library multiplies computed intensities by `rescale` and converts them to the output pixel type. `rescale` is 1 unless exponent normalization writes an integer type; it is recorded as `/entry1/microDiffraction/norm_rescale` when exponent normalization ran, and it is applied once. For a floating-point type, the conversion equals a NumPy cast. For an integer type, the conversion truncates toward zero and then saturates at the limits of the type. Positive and negative infinity saturate. NaN stores 0 in an integer type.

The native library converts pixels and computes reductions in one pass. For finite values and infinities, conversion matches the HDF5 conversion used by per-depth output. Integer conversion of NaN depends on the HDF5 destination type and can differ from the scan format's defined value of zero.

```{warning}
Reconstructed intensities can be negative with any `wire_edge` setting. In an unsigned output type, each negative pixel is stored as 0, so the stored totals are larger than the computed totals. For the recorded leading-edge reference, the `numpy.uint16` frames sum to 573801 and the computed frames sum to 512684.9.
```

### Reductions

Stored-pixel totals are calculated after conversion, from the same values that a read of the file returns.

- Integer stored pixels accumulate in `numpy.int64` and are exact. A sum of `n` values of at most 4 bytes cannot overflow while `n` is below 2³¹, which exceeds the pixel count of a 4096 by 4096 frame.
- Floating-point stored pixels accumulate in `numpy.float64`. The summation order is not specified. Reconstructed pixels are signed, so a total can be much smaller than the pixels that produce it, and a tolerance relative to the total is not meaningful. Compare a reduction with a recomputed one at an absolute tolerance of 10⁻¹² times the sum of the absolute pixel values in the reduction.

`/entry1/reconstruction/computed_depth_intensity` contains the totals before storage conversion, which are also in the LaueGo-compatible summary. `/entry1/reconstruction/depth_intensity` contains the stored-pixel totals, which inspection uses. Because of conversion and `rescale`, the two totals can differ.

### Reference images

Three reference images are saved with each point for inspection after the raw input becomes unavailable. Each has the same shape as a detector frame.

| Image | Values | Definition |
| --- | --- | --- |
| `/entry1/reconstruction/first_raw` | raw | The first selected scan frame. |
| `/entry1/reconstruction/sum_raw` | raw | The sum of every selected scan frame. |
| `/entry1/reconstruction/sum_reconstructed` | stored | The sum of the stored frames through depth. |

`/entry1/reconstruction/raw_slices` contains the half-open range of the selected input slices. For a 34-ID-E multi-image file with `n` stored slices the range is `[frame_skip, n - 1)`: the first `frame_skip` slices are bookkeeping, and the last slice is never differenced. See "How a point file is read" in [Reconstruct a wire scan](../guides/reconstruction.md). The raw stack itself is not copied.

## File layout

In the tables below, `n_points` is the number of points in the scan. The depth count (`n_depths`), frame dimensions (`rows`, `columns`), and output pixel type can vary between points. The following dtype rules apply separately to each point:

| Rule | Dtype |
| --- | --- |
| `stored` | The point's output pixel type. |
| `input` | The dtype of the point's raw frames. |
| `stored_accumulator` | `<i8` when `stored` is an integer type, otherwise `<f8`. |
| `raw_accumulator` | `<i8` when `input` is an integer type of at most 4 bytes, otherwise `<f8`. |

In the catalog tables, the **Fill** column gives the value that is stored when a quantity does not apply or is not available. For a numeric dataset, this value is also the HDF5 fill value. `enum` identifies an HDF5 enumerated type, which HDF5 tools display by name.

### Root attributes

In addition to the attributes common to lauelab files, the root of `scan.h5` has these attributes:

| Attribute | Type | Meaning |
| --- | --- | --- |
| `program` | string | `lauelab` |
| `native_version` | string | Version of the native library that prepared the scan |
| `status` | `enum u1` | Run status; the codes are listed in [Reconstruction scan protocol](reconstruction-scan-protocol.md) |

### Parameters and geometry

These datasets contain the requested settings, with the names of the {class}`~lauelab.reconstruct.Reconstructor` arguments. A point file contains the settings that were used for that point; see the point-file table below. `/geometry` also has a `path` attribute, which contains the geometry file path as given.

| Path | Dtype | Shape | Units | Fill | Meaning |
| --- | --- | --- | --- | --- | --- |
| `/parameters/detector` | `<i4` | `()` | | | Physical detector slot in the geometry |
| `/parameters/depth_range` | `<f8` | `(2,)` | um | | Requested inclusive depth range |
| `/parameters/resolution` | `<f8` | `()` | um | | Spacing between depths |
| `/parameters/wire_edge` | `string` | `()` | | | `leading`, `trailing`, or `both` |
| `/parameters/percent_brightest` | `<f8` | `()` | | | Percentage of intensity-map pixels reconstructed |
| `/parameters/normalization` | `string` | `()` | | empty | Name of the normalization vector |
| `/parameters/norm_exponent` | `<f8` | `()` | | NaN | Exponent normalization |
| `/parameters/norm_threshold` | `<f8` | `()` | | NaN | Requested threshold; NaN selects it automatically for each point |
| `/parameters/cosmic_filter` | `u1` | `()` | | | 1 when the cosmic-ray filter is requested |
| `/parameters/frame_skip` | `<i4` | `()` | | | Leading stored slices skipped |
| `/parameters/wire_skip` | `<i4` | `()` | | | Stored wire-vector entries skipped after entry 0 |
| `/parameters/output_pixel_type` | `<i4` | `()` | | -1 | Requested pixel type code; -1 selects it from the input |
| `/parameters/rows_per_stripe` | `<i4` | `()` | | -1 | Requested rows per stripe; -1 selects them automatically |
| `/parameters/memory_limit_mb` | `<i4` | `()` | MiB | | Stripe-buffer limit of one worker |
| `/geometry/xml` | `string` | `()` | | | Complete text of the geometry file, read once when the scan is prepared |

### Points

Each dataset under `/points` has one entry per point in manifest order. Listing a scan reads `scan.h5` only; no point file is opened.

| Path | Dtype | Shape | Units | Fill | Meaning |
| --- | --- | --- | --- | --- | --- |
| `/points/point_ids` | `string` | `(n_points,)` | | | Point ID |
| `/points/paths` | `string` | `(n_points,)` | | | Point file relative to the scan directory |
| `/points/statuses` | `enum u1` | `(n_points,)` | | | Point status |
| `/points/errors` | `string` | `(n_points,)` | | empty | Error text of a failed point |
| `/points/source_paths` | `string` | `(n_points,)` | | | Absolute input path |
| `/points/scan_numbers` | `<i4` | `(n_points,)` | | -1 | Acquisition scan number |
| `/points/sample_positions` | `<f8` | `(n_points, 3)` | um | NaN | Sample `(x, y, z)` in the acquisition coordinate system |
| `/points/energies_kev` | `<f8` | `(n_points,)` | keV | NaN | Incident energy |
| `/points/shapes` | `<i4` | `(n_points, 3)` | | 0 | `(n_depths, ny, nx)` of the stored frames; zeros when the input could not be read |
| `/points/depth_bounds` | `<f8` | `(n_points, 2)` | um | NaN | First and last physical depth |
| `/points/dtypes` | `enum i1` | `(n_points,)` | | -1 | Output pixel type: `float32`, `int32`, `int16`, `uint16`, `float64`, `int8`, or `uint8`, stored as the LaueGo pixel-type code; `none` when the input could not be read |

The input path is stored for provenance. It does not identify the contents of the input, and a scan cannot be resumed from it. Each point file also contains the size and modification time of its input. A file-status call reads these values without reading the file, so an unchanged file can have a new modification time, and changed contents can have the same size and modification time.

### Point file

Each completed point file contains the datasets below and the metadata copied from its input. The **When** column gives the condition for an optional dataset; the other datasets are always present. The file is at its final path only after writing and validation succeed.

| Path | Dtype | Shape | Units | When | Meaning |
| --- | --- | --- | --- | --- | --- |
| `/entry1/data/data` | `stored` | `(n_depths, rows, columns)` | | | Stored frames; attribute `signal=1` |
| `/entry1/depth` | `<f8` | `(n_depths,)` | micron | | Physical depth of each frame |
| `/entry1/microDiffraction/cosmic_filter` | `<i4` | `(1,)` | | cosmic-ray filter ran | 1 |
| `/entry1/microDiffraction/norm_exponent` | `<f8` | `(1,)` | | exponent normalization ran | Normalization exponent |
| `/entry1/microDiffraction/norm_threshold` | `<f8` | `(1,)` | | exponent normalization ran | Threshold used |
| `/entry1/microDiffraction/norm_rescale` | `<f8` | `(1,)` | | exponent normalization ran | Factor applied before conversion |
| `/entry1/reconstruction/detector` | `<i4` | `(1,)` | | | Physical detector slot in the geometry |
| `/entry1/reconstruction/depth_range` | `<f8` | `(2,)` | micron | | Requested inclusive depth range |
| `/entry1/reconstruction/resolution` | `<f8` | `(1,)` | micron | | Spacing between depths |
| `/entry1/reconstruction/wire_edge` | `text` | `(1,)` | | | `leading`, `trailing`, or `both` |
| `/entry1/reconstruction/percent_brightest` | `<f8` | `(1,)` | | | Percentage of intensity-map pixels reconstructed |
| `/entry1/reconstruction/normalization` | `text` | `(1,)` | | a vector was used | Name of the normalization vector |
| `/entry1/reconstruction/frame_skip` | `<i4` | `(1,)` | | | Leading stored slices skipped |
| `/entry1/reconstruction/wire_skip` | `<i4` | `(1,)` | | | Stored wire-vector entries skipped after entry 0 |
| `/entry1/reconstruction/geometry_path` | `text` | `(1,)` | | | Geometry file path as given |
| `/entry1/reconstruction/geometry_xml` | `text` | `(1,)` | | | Complete text of the geometry file used |
| `/entry1/reconstruction/point_id` | `text` | `(1,)` | | | Point ID |
| `/entry1/reconstruction/manifest_index` | `<i4` | `(1,)` | | the point belongs to a scan | Manifest index in the scan |
| `/entry1/reconstruction/source_path` | `text` | `(1,)` | | | Absolute input path |
| `/entry1/reconstruction/source_size` | `<i8` | `(1,)` | bytes | the input could be examined | Input size in bytes |
| `/entry1/reconstruction/source_mtime_ns` | `<i8` | `(1,)` | | the input could be examined | Input modification time in ns since the epoch |
| `/entry1/reconstruction/raw_slices` | `<i4` | `(2,)` | | | Half-open range of selected input slices |
| `/entry1/reconstruction/native_version` | `text` | `(1,)` | | | Version of the native library that reconstructed the point |
| `/entry1/reconstruction/num_threads` | `<i4` | `(1,)` | | | OpenMP threads used |
| `/entry1/reconstruction/rows_per_stripe` | `<i4` | `(1,)` | | | Rows per stripe used |
| `/entry1/reconstruction/memory_limit_mb` | `<i4` | `(1,)` | MiB | | Stripe-buffer limit |
| `/entry1/reconstruction/elapsed_seconds` | `<f8` | `(1,)` | s | | Wall time of the reconstruction |
| `/entry1/reconstruction/first_raw` | `input` | `(rows, columns)` | | | First selected raw frame |
| `/entry1/reconstruction/sum_raw` | `raw_accumulator` | `(rows, columns)` | | | Sum of the selected raw frames |
| `/entry1/reconstruction/sum_reconstructed` | `stored_accumulator` | `(rows, columns)` | | | Sum of the stored frames |
| `/entry1/reconstruction/depth_intensity` | `stored_accumulator` | `(n_depths,)` | | | Stored total of each frame |
| `/entry1/reconstruction/computed_depth_intensity` | `<f8` | `(n_depths,)` | | | Computed total of each frame |

`text` is fixed-length UTF-8 text whose length is that of the value. The settings in the point file are the values used. The output pixel type is the dtype of `/entry1/data/data`. A threshold or stripe height that was selected automatically is recorded as the value selected.

`num_threads`, `rows_per_stripe`, `memory_limit_mb`, and `elapsed_seconds` are execution values. Runs with identical results can have different execution values.

The writer rejects an input that contains `/entry1/reconstruction` or one of the four `/entry1/microDiffraction` datasets above, because such a file is the output of a reconstruction. Per-depth export copies the point file without `/entry1/reconstruction` and writes one frame and its physical depth to each file.

A point file can be inspected, indexed, and exported to per-depth files without its catalog, raw input, or geometry file. It does not contain the raw frames, so the input is necessary to reconstruct the point again.

`/entry1/data/data` is stored in HDF5 chunks. The chunk shape and the compression are not part of the format.

## Per-depth output

The in-process and executable paths are cross-checks of each other. The recorded references under `tests/data/reconstruction/` are the expected output of both. The stored frames of a point file equal the per-depth frames for every recorded variant.

{func}`~lauelab.reconstruct.export_per_depth` reads the stored pixels from a point file and writes them in their stored dtype, with their physical depth, the acquisition metadata, and the normalization values. It does not run reconstruction and does not apply `rescale` again. Its LaueGo-compatible summary contains `/entry1/reconstruction/computed_depth_intensity` and the execution time and stripe height of the reconstruction. A failed export does not change the point file.

## Evidence

`tests/test_reconstruction_contract.py` checks the conversion rule against HDF5, records what each reference variant contains, and verifies the hand-worked fixtures in `tests/data/reconstruction_contract/fixtures.py` with brute-force oracles. The fixtures contain the cases that are not in any recorded reference: a zero frame, a frame with no positive pixel, saturation at both limits, and integer output with `rescale` above 1. `tests/test_point_files.py` compares point files with the output of `reconstructN_cpu` and checks the point validator. `tests/test_scan_preparation.py` checks preparation, tasks, destinations, and the catalog validator. `tests/test_scan_layout.py` compares the tables on this page and the status codes in [Reconstruction scan protocol](reconstruction-scan-protocol.md) with the layout module.
