# Q-distribution HDF5 format

An HDF5 Q-distribution file contains one profile, its processing settings and its provenance. Source image, dark and mask pixels are stored separately. Use {func}`~lauelab.energy.save_q_distribution`, {func}`~lauelab.energy.load_q_distribution` and {func}`~lauelab.energy.validate_q_distribution_file` to write, read and validate a profile. Loading and plotting require no source files.

The format follows the shared [HDF5 conventions](hdf5-conventions.md). Root attributes are `format="lauelab-q-distribution"`, integer `version=1`, `lauelab_version`, and UTC `created`. Readers check the format and version first, ignore unknown fields, and reject unsupported versions. The complete dataset registry is `lauelab.energy._layout`.

## Profile arrays

All five datasets have shape `(n,)`, with `n >= 2`. The validator checks `max_bins`, default 1,000,000, before reading profile values. Counts are dimensionless and have no units attribute.

| Path | dtype | Units | Meaning |
|---|---|---|---|
| `profile/q_nm_inv` | float64 | `1/nm` | Uniform, increasing Q sample centers |
| `profile/intensity_mean` | float64 | By correction mode | Mean corrected detector intensity |
| `profile/intensity_sum` | float64 | By correction mode | Sum of corrected pixel intensities |
| `profile/pixel_count` | int64 | None | All unmasked assigned pixels |
| `profile/nonfinite_pixel_count` | int64 | None | Assigned nonfinite corrected intensities |

Intensity units are the recorded input units for raw mode, input units divided by seconds for exposure-only mode, and `unspecified` when input units are unknown or for conventional I0-relative intensity. Each intensity dataset also has a `normalization` attribute. No absolute intensity unit is inferred.

Empty bins have zero sum/counts and NaN mean. Contaminated bins have positive nonfinite count and NaN sum/mean. Other occupied bins have finite sum/mean, including zeros and negatives. The validator checks these relationships, uniform centers and exact agreement between occupancy totals and diagnostics.

## Dimensioned settings and geometry

| Path | dtype | Shape | Units |
|---|---|---|---|
| `parameters/energy_kev` | float64 | scalar | `keV` |
| `parameters/depth_um` | float64 | scalar | `um` |
| `parameters/start` | int64 | `(2,)` | `pixel` |
| `parameters/group` | int64 | `(2,)` | None |
| `parameters/roi` | int64 | `(4,)` | `pixel` |
| `parameters/hc_kev_nm` | float64 | scalar | `keV nm` |
| `parameters/normalization_factor` | float64 | scalar | None for raw, `1/s` for exposure-only, `unspecified` for I0 |
| `parameters/exposure_seconds` | float64 | scalar | `s` |
| `parameters/i0_counts` | float64 | scalar | `unspecified` |
| `parameters/scaler_count_time_seconds` | float64 | scalar | `s` |
| `parameters/i0_gain` | float64 | scalar | `unspecified` |
| `parameters/normalization` | UTF-8 text | scalar | None |
| `geometry/pixels` | int64 | `(2,)` | `pixel` |
| `geometry/size_um` | float64 | `(2,)` | `um` |
| `geometry/translation_um` | float64 | `(3,)` | `um` |
| `geometry/rotation_vector_rad` | float64 | `(3,)` | `rad` |
| `geometry/rotation` | float64 | `(3,3)` | None |
| `geometry/detector_id` | UTF-8 text | scalar | None |
| `geometry/detector_index` | int64 | scalar | None |
| `geometry/xml` | UTF-8 text | scalar | None |
| `provenance/json` | UTF-8 text | scalar | None |

Start, group and pixel dimensions use `(x, y)` order. ROI bounds are half-open `(y0, y1, x0, x1)`. Unused correction settings are NaN. Geometry is stored as numerical calibration values and an XML snapshot captured at geometry loading, with normalized XML formatting. For non-XML geometry, `geometry/xml` is empty and the provenance includes the reason.

## Provenance

The JSON object contains algorithm and package versions, original and effective metadata, explicit overrides and beam declarations, selected point and depth, geometry, ROI and grid, correction factors and assumptions, reconstruction settings, input identities and diagnostics. The validator compares the duplicated settings with this snapshot and recalculates normalization from the effective metadata.

The writer and validator also check that the selection and calibration values are valid. The source shape must describe a nonempty 2D image. Start values are nonnegative integers; grouping and detector pixel dimensions are positive integers. These values must fit `int64`. The full frame must fit the detector after applying start and group. The half-open ROI must fit the source shape, and its area must equal the recorded ROI pixel count. Detector physical sizes must be finite and positive. Translation, rotation vector and rotation matrix must have their documented shapes and finite real values. The detector index must be a supported physical slot: 0, 1 or 2. These checks require no source image or geometry file.

Input content hashes are SHA256 values calculated from logical C-order pixel bytes in the recorded source dtype. The provenance also contains shape and dtype. The hash covers the selected source image or the supplied full-frame mask or dark, not the HDF5 container. Recomputing the profile requires the original inputs. Validation checks the saved arrays and settings without reading those inputs; it does not verify the experimental metadata against the measurement.

JSON is standard UTF-8 without nonstandard numeric NaN or Infinity tokens. Unused or nonfinite original metadata is represented by tagged objects such as `{"nonfinite":"NaN"}`. Extended floating-point metadata outside the float64 range is stored as value text and dtype. These tagged values require conversion before reuse as computation inputs. Unavailable profile values are stored as floating-point NaN in the numeric datasets.

## Publication and CSV

The writer creates a unique `.partial` file in the destination directory, closes and validates it, then publishes it with `publish_file`. Publication raises `FileExistsError` for an existing destination unless `overwrite=True`. On filesystems supporting hard links, this check is atomic against a competing publisher. The helper uses a checked rename fallback on other filesystems. If writing, validation or publication fails, the writer removes only its own temporary file.

CSV output contains the five profile columns, a comment header for units and corrections, and `nan` for unavailable values. Skip lines beginning with `#` when parsing. Use HDF5 when you need the complete provenance; CSV contains only numerical values and the correction summary.
