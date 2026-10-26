# Strain-result format

Standalone HDF5 files use root `format="lauelab-strain-results"` and integer `version=1`. Root metadata identifies creation time, package version, `strain_measure="deviatoric Biot stretch"`, `scale_convention="reference volume"`, `shear_convention="tensor"` and result count.

The ordered groups `/results/0`, `/results/1`, and so on each contain one successful or failed fit. Saving a single result is a one-item collection; loading always returns a list. An empty collection is valid.

| Fields | Storage | Meaning |
|---|---|---|
| `status` | Scalar int8 HDF5 enum | `converged=0`, `insufficient_support=1`, `ill_conditioned=2`, `nonconverged=3`, `invalid_input=4` |
| `message`, `coordinates`, `fit_mask`, `provenance_json` | Scalar UTF-8 strings | Diagnostic explanation, conventions and effective settings |
| `n_parameters`, `rank`, `nfev` | int64 scalars | Fit/support diagnostics |
| `condition_number` | float64 scalar | Dimensionless joint Jacobian condition, infinity if unavailable |
| `tensor`, `rotation` | float64 `(3,3)` | Dimensionless tensor and fitted orientation |
| `reciprocal`, `reference_reciprocal` | float64 `(3,3)` | Reciprocal rows in `1/nm`, including `2*pi` |
| `cell_parameters`, `working_cell_parameters` | float64 `(6,)` | Final volume-normalized and pre-normalization fitted `a,b,c,alpha,beta,gamma`, respectively; lengths in nm and angles in degrees |
| `predicted_qhat`, `predicted_xy`, `angular_residual_deg` | float64 `(n,3)`, `(n,2)`, `(n,)` | Predictions and errors aligned to measurement rows |
| `measurements/qhat`, `measurements/xy` | float64 `(n,3)`, `(n,2)` | Actual direction inputs and available zero-based frame centroids |
| `measurements/hkl`, `measurements/peak_indices` | int64 `(n,3)`, `(n,)` | Miller indices and original measured-peak IDs |
| `measurements/initial_reciprocal` | float64 `(3,3)` | Starting reciprocal rows; availability recorded separately |
| `measurements/frame_id_json`, `measurements/pattern_index` | UTF-8 JSON scalar, int64 scalar | Source identity without conflating string/integer frame IDs |
| `measurements/provenance_json` | UTF-8 scalar | Source precision, selections, effective calibration snapshot and input fingerprint |

NaN marks unavailable numerical arrays, including unsuccessful tensors and direction-only pixel columns. Missing initial lattices for invalid-input batch records use a false boolean `has_initial` attribute and an all-NaN stored matrix. These are not zero-strain measurements.

Dimensioned arrays have a `units` attribute; dimensionless tensors, rotations, directions and indices have none. The mixed cell-parameter arrays have `component_units=["nm","nm","nm","deg","deg","deg"]` instead of a single unit. Effective native reference parameters, frozen-value policy, initial projection, optimizer controls and empirical-parity status are retained. Pixel-based measurements include detector calibration parameters and the loaded geometry XML snapshot where available; saved results do not depend on a mutable source geometry path.

Loading validates the marker/version, group counts, scalar types, array shapes/dtypes/units, identity/HKL validity and owned result model. Successful results additionally require finite scientific fields, a proper fitted rotation, reference-volume agreement, and consistency among lattices, tensors, cell parameters, predicted directions and angular errors. This is structural/numerical validation, not an independent check of calibration quality or phase correctness.

Saving writes a temporary file beside the destination, validates by loading, closes the file, then uses the package publication helper. Default publication refuses to overwrite an existing destination. Original indexing inputs cannot be used as destinations.

CSV exports have two row granularities: one per result and one per reflection. Both identify result index, source frame/pattern identity, status, coordinates, strain measure, scale and shear convention. The catalogue includes fitted orientation/reciprocal matrices, tensor components and support diagnostics. The reflection table includes measured/predicted pixels, angular residuals and predicted energy. CSV is an exchange view; HDF5 retains the complete evidence and provenance. The two CSV files are individually published, not an atomic pair.

This standalone collection uses per-fit groups rather than the flat ragged tables in an indexing catalogue. Each group is an independent result with its own reference, model, calibration snapshot and available/failed arrays. The format is written as a complete collection, not an appendable scan. Effective settings and provenance are stored as JSON objects; the scientific arrays and diagnostics remain directly accessible HDF5 datasets.
