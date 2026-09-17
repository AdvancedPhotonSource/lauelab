# Parameters

{class}`~lauelab.indexing.PeakParams` controls peak detection and fitting. {class}`~lauelab.indexing.IndexParams` controls orientation indexing. Both are immutable dataclasses that the `Indexer` validates during construction.

`cosmic_filter` must be `False` for indexing. `Indexer`, `index_frame`, and `lauego` reject `True` because they do not implement cosmic-ray filtering. Earlier versions accepted the setting and recorded it without filtering the image. Historical result files retain that recorded value when loaded; it does not establish that filtering occurred. The separate reconstruction cosmic-ray filter remains supported.

## Start with the defaults

Use default values for an initial API check, but do not assume that they are appropriate for every detector, exposure, or material.

```python
from lauelab.indexing import Indexer, IndexParams, PeakParams

indexer = Indexer(
    "geometry.xml",
    "crystal.xml",
    peak_params=PeakParams(),
    index_params=IndexParams(),
)
```

Record the complete parameter objects with analysis output. A result alone does not contain every setting needed to reproduce processing.

## Peak-search parameters

| Parameter | Default | Units | Effect and constraint |
|---|---:|---|---|
| `boxsize` | `5` | px | Half-width of the square fitting region. Must be a positive whole number. |
| `max_rfactor` | `2.0` | dimensionless | Maximum accepted fit residual factor. Must be positive. |
| `min_size` | `3` | px | Minimum accepted peak size. Must be a positive whole number. |
| `min_separation` | `10` | px | Minimum separation between accepted peaks. Must be a positive whole number. |
| `threshold` | `100.0` | detector counts | Absolute detection threshold. Use `None` for an automatically derived threshold. |
| `threshold_ratio` | `None` | dimensionless | Scale applied to the frame standard deviation for automatic thresholding. `None` resolves to the native default, `4.0`. |
| `peak_shape` | `"Lorentzian"` | none | Fit model. Exactly `"Lorentzian"` or `"Gaussian"`. |
| `max_peaks` | `50` | peaks | Maximum number of returned peaks, a positive whole number, or `None` for no limit: every blob above the threshold is fitted. |
| `smooth` | `False` | none | Applies native image smoothing before detection and fitting. Frame sums continue to describe the raw input image; an automatically derived threshold is computed from the smoothed image. |

Whole-number parameters accept an integral float: `min_size=3.0` is the same as `min_size=3`, and the `Indexer` stores the `int`. A fractional value such as `3.5` raises {class}`~lauelab.indexing.InputError` naming the parameter; nothing is rounded silently.

With `max_peaks=None`, a frame with many blobs takes longer and returns more peaks. XML output then omits the `max_number` attribute, and the results file records `max_peaks` as `NaN`.

When `threshold` is not `None`, `threshold_ratio` does not determine the threshold. When `threshold` is `None`, the native stage calculates the threshold from frame statistics and the resolved `threshold_ratio`. XML provenance records the resolved value (`4.0` when configured as `None`).

## Indexing parameters

| Parameter | Default | Units | Effect and constraint |
|---|---:|---|---|
| `kev_max_calc` | `30.0` | keV | Maximum energy used to calculate candidate reflections. Must be positive. |
| `kev_max_test` | `35.0` | keV | Maximum energy used when testing candidate reflections. Must be positive. |
| `angle_tolerance_deg` | `0.12` | deg | Angular matching tolerance. Must be positive. |
| `cone_deg` | `72.0` | deg | Search-cone angle. Must be positive. |
| `hkl_prefer` | `(0, 0, 1)` | Miller indices | Preferred direction. Must contain exactly three whole numbers. |
| `max_data` | `250` | peaks | Maximum detected peaks supplied to orientation indexing. Must be a whole number of at least two. |

These fields configure the native orientation search. Their scientifically appropriate values depend on the experiment and crystal. This guide does not prescribe universal tuning values.

## Derive a configuration

Use {func}`dataclasses.replace` instead of mutating a parameter object:

```python
from dataclasses import replace

base_peaks = PeakParams()
automatic_threshold = replace(
    base_peaks,
    threshold=None,
    threshold_ratio=4.0,
    max_peaks=200,
)
```

Construct a new indexer or use `Indexer.replace()` to validate changed settings:

```python
updated = indexer.replace(peak_params=automatic_threshold)
```

`Indexer.replace()` constructs independent native geometry and crystal state for the new instance.

## Parameter interactions

Several interactions follow directly from processing behavior:

- `threshold` selects absolute or automatic thresholding.
- `threshold_ratio` affects automatic thresholding only.
- `max_peaks` limits the output of peak search; `None` removes the limit.
- `max_data` limits how many detected peaks enter orientation indexing.
- Orientation indexing runs only when a crystal is present and at least two peaks were detected.
- `start`, `group`, and `depth` affect pixel-to-q conversion rather than peak fitting.
- A mask changes which frame pixels peak search can use.

Changes to peak acceptance can change the scattering vectors available to indexing. Compare the intermediate peak count and fit fields before attributing a changed pattern result only to `IndexParams`.

## A conservative tuning workflow

1. Save the original frame, geometry, crystal description, and parameter objects.
2. Run peak search with a fixed configuration.
3. Inspect `threshold_used`, `n_peaks`, fitted positions, widths, residuals, and masked regions.
4. Change one peak-search parameter at a time.
5. Hold the accepted peak set fixed before comparing indexing parameters.
6. Compare pattern count, assignments, and angular errors across runs.
7. Record the chosen values and the reason for each change.

This process isolates parameter effects. Numerical thresholds and acceptance criteria still require experiment-specific review.

## Performance and reproducibility

Do not infer speed from `elapsed_seconds` alone. It sums recorded peak-search and orientation-indexing time but excludes pixel-to-q conversion and Python setup.

For a reproducible comparison, record:

- Package version or Git commit
- Frame identifier and checksum when possible
- Geometry and crystal file versions
- Detector selection
- `start`, `group`, and `depth`
- Mask identity
- Complete `PeakParams` and `IndexParams`
- Hardware and process configuration for timing comparisons

## Invalid configurations

The `Indexer` raises {class}`~lauelab.indexing.InputError` for invalid parameter ranges, unsupported peak models, malformed `hkl_prefer`, and `max_data` below two. See [Configuration](../reference/configuration.md) for exact field definitions.
