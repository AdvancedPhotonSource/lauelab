# Results

`FrameResult` owns Python copies of native indexing output. A result with no
identified patterns is valid and can still contain detected peaks and frame
statistics.

## Results files

See [Results files](../guides/results-file.md) for writing, loading, and converting.

```{eval-rst}
.. currentmodule:: lauelab.indexing

.. autoclass:: ResultsWriter
   :members: append

.. currentmodule:: lauelab

.. autofunction:: is_results_file
```

(results-file-layout)=
### File layout

Format `lauelab-indexing-results`, version 1. Root attributes are `format`, `version`, `lauelab_version`, `created`, and, for a converted file, `source`. The `run` group has no datasets; its attributes are `program`, `detector_index`, `detector_id`, `cosmic_filter`, and one attribute per {class}`~lauelab.indexing.PeakParams` and {class}`~lauelab.indexing.IndexParams` field. The `crystal` group carries `name`, `space_group`, `setting`, and `source` as attributes and is absent when the run had no crystal. The `geometry` group carries `path` as an attribute.

A converted file carries only the run attributes the XML recorded, when present: `program`, `peak_program`, `cosmic_filter`, `max_rfactor`, `max_peaks`, `min_separation`, `peak_shape`, `mask_file`, `kev_max_calc`, `kev_max_test`, `angle_tolerance_deg`, `cone_deg`, and `hkl_prefer`.

`n` is the row count of the group. Rows of `peaks`, `patterns`, and `assignments` are grouped by owner; the rows of owner `i` are `offsets[i]` to `offsets[i + 1]` of the matching offsets dataset, which has one more entry than there are owners. Absent values are `NaN` for floating-point datasets, `-1` for `scan_numbers`, `beam_bad`, and `light_on`, and the empty string for string datasets. Coordinates in `peaks` are zero-based frame pixel `(x, y)` values.

| Dataset | dtype | Shape | Units | Meaning |
|---|---|---|---|---|
| `crystal/lattice_parameters` | `float64` | `(6,)` | nm, deg | `a`, `b`, `c`, `alpha`, `beta`, `gamma`; attribute `angle_units` is `deg` |
| `crystal/atom_symbols` | string | `(n,)` | | Element symbol per atom |
| `crystal/atom_labels` | string | `(n,)` | | Site label per atom |
| `crystal/atom_positions` | `float64` | `(n, 3)` | fractional | Fractional coordinates |
| `crystal/atom_occupancies` | `float32` | `(n,)` | | Site occupancy |
| `geometry/xml` | string | scalar | | Geometry file text; present when the file was readable at write time |
| `frames/frame_ids` | `int32` or string | `(n,)` | | Frame identifier |
| `frames/sample_positions` | `float64` | `(n, 3)` | um | Sample `(x, y, z)` in the acquisition coordinate system |
| `frames/depths` | `float64` | `(n,)` | um | Sample depth passed to the geometry conversion |
| `frames/scan_numbers` | `int32` | `(n,)` | | Scan number |
| `frames/energies_kev` | `float32` | `(n,)` | keV | Incident energy |
| `frames/exposure_seconds` | `float32` | `(n,)` | s | Detector exposure time |
| `frames/beam_bad`, `light_on` | `int32` | `(n,)` | | Acquisition flags, `-1` when absent |
| `frames/hutch_temperature`, `sample_distance` | `float32` | `(n,)` | unspecified | Acquisition values without an established unit; no conversion applied |
| `frames/detector_ids` | string | `(n,)` | | Detector identifier |
| `frames/input_images` | string | `(n,)` | | Source HDF5 path |
| `frames/titles`, `sample_names`, `user_names`, `beamlines`, `dates_exposed`, `ccd_shutters`, `mono_modes` | string | `(n,)` | | Acquisition metadata strings |
| `frames/image_shapes` | `int32` | `(n, 2)` | | Frame shape as `(rows, columns)` |
| `frames/roi_starts` | `int32` | `(n, 2)` | | Full-detector `(x, y)` origin of the frame |
| `frames/roi_groups` | `int32` | `(n, 2)` | | Pixel grouping factors `(x, y)` |
| `frames/n_peaks` | `int32` | `(n,)` | | Detected peaks |
| `frames/n_patterns` | `int16` | `(n,)` | | Identified patterns |
| `frames/threshold_used` | `float32` | `(n,)` | | Peak-search threshold |
| `frames/threshold_ratio` | `float32` | `(n,)` | | Automatic-threshold ratio; `NaN` when inactive |
| `frames/total_sum` | `float64` | `(n,)` | | Sum of unmasked pixel values |
| `frames/sum_above_threshold` | `float64` | `(n,)` | | Sum of pixel values above the threshold |
| `frames/num_above_threshold` | `int64` | `(n,)` | | Pixels above the threshold |
| `frames/peak_minwidth`, `peak_maxwidth`, `peak_max_cent_to_fit` | `float32` | `(n,)` | | Effective peak-fit parameters |
| `frames/peak_boxsize` | `int32` | `(n,)` | | Effective fit box size |
| `frames/peaksearch_seconds`, `indexing_seconds` | `float32` | `(n,)` | s | Stage timing |
| `frames/peak_offsets` | `int64` | `(n + 1,)` | | Row range of each frame in `peaks` |
| `frames/pattern_offsets` | `int64` | `(n + 1,)` | | Row range of each frame in `patterns` |
| `peaks/fit_x`, `fit_y` | `float32` | `(n,)` | pixel | Fitted peak coordinate |
| `peaks/intens`, `integral`, `background` | `float32` | `(n,)` | | Fitted intensity, integral, and background |
| `peaks/hwhm_x`, `hwhm_y` | `float32` | `(n,)` | pixel | Fitted half-widths |
| `peaks/tilt` | `float32` | `(n,)` | deg | Fitted tilt |
| `peaks/chisq` | `float32` | `(n,)` | | Normalized fit residual |
| `peaks/qhat` | `float32` | `(n, 3)` | | Unit scattering vector |
| `patterns/rank` | `int16` | `(n,)` | | Pattern index within its frame |
| `patterns/reciprocal` | `float64` | `(n, 3, 3)` | 1/nm | Rows `a*`, `b*`, `c*` including the factor of two pi; attributes `rows` and `includes_two_pi` |
| `patterns/goodness` | `float32` | `(n,)` | | Native goodness score |
| `patterns/rms_error_deg` | `float32` | `(n,)` | deg | Root-mean-square angular error |
| `patterns/n_indexed` | `int32` | `(n,)` | | Assignments in the pattern |
| `patterns/assignment_offsets` | `int64` | `(n + 1,)` | | Row range of each pattern in `assignments` |
| `assignments/peak_index` | `int32` | `(n,)` | | Zero-based index into the owning frame's peaks |
| `assignments/hkl` | `int16` | `(n, 3)` | | Miller indices |
| `assignments/error_deg` | `float32` | `(n,)` | deg | Angular error |
| `assignments/energy_kev` | `float32` | `(n,)` | keV | Photon energy |
| `assignments/pred_intens` | `float32` | `(n,)` | | Predicted intensity |

## Frame Result

```{eval-rst}
.. currentmodule:: lauelab.indexing

.. autoclass:: FrameResult(peaks, patterns, threshold_used, total_sum, sum_above_threshold, num_above_threshold, peaksearch_seconds, indexing_seconds, threshold_ratio=4.0, peak_minwidth=0.0, peak_maxwidth=0.0, peak_max_cent_to_fit=0.0, peak_boxsize=0, metadata={}, input_image=None, image_shape=(0, 0), start=(0, 0), group=(1, 1), depth=None, image=None)
   :members: indexed, n_peaks, n_indexed, n_patterns, elapsed_seconds, indexed_peak_indices, unindexed_peak_indices, write_xml
```

## Indexed Pattern

See the [results guide](../guides/results.md) for the reciprocal-basis convention used by `Pattern.reciprocal`.

```{eval-rst}
.. currentmodule:: lauelab.indexing

.. autoclass:: Pattern
   :members: n_indexed
```
