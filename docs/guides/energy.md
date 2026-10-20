# Plot intensity versus Q from a monochromatic image

Use {func}`~lauelab.energy.q_distribution` to calculate mean intensity versus scattering-vector magnitude Q from one monochromatic detector image. The result contains intensity sums, pixel counts, nonfinite-pixel counts, and processing settings. Q is in `1/nm`, including the factor of two pi.

You need the image's photon energy and calibrated detector geometry. The calculation runs in Python without crystal indexing or an Igor installation. It processes one image and does not fit peaks, calculate strain, or construct volumes.

## Run the synthetic example

The synthetic fixtures in `tests/data/energy/` contain an image with rings centered at 6.5 and 13 `1/nm`, a matching dark image, and an exclusion mask. Run this example from the repository root with the package installed:

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np

from lauelab.energy import (
    q_distribution, save_q_distribution, load_q_distribution,
    export_q_distribution_csv,
)
from lauelab.indexing import load_mask
from lauelab.visualization import plot_q_distribution

data = Path("tests/data/energy")
profile = q_distribution(
    data / "rings.h5",
    geometry=data / "geometry.xml",
    dark=data / "dark.h5",
    mask=load_mask(data / "mask.h5"),
    roi=(5, 73, 7, 91),
    q_first=0.0, q_last=20.0, q_points=201,
)
assert profile.pixel_count.sum() == 5596
assert profile.nonfinite_pixel_count.sum() == 0
assert profile.provenance["normalization"]["factor"] == 5.0
figure = plot_q_distribution(profile)

with TemporaryDirectory() as output:
    output = Path(output)
    save_q_distribution(profile, output / "profile.h5")
    export_q_distribution_csv(profile, output / "profile.csv")
    reopened = load_q_distribution(output / "profile.h5")
    np.testing.assert_array_equal(reopened.intensity_mean, profile.intensity_mean)
    reopened_figure = plot_q_distribution(reopened)
```

In a notebook, display `figure` to inspect the profile. The plot uses blue points and a dashed line, linear axes, Q in `nm⁻¹`, inward ticks, mirrored axes, and a y minimum of zero. Negative intensities are included in the arrays; change the returned figure's y range to display them.

## Select an input and detector

Pass a 2D NumPy array, a supported 34-ID-E HDF5 frame, or a {class}`~lauelab.indexing.ScanFrame` for either the primary image or the dark. A `ScanFrame` identifies one reconstructed point and depth index. Raw 3D stacks are not accepted.

Images have shape `(ny, nx)`, and `image[y, x]` accesses pixel `(x, y)`. Full-detector `start` and `group` use `(x, y)`. Coordinate conversion uses the fractional center of each grouped pixel. `depth` is in µm in the [detector geometry convention](geometry.md). Stored `start` and `group` take precedence over call arguments. An explicit `depth`, including zero, overrides the stored depth. See [Frame input](frame-input.md) for the input layout.

An HDF5 origin or binning pair must contain both axes or neither: `startx` with `starty`, and `binx` with `biny`. The function raises `InputError` for an incomplete pair in either a primary or dark image. When both axes of a primary field are absent, call arguments or their defaults apply. If you supply a dark without coordinate metadata, the function assumes it matches the primary image. Missing or nonfinite stored depth is treated as absent; the default is zero unless you pass an explicit finite `depth`.

The function selects a detector using the ID in the effective metadata or the sole active detector in the geometry. If neither is available, supply `detector_id` or `detector_index`. The function raises `InputError` for conflicting selections.

```{warning}
`detector_index` is a physical geometry slot, not its position among active detectors. Slots can be sparse; the synthetic geometry uses slot 2.
```

## Establish photon energy and beam mode

Photon energy must be finite and positive. The `energy_kev` argument takes precedence over energy in supplied metadata, which takes precedence over file energy. Supplied {class}`~lauelab.indexing.FrameMetadata` or mapping fields override other file metadata. The provenance contains the original values and explicit overrides.

Confirm monochromatic illumination with reliable `mono_mode="monochromatic"` metadata or an explicit `monochromatic=True` declaration. An `incident_energy` value alone is insufficient. If white-beam mode is recorded in either the file or supplied metadata, pass both `monochromatic=True` and a nonempty `monochromatic_reason`. This requirement applies even when you override the photon energy or mode metadata.

## Choose corrections

Choose a normalization mode. The default is `normalization="i0"`; all quantities required by the selected mode must be positive and finite:

| Mode | Required metadata | Effect after direct dark subtraction |
|---|---|---|
| `"none"` or `None` | None | No scalar correction |
| `"exposure"` | `exposure_seconds` | Divide by exposure in seconds |
| `"i0"` | `exposure_seconds`, `i0_counts`, `scaler_count_time_seconds` | Normalize to exposure and I0, including the `1e5` scale |

When `i0_gain` is omitted, the function uses a nominal gain of `1e9` and records the assumption in the provenance. Supplied `i0_gain` must be positive and finite. Detector gain is a separate quantity and cannot substitute for I0 gain. Monitor metadata must be scalar or length one. For longer vectors, supply an explicit scalar override, including when processing a `ScanFrame`.

The function raises `InputError` for missing or invalid required quantities. For an array without metadata, supply the required fields or select `normalization="none"`. [The algorithm](../concepts/algorithms/energy.md) defines the correction formula.

A dark must match the entire supplied frame in shape, stored coordinates, binning, units and exposure. The function raises `InputError` for known mismatches and does not rescale or resample the dark. Use `dark_metadata` to override a dark file's metadata. If you supply a dark array without metadata, you are responsible for checking these conditions.

```{warning}
Reconstructed images can contain normalized or scaled intensities. Inspect the recorded reconstruction settings and select corrections appropriate to the stored values. Energy reduction does not undo reconstruction scaling or identify every correction applied to an image.
```

The function does not apply flat-field, detector distortion, solid-angle, polarization, Lorentz or automatic background corrections. Applied corrections and references to unapplied distortion maps are recorded in the provenance. The output is corrected detector intensity; it is not calibrated as an absolute scattering cross section.

(energy-pixels)=
## Select pixels and Q centers

Pass `roi=(y0, y1, x0, x1)` to select the half-open slice `image[y0:y1, x0:x1]`. A mask must match the full supplied image before ROI selection: zero includes a pixel, and nonzero excludes it. Excluded pixels are omitted from both intensity sums and occupancy, including when their intensities are NaN or Inf. The function does not resize or broadcast masks.

Igor's energy-scan routine uses an inclusion mask: `1` includes a pixel and `0` excludes it. To reuse that mask, arrange it in NumPy `(ny, nx)` order and pass `mask=(igor_mask == 0)`. The {func}`~lauelab.indexing.load_mask` helper reads nonzero entries as exclusions, following the peaksearch convention. The `maskNorm` ("Suppress") pedestal correction is not supported. Use `dark` for subtraction of a supplied dark image.

Omit all Q controls to calculate spacing from the selected detector pixels. Alternatively, supply `q_first`, `q_last` and `q_points` together: a finite nonnegative first center, a larger finite last center, and an integer count of at least two. These values specify **sample centers**. Nearest-center assignment rounds half ties away from zero. Pixels whose rounded indices fall outside the grid are omitted and counted in the diagnostics.

If automatic spacing is undefined or the Q span is zero, the function raises `InputError` and requests an explicit grid. The default allocation limit is `max_bins=1_000_000`. Computation and loading check this limit before allocating profile arrays; raise it explicitly for a larger profile. Pixel calculations use bounded blocks, and one source frame is loaded into memory.

## Interpret missing and negative values

| Bin contents | Mean | Sum | Counts |
|---|---|---|---|
| No assigned pixels | NaN | Zero | Zero occupancy |
| Assigned finite pixels, including all zeros | Sum / occupancy | Corrected sum | Positive occupancy |
| An assigned nonfinite intensity or dark value | NaN | NaN | Positive occupancy and nonfinite count |

Nonfinite intensities affect only their assigned bins. You can inspect a profile in which every occupied bin has unavailable intensity by using its counts and diagnostics. The function raises `InputError` when all selected pixels are masked or no pixels map to the requested grid. Zero and finite negative intensities are valid. Pixel counts are diagnostics; the result does not include statistical uncertainties or error bars.

The function accepts real integer and floating-point dtypes, including `uint32`, `int64`, `uint64`, byte-swapped arrays and noncontiguous arrays. Arithmetic and floating-point outputs use `numpy.float64`. Integers beyond `2**53` can round during conversion; calculations are not exact for arbitrary wide integers. Finite arithmetic overflow raises `InputError`. The function does not modify its inputs. The result contains independent, read-only arrays, and each access to `provenance` returns a copy of the metadata and settings.

## Save or exchange a result

Use {func}`~lauelab.energy.save_q_distribution` to write an HDF5 file containing profile arrays, geometry, dimensioned settings, original and effective metadata, assumptions, diagnostics, and input, dark and mask content hashes. Source pixel arrays are not embedded in the file. You can load and plot the saved profile without source files; recomputation requires the original inputs.

Saving, validation and loading check the selection and calibration settings. They reject nonpositive binning or detector dimensions, nonpositive physical sizes, nonfinite transforms, and a frame or ROI outside its recorded bounds.

Use {func}`~lauelab.energy.export_q_distribution_csv` to write five numeric columns with comment lines identifying units and normalization. Skip lines beginning with `#` when reading the CSV. Both export functions raise `FileExistsError` for an existing destination unless you pass `overwrite=True`.

See the [API reference](../reference/energy.md) for signatures, the [format description](../development/energy-format.md) for the file layout, and the [algorithm](../concepts/algorithms/energy.md) for equations and validation evidence.
