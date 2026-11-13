# Calibrate a detector

Use `lauelab.calibration` to fit the rotation and translation of a detector to one frame of a calibration crystal, and to write the fitted values into a copy of the geometry file. The module is a port of LaueGo's detector calibration (`microN/Calibration.ipf`, version 1.01). A calibration has these steps:

1. Find the peaks in the frame, and add by hand the peaks that peak search missed.
2. Index the peaks with the starting geometry and the crystal description.
3. Enter the measured energy of some reflections.
4. Fit the detector. For detector 0, the fit also includes the sample rotation.
5. Compare the orientation of the crystal with the expected normal of its surface.
6. Write the refined detector into a new geometry file.

The tests cover controlled numerical cases, including a port of the fake-data test in `Calibration.ipf`. **Empirical agreement with a calibration made in Igor remains pending**; see [calibration validation](../development/calibration-validation.md).

## Conventions

The calibration uses the conventions of the rest of lauelab:

- Peak positions in a calibration table and in a result are zero-based full-detector pixel coordinates `(x, y)`.
- Detector translations are in µm. The geometry file contains them in mm.
- A reciprocal matrix $B$ contains the vectors a\*, b\* and c\* as rows in `1/nm`, including $2\pi$, so that $q = hkl\,B$ for a row $hkl$.
- Rotation vectors are axis times angle in radians, as in the geometry file. They are not the Rodrigues vectors (axis times $\tan(\theta/2)$) that {func}`~lauelab.analysis.orientation_to_rodrigues` returns.

A sample rotation $U$ rotates the crystal reference basis to the laboratory frame: $q = U B_0^{\mathsf T}\,hkl$ for a column $hkl$, where the rows of $B_0$ are the reciprocal vectors of the reference basis. Igor's `rhoSample` has the same direction: `Calibration.ipf` multiplies the reciprocal lattice of the standard orientation by it.

```{warning}
Igor writes a matrix in a wave note, such as `rotation_matrix0`, with one column in each brace group. If you read the groups as rows, you get the transpose, which is the inverse rotation. Read such text with {func}`~lauelab.calibration.matrix_from_igor`.
```

## Run a controlled example

This example uses simulated peaks and no detector image. It simulates the peaks of a Si crystal on detector 0 of a geometry file from the test data, changes the rotation and translation of that detector, and fits them back from the peaks.

```python
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from lauelab.analysis import lattice_params_to_reciprocal, simulate_reflections
from lauelab.calibration import (
    calibrate_detector, calibration_table, peak_directions, rotation_matrix,
    surface_check, write_geometry,
)
from lauelab.indexing import index_orientations, load_crystal, load_geometry

crystal = load_crystal("tests/config/Si.xml")
geometry_path = Path("tests/config/geoN_2023-04-06_03-07-11.xml")
truth = load_geometry(geometry_path).detector(0)

# Simulate the peaks of a Si (001) crystal in Igor's "perfect Si" orientation, from 6 keV to 20 keV.
perfect = rotation_matrix([-2.2256407716336, -0.884037203122317, -0.38914203759791])
reference = lattice_params_to_reciprocal(*[0.5431020511] * 3, 90, 90, 90, space_group=227)
simulated = simulate_reflections(crystal, reference @ perfect.T, truth, energy_range_kev=(6, 20))
peaks = simulated.detector_xy

# Starting detector: rotated by 0.12 degree and translated by 0.54 mm
# (components in µm) from the true one.
rotation_vector = truth.rotation_vector + [0.002, 0.001, -0.001]
start = replace(
    truth, rotation_vector=rotation_vector, rotation=rotation_matrix(rotation_vector),
    translation=truth.translation + [300.0, 200.0, -400.0],
)

# Index the peaks with the starting detector.
patterns = index_orientations(
    peak_directions(start, peaks), crystal,
    kev_max_calc=20, kev_max_test=25, angle_tolerance_deg=0.3,
)
table = calibration_table(peaks, patterns[0])

# Enter one measured energy: the simulated energy of the first peak. Indexing can
# give the peak the Miller indices of another harmonic, so set the simulated harmonic.
row = int(np.argmin(np.linalg.norm(table.pixels - peaks[0], axis=1)))
order = np.gcd.reduce(np.abs(simulated.hkl[0]))
direction = table.hkl[row] // np.gcd.reduce(np.abs(table.hkl[row]))
table = table.with_hkl(row, direction * order)
table = table.with_energy(row, simulated.energy_kev[0], "energy scan 123")

# Fit detector 0 and the sample rotation.
result = calibrate_detector(table, crystal, patterns[0].reciprocal, start)
assert result.success, result.message
assert np.allclose(result.refined.translation, truth.translation, atol=2.0)
print(result.refined.translation / 1000, "mm", result.rms_pixel_error, "pixel")

# Compare the crystal orientation with the expected surface normal.
check = surface_check(crystal, result.refined_reciprocal)
print(check.hkl, round(check.angle_to_expected_deg, 2), check.between)

# Write the refined detector into a copy of the geometry file and read it back.
with TemporaryDirectory() as folder:
    written = write_geometry(
        geometry_path, Path(folder) / "geoN_calibrated.xml", [result.refined],
        note=result.summary_note(), file_note="simulated Si frame",
    )
    assert np.allclose(load_geometry(written).detector(0).translation, result.refined.translation)
```

The example prints the refined translation in mm and the root-mean-square pixel error, and then the result of the orientation check. It also asserts that the fit succeeds, that the refined translation agrees with the translation in the geometry file, and that the written geometry file contains the refined translation.

## Prepare the reflections

{func}`~lauelab.calibration.calibration_table` returns a table with one row for each peak that a pattern indexed. Pass the peaks in the order in which their scattering vectors were indexed, because `pattern.pk_index` refers to that order. Peak search returns frame coordinates. If the frame is a region of the detector or is binned, also pass `start` and `group`, so that the table contains full-detector coordinates.

To add a peak that peak search missed, fit it with {func}`~lauelab.calibration.fit_peak` and append the result to the peak array before you index. The function fits a two-dimensional Gaussian around the brightest pixel near the position that you give, in frame coordinates. It returns `None` when it cannot fit a peak.

{func}`~lauelab.calibration.peak_directions` converts full-detector pixel coordinates to unit scattering vectors. It accepts any {class}`~lauelab.indexing.DetectorGeometry`, including a refined one. To match more peaks after a fit:

1. Index the peaks again with `result.refined`.
2. Call `calibration_table` again and pass the earlier table as `previous`.

The new table then contains each measured energy in the row of the same peak.

## Enter measured energies

The measured energy of a reflection determines the length of its scattering vector, and therefore the detector distance. A fit of the sample rotation (detector 0) requires at least one measured energy, as in LaueGo. In the fit, the energy of a row has the weight $\min((n/n_E)^2, 40)$, where $n$ is the number of rows in the table and $n_E$ is the number of rows with a measured energy. The [algorithm](../concepts/algorithms/calibration.md) page defines the complete error.

The energy of a spot depends on the harmonic: the 008 reflection has the same direction as 004 and twice the energy. Indexing gives each spot the Miller indices of one harmonic. If a measured energy belongs to another harmonic, set the Miller indices of that row with {meth}`~lauelab.calibration.CalibrationTable.with_hkl`. {func}`~lauelab.calibration.calibrate_detector` does not start the fit and returns the status `"energy_mismatch"` when a measured energy differs by more than 25 % from the energy calculated for its row.

## Fit detector 0 and other detectors

For detector 0, `calibrate_detector` fits the sample rotation together with the detector. A rotation of the sample and the detector together about the incident beam changes no measured value. The error therefore has a term for the difference between the detector azimuth about the incident beam and `azimuth_deg`. The default of 90 degrees is LaueGo's beamline preference for 34-ID-E.

For another detector, pass `fit_sample_rotation=False` and the `refined_reciprocal` of the detector-0 calibration of the same sample. The sample rotation is then fixed, and no measured energy is required.

{class}`~lauelab.calibration.DetectorCalibration` contains the refined detector, the sample rotation, the pixel error and energy error of each row, and the standard errors of the parameters. The standard errors are calculated from the fit residuals with Igor's weights. They are not a calibrated measurement uncertainty. When a fit cannot run or does not converge, `success` is false, `message` contains the reason, and the refined values are equal to the initial values.

## Check the crystal orientation

{func}`~lauelab.calibration.surface_check` finds the member of a reflection family, by default {001}, whose scattering vector is closest to the expected surface normal. The default normal is at 45 degrees to both the reversed incident beam and the +y axis, between the two. This is the assumed mounting of the Si calibration standard at 34-ID-E.

`angle_to_expected_deg` is the angle between that scattering vector and the expected normal. For Igor's perfect-Si orientation the angle is 1.1 degrees. For the transposed rotation it is 29.9 degrees.

```{warning}
Use `angle_to_expected_deg`, not `between`, to detect a rotation that was applied in the wrong direction. For a cubic crystal, some member of a family such as {001} is always within about 55 degrees of any direction, so `between` can be true for a wrong rotation.
```

## Write the geometry file

{func}`~lauelab.calibration.write_geometry` writes a copy of the starting geometry file. In the copy, only these elements differ from the source:

- `<R>`, `<P>`, `<timeMeasured>` and `<note>` of each calibrated detector.
- The times in the file header: `<dateWritten>`, `<timeWritten>` and `<EPOCH>`.
- `<fileNote>`, if you pass `file_note`.

The C programs and Igor find an element by a search of the file text for its tag. For this reason, the function adds no new element names, and these constraints apply:

- `note` and `file_note` must be printable ASCII without `<`, `>` or `&`.
- `note` has at most 99 characters, because Igor stores a detector note in 100 bytes.
- The destination file must not exist. The function does not replace an existing file.
- The source file must not contain an XML tag inside a comment. The C programs would read the tag, while Igor removes comments first.

See the [API reference](../reference/calibration.md) for the complete signatures and result fields.
