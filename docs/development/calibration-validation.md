# Calibration validation

The evidence for the calibration has two separate gates, as for [strain](strain-validation.md):

1. **Controlled numerical correctness.** The tests pass for the controlled cases that are described below.
2. **Empirical Igor parity.** A calibration is made in Igor and repeated in lauelab with the same inputs, and the results are compared. This gate remains **pending**: the development environment has no runnable Igor installation and no exported Igor calibration. `DetectorCalibration.provenance()` returns `"empirical_igor_parity": "pending"`.

The controlled tests do not establish empirical parity.

## Controlled cases

`tests/calibration_cases.py` contains the controlled cases, which are ported from Igor:

- `MakeFakeCalibrationData`, with its 25 pixel positions for detector 0, its 25 pixel positions for the side detectors, and its sample rotation of -135 degrees about x.
- `GeoReferenceOrientation`, which defines the reference detectors.
- `CalibrationErrorRPinternal` and the azimuth term of `CalibrationErrorRPrho`, ported line by line and not vectorized, so that they check the vectorized residuals independently.

The tests check that:

- The vectorized residuals give Igor's error.
- The fit recovers detector 0 together with the sample rotation.
- The fit recovers the side detectors with a fixed sample rotation.
- A fit that starts with the detector rotated by 0.66 degree and translated by 3.4 mm recovers detector 0 of the 2023-04-06 geometry file.
- `least_squares` and BFGS on Igor's scalar error reach the same minimum.
- The native geometry reader reads a written geometry file back.

## Export an Igor calibration

Keep these items from one Igor calibration of detector 0:

- The frame.
- `FullPeakList` and `FullPeakIndexed` with their wave notes. The note of `FullPeakIndexed` contains `rotation_matrix0`.
- The `Emeasured` wave.
- The geometry file that was loaded before the calibration and the geometry file that was written after it.
- The `CalibrationList` wave after `OptimizeAll`, with its wave note (`rhox`, `rhoy`, `rhoz`, `azimuth`, `errStart`, `err`).

To compare the Igor calibration with lauelab:

1. Read `rotation_matrix0` with {func}`~lauelab.calibration.matrix_from_igor`.
2. Compare the starting error, the refined $R$ and $P$, the sample rotation and the `calculated_keV` column.
3. Only then compare values that are derived from them.

Choose tolerances from the measurement precision and record the justification. Never edit the exported values by hand.
