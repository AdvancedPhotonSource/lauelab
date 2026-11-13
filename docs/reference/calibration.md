# Calibration

`lauelab.calibration` fits the rotation and translation of a detector to reflections of a crystal with a known lattice, and writes the result into a copy of a geometry file. The [guide](../guides/calibration.md) contains a complete synthetic workflow. The [algorithm](../concepts/algorithms/calibration.md) page defines the error that the fit minimizes and its weights.

The module consists of functions and has no configured object. A calibration is made a few times a year from one frame, so there is no configuration to reuse between calls. Each function returns a new value and keeps no state.

```{eval-rst}
.. currentmodule:: lauelab.calibration

.. autofunction:: calibration_table

.. autoclass:: CalibrationTable
   :members: n_energies, with_energy, with_hkl

.. autofunction:: calibrate_detector

.. autoclass:: DetectorCalibration
   :members: success, refined_reciprocal, sample_rotation_vector, rms_pixel_error, rms_energy_error_ev, summary_note, provenance

.. autofunction:: surface_check

.. autoclass:: SurfaceCheck

.. autofunction:: write_geometry

.. autofunction:: fit_peak

.. autofunction:: peak_directions

.. autofunction:: detector_azimuth_deg

.. autofunction:: rotation_matrix

.. autofunction:: rotation_vector

.. autofunction:: matrix_from_igor

.. autodata:: HC_KEV_NM

.. autodata:: DEFAULT_SURFACE_NORMAL

.. autodata:: STATUSES
```

## Units and shapes

| Quantity | Convention |
| --- | --- |
| Pixel positions | Arrays of shape `(n, 2)` in zero-based full-detector `(x, y)` coordinates. `NaN` marks a missing value. The `peaks` argument of {func}`calibration_table` and the positions that {func}`fit_peak` accepts and returns are frame coordinates. |
| Translations | µm |
| Energies | keV |
| Energy errors | eV |
| Angles | Degrees. Rotation vectors, the standard errors `rotation_vector_sigma` and `sample_rotation_sigma`, and values whose name ends in `_rad` are in radians. |
| Rotation vectors | Axis times angle in radians |
| Reciprocal matrices | Rows a\*, b\*, c\* in `1/nm`, including $2\pi$ |

## Ownership and errors

The arrays of a {class}`CalibrationTable`, a {class}`DetectorCalibration` and a {class}`SurfaceCheck` are read-only.

Invalid arguments raise `InputError`. {func}`write_geometry` raises `FileExistsError` when the destination exists.

A fit that cannot run or does not converge raises no exception. {func}`calibrate_detector` returns a {class}`DetectorCalibration` whose `status` is not `"ok"`, and the refined values are equal to the initial values.
