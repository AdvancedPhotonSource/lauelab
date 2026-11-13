# Detector calibration

{func}`~lauelab.calibration.calibrate_detector` fits the rotation vector $R$ (rad) and the translation $P$ (µm) of one detector. For detector 0, it also fits the sample rotation $U$. The fit follows `OptimizeAll`, `OptimizeDetectorCalibration` and `CalibrationErrorRPinternal` in LaueGo `microN/Calibration.ipf` (version 1.01).

This page defines the error that the fit minimizes, so that you can interpret the errors in a result and the statuses of a fit that did not run. The [guide](../../guides/calibration.md) shows the procedure.

## Model

Each row of the calibration table is one reflection with Miller indices $hkl$. A row can have a pixel position, a measured energy $E$, or both. A position that is `NaN` or outside the detector counts as no pixel.

The calculated scattering vector of a row is

$$q_c = U B_0^{\mathsf T}\,hkl,$$

where the rows of $B_0$ are the reciprocal vectors of the crystal's reference basis (the native basis, with c parallel to z) in `1/nm`, including $2\pi$.

The measured scattering vector has a direction $\hat q_m$ and a length $|q_m|$:

- For a row with a pixel, $\hat q_m$ is the unit vector of $\hat k_f - \hat z$. Here $\hat z$ is the direction of the incident beam, and $\hat k_f$ is the unit vector from the origin to the pixel on the detector with rotation $R$ and translation $P$. For a row without a pixel, $\hat q_m$ is the direction of $q_c$.
- For a row with a measured energy, $|q_m| = 4\pi \sin\theta\, E / hc$ with $\sin\theta = -\hat z\cdot\hat q_m$. For a row without a measured energy, $|q_m| = |q_c|$. The constant $hc$ is 1.239841984 keV nm, the value in `Calibration.ipf`.

The difference $\Delta q = q_c - |q_m|\,\hat q_m$ has a radial part $\Delta q_r = \hat q_m\cdot\Delta q$ and a perpendicular part $\Delta q_\perp = \Delta q - \Delta q_r\,\hat q_m$. The squared error of row $i$ is

$$e_i^2 = \frac{[\text{pixel}]\,|\Delta q_\perp|^2 + [\text{energy}]\,w_E\,\Delta q_r^2}{|q_m|^2},$$

where $[\text{pixel}]$ is 1 for a row with a pixel and 0 otherwise, and $[\text{energy}]$ is 1 for a row with a measured energy and 0 otherwise. The energy weight is $w_E = \min((n/n_E)^2, 40)$, where $n$ is the number of rows and $n_E$ is the number of rows with a measured energy. As in Igor, $n$ is the number of all rows of the table, including the rows that have neither a pixel on the detector nor a measured energy.

Igor's error is

$$\epsilon = \sqrt{\frac{\sum_i e_i^2}{\sum_i w_i} + \left(10\,\Delta\phi\right)^2},$$

where $w_i$ is $w_E$ for a row with a measured energy and 1 for a row with only a pixel. The sums include only the rows that have a pixel or a measured energy. $\epsilon$ is dimensionless.

### Azimuth term

The azimuth term $\Delta\phi$ (rad) is in the error only when the sample rotation is fitted. Without it, a rotation of the sample and the detector together about the incident beam would not change the error.

$\Delta\phi$ is the detector azimuth about the incident beam minus the requested azimuth (`azimuth_deg`). The detector azimuth is Igor's `AzimuthAngle`, which {func}`~lauelab.calibration.detector_azimuth_deg` returns:

- For a detector far from the incident beam, it is the azimuth of the detector's z axis.
- For a detector close to forward or back scattering, it is the azimuth of the detector's +x pixel direction.

Unlike Igor, lauelab wraps the difference to $[-180°, 180°)$. The wrap changes nothing for azimuths away from ±180 degrees.

## Optimization

Igor minimizes $\epsilon$ with a quasi-Newton method (`Optimize /M={0,0}`) over $R$, $P$ and the sample rotation vector. lauelab minimizes the same sum of squares with `scipy.optimize.least_squares` (trust-region reflective). The residual vector contains these values, and its norm is $\epsilon$:

- For each row with a pixel, the three components of $\Delta q_\perp/|q_m|$, divided by $\sqrt{\sum_i w_i}$.
- For each row with a measured energy, $\sqrt{w_E}\,\Delta q_r/|q_m|$, divided by $\sqrt{\sum_i w_i}$.
- When the sample rotation is fitted, $10\,\Delta\phi$.

lauelab varies the sample rotation as a small rotation vector that is applied to the starting rotation. This parameter has no singularity when the rotation vector of the sample is near 180 degrees. The tests compare the minimum with the minimum that SciPy's BFGS finds for Igor's scalar error over Igor's parameters.

The result is rounded as Igor rounds it:

- $R$ is rounded to $10^{-8}$ rad.
- $P$ is rounded to the power of ten nearest 1 % of a pixel (1 µm for 200 µm pixels).

The standard errors are the square roots of the diagonal of $(J^{\mathsf T}J)^{-1}\,\epsilon^2/(m-p)$, where $J$ is the Jacobian of the residuals, $m$ is the number of measured values (two per pixel, one per energy, one for the azimuth) and $p$ is the number of parameters (9 when the sample rotation is fitted, 6 otherwise). They describe only the scatter of the residuals under Igor's weights. They are not a calibrated measurement uncertainty.

## Checks before and after the fit

`calibrate_detector` makes these checks before the fit. If one fails, the fit is not started.

- A fit of the sample rotation needs at least one measured energy, as in LaueGo. Otherwise the status is `"insufficient_data"`.
- $m$ must be larger than $p$. Otherwise the status is `"insufficient_data"`.
- Each measured energy must be within 25 % of the energy calculated for its row from the starting sample rotation. A measured energy that differs more most likely belongs to another harmonic. The status is then `"energy_mismatch"`.

After the fit, the function calculates the condition number of the Jacobian, with each parameter in units of its typical step (1 mrad for rotations, one pixel for the translation). The condition number must not be larger than $10^8$. A larger value means that the reflections do not determine some combination of parameters, for example when all rows repeat one reflection or no row has a pixel. The status is then `"insufficient_data"`. The controlled cases have condition numbers from about 60 to 1,300.

A fit that does not converge has the status `"not_converged"`.
