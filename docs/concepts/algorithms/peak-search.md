# Peak search

Peak search converts a detector intensity array into fitted peak positions and shape measurements. It runs before geometry conversion and does not use the crystal description.

## Input image

The stage receives a two-dimensional frame with shape `(ny, nx)` and one of the [supported dtypes](../../guides/frame-input.md); each pixel enters as its exact double value. If `smooth=True`, the native implementation applies Gaussian smoothing before thresholding and fitting.

A mask has the same shape as the frame. Zero mask values leave pixels available. Nonzero values replace the corresponding working pixels with a background estimate before detection. The source array retained in `FrameResult.image` is not this modified working buffer.

## Threshold selection

With a numeric `PeakParams.threshold`, the stage uses that value as an absolute detector-count threshold.

With `threshold=None`, it calculates the mean and population standard deviation of unmasked, nonzero working pixels. The threshold is:

```{math}
T = \bar{I} + r\sigma
```

Here, `T` is the threshold, `\bar{I}` is the selected-pixel mean, `\sigma` is the selected-pixel population standard deviation, and `r` is `threshold_ratio`.

Automatic thresholding returns a valid empty result when the image has no unmasked, nonzero pixels. In that case, `FrameResult.threshold_used` is `NaN`. Otherwise, the selected threshold is returned in `FrameResult.threshold_used`.

## Detection and fitting

The native stage finds connected regions above the threshold, fits each candidate in a square neighborhood, and accepts fits that satisfy configured width, displacement, and residual checks. `peak_shape` selects a Lorentzian or Gaussian model.

`boxsize` is the half-width of the fitting neighborhood. The implementation clips a neighborhood at image boundaries rather than reading outside the frame.

After fitting, nearby accepted peaks are filtered according to `min_separation`. The implementation compares their fitted x and y separations and removes the lower-intensity candidate when both are below the configured value.

`max_peaks` limits the number of candidate peaks processed; `None` fits every blob above the threshold. Do not treat it as a scientific acceptance threshold.

## Coordinates and fields

The native fitting code uses one-based coordinates internally. The public API subtracts one before constructing `FrameResult`, so `fit_x` and `fit_y` are zero-based frame coordinates.

The returned peak row contains:

- Fitted `(x, y)` position
- Fitted intensity and background
- Integrated intensity
- Half-widths and tilt
- Normalized fit residual in `chisq`
- A `qhat` field populated by the next stage

See [Results](../../guides/results.md) for the complete schema.

## Limits of interpretation

The implementation defines how parameters enter detection and fit acceptance. It does not establish universal values for a detector, exposure, or sample. Validate changes against representative frames and preserve the accepted-peak set when comparing orientation-indexing settings.