# Wire-scan reconstruction

A wire scan records a stack of detector frames while a wire moves across the diffracted beams. Each diffracted beam leaves the sample from a specific depth along the incident beam. As the wire edge crosses that beam, the intensity it occludes disappears from the detector between one frame and the next. The geometry of the wire, the detector pixel, and the incident beam determines the depth at which that intensity originated.

Reconstruction resolves the frame stack into one detector image for each requested depth. `lauelab` performs this work in-process with {class}`~lauelab.reconstruct.Reconstructor` and returns a {class}`~lauelab.reconstruct.ReconstructionResult`.

The differential-aperture X-ray microscopy method is described by [Larson et al. (2002)](https://doi.org/10.1038/415887a) and [Yang et al. (2004)](https://doi.org/10.1016/j.micron.2004.02.004).

## Inputs and outputs

A reconstruction call needs:

- A wire-scan point: `N` detector frames with shape `(N, rows, columns)` and the wire position before and after each frame, with shape `(N + 1, 3)`
- A geometry file with a complete wire section
- A detector slot in that geometry
- A depth range and depth resolution in µm
- Reconstruction options, or their defaults

Pass a 34-ID-E multi-image HDF5 file or in-memory arrays for the point. For file input, reconstruction reads frames, wire positions and acquisition metadata using the conventions in [Reconstruct a wire scan](../guides/reconstruction.md). For array input, pass aligned frames and wire positions; file-format bookkeeping does not apply.

The result contains the reconstructed depth coordinates, the summed intensity at each depth, per-stripe timings, and optional retained images with shape `(n_depths, rows, columns)` and dtype `numpy.float64`. File reconstruction can also write one HDF5 file for each depth and a summary text file.

## Pixel depth

For one detector pixel and one wire position, the implementation calculates the depth along the incident beam at which the wire edge occludes the pixel. It transforms the pixel to a laboratory position through the detector geometry, transforms the raw wire position through the wire geometry, and intersects the line from the pixel past the wire edge with the incident beam.

The `wire_edge` option selects the leading edge, the trailing edge, or both. Each edge produces its own depth for the same pixel and wire position.

The depth of a pixel changes as the wire moves. Consecutive wire positions therefore bound a depth interval for each pixel, and the intensity that disappears between the corresponding frames is assigned to that interval. See [Depth reconstruction](algorithms/depth-reconstruction.md) for the transformations and the tangent construction.

## Differencing and depth binning

For each pixel that passes the intensity mask, the implementation reads the pixel's value from every scan frame, applies the selected normalizations, and forms the difference between consecutive frames. Each nonzero difference is distributed across the output depth bins that overlap the pixel's depth interval for that frame pair, weighted by a trapezoid that accounts for the finite pixel width.

The output depths form a regular grid. `depth_range` endpoints are rounded to multiples of `resolution`, and each output image corresponds to one grid point. `ReconstructionResult.depth_um` lists the grid in µm.

The intensity mask retains the brightest pixels of the intensity map. Pixels outside the mask are not reconstructed and remain zero in every output image.

## In-process execution

The native kernel processes the image in stripes of rows. For each stripe, the driver reads the input rows, runs the kernel across OpenMP threads, and writes the finished stripe to the output files while the next stripe computes. Input and output for one stripe therefore overlap with computation.

```text
Wire-scan point (frames + wire positions)
    |
    v
Intensity mask and normalization <--- Intensity map
    |
    v
Per-pixel depth mapping <------------- Geometry (detector + wire)
    |
    v
Frame differencing and depth binning
    |
    v
ReconstructionResult ----------------> Per-depth HDF5 files (optional)
```

The result does not depend on the thread count or the stripe size. Those options change memory use and elapsed time only.

## One point or many

Use {class}`~lauelab.reconstruct.Reconstructor` directly for one point or for a sequence of points processed in one process. Create it once and reuse it for every point that shares the detector layout and reconstruction options.

Use {func}`~lauelab.reconstruct.reconstruct_points` to process several point files in worker processes. Each worker holds one reconstructor and a fixed OpenMP thread count.

## Failure model

Invalid arguments and setup failures raise before any stripe is processed:

- {class}`~lauelab.indexing.InputError` reports invalid options, geometry, file metadata, or array shapes.
- {class}`MemoryError` reports a native allocation failure during setup.

After the first stripe starts, an expected reconstruction or I/O failure returns a result with `success=False`, the error message, and the progress made so far. {class}`~lauelab.indexing.ReconstructionError` identifies a native failure in that message. Check `success` on every result before using its arrays or files.
