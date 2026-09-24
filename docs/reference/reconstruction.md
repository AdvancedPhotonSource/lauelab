# Reconstruction

`lauelab.reconstruct` provides an in-process {class}`~lauelab.reconstruct.Reconstructor` and supported subprocess wrappers. The CPU executable, `reconstructN_cpu`, ships with the package. The CUDA executable, `reconstructN_gpu`, requires a separate CUDA build and must be available on `PATH` unless you pass its path explicitly.

See [Reconstruct a wire scan](../guides/reconstruction.md) for usage and [Depth reconstruction](../concepts/algorithms/depth-reconstruction.md) for the calculation.

Use {func}`~lauelab.reconstruct.reconstruct_scan` to reconstruct points in local worker processes. It writes one HDF5 file per point and a shared `scan.h5` catalog, then returns a {class}`~lauelab.reconstruct.ScanResult`.

For an external scheduler, call {func}`~lauelab.reconstruct.prepare_scan` to get a {class}`~lauelab.reconstruct.PreparedScan` coordinator and serializable {class}`~lauelab.reconstruct.PointTask` objects. Workers execute tasks with {func}`~lauelab.reconstruct.reconstruct_point` and return a {class}`~lauelab.reconstruct.PointOutcome`. You can also call `reconstruct_point` directly to write a standalone point.

To index a depth image, select it with {class}`~lauelab.indexing.ScanFrame`, also importable from this module. To produce per-depth files for other tools, use {func}`~lauelab.reconstruct.export_per_depth`. [Reconstruction scan format](../development/reconstruction-scan-format.md) specifies the layouts, statuses, and pixel values.

Every other path returns a {class}`~lauelab.reconstruct.ReconstructionResult`. Runtime failures are recorded in that result. Invalid arguments and setup failures that occur before processing raise exceptions. The native path does not support the executable's `-F` parameter-file option or distortion maps.

The subprocess environment defaults `OPENBLAS_NUM_THREADS` to `1`. The reconstruction programs link OpenBLAS through GSL but do not call BLAS, so extra OpenBLAS workers only consume CPU time. An existing caller setting is preserved. The `num_threads` argument independently controls the CPU program's OpenMP reconstruction threads. The executable's `image_range` (`-f`/`-l`) options apply only to scans stored as one file per image; multi-image HDF5 input used by `Reconstructor` has no file range.

## Behaviour changes in this release

The `reconstructN_cpu` option `-n <tag>` previously had no effect because the normalization-vector length was taken from an HDF5 status code. It now scales each scan frame by `entry1/<tag>`, falling back to `<tag>` at the file root, divided by 102 for `mA` and 88100 for `cnt3`. The in-process path applies the same divisors but reads only `entry1/<tag>`, and it raises {class}`~lauelab.indexing.InputError` when that vector is missing or shorter than the scan. The executable silently skips normalization in the same case. This asymmetry is deliberate: the in-process path reports a request it cannot honour.

A wire axis exactly parallel to the positioner x axis, with a zero wire rotation vector, now uses the identity transformation in both reconstruction paths. Earlier `reconstructN_cpu` releases produced non-finite values for this valid zero-rotation geometry.

```{eval-rst}
.. currentmodule:: lauelab.reconstruct

.. autoclass:: Reconstructor
   :members: reconstruct, reconstruct_array

.. autoclass:: ImageGeometry
   :members:

.. autoclass:: StripeTiming
   :members:

.. autoclass:: ReconstructionResult
   :members:

``ReconstructionResult`` is a dataclass with the same six leading fields as the former named tuple. Attribute access and positional construction are unchanged; tuple unpacking is not supported.

.. autofunction:: reconstruct_points

.. autofunction:: prepare_scan

.. autoclass:: PreparedScan
   :members: record_dispatch, record, snapshot, finish, close

.. autoclass:: PointTask

.. autofunction:: reconstruct_point

.. autofunction:: reconstruct_scan

.. autoclass:: ScanResult
   :members:

.. autoclass:: PointOutcome
   :members:

.. autoclass:: ScanReader
   :members:

.. autoclass:: PointReader
   :members:

.. autoclass:: PerDepthReader
   :members:

.. autoclass:: PointEntry
   :members:

.. autofunction:: validate_scan_file

.. autofunction:: export_per_depth

.. autoclass:: ScanFileSummary
   :members:

.. autofunction:: reconstruct

.. autofunction:: reconstruct_gpu

.. autofunction:: find_executable

.. autofunction:: find_gpu_executable

.. autofunction:: gpu_available

Wire metadata used by reconstruction is documented as
:class:`lauelab.indexing.WireGeometry`.
```

## Inspection

`lauelab.reconstruct.inspection` provides square ROI placement, depth traces, normalization, positive-sample selection for logarithmic axes, and reference images. See [Inspect a reconstructed point](../guides/depth-inspection.md) for examples. These functions operate independently of the plotting library. Every returned array is a new, read-only array that the caller owns. ROI bounds are half-open `(y0, y1, x0, x1)` in stored-image pixels.

```{eval-rst}
.. currentmodule:: lauelab.reconstruct.inspection

.. autofunction:: square_bounds

.. autofunction:: bounds_center

.. autofunction:: depth_trace

.. autofunction:: roi_traces

.. autofunction:: reference_image

.. autoclass:: DepthTrace
   :members:

.. autoclass:: NormalizedTrace
   :members:

.. autoclass:: LogSamples
   :members:

.. autoclass:: ReferenceImage
   :members:

.. autoclass:: ArrayPoint
   :members:
```
