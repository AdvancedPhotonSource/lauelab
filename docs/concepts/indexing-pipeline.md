# Indexing pipeline

Indexing converts intensity measurements on a detector into one or more candidate crystal orientations. `lauelab` performs this work in three stages and returns their outputs together in a {class}`~lauelab.indexing.FrameResult`.

## Inputs and outputs

A complete orientation-indexing call needs:

- A two-dimensional detector frame, `numpy.uint16` for raw frames or another [supported dtype](../guides/frame-input.md) for reconstructed ones
- A geometry that maps detector pixels into the 34-ID-E laboratory frame
- A crystal description with a unit cell and space group
- Peak-search and indexing parameters, or their defaults

The frame can be an in-memory NumPy array or a supported 34-ID-E HDF5 file. You can omit the crystal to run only peak search and pixel-to-q conversion.

The result contains fitted peaks, frame statistics, and any indexed patterns. It also records the frame shape, detector region, grouping, depth, metadata, and elapsed times needed to interpret the processing call.

## Peak search

Peak search starts from detector intensities. It detects candidate regions, fits each accepted peak, and returns subpixel positions and fit measurements. The configurable inputs control the intensity threshold, fit region, accepted size and residual, separation, peak model, smoothing, and maximum count.

An absolute threshold is used by default. Set `PeakParams.threshold` to `None` to derive a threshold from frame statistics and `threshold_ratio`.

Peak coordinates are zero-based `(x, y)` values in the supplied frame. A NumPy image uses `[y, x]` indexing, so `image[y, x]` addresses the pixel at coordinate `(x, y)`.

See [Peak search](algorithms/peak-search.md) for stage details and [Parameters](../guides/parameters.md) for configuration.

## Pixel-to-q conversion

The geometry stage maps each fitted frame coordinate to its full-detector position. It applies the frame's `start` and `group`, then uses the selected detector geometry to calculate a unit scattering vector named `qhat`.

A geometry can contain more than one detector. The selected index identifies a physical detector slot in the geometry file, not its ordinal position among active detectors.

The returned `qhat` values use the 34-ID-E laboratory convention implemented by the geometry conversion. See [Pixel-to-q conversion](algorithms/pixel-to-q.md) for the transformation and the limits of the currently verified coordinate description.

## Crystal indexing

When you supply a crystal and the frame contains at least two peaks, the orientation indexer compares the measured scattering-vector directions with reflections calculated from the crystal description. It can return zero, one, or multiple candidate patterns.

Each {class}`~lauelab.indexing.Pattern` contains an orientation, reciprocal-lattice information, assigned Miller indices, and zero-based indices back into the frame's peak array. A successful call can return zero patterns.

See [Crystal indexing](algorithms/crystal-indexing.md) for supported algorithm details.

## In-process execution

The preferred API calls the native library in the Python process. It does not start a subprocess or create intermediate peak, pixel-to-q, and indexing text files.

```text
Detector frame
    |
    v
Peak search
    |
    v
Pixel-to-q conversion <--- Detector geometry
    |
    v
Crystal indexing <--------- Crystal description
    |
    v
FrameResult --------------> LaueGo XML (explicit export)
```

Native memory is released before `FrameResult` is returned. Peak and pattern data are copied into NumPy arrays owned by Python.

## One frame or many

Use {func}`~lauelab.indexing.index_frame` for a one-off call. It creates a temporary `Indexer` and retains the image by default.

Use {class}`~lauelab.indexing.Indexer` when frames share configuration. `Indexer.index()` processes one frame. `Indexer.index_many()` processes an iterable sequentially, preserves order, and does not retain images by default.

## Failure model

The in-process API distinguishes three failure classes:

- {class}`~lauelab.indexing.InputError` reports invalid configuration, detector selection, frame data, or metadata.
- {class}`MemoryError` reports a native allocation failure.
- {class}`~lauelab.indexing.IndexingError` reports a native numerical or internal failure.

A frame with no peaks or no indexed patterns is a successful result. Check `n_peaks`, `n_patterns`, and `indexed` instead of treating an empty result as an exception.
