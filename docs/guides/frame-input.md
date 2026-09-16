# Frame input

The indexing API accepts either an in-memory NumPy array or a path to the supported 34-ID-E HDF5 layout. Both forms produce the same `FrameResult` model.

## NumPy frames

An array must be two-dimensional with one of the supported dtypes:

```python
import numpy as np

frame = np.zeros((2048, 2048), dtype=np.uint16)
```

The shape is `(ny, nx)`. NumPy accesses a pixel as `frame[y, x]`, while fitted peak coordinates are reported as `(x, y)`.

Raw 34-ID-E frames are `numpy.uint16`. Reconstructed frames can be `numpy.int32` (both wire edges), `numpy.float64` (in-memory reconstruction), or another type chosen at write time, so peak search also accepts `uint8`, `int8`, `int16`, `int32`, `float32`, and `float64`. {data}`~lauelab.indexing.indexer.SUPPORTED_FRAME_DTYPES` lists them. Every supported element converts to a double exactly, so an `int32` frame and a `uint16` frame with the same values give identical peaks. A frame stored in the non-native byte order is byte-swapped to a native copy of the same dtype. Floating-point frames must be finite.

Any other dtype, such as `uint32`, `int64`, or `bool`, raises `InputError`. Nothing is cast, clipped, or rescaled implicitly: convert intentionally before the call so that the change is visible in your application.

The indexer makes a C-contiguous copy only when the supplied array is not already contiguous.

## HDF5 frames

Pass a path to read a frame from `entry1/data/data`:

```python
result = indexer.index("frame.h5")
```

This is support for a specific acquisition layout, not arbitrary HDF5. A missing image dataset raises `KeyError`, and a file-open failure raises `OSError`.

When present, the loader reads processing values from:

| Value | HDF5 dataset |
|---|---|
| `start_x` | `entry1/detector/startx` |
| `start_y` | `entry1/detector/starty` |
| `group_x` | `entry1/detector/binx` |
| `group_y` | `entry1/detector/biny` |

HDF5 `start` and `group` values take precedence over values passed to `index()`.

Depth works the other way. When `entry1/depth` is present and its first value is finite, that value is the frame's physical depth in µm unless `depth` is passed explicitly. An explicit finite `depth`, including `0.0`, overrides the file. Reconstructed per-depth files written by {class}`~lauelab.reconstruct.Reconstructor` and by `reconstructN_cpu` carry this dataset; raw frames normally do not, and a missing or non-finite value means no depth. `FrameResult.depth` records the value that was used. This matches the LaueGo `peaksearch` program, which reads the same dataset into its `$depth` header.

Selecting a reconstructed frame by its position in a depth stack is not the same as its physical depth. Read the depth from the file or the reconstruction result, and pass `depth` only when you intend to override it.

## Region and grouping

For an in-memory frame, `start=(x, y)` identifies the frame's zero-based origin on the full detector. `group=(x, y)` gives the number of detector pixels represented by one frame pixel along each axis.

```python
region = np.zeros((512, 512), dtype=np.uint16)
result = indexer.index(
    region,
    start=(100, 200),
    group=(2, 2),
)
```

Both `start` values must be nonnegative integers. Both `group` values must be positive integers. The transformed frame extent must remain within the selected detector.

The pixel-to-q conversion uses the center of each grouped detector region. See [Geometry](geometry.md) for the mapping.

`depth` is an optional finite sample depth in micrometres passed to geometry conversion. The physical sign convention requires 34-ID-E domain verification and is not inferred by this documentation.

## Masks

A mask must have the same shape as the frame:

```python
mask = np.zeros(frame.shape, dtype=np.uint8)
mask[100:120, 300:340] = 1

result = indexer.index(frame, mask=mask)
```

The API converts the mask to contiguous `uint8`. Zero pixels remain available to peak search. Nonzero pixels are masked.

A mask saved as a 34-ID-E HDF5 image, the file `peaksearch -K` accepts, loads with {func}`~lauelab.indexing.load_mask`:

```python
from lauelab.indexing import load_mask

mask = load_mask("mask.h5")
result = indexer.index(frame, mask=mask)
```

It returns a boolean array in which `True` marks an excluded pixel. `index()` still checks the shape against the frame.

## Metadata

Pass {class}`~lauelab.indexing.FrameMetadata` or a mapping when the result needs experiment provenance:

```python
from lauelab.indexing import FrameMetadata

metadata = FrameMetadata(
    sample_name="synthetic nickel",
    scan_number=42,
    detector_id="PE1621 723-3335",
    exposure_seconds=0.25,
)

result = indexer.index(frame, metadata=metadata)
```

For HDF5 input, the loader reads recognized metadata fields when their datasets exist. Explicit metadata values override values loaded from the file.

The HDF5 detector identifier is also checked against the selected geometry detector. A mismatch raises `InputError`. For an in-memory frame, `metadata.detector_id` is provenance only and is not used for this validation.

See the {class}`~lauelab.indexing.FrameMetadata` reference for the complete field list.

## Image ownership

`index_frame()` and `Indexer.index()` retain the contiguous image, in its input dtype, in `result.image` by default. Pass `keep_image=False` when later processing only needs peaks and patterns.

```python
result = indexer.index(frame, keep_image=False)
assert result.image is None
```

`Indexer.index_many()` defaults to `keep_images=False` to limit batch memory use. Pass `keep_images=True` only when every result needs its source image.

The result's peak and pattern arrays are Python-owned copies. Native result storage is released before the method returns.

## Common failures

`InputError` reports:

- A frame with the wrong number of dimensions, an unsupported dtype, or non-finite values
- Invalid `start`, `group`, or `depth`
- A frame region outside the selected detector
- A mask with a different shape
- An HDF5 detector identifier that does not match the selected detector

A no-peak result is not an input failure. It returns an empty peak array and no patterns.
