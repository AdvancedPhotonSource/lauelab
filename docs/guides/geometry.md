# Geometry

A geometry file describes the physical detectors used to convert fitted pixel positions into scattering-vector directions. Load it once when several frames share the same detector configuration.

## Load geometry

{func}`~lauelab.indexing.load_geometry` parses and validates detector geometry:

```python
from lauelab.indexing import load_geometry

geometry = load_geometry("geometry.xml")
print(geometry.detector_count)
```

You can pass the returned {class}`~lauelab.indexing.Geometry` directly to `Indexer`. This avoids parsing the file again when you construct related indexers.

```python
from lauelab.indexing import Indexer

indexer = Indexer(geometry, "crystal.xml")
```

## Select a detector

Select a detector by physical slot or exact detector identifier:

```python
slot = geometry.find_detector("PE1621 723-3335")
if slot < 0:
    raise ValueError("detector is not present in the geometry")

indexer = Indexer(geometry, "crystal.xml", detector_index=slot)
```

`Indexer(..., detector_id="PE1621 723-3335")` performs the lookup and reports an {class}`~lauelab.indexing.InputError` if the identifier is absent. If both selection arguments are supplied, `detector_id` determines the selected slot.

```{warning}
A detector index is a physical slot from the geometry file. It is not a zero-based position in the list of active detectors. If slots 0 and 2 are active, `detector_index=1` is invalid rather than a reference to slot 2.
```

## Pixel coordinates and grouping

Frame coordinates use zero-based `(x, y)` order. NumPy arrays use `[y, x]` indexing and have shape `(ny, nx)`.

`start=(start_x, start_y)` gives the full-detector origin of an in-memory frame. `group=(group_x, group_y)` gives the positive integer detector-pixel grouping factor. The complete grouped frame must fit inside the selected detector. See [Pixel-to-q conversion](../concepts/algorithms/pixel-to-q.md) for the coordinate equations and physical transformation.

## Inspect detector metadata

`Geometry.detector(slot)` returns immutable metadata for one active slot:

```python
detector = geometry.detector(slot)

print(detector.detector_id)
print(detector.nx, detector.ny)
print(detector.size_x, detector.size_y)
```

`nx` and `ny` are full-detector dimensions in unbinned pixels. `size_x`, `size_y`, and `translation` use micrometres. `rotation_vector` is an axis-angle vector in radians, and `rotation` is the corresponding `(3, 3)` matrix.

`detector_count` reports the number of active detectors. It does not identify their slots.

## Inspect wire metadata

`Geometry.wire` returns immutable {class}`~lauelab.indexing.WireGeometry` metadata when the geometry file has a complete wire section. It returns `None` for a detector-only geometry. The diameter, focal distance, and origin use µm; the rotation magnitude uses degrees.

```python
wire = geometry.wire
if wire is not None:
    print(wire.diameter_um, wire.origin_um)
```

Reconstruction requires complete wire geometry and raises {class}`~lauelab.indexing.InputError` when it is absent. Indexing accepts detector-only geometry files.

## Validation errors

Geometry loading raises `ValueError` for malformed, incomplete, duplicated, or physically invalid detector declarations. Detector lookup returns `-1` for an unknown identifier. `Geometry.detector()` raises `ValueError` for an inactive or out-of-range slot. `Indexer` converts detector-selection failures to `InputError`.

Do not continue with a different detector after a selection failure. Detector geometry determines every `qhat` value and therefore affects orientation indexing.

## Convert known pixel positions

Use `Geometry.pixels_to_q()` when peak positions already exist and you only need geometry conversion:

```python
import numpy as np

peaks_xy = np.array([
    [720.25, 512.75],
    [1311.50, 984.00],
])

qhat = geometry.pixels_to_q(
    peaks_xy,
    detector_index=slot,
    start=(0, 0),
    group=(1, 1),
)

assert qhat.shape == (2, 3)
```

Each row of `qhat` is a unit scattering vector in the 34-ID-E laboratory convention.
