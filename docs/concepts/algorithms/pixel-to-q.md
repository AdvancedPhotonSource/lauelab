# Pixel-to-q conversion

Pixel-to-q conversion maps each fitted frame coordinate to a unit scattering vector. It uses the selected detector's dimensions, physical size, rotation, and translation from the geometry file.

## From frame to detector pixels

For a fitted frame coordinate `(x_f, y_f)`, the corresponding full-detector coordinate is the center of its grouped region:

```{math}
x_d = start_x + x_f group_x + \frac{group_x - 1}{2}
```

```{math}
y_d = start_y + y_f group_y + \frac{group_y - 1}{2}
```

Coordinates are zero-based. With `start=(0, 0)` and `group=(1, 1)`, frame and full-detector coordinates are equal.

## Detector-local position

Let the detector have `Nx` by `Ny` unbinned pixels and physical dimensions `sizeX` by `sizeY`. The implementation calculates detector-local coordinates relative to the detector center:

```{math}
x' = \left(x_d - \frac{N_x - 1}{2}\right)\frac{sizeX}{N_x} + P_x
```

```{math}
y' = \left(y_d - \frac{N_y - 1}{2}\right)\frac{sizeY}{N_y} + P_y
```

```{math}
z' = P_z
```

The geometry rotation matrix maps this position into the 34-ID-E laboratory frame. If `depth` is supplied in micrometres, the implementation subtracts it from the transformed laboratory z component.

```{warning}
The physical sign convention for `depth` and the names and positive directions of the laboratory axes require 34-ID-E domain verification. Use a geometry and depth convention from the same workflow.
```

## Scattering direction

The transformed detector position is first normalized to the outgoing ray direction:

```{math}
\hat{k}_f = \frac{(x, y, z)}{\sqrt{x^2 + y^2 + z^2}}
```

The implementation treats the incident beam direction as positive laboratory z. It forms and normalizes the scattering-vector direction:

```{math}
\hat{q} = \frac{\hat{k}_f - (0, 0, 1)}{\left\lVert\hat{k}_f - (0, 0, 1)\right\rVert}
```

The three components are stored in the peak's `qhat` field. `qhat` is dimensionless and has unit length, apart from floating-point error.

## Direct conversion

Use `Geometry.pixels_to_q()` to convert known peak coordinates without running peak search:

```python
import numpy as np

from lauelab.indexing import load_geometry

geometry = load_geometry("geometry.xml")
peaks_xy = np.array([[720.25, 512.75], [1311.50, 984.00]])
qhat = geometry.pixels_to_q(peaks_xy)
```

The method validates shape, finite coordinates, `start`, `group`, `depth`, detector selection, and detector bounds.

## Coordinate boundary

This page describes the transformation implemented by the library. It does not assign physical labels such as vertical, outboard, upstream, or downstream to laboratory axes. Do not exchange vectors with another coordinate convention without an explicit basis transformation.
