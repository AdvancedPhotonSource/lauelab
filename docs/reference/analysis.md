# Analysis

`lauelab.analysis` contains orientation, projection, coloring, and reflection-simulation functions. The functions accept NumPy-compatible arrays and return NumPy arrays or package data classes.

## Reciprocal and orientation conventions

See the [results guide](../guides/results.md) for the reciprocal-basis convention. {func}`~lauelab.analysis.lattice_params_to_reciprocal` takes cell lengths in nm and angles in degrees.

An orientation matrix maps vectors from the reference crystal basis to the measured basis. {func}`~lauelab.analysis.reciprocal_to_orientation` calculates that matrix as `measured.T @ inv(reference.T)`. {func}`~lauelab.analysis.crystal_direction` applies the inverse orientation to a laboratory-frame direction.

Rotation matrices have shape `(3, 3)` and are dimensionless. {func}`~lauelab.analysis.orientation_to_rodrigues`, {func}`~lauelab.analysis.symmetry_reduce_orientation`, {func}`~lauelab.analysis.misorientation_matrix`, and {func}`~lauelab.analysis.crystal_direction` also accept a stack with shape `(..., 3, 3)` and return results with the matching leading shape. Rodrigues vectors use the dimensionless $\hat{a}\tan(\theta/2)$ convention. At the 180-degree singularity, {func}`~lauelab.analysis.orientation_to_rodrigues` clamps the effective angle to $\pi - 10^{-7}$ radians while retaining a deterministic rotation axis. Misorientation angles use degrees.

## Lattice and orientation

```{eval-rst}
.. currentmodule:: lauelab.analysis

.. py:data:: CUBIC_SYMMETRY
   :type: numpy.ndarray

   Cubic proper rotations with shape ``(24, 3, 3)``.

.. py:data:: HEXAGONAL_SYMMETRY
   :type: numpy.ndarray

   Hexagonal proper rotations with shape ``(12, 3, 3)``.

.. autofunction:: lattice_params_to_reciprocal

.. autofunction:: reciprocal_to_orientation

.. autofunction:: orientation_to_rodrigues

.. autofunction:: crystal_direction

.. autofunction:: symmetry_operations

.. autofunction:: symmetry_reduce_orientation

.. autofunction:: misorientation_matrix

.. autofunction:: misorientation_angle

.. autofunction:: misorientation_from_reference

.. autofunction:: pairwise_misorientation
```

The symmetry constants and arrays returned by {func}`~lauelab.analysis.symmetry_operations` contain proper rotation matrices. Cubic space groups are 195 through 230. Hexagonal space groups are 168 through 194.

## Pole projection

```{eval-rst}
.. currentmodule:: lauelab.analysis

.. autoclass:: SurfaceFrame
   :members: from_vectors, aps_34ide
   :no-index:

.. autofunction:: cubic_hkl_family

.. autofunction:: pole_figure_points

.. autofunction:: pole_color_radius
```

Pole-figure points have shape `(n, 2)` and use dimensionless stereographic coordinates. Angular inputs use degrees. `SurfaceFrame` vectors are dimensionless laboratory-frame directions.

## Coloring

```{eval-rst}
.. currentmodule:: lauelab.analysis

.. autofunction:: cubic_ipf_colors

.. autofunction:: rodrigues_colors

.. autofunction:: hsv_position_colors

.. autofunction:: closest_pole_colors

.. autofunction:: cubic_ipf_key

.. autofunction:: hsv_key
```

Color functions return RGB values in `[0, 1]` unless the function returns a reference image. Reference images have shape `(height, width, 4)`, dtype `numpy.uint8`, and RGBA channel order. Angular color limits use degrees.

## Reflection simulation

```{eval-rst}
.. currentmodule:: lauelab.analysis

.. autoclass:: SimulationResult
   :members: missing_from
   :no-index:

.. autofunction:: simulate_reflections
   :no-index:
```

Simulation uses reciprocal rows in `1/nm`, photon energy in keV, sample depth in micrometres, and zero-based full-detector coordinates `(x, y)`. See the [reflection simulation reference](simulation.md) and [Simulate detector reflections](../guides/simulation.md) for a complete workflow.
