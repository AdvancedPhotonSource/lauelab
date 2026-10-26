# Strain

`lauelab.strain` calculates reference-volume-normalized deviatoric Biot lattice strain from assigned measured directions. The [guide](../guides/strain.md) gives a complete synthetic workflow; the [algorithm](../concepts/algorithms/strain.md) defines conventions and numerical support checks.

This public surface adds a configured refiner and functional stages because users need both repeated notebook/script configuration and direct numerical access. The object composes the same stage implementation. No implicit current pattern or accumulated fit state is retained.

```{eval-rst}
.. currentmodule:: lauelab.strain

.. autoclass:: StrainRefiner
   :members: refine, refine_many, replace

.. autofunction:: prepare_reflections

.. autofunction:: refine_deviatoric

.. autofunction:: strain_from_reciprocal

.. autofunction:: transform_strain

.. autoclass:: StrainReflections
   :members: provenance

.. autoclass:: StrainResult
   :members: success, provenance, frame_id, pattern_index, rms_error_deg, tensor_in, von_mises_strain, sum_abs_epsilon_crystal, predicted_energy_kev, reflections

.. autofunction:: save_strain_results

.. autofunction:: load_strain_results

.. autofunction:: export_strain_csv
```

The numerical input arrays have shapes `(n, 3)` for unit measured directions and integer HKLs; reciprocal matrices have shape `(3, 3)` with rows in `1/nm`, including `2*pi`. Prepared centroids and predicted pixels have shape `(n, 2)` in zero-based `(x, y)` frame coordinates. Depth is in µm. Cell lengths are stored in nm and angles in degrees. Rotations are proper, dimensionless matrices. Tensor arrays have shape `(3, 3)` and use tensor shear.

Full fit results and their measurements own read-only arrays. Accessing `provenance` returns an independent dictionary. Pure conversion and transformation functions return NumPy arrays without inventing measurement-fit diagnostics.

Expected scientific failures return an unsuccessful result with a NaN tensor. Invalid single inputs raise `InputError`, `ValueError` or `TypeError`; missing/unreadable files raise their I/O errors. Serial batches record expected item-input errors and propagate shared-source or unexpected internal errors. Saving refuses existing destinations by default.
