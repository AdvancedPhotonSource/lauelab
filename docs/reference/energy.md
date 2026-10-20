# Monochromatic energy reduction

Use `lauelab.energy` to calculate mean intensity versus Q from one monochromatic image. See [Plot intensity versus Q](../guides/energy.md) for an example, [the algorithm](../concepts/algorithms/energy.md) for scientific conventions, and [the file format](../development/energy-format.md) for storage.

## Compute and inspect

```{eval-rst}
.. currentmodule:: lauelab.energy

.. autofunction:: q_distribution

.. autoclass:: QDistribution
   :members: provenance, intensity_label, intensity_units
```

The five read-only arrays have the same shape `(n,)`: `q_nm_inv`, `intensity_mean`, `intensity_sum`, `pixel_count`, and `nonfinite_pixel_count`. Floating-point arrays use `numpy.float64`; count arrays use `numpy.int64`. Mean intensity is NaN in empty bins. Bins containing nonfinite intensities have NaN sums and means. Use `pixel_count` and `nonfinite_pixel_count` to distinguish these cases.

## Save, load and export

```{eval-rst}
.. currentmodule:: lauelab.energy

.. autofunction:: save_q_distribution

.. autofunction:: load_q_distribution

.. autofunction:: validate_q_distribution_file

.. autofunction:: export_q_distribution_csv
```

The validator checks arrays in bounded slices. Computation and validation raise `InputError` for invalid requests or malformed profile content. Opening a saved file can raise `OSError`. Saving a manually constructed result requires complete provenance in the schema produced by `q_distribution`.

## Plot

Use {func}`~lauelab.visualization.plot_q_distribution` to plot a computed or loaded result. The function does not read source files, recalculate the profile or write a file. Customize the returned Plotly figure with `layout_update` or per-role `trace_update` options.
