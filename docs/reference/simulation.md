# Reflection simulation

Use `simulate_reflections` to predict reflections from a crystal orientation and detector geometry. The returned `SimulationResult` contains NumPy arrays of reflection indices, positions, energies, and relative intensities.

See [Simulate detector reflections](../guides/simulation.md) for input selection, numerical behavior, and examples.

```{eval-rst}
.. currentmodule:: lauelab.analysis

.. autoclass:: SimulationResult
   :members: missing_from

.. autofunction:: simulate_reflections
```

Simulation failures raise exceptions. See the guide for input requirements and calculation limits.
