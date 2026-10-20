# Concepts

These pages explain the models behind the processing workflows. Indexing connects detector pixels to candidate crystal orientations. Wire-scan reconstruction connects a stack of frames to sample depths. Monochromatic image reduction assigns scattering-vector magnitudes to pixels and averages intensity over Q bins. Start with the overview for the workflow you are using, then read the algorithm pages when you need the equations, assumptions, or output definitions behind a stage.

```{toctree}
:maxdepth: 2

indexing-pipeline
wire-scan-reconstruction
algorithms/index
```

Reflection simulation and the preparation of maps, pole figures, detector views, and tables are task-oriented APIs. They are covered in [Simulation](../guides/simulation.md) and [Visualization data](../guides/visualization.md).
