# User guides

These guides cover the decisions and data conventions needed for indexing, reconstruction and monochromatic image reduction.

```{toctree}
:maxdepth: 1

geometry
crystals
frame-input
parameters
stages
strain
results
results-file
batch-indexing
reconstruction
depth-inspection
energy
visualization
simulation
detector-simulation
error-handling
```

Read [Geometry](geometry.md), [Crystals](crystals.md), and [Frame input](frame-input.md) to prepare experiment inputs. Use [Parameters](parameters.md) to configure processing. To compare settings or investigate results at each stage, see [Run the indexing stages separately](stages.md). [Results](results.md) defines the returned arrays and their relationships, and [Results files](results-file.md) describes the HDF5 output for a whole run.

Use [Reconstruct a wire scan](reconstruction.md) for depth-resolved reconstruction and [Inspect a reconstructed point](depth-inspection.md) for reference images, square ROIs and intensity through depth. [Plot intensity versus Q](energy.md) describes monochromatic image reduction. [Simulation](simulation.md), [detector-view simulation](detector-simulation.md) and [Visualization data](visualization.md) cover reflection prediction and plotting.
