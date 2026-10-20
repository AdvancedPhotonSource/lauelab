# lauelab

`lauelab` provides Python interfaces for processing Laue diffraction data at APS beamline 34-ID-E. It indexes crystal orientations, reconstructs wire scans into depth-resolved images, reduces monochromatic images to intensity-versus-Q profiles, predicts detector reflections, and prepares maps, pole figures, detector overlays, and tables. Visualization functions accept in-memory indexing results, results files, or LaueGo XML.

Built on [LaueGo](https://github.com/34IDE/LaueGo), developed by Jon Z. Tischler and collaborators. See [acknowledgments and citations](acknowledgments.md).

```{note}
`lauelab` is alpha software. The public APIs documented here are supported interfaces, and their behavior can change before a stable release. The source build targets Linux and requires native C build tools and scientific libraries.
```

```{toctree}
:hidden:
:maxdepth: 2

installation
quickstart
concepts/index
guides/index
reference/index
lauego/index
development/index
acknowledgments
```

## Start here

Install the package from source, then follow one frame through indexing, visualization, and optional reflection simulation.

- [Install lauelab](installation.md)
- [Index your first frame](quickstart.md)
- [Reconstruct a wire scan](guides/reconstruction.md)
- [Plot monochromatic intensity versus Q](guides/energy.md)
- [Prepare maps, figures, and tables](guides/visualization.md)
- [Simulate reflections for an indexed orientation](guides/simulation.md)

The in-process API accepts a two-dimensional detector frame, as a NumPy array in a supported dtype or a supported 34-ID-E HDF5 file. It returns a {class}`~lauelab.indexing.FrameResult` containing detected peaks, scattering vectors, candidate crystal patterns, timing, and provenance.

## Understand indexing

The [indexing pipeline](concepts/indexing-pipeline.md) explains the three processing stages and the data passed between them. The [algorithm pages](concepts/algorithms/index.md) explain the calculations and parameter effects in more detail.

## Understand reconstruction

A wire scan records frames while a wire crosses the diffracted beams. [Wire-scan reconstruction](concepts/wire-scan-reconstruction.md) explains how the frame differences are mapped to sample depths, and [Depth reconstruction](concepts/algorithms/depth-reconstruction.md) gives the transformations and binning. {class}`~lauelab.reconstruct.Reconstructor` runs the calculation in-process and returns a {class}`~lauelab.reconstruct.ReconstructionResult` with depth coordinates, intensity totals, and optional images.

## Configure a workflow

Use the task-oriented guides when you need to select a detector, define a crystal, prepare frame data, tune parameters, process a sequence, or reconstruct wire-scan points.

- [User guides](guides/index.md)
- [API reference](reference/index.md)

The API reference contains the exact supported signatures and result fields. The guides explain when to use them and which conventions affect the result.

## Subprocess indexing

Use `lauego` to run the `peaksearch`, `pix2qs` and `euler` programs as subprocesses. For in-process indexing with NumPy results, use {func}`~lauelab.indexing.index_frame` or {class}`~lauelab.indexing.Indexer`. See [LaueGo interfaces](lauego/index.md) to compare arguments, outputs and error handling.
