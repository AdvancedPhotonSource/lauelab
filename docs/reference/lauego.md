# LaueGo API

The LaueGo API invokes command-line executables and writes intermediate files.
New integrations should use `index_frame` or `Indexer` from the
{doc}`processing` API.

Each program has a corresponding in-process stage:
`peaksearch` corresponds to {func}`~lauelab.indexing.peak_search`, `pix2qs` to
{meth}`Geometry.pixels_to_q <lauelab.indexing.Geometry.pixels_to_q>`, and
`euler` to {func}`~lauelab.indexing.index_orientations`. The `lauego` function
runs all three programs in one call; the programs are not exposed as separate
Python functions.

## Subprocess pipeline

```{eval-rst}
.. currentmodule:: lauelab.indexing

.. autofunction:: lauego
```

## LaueGo result

```{py:class} lauelab.indexing.index.IndexingResult(success, output_files, n_peaks_found, n_indexed, n_patterns_found, indexing_data, step_data, xml_file, log, error=None, command_history=())

Result returned by the LaueGo subprocess indexing interface.

The result contains the overall status, generated output paths, peak and
pattern counts, optional parsed LaueGo data, the XML path, logs, an optional
error message, and the command history. New code should use `FrameResult`.
```
