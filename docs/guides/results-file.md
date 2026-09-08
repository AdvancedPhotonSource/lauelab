# Results files

A results file stores every {class}`~lauelab.indexing.FrameResult` from one indexing run in one HDF5 file. It is the output to keep from a scan: visualization functions load it directly, and any HDF5 reader opens it. XML output remains available for software that reads that format; see {ref}`Write XML <results-write-xml>`.

The XML document stores every number as text in a nested element tree, so reading a 100,000-frame wire scan takes tens of seconds and several gigabytes. A results file stores each quantity as one typed array across all frames, and the same scan loads in a fraction of a second.

## Write a results file

`Indexer.write_results()` writes an iterable of results in iteration order, together with the crystal, the geometry, and the peak and indexing parameters from the indexer:

```python
from pathlib import Path

from lauelab.indexing import Indexer

indexer = Indexer("geometry.xml", "crystal.xml")
frames = sorted(Path("frames").glob("*.h5"))
results = indexer.index_many(frames)
indexer.write_results(results, "results.h5")
```

An existing file is replaced only with `overwrite=True`. Pass `frame_ids` to record an identifier for each frame instead of its zero-based position; identifiers must be unique and all strings or all integers:

```python
indexer.write_results(results, "results.h5", frame_ids=[path.stem for path in frames])
```

### Write while indexing

`Indexer.results_writer()` returns a streaming writer for scans too large to hold every `FrameResult` in memory. Each `append()` extends the file in order, and the file is complete when the `with` block exits:

```python
with indexer.results_writer("results.h5") as writer:
    for path in frames:
        result = indexer.index(path, keep_image=False)
        writer.append(result, frame_id=path.stem)
```

(results-file-process-pool)=
### Write from a process pool

`Indexer` is not picklable, so each worker constructs its own. A `FrameResult` is picklable, so workers can return results to one writer:

```python
from multiprocessing import Pool

def index_one(path):
    global indexer
    try:
        indexer
    except NameError:
        indexer = Indexer("geometry.xml", "crystal.xml")
    return indexer.index(path, keep_image=False)

with Pool(8) as pool, indexer.results_writer("results.h5") as writer:
    for path, result in zip(frames, pool.imap(index_one, frames)):
        writer.append(result, frame_id=path.stem)
```

`Pool.imap()` preserves input order. Pass `keep_image=False` in the worker; a retained image adds one detector-sized array to every result sent between processes.

## Load a results file

{func}`~lauelab.visualization.load_results` returns the same {class}`~lauelab.visualization.VisualizationDataset` as the other loaders:

```python
from lauelab.visualization import load_results, plot_map

dataset = load_results("results.h5")
figure = plot_map(dataset, axes=("X", "Y"), color="cubic_ipf")
```

The crystal comes from the file. The geometry is, in order, an explicit `geometry` argument, the geometry text embedded in the file, or the recorded geometry path. An updated external calibration therefore does not change an existing results file; missing geometry prevents only detector views. Rotation matrices are not stored; the loader derives them from each reciprocal lattice and the crystal in the basis described in {ref}`Patterns <results-patterns>`.

{func}`~lauelab.is_results_file` distinguishes a results file from other HDF5 files, such as detector frames.

## Convert an existing XML document

{func}`~lauelab.visualization.convert_xml` reads a LaueGo `AllSteps` document once and writes the equivalent results file, by default beside the XML with the `.h5` suffix:

```python
from lauelab.visualization import convert_xml

output = convert_xml("indexed-scan.xml")
```

An existing output raises `FileExistsError` unless you pass `output_path` or `overwrite=True`. Conversion costs one XML load, so convert once and load the results file afterwards.

The converter records the geometry path from the XML and embeds the geometry text only when that path is readable. A document without crystal information converts without a crystal, and the loaded dataset then rejects pole figures and orientation colors. Run parameters recorded in the XML become `run` attributes; parameters absent from the XML stay absent, and parameters that conflict between steps raise `ValueError`, so convert separate configurations separately.

## File contents

| Group | One row per | Contents |
|---|---|---|
| `crystal` | | Space group, cell, and atoms; absent without a crystal |
| `geometry` | | Geometry path and, when available, the geometry XML text |
| `run` | | Program name, detector selection, and every peak and indexing parameter, as attributes |
| `frames` | frame | Sample position, depth, energy, acquisition metadata, image shape and region, frame statistics, timing, and offsets into `peaks` and `patterns` |
| `peaks` | detected peak | The fields of `FrameResult.peaks` |
| `patterns` | pattern | Rank within its frame, reciprocal lattice, quality values, and offsets into `assignments` |
| `assignments` | pattern-to-peak assignment | Peak index, `hkl`, angular error, energy, and predicted intensity |

Rows of `peaks`, `patterns`, and `assignments` are grouped by owner. The peaks of frame `i` are rows `peak_offsets[i]` to `peak_offsets[i + 1]`; `pattern_offsets` and `assignment_offsets` follow the same rule.

Reciprocal lattices, cell parameters, atom positions, sample positions, depths, and intensity sums are `numpy.float64`. Other floating-point values are `numpy.float32`, which keeps about seven significant digits, or about 0.0001 px for a coordinate on a 2048-pixel detector. Hutch temperature and sample distance keep the values supplied by acquisition with `units="unspecified"`, because the input metadata does not establish a unit. Missing flags are `-1` and missing floating-point values are `NaN`.

The {ref}`reference <results-file-layout>` lists every dataset with its dtype, shape, and units. Provenance that `VisualizationDataset` does not carry, such as the `run` attributes, is read directly with h5py.
