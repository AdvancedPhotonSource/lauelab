# Results files

A results file contains every {class}`~lauelab.indexing.FrameResult` from one indexing run in one HDF5 file. Save this file for analysis: visualization functions load it directly, and its datasets are accessible with standard HDF5 tools. To write XML for tools that read that format, see {ref}`Write XML <results-write-xml>`.

In XML, every number is text in a nested element tree, so reading a 100,000-frame wire scan takes tens of seconds and several gigabytes. In a results file, each quantity is one typed array across all frames, and the same scan loads in a fraction of a second.

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

`append()` checks the result type and frame ID before writing. If either is invalid, such as a duplicate ID, it raises an exception and the writer remains usable.

A failure during writing can leave a partial frame in the file. In that case, `writer.failed` becomes `True`, `writer.error` contains the exception, and further appends raise `RuntimeError`. Stop writing and recreate the results file from the beginning. Validate the completed file as described below.

(results-file-xml-alongside)=
### Write XML alongside

{class}`~lauelab.indexing.XmlResultsWriter` appends each result to one LaueGo `AllSteps` document as it arrives, with the same bytes `Indexer.write_many_xml()` would produce and without holding earlier steps in memory. Feed both writers from one stream of results:

```python
import warnings

from lauelab.indexing import XmlResultsWriter

with indexer.results_writer("results.h5") as writer, \
        XmlResultsWriter("results.xml") as xml:
    for path in frames:
        result = indexer.index(path, keep_image=False)
        writer.append(result, frame_id=path.stem)
        if not xml.failed:
            try:
                xml.append(result)
            except OSError as error:
                warnings.warn(f"XML output stopped: {error}")
```

This example continues writing HDF5 results if XML output fails. Each XML step is flushed after writing. Following a failed append, the XML writer rejects further steps. On close, it attempts to truncate the document to the last complete step and add the closing tag; this recovery depends on the filesystem remaining writable. An HDF5 write failure stops the loop.

(results-file-validate)=
## Validate a results file

{func}`~lauelab.indexing.validate_results_file` checks a closed file's structure with bounded reads and returns a {class}`~lauelab.indexing.ResultsFileSummary`:

```python
from lauelab.indexing import validate_results_file

summary = validate_results_file("results.h5", frame_ids=[path.stem for path in frames])
print(summary.n_frames, summary.n_peaks, summary.n_patterns)
```

Validation checks the format and version, required datasets, dtypes, shapes, and lengths. It also checks that offsets and counts agree, pattern ranks start at zero in each frame, and frame IDs are unique. Assignment peak indices are checked in bounded chunks to ensure they refer to peaks in the corresponding frame. Peak measurements, reciprocal lattices, and other assignment values are excluded from this structural check.

Pass `frame_ids` or `n_frames` to compare the output with the expected inputs. An ID mismatch reports the first differing frame.

The validator raises {class}`~lauelab.indexing.InvalidResultsFile` for a structural defect, with the file path and the failing check. {func}`lauelab.is_results_file` reads only the format marker. An interrupted file can contain this marker, so use the validator before treating the file as complete. A structurally valid file can contain only part of a run. Compare its frame IDs or count with the expected inputs to check completeness.

(results-file-process-pool)=
### Write from a process pool

{meth}`~lauelab.indexing.Indexer.iter_index` runs frames in worker processes and yields outcomes in input order, so one writer can append them as they arrive:

```python
from lauelab.indexing import FrameInput

inputs = [FrameInput(path, input_id=path.stem) for path in frames]
failures = []

with indexer.iter_index(inputs, workers=8) as outcomes, \
        indexer.results_writer("results.h5") as writer:
    for outcome in outcomes:
        if outcome.ok:
            writer.append(outcome.result, frame_id=outcome.input_id)
        else:
            failures.append((outcome.input_id, outcome.error))
```

Results arrive without images by default, so only peaks, patterns, and provenance cross between processes. See [Batch indexing](batch-indexing.md) for the in-flight bound, error handling, and cooperative stopping.

## Merge results from independent workers

Combine results files with
{func}`~lauelab.indexing.merge_results_files`. Assign globally unique frame IDs
when writing the worker files; each writer's default IDs restart at zero.

```python
from lauelab.indexing import merge_results_files, merge_xml_files

summary = merge_results_files(
    ["worker-0.h5", "worker-1.h5"], "results.h5",
    frame_ids=["point-a/depth-0", "point-a/depth-1", "point-b/depth-0"],
)
print(summary.n_frames, summary.n_patterns)

steps = merge_xml_files(
    ["worker-0.xml", "worker-1.xml"], "results.xml", n_steps=summary.n_frames,
)
```

Files are concatenated in the supplied order, retaining each file's frame
order. `frame_ids` checks that exact sequence; it does not sort or redistribute
frames. All HDF5 inputs and the merged output are structurally validated.
Duplicate identities, mixed integer and string identities, incompatible run
settings, and differences in geometry or crystal metadata raise
{class}`~lauelab.indexing.InvalidResultsFile`. Metadata must match exactly,
including paths and units. Empty input files are allowed. For inputs without
the optional scan-source columns, the merger writes the documented absent values.

HDF5 arrays are copied in bounded blocks into an uncompressed file, and ragged
offsets are rebased. Detector images are not read and indexing is not repeated.
Frame IDs and validation bookkeeping use memory proportional to the numbers of
frames and patterns. The result contains its own datasets, so you can remove
worker files after checking the result.

The XML merger checks well-formed XML, the `AllSteps` root, its `step` children,
and the optional expected count. It preserves each document's body as bytes,
including numeric text. This structural check does not establish scientific
compatibility or unique frame identities. Inputs must use XML 1.0 with UTF-8
or ASCII encoding, an `AllSteps` root without attributes, and no DTD. Parsing
and copying use bounded buffers without constructing a tree for the scan.

Keep all inputs closed and unchanged during merging. Both functions publish
through a unique temporary file; any failure removes that temporary and leaves
an existing destination untouched. Pass `overwrite=True` to replace an
existing result after a successful merge. The destination must always differ
from every input. Worker scheduling and communication are managed by your
application; these functions have no MPI dependency.

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

`convert_xml` raises `FileExistsError` for an existing destination. Pass a different `output_path` or `overwrite=True`. Conversion requires one XML load; subsequent analysis can load the results file directly.

Conversion writes a unique `<destination>.partial-*` file in the destination directory, then closes and validates it before renaming it to the requested path. A failure removes the partial file and preserves any existing destination. If two conversions target the same path, the second to finish raises `FileExistsError` unless `overwrite=True`.

The converter raises `FileNotFoundError` for a missing document, `xml.etree.ElementTree.ParseError` for malformed XML, `ValueError` for conflicting run parameters, and {class}`~lauelab.indexing.InvalidResultsFile` if the output fails validation. It accepts documents without geometry or crystal context; `validate_results_file` reports `has_geometry_text` and `has_crystal` for the output.

The same publication step is available for your own outputs: write to {func}`lauelab.partial_path`, validate, then {func}`lauelab.publish_file` renames it to the final name and refuses, by default, to replace a file that appeared in the meantime.

The converter records the geometry path from the XML and embeds the geometry text only when that path is readable. It accepts documents without crystal information; plotting functions require crystal information for pole figures and orientation colors. Run parameters recorded in the XML become `run` attributes. Parameters absent from the XML are omitted. The converter raises `ValueError` for conflicting parameters between steps, so convert separate configurations separately.

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

Peak and pattern rows are grouped by frame; assignment rows are grouped by pattern. The peaks of frame `i` are rows `peak_offsets[i]` to `peak_offsets[i + 1]`; `pattern_offsets` and `assignment_offsets` follow the same rule.

Reciprocal lattices, cell parameters, atom positions, sample positions, depths, and intensity sums are `numpy.float64`. Other floating-point values are `numpy.float32`, with about seven significant digits, or about 0.0001 px for a coordinate on a 2048-pixel detector. Hutch temperature and sample distance are stored as supplied by acquisition with `units="unspecified"`, because the input metadata does not establish a unit. Missing flags are `-1` and missing floating-point values are `NaN`. In the `run` attributes, a parameter given as `None` is also stored as `NaN`: `threshold` for automatic thresholding, `threshold_ratio` for the native default, and `max_peaks` for an unlimited peak search.

The {ref}`reference <results-file-layout>` lists every dataset with its dtype, shape, and units. Use h5py to read provenance outside `VisualizationDataset`, such as the `run` attributes.
