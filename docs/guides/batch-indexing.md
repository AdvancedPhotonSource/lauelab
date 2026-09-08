# Batch indexing

Use one {class}`~lauelab.indexing.Indexer` for frames that share geometry, crystal, detector selection, and processing parameters. This retains parsed configuration and avoids rebuilding native state for every frame.

## Reuse an indexer

```python
from pathlib import Path

from lauelab.indexing import Indexer

indexer = Indexer("geometry.xml", "crystal.xml")
frames = sorted(Path("frames").glob("*.h5"))
```

Create a separate indexer when a frame needs a different geometry, crystal, detector, or parameter set. `Indexer.replace()` is a convenient way to construct and validate a related configuration, but the new object has independent native state.

## Index frames in order

`index_many()` accepts any iterable of NumPy frames or supported HDF5 paths:

```python
results = indexer.index_many(frames)

for path, result in zip(frames, results):
    print(path.name, result.n_peaks, result.n_patterns)
```

The method processes frames sequentially and returns results in input order.

## Control memory use

Batch processing uses `keep_images=False` by default. Each `FrameResult.image` is therefore `None` unless you opt in:

```python
results = indexer.index_many(frames, keep_images=True)
```

Retaining images adds the storage of every contiguous `uint16` frame to the result list. Keep them only when later analysis requires direct pixel access.

Peak and pattern arrays remain available regardless of image retention.

## Handle per-frame failures

`index_many()` stops at the first exception and does not return its partially built result list. Process frames individually when the application must record a failure and continue:

```python
from lauelab.indexing import IndexingError, InputError

results = []
failures = []

for frame in frames:
    try:
        results.append(indexer.index(frame, keep_image=False))
    except (InputError, IndexingError, MemoryError, OSError, KeyError) as error:
        failures.append((frame, error))
```

Choose the caught exceptions deliberately. For example, an application may stop on `MemoryError` rather than continue with other frames.

## Parallelism boundaries

`index_many()` is sequential. The public API does not promise thread safety or built-in process parallelism. If an application adds process-level parallelism, each worker should construct and own its `Indexer` until stronger sharing guarantees are documented. `Indexer` is not picklable; `FrameResult` is, so workers can return results to the parent process. {ref}`Write from a process pool <results-file-process-pool>` shows that pattern with a streaming writer.

Measure process count and memory use with representative detector frames before using parallel execution in production.

## Write combined output

Write successful results to one results file:

```python
indexer.write_results(results, "indexed-scan.h5")
```

Results are written in iteration order, and an existing destination raises `FileExistsError` unless you pass `overwrite=True`. For a scan too large to hold in memory, `indexer.results_writer()` appends each result as it is produced; see [Results files](results-file.md).

Write a LaueGo XML document when other software requires that format:

```python
indexer.write_many_xml(results, "indexed-scan.xml")
```

The XML destination is replaced if it exists. A result constructed manually without an XML snapshot raises `RuntimeError`.

## Measure performance

Use a warm indexer, representative frames, and explicit image-retention settings:

```python
from time import perf_counter

started = perf_counter()
results = indexer.index_many(frames, keep_images=False)
elapsed = perf_counter() - started

print(f"frames: {len(results)}")
print(f"total seconds: {elapsed:.3f}")
print(f"seconds per frame: {elapsed / len(results):.3f}")
```

Record hardware, package version, frame dimensions, peak and indexing parameters, detector selection, and input storage. `FrameResult.elapsed_seconds` excludes setup and pixel-to-q time, so use wall-clock timing for end-to-end comparisons.
