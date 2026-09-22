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

Retaining images adds the storage of every contiguous frame to the result list. Keep them only when later analysis requires direct pixel access.

Peak and pattern arrays remain available regardless of image retention.

## Handle per-frame failures

`index_many()` stops at the first exception and does not return its partially built result list. Process frames individually when the application must record a failure and continue:

```python
from lauelab.indexing import NumericalIndexingError, InputError

results = []
failures = []

for frame in frames:
    try:
        results.append(indexer.index(frame, keep_image=False))
    except (InputError, NumericalIndexingError, MemoryError) as error:
        failures.append((frame, error))
```

Choose the caught exceptions deliberately. For example, an application may stop on `MemoryError` rather than continue with other frames.

## Index in parallel with bounded buffering

Use a separate process for each parallel indexing worker. Concurrent calls on one `Indexer` from multiple threads are unsupported. For parallel work, use {meth}`~lauelab.indexing.Indexer.iter_index`, which runs frames in spawn-based worker processes and yields one {class}`~lauelab.indexing.FrameOutcome` per input, in input order:

```python
from lauelab.indexing import FrameInput

inputs = [FrameInput(path, input_id=path.stem) for path in frames]

with indexer.iter_index(inputs, workers=4) as outcomes:
    for outcome in outcomes:
        if outcome.ok:
            print(outcome.input_id, outcome.result.n_peaks, outcome.result.n_patterns)
        else:
            print(outcome.input_id, "failed:", outcome.error)
```

Each worker builds a local `Indexer` from the parent's geometry path, crystal, parameters, and detector selection. A {class}`~lauelab.indexing.FrameInput` carries the frame, an optional stable `input_id`, and the per-frame `start`, `group`, `depth`, and `metadata` values of `index()`. A shared `mask` is passed once to `iter_index()` and sent to each worker once. Plain paths or arrays are accepted in place of `FrameInput` when no identity or per-frame values are needed.

`max_in_flight` limits the number of submitted inputs whose outcomes have not yet been yielded; the default is `2 * workers`. If an early frame is slow, later results wait in this bounded buffer and further submissions pause until space is available. Results come back without images unless `keep_images=True`.

An expected input problem, one of {data}`~lauelab.indexing.EXPECTED_INPUT_ERRORS`, becomes an outcome with `error` set and the iteration continues. A processed frame with no peaks or no patterns is a successful outcome. A worker that cannot initialize, an unexpected exception inside a worker, or a killed worker raises {class}`~lauelab.indexing.WorkerError` from the iteration instead; that run cannot continue.

Pass `should_stop` to stop cooperatively. It is polled before each submission and at least every `poll_seconds` while waiting. Once it returns `True`, no further inputs are admitted, unstarted inputs are withdrawn, frames already running finish and are yielded, and the iteration ends with `outcomes.stopped` set. The executor keeps up to `workers + 1` submitted inputs ready to start, so that many inputs can still run and be yielded after a stop even though they had not started when it was requested. The outcomes object reports progress through `n_submitted`, `n_yielded`, and `n_cancelled`. Inputs cancelled before starting produce no outcome.

Always iterate inside the `with` block. Leaving it, including through a `break` or an exception in your loop body, shuts the workers down. Because the workers use the `spawn` start method, a script that calls `iter_index()` at module level needs the usual `if __name__ == "__main__":` guard.

In the consuming loop, write successful results, record failures, and handle a requested stop. See {ref}`Write from a process pool <results-file-process-pool>` for an example using a streaming writer.

Measure process count and memory use with representative detector frames before choosing `workers` for production. `tests/perf_testing/run_iter_index_perf.py` reports wall time and peak resident memory of the parent and workers on the synthetic frames.

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

The XML destination is replaced if it exists. A result constructed manually without an XML snapshot raises `RuntimeError`. Results are serialized one at a time, so the call does not hold the whole document in memory; to write XML while indexing, use {class}`~lauelab.indexing.XmlResultsWriter` as shown in {ref}`Write XML alongside <results-file-xml-alongside>`.

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
