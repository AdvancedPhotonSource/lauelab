# Error handling

The in-process API separates invalid input from native processing failures. Catch the narrowest exception that your application can handle correctly.

## Exception hierarchy

{class}`~lauelab.indexing.LaueError` is the package base class.

- {class}`~lauelab.indexing.InputError` inherits from both `LaueError` and `ValueError`.
- {class}`~lauelab.indexing.IndexingError` inherits from both `LaueError` and `RuntimeError`.
- {class}`~lauelab.indexing.ReconstructionError` inherits from both `LaueError` and `RuntimeError`.
- {class}`~lauelab.indexing.WorkerError` inherits from both `LaueError` and `RuntimeError`.
- {class}`~lauelab.indexing.InvalidResultsFile` inherits from both `LaueError` and `ValueError`.
- Native allocation failures use Python's built-in {class}`MemoryError`.

`ValueError`, XML parse errors, `OSError`, and `KeyError` can also occur when calling geometry, crystal, or HDF5 loaders directly. `Indexer.index` normalizes HDF5 reading failures to `InputError`, with the original exception as its cause. See the relevant API reference for each loader.

## Invalid input

`InputError` reports invalid processing configuration, including:

- Peak or indexing parameters outside supported ranges, or a fractional value such as `min_size=3.5` where a whole number is required
- An unknown detector identifier or inactive detector slot
- A frame that is not a two-dimensional array of a supported dtype, or a floating-point frame with non-finite values
- Invalid `start`, `group`, or `depth`
- A frame region outside detector bounds
- A mask shape that does not match the frame
- An HDF5 detector identifier that does not match the selected geometry

Fix the input before retrying. The exception message names the failed check and includes the received value when useful.

```python
import numpy as np

from lauelab.indexing import InputError

bad_frame = np.zeros((128, 128), dtype=np.float32)

try:
    indexer.index(bad_frame)
except InputError as error:
    print(error)
```

## Memory failure

A native stage raises `MemoryError` when it cannot allocate required storage. The message identifies the stage and includes its diagnostic.

Do not assume that immediate retry will succeed. Release unneeded arrays and results, reduce concurrent work, or move the workload to a process with sufficient memory before retrying.

## Native indexing failure

`NumericalIndexingError`, a subclass of `IndexingError`, reports native numerical failures for one frame. Other `IndexingError` exceptions report internal failures in a native processing stage. Its message begins with the stage name, such as `pixel-to-q conversion failed` or `orientation indexing failed`.

Preserve the complete message. It can distinguish a geometry conversion problem from an orientation-indexing problem without exposing native status values as a public API.

A successful call can return zero peaks or zero patterns. Check these counts against the scientific acceptance criteria for your experiment.

## Parallel indexing

An exception in {data}`~lauelab.indexing.EXPECTED_INPUT_ERRORS` (`InputError`, `NumericalIndexingError`, or `MemoryError`) is returned in `FrameOutcome.error` and the next input is processed. `Indexer.index` wraps unreadable or malformed HDF5 input as `InputError`, preserving the underlying exception as its cause. Other `IndexingError` failures and bare `ValueError`, `KeyError`, or `OSError` raised elsewhere are fatal worker errors. Decide per outcome whether to continue after `MemoryError`; an application may treat it as a reason to stop.

`WorkerError` means the run itself failed: a worker could not build its indexer, a worker raised an exception outside the expected set, or the pool broke because a process died. The message names the input where possible and includes the worker traceback. The iteration cannot continue after it; the workers are shut down before it propagates.

## Output files

After a write failure, {class}`~lauelab.indexing.ResultsWriter` rejects further appends with `RuntimeError`. Recreate the incomplete results file before using it. {func}`~lauelab.indexing.validate_results_file` raises `InvalidResultsFile` for such a file and for any other structural defect; a file that cannot be opened raises `OSError`. An {class}`~lauelab.indexing.XmlResultsWriter` failure concerns the auxiliary XML document only. See [Results files](results-file.md).

## Reconstruction failure

{class}`~lauelab.reconstruct.Reconstructor` raises `InputError` and `MemoryError` for failures before the first stripe is processed. A failure after that point does not raise. It returns a {class}`~lauelab.reconstruct.ReconstructionResult` with `success=False`, the message in `error`, and the progress made so far in `last_completed_stripe`. `ReconstructionError` names a native failure in that message; an I/O failure carries the underlying `OSError` text.

Check `success` on every result. {func}`~lauelab.reconstruct.reconstruct_points` applies the same rule to each point and continues the batch after a failed point. See [Reconstruct a wire scan](reconstruction.md).

## Batch strategy

Use `index_many()` when the batch should stop on its first failure. Use an explicit loop when each frame needs an independent status:

```python
from lauelab.indexing import IndexingError, InputError

results = {}
failures = {}

for frame_id, frame in frames.items():
    try:
        results[frame_id] = indexer.index(frame, keep_image=False)
    except (InputError, IndexingError) as error:
        failures[frame_id] = str(error)
```

Decide separately whether to catch `MemoryError`, `OSError`, and `KeyError`. Continuing after those failures may hide a system-wide resource problem or a repeated file-layout error.

## Diagnostic context

Record enough context to reproduce the call:

- Package version or Git commit
- Input identifier, shape, and dtype
- Geometry and crystal identifiers
- Detector slot and detector ID
- `start`, `group`, and `depth`
- Peak and indexing parameters
- Mask identity or generation method
- Exception type and complete message

Do not log full frame arrays. Remove user names, sample names, local paths, and other sensitive acquisition metadata before sharing a report.

## Reflection simulation failures

{func}`~lauelab.analysis.simulate_reflections` uses built-in exception types because its public result does not expose backend status:

- `TypeError` reports unsupported package objects or numeric types.
- `ValueError` reports invalid scientific inputs, including array shapes, non-finite values, atomless crystals, and invalid energy intervals.
- `RuntimeError` reports private simulator loading, resource, execution, numerical, projection, or candidate-limit failures.

A successful simulation with no on-detector reflections returns an empty {class}`~lauelab.analysis.SimulationResult`. A failed simulation raises `RuntimeError` without attempting a fallback calculation.

Detector-view preparation and rendering propagate these exceptions. Missing crystal context raises only when `simulation_energy_range_kev` is not `None`.
