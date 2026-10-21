# Reconstruction scan protocol

This page describes how a reconstruction scan is prepared, run, recorded, and read. [Reconstruction scan format](reconstruction-scan-format.md) specifies the files. For examples, see [Reconstruct a wire scan](../guides/reconstruction.md).

{func}`~lauelab.reconstruct.prepare_scan` checks the inputs and publishes the first catalog. {func}`~lauelab.reconstruct.reconstruct_point` writes one point file, and {func}`~lauelab.reconstruct.reconstruct_scan` runs the full workflow using local worker processes. {class}`~lauelab.reconstruct.ScanReader` reads a catalog, {class}`~lauelab.reconstruct.PointReader` reads a point file, and {func}`~lauelab.reconstruct.validate_scan_file` checks a catalog.

## Preparation and tasks

{func}`~lauelab.reconstruct.prepare_scan` takes the input files in manifest order, the scan directory, the geometry, and the reconstruction settings. Before writing, it validates the settings, detector slot, point IDs, output filenames, and destination. It then reads each input's metadata without loading frames and creates one {class}`~lauelab.reconstruct.PointTask` for each point whose input can be read. Input paths are made absolute against the current directory; symbolic links are not resolved.

The first catalog is published before any point is computed, and it contains every input. An input that cannot be read, or that has invalid metadata, is recorded as a `failed` point with its error and gets no task. The other points continue.

For inputs that arrive during acquisition, `prepare_scan([], directory, **settings)` publishes an empty running catalog. `PreparedScan.add(path, point_id=...)` inspects one ready input and appends its task. The caller establishes that the input is complete and immutable. Admission reads metadata, uses the frozen settings and geometry, and does not publish or dispatch. Persist the updated control state before publishing or handing the task to a worker. A failed admission raises without changing the registry or consuming an index. Duplicate identities, sources, conflicting output names, and existing point outputs or private files are refused.

An appended task keeps its registration index even when inputs arrive out of acquisition order. `PreparedScan.outcomes` returns a tuple of current point statuses in this order without opening files. Earlier task and outcome tuples remain snapshots. An empty run can be cancelled or closed as failed, but cannot finish successfully. The convenience runner `reconstruct_scan` still requires at least one input.

Each task contains the absolute input and output paths, the point identity, the detector slot, the validated reconstruction settings, and the complete geometry XML. It also contains the values found during inspection: the frame shape, the depth count and bounds, the pixel type, the selected raw-slice range, the scan number, the sample position, and the incident energy. A missing acquisition value is `None` in the task and the fill value of its catalog column. Tasks contain only strings, numbers, and plain containers, so they can be pickled or converted to JSON.

The worker process opens its own input and creates its own native state. Before it creates an output file, it compares the output description, raw-slice range, and acquisition values with the prepared task. If they differ, the point fails. The comparison reads only metadata; raw pixel values are not compared. Treat tasks as read-only. The coordinator keeps its own copy of the prepared request and rejects changed tasks, and completed files that do not match the request.

Exactly one coordinator, the {class}`~lauelab.reconstruct.PreparedScan` returned by `prepare_scan`, writes `scan.h5`. Workers write only the point files of their tasks. A scheduler other than {func}`~lauelab.reconstruct.reconstruct_scan` uses the coordinator in this order:

1. Call {meth}`~lauelab.reconstruct.PreparedScan.record_dispatch` when handing a task to a worker. This sets its status to `writing`.
2. The worker calls {func}`~lauelab.reconstruct.reconstruct_point` with the task. It writes the frames to the point file and returns only a small {class}`~lauelab.reconstruct.PointOutcome`.
3. Call {meth}`~lauelab.reconstruct.PreparedScan.record` as outcomes arrive, in any order.
4. Call {meth}`~lauelab.reconstruct.PreparedScan.finish` after every dispatched task has an outcome. Pass `cancelled=True` if some tasks were never dispatched; they become `unattempted`.

The scheduler assigns each task once and records a failed outcome for a task whose worker was lost; the coordinator provides no retry. Leaving the coordinator's context without `finish` records the run as `failed`, dispatched points without an outcome as `interrupted`, and the other points as `unattempted`. A standalone point, reconstructed without a scan, is prepared with the same rules and written by the same writer. It has no manifest index.

## Status and completeness

`/points/statuses` and the root `status` attribute of `scan.h5` are HDF5 enumerated types with these codes and names.

| Code | Point status | Meaning |
| --- | --- | --- |
| 0 | `pending` | Not started. This is also the value of an entry that was never written. |
| 1 | `writing` | Dispatched for reconstruction; awaiting a recorded outcome. |
| 2 | `complete` | The point file is published. |
| 3 | `failed` | An expected input or reconstruction failure. The error text is in `/points/errors`. |
| 4 | `interrupted` | Dispatched, but the run ended before an outcome was recorded. |
| 5 | `unattempted` | The run ended before the point was dispatched. |

| Code | Run status | Meaning |
| --- | --- | --- |
| 0 | `running` | The coordinator is running the scan. |
| 1 | `finished` | Every point was attempted. Individual points can have failed. |
| 2 | `cancelled` | The run stopped on request after its running points finished. |
| 3 | `failed` | A shared failure stopped the run, or the coordinator was closed before the run was finished. |

The coordinator keeps the run state in memory and updates it at each dispatch and outcome. It publishes pending changes at most once every five seconds. After preparation, successful completion, cancellation, and failure, it publishes immediately. It does not rewrite an unchanged catalog.

{func}`~lauelab.reconstruct.reconstruct_scan` checks for a due snapshot while it waits for workers. An external scheduler should call {meth}`~lauelab.reconstruct.PreparedScan.snapshot` in its polling loop, in addition to recording dispatches and outcomes. The method returns quickly when no snapshot is due. `snapshot(force=True)` publishes pending changes immediately. No background thread writes the catalog, so a coordinator that is blocked outside these calls does not publish on schedule.

The coordinator writes each snapshot to a private `.partial` file in the scan directory, closes and validates it, and moves it over `scan.h5`. The first snapshot is published before any point is computed and has the run status `running`, so a published catalog can have any run status. A `finished` or `cancelled` catalog contains only the point statuses `complete`, `failed`, `interrupted`, and `unattempted`. A catalog reader keeps the snapshot that it loaded; create a new reader to see later changes.

A worker writes each point file to a private `.partial` file beside its final path, closes it, validates it against its task, and publishes it without replacing another file. A published point file is complete and is not changed again. {func}`~lauelab.reconstruct.reconstruct_point` handles input and reconstruction failures differently from output failures:

| Failure | Examples | Result |
| --- | --- | --- |
| Input or reconstruction of the point | Missing or unreadable input, invalid metadata, an input changed since preparation, a stripe-buffer budget too small for one row, a native reconstruction error | A `failed` outcome with the error. Nothing is published, and other points continue. |
| Output of the point | The private file cannot be created or written, validation or publication fails, the final file or another worker's private file already exists | {class}`~lauelab.indexing.ReconstructionError` is raised. The worker removes its own private file and never replaces another file. |

When you call {meth}`~lauelab.reconstruct.PreparedScan.record` with a complete outcome, the coordinator validates the published file against the prepared task. It checks the point ID, manifest index, shape, depth bounds, pixel type, the settings recorded as given, the embedded geometry, the source path, selected raw-slice range, and acquisition metadata without reading pixels. Outcomes for unknown tasks and duplicate outcomes are rejected.

Point files and the catalog are published separately. If a run stops after a point file is published but before the next catalog snapshot, the file can be read directly, although the catalog does not list it as complete. Readers use the catalog status; they do not search the directory for other completed files. Without caller-managed checkpoints and receipts, a process crash loses changes since the last snapshot. Durability after a power loss is not guaranteed.

Structural validation of a catalog checks the root attributes, the required datasets, their dtypes, enumerated types, shapes, units, and fill values, the catalog length, the status codes, the point IDs and paths, the consistency of each catalog row, and that `/point_files` contains a link to each complete point and to no other. It reads only `scan.h5` and does not change it. A scan is complete when it has at least one point and every point has the status `complete`, so the catalog of a cancelled or partly failed run can pass validation and contain incomplete points.

(checkpoints-and-receipt-replay)=
## Checkpoints and receipt replay

{meth}`~lauelab.reconstruct.PreparedScan.export_state` returns detached control state that round-trips through strict JSON (`allow_nan=False`). It contains the frozen recipe and geometry XML, registered task descriptions, identities, outcomes, absolute paths, and package/native version tags. It contains no image arrays, open files, or native handles. Treat its versioned structure as opaque. The method does not write a sidecar or publish a catalog.

The application owns checkpoint storage, durable worker receipts, queue reconciliation, and exclusive access to each run. Save registrations before dispatch or publication, and persist each worker's `PointOutcome` before calling `record`. Reconstruct JSON receipts with `PointOutcome(**values)` when loading them.

{meth}`~lauelab.reconstruct.PreparedScan.restore` accepts an unfinished checkpoint and optional durable outcomes. It validates completed point files against their saved tasks without reopening raw inputs or the original geometry file. It permits an older catalog snapshot and exact repeated receipts; it rejects conflicting or unknown receipts, changed identities, unavailable completed files, and terminal catalog progress absent from the receipts. Restore requires the same mount paths and package/native version tags. It does not migrate state or establish that running workers have stopped.

Restore does not write files, dispatch tasks, or remove private files. An unrecorded final output requires explicit receipt reconciliation; it is never silently adopted or overwritten. A surviving `.partial` file is preserved, and a later reconstruction attempt refuses to replace it. Once restoration succeeds, reconcile pending and dispatched tasks with the scheduler before submitting more work, and call `snapshot` to publish the recovered view. This is a caller-managed recovery mechanism, not an automatic retry service.

## Readers

{class}`~lauelab.reconstruct.PointReader`, {class}`~lauelab.reconstruct.PerDepthReader`, and {class}`~lauelab.reconstruct.inspection.ArrayPoint` have the same frame and region methods for inspection and plotting. They read a point file, per-depth files, and an in-memory array.

{class}`~lauelab.reconstruct.ScanReader` loads a catalog snapshot when it is created, then closes the catalog file. It lists the point IDs, statuses, and metadata in manifest order without opening point files, and resolves each point path against the scan directory. {meth}`~lauelab.reconstruct.ScanReader.point` opens only the file of that point, and compares its point ID, manifest index, shape, dtype, and physical depth bounds with the catalog.

`PointReader` owns the handle of one point file. It reads the scan number, sample position, incident energy, and detector region from the metadata copied from the input. For a missing field, it uses the default that reconstruction uses: the frame size for `Nx` and `Ny`, `(0, 0)` for the origin, and 1 for each binning factor. Open it as a context manager in the process that uses it; it cannot be sent to another process. A point opened through a `ScanReader` is closed when the caller closes it or closes that `ScanReader`. `PerDepthReader` orders the files by their physical depths and opens at most one file at a time. It calculates the totals and the reconstructed reference from the stored pixels when they are requested; raw references are not available.

`frame()` reads one frame, `region()` reads a rectangular selection over a depth range, and `iter_blocks()` reads successive depth blocks within a byte budget that the caller gives. A request for a reference image that is not available raises `InputError`. Returned arrays belong to the caller.

`region()` allocates the requested selection, so choose bounded depth slices for a large region. The inspection functions in [Inspect a reconstructed point](../guides/depth-inspection.md) read bounded depth blocks.
