# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Bounded incremental indexing across spawn-based worker processes."""

from __future__ import annotations

from collections import deque
from concurrent.futures import CancelledError, Future, ProcessPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
import multiprocessing
from pathlib import Path
from time import perf_counter
import traceback
from typing import Callable, Hashable, Iterable, Iterator, Mapping

import numpy as np

from .errors import InputError, NumericalIndexingError, WorkerError

# File-reading errors are normalized to InputError at the reader boundary.
# Bare ValueError/KeyError/OSError and native internal failures remain fatal.
EXPECTED_INPUT_ERRORS = (InputError, NumericalIndexingError, MemoryError)


@dataclass(frozen=True)
class FrameInput:
    """One frame to index, with its identity and per-frame processing values.

    Parameters
    ----------
    frame
        Two-dimensional array with a supported dtype, or path to a supported
        HDF5 frame. An array is sent to the worker with the task.
    input_id
        Caller-defined stable identity carried unchanged into the outcome.
        `None` when the caller has no identity beyond the input position.
    start, group
        Full-detector ROI origin and pixel grouping in ``(x, y)`` order, as for
        :meth:`Indexer.index`.
    depth
        Physical sample depth in micrometres, or `None` to use the HDF5 frame's
        ``entry1/depth`` when present.
    metadata
        Optional frame metadata object or mapping.
    """

    frame: np.ndarray | str | Path
    input_id: Hashable | None = None
    start: tuple[int, int] = (0, 0)
    group: tuple[int, int] = (1, 1)
    depth: float | None = None
    metadata: object | Mapping[str, object] | None = None


@dataclass(frozen=True)
class FrameOutcome:
    """The result of attempting one input.

    Parameters
    ----------
    input_index
        Zero-based position of the input in the iterable passed to
        :meth:`Indexer.iter_index`.
    input_id
        The ``input_id`` of the corresponding :class:`FrameInput`, or `None`.
    result
        The :class:`FrameResult` when the frame was processed, including a
        processed frame with no peaks or no patterns.
    error
        The expected input error when the frame could not be processed, one of
        :data:`EXPECTED_INPUT_ERRORS`. Exactly one of ``result`` and ``error``
        is set.
    seconds
        Wall time of the indexing call inside the worker.
    """

    input_index: int
    input_id: Hashable | None
    result: object | None = None
    error: Exception | None = None
    seconds: float = 0.0

    @property
    def ok(self) -> bool:
        """Whether the frame was processed."""
        return self.error is None


_worker_indexer = None
_worker_mask = None
_worker_init_failure = None


def _initialize_worker(geometry_path, crystal, peak_params, index_params,
                       detector_index, cosmic_filter, mask):
    global _worker_indexer, _worker_mask, _worker_init_failure
    from .indexer import Indexer

    try:
        _worker_indexer = Indexer(
            geometry_path, crystal, peak_params=peak_params, index_params=index_params,
            detector_index=detector_index, cosmic_filter=cosmic_filter,
        )
        _worker_mask = mask
    except Exception as error:
        _worker_init_failure = (type(error).__name__, str(error), traceback.format_exc())


def _index_task(task):
    index, frame, start, group, depth, metadata, keep_image = task
    if _worker_init_failure is not None:
        return ("init", index, *_worker_init_failure)
    started = perf_counter()
    try:
        result = _worker_indexer.index(
            frame, start=start, group=group, depth=depth, mask=_worker_mask,
            metadata=metadata, keep_image=keep_image,
        )
    except EXPECTED_INPUT_ERRORS as error:
        return ("error", index, error, perf_counter() - started)
    except Exception as error:
        return ("failure", index, type(error).__name__, str(error), traceback.format_exc())
    return ("ok", index, result, perf_counter() - started)


class FrameOutcomes:
    """Ordered outcomes of a bounded parallel indexing run.

    Instances are created by :meth:`Indexer.iter_index`. Use one as a context
    manager and iterate over it; leaving the ``with`` block shuts the worker
    processes down whether the iteration finished, stopped early, or raised.

    Attributes
    ----------
    stopped : bool
        `True` once ``should_stop`` returned `True` and admission ended.
    n_submitted : int
        Inputs handed to the worker pool.
    n_yielded : int
        Outcomes returned so far.
    n_cancelled : int
        Submitted inputs withdrawn before a worker started them, after a stop.
    peak_in_flight : int
        Highest number of submitted-but-unconsumed inputs observed. It never
        exceeds ``max_in_flight``.

    Notes
    -----
    Outcomes come back in input order. An input whose worker call is slow holds
    back later outcomes, which wait in the bounded in-flight window; no further
    inputs are admitted until the head is consumed. Inputs that were never
    started produce no outcome; compare ``n_submitted - n_cancelled`` with the
    number of inputs to find them.

    Workers use the ``spawn`` start method, so a script calling
    :meth:`Indexer.iter_index` at module level must guard that call with
    ``if __name__ == "__main__":``.
    """

    def __init__(self, indexer, inputs: Iterable, *, mask, workers: int,
                 max_in_flight: int | None, should_stop: Callable[[], bool] | None,
                 keep_images: bool, poll_seconds: float):
        if not isinstance(workers, (int, np.integer)) or isinstance(workers, bool) or workers < 1:
            raise InputError(f"workers must be a positive integer; received {workers!r}")
        if max_in_flight is None:
            max_in_flight = 2 * int(workers)
        if (
            not isinstance(max_in_flight, (int, np.integer))
            or isinstance(max_in_flight, bool)
            or max_in_flight < workers
        ):
            raise InputError(
                f"max_in_flight must be an integer of at least workers ({workers}); received {max_in_flight!r}"
            )
        if should_stop is not None and not callable(should_stop):
            raise InputError("should_stop must be callable or None")
        if not poll_seconds > 0:
            raise InputError("poll_seconds must be positive")
        if mask is not None:
            mask = np.ascontiguousarray(np.asarray(mask) != 0, dtype=np.uint8)
            if mask.ndim != 2:
                raise InputError(f"mask must be two-dimensional; received shape {mask.shape}")
        self._init_args = (
            str(indexer.geometry_path), indexer.crystal, indexer.peak_params,
            indexer.index_params, indexer.detector_index, indexer.cosmic_filter, mask,
        )
        self._inputs: Iterator[tuple[int, FrameInput]] = iter(
            (index, item if isinstance(item, FrameInput) else FrameInput(item))
            for index, item in enumerate(inputs)
        )
        self._workers = int(workers)
        self._max_in_flight = int(max_in_flight)
        self._should_stop = should_stop
        self._keep_images = bool(keep_images)
        self._poll_seconds = float(poll_seconds)
        self._pool: ProcessPoolExecutor | None = None
        self._pending: deque[Future] = deque()
        self._ids: dict[int, Hashable | None] = {}
        self._exhausted = False
        self._closed = False
        self.stopped = False
        self.n_submitted = 0
        self.n_yielded = 0
        self.n_cancelled = 0
        self.peak_in_flight = 0

    # -- lifecycle -----------------------------------------------------------

    def __enter__(self) -> "FrameOutcomes":
        self._start()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
        self.close()

    def __iter__(self) -> "FrameOutcomes":
        return self

    def close(self) -> None:
        """Withdraw unstarted work, wait for running work, and stop the workers.

        Safe to call more than once. After ``close()`` the iterator is
        finished.
        """
        self._closed = True
        pool, self._pool = self._pool, None
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=True)
        self._pending.clear()
        self._ids.clear()

    def _start(self) -> None:
        if self._pool is None and not self._closed:
            self._pool = ProcessPoolExecutor(
                max_workers=self._workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize_worker,
                initargs=self._init_args,
            )

    # -- iteration -------------------------------------------------------------

    def __next__(self) -> FrameOutcome:
        if self._closed:
            raise StopIteration
        try:
            return self._next_outcome()
        except StopIteration:
            self.close()
            raise
        except BaseException:
            self.close()
            raise

    def _next_outcome(self) -> FrameOutcome:
        self._start()
        self._fill()
        while self._pending:
            head = self._pending[0]
            try:
                payload = head.result(timeout=self._poll_seconds)
            except FutureTimeoutError:
                self._check_stop()
                continue
            except CancelledError:
                self._pending.popleft()
                continue
            except BrokenProcessPool as error:
                raise WorkerError(f"the indexing worker pool is broken: {error}") from error
            except Exception as error:
                raise WorkerError(
                    f"the indexing worker pool failed to return an outcome: {error}"
                ) from error
            self._pending.popleft()
            outcome = self._to_outcome(payload)
            self.n_yielded += 1
            self._fill()
            return outcome
        raise StopIteration

    def _fill(self) -> None:
        while not self._exhausted and len(self._pending) < self._max_in_flight:
            self._check_stop()
            if self._exhausted:
                return
            try:
                index, item = next(self._inputs)
            except StopIteration:
                self._exhausted = True
                return
            task = (
                index, item.frame, item.start, item.group, item.depth, item.metadata,
                self._keep_images,
            )
            self._ids[index] = item.input_id
            try:
                future = self._pool.submit(_index_task, task)
            except (BrokenProcessPool, RuntimeError) as error:
                # submit() itself raises once a worker has died, even when the
                # head outcome was delivered normally.
                raise WorkerError(f"the indexing worker pool is broken: {error}") from error
            self._pending.append(future)
            self.n_submitted += 1
            self.peak_in_flight = max(self.peak_in_flight, len(self._pending))

    def _check_stop(self) -> None:
        if self.stopped or self._should_stop is None:
            return
        if self._should_stop():
            self.stopped = True
            self._exhausted = True
            kept = deque()
            for future in self._pending:
                if future.cancel():
                    self.n_cancelled += 1
                else:
                    kept.append(future)
            self._pending = kept

    def _to_outcome(self, payload) -> FrameOutcome:
        kind, index = payload[0], payload[1]
        input_id = self._ids.pop(index, None)
        if kind == "ok":
            return FrameOutcome(index, input_id, result=payload[2], seconds=payload[3])
        if kind == "error":
            return FrameOutcome(index, input_id, error=payload[2], seconds=payload[3])
        if kind == "init":
            name, message, trace = payload[2:]
            raise WorkerError(f"indexing worker initialization failed: {name}: {message}\n{trace}")
        name, message, trace = payload[2:]
        raise WorkerError(f"indexing worker failed on input {index}: {name}: {message}\n{trace}")
