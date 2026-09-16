# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Public exceptions for in-process indexing."""


class LaueError(Exception):
    """Base class for errors reported by the in-process indexing API.

    Notes
    -----
    Native allocation failures are reported as the built-in :class:`MemoryError`
    rather than as a ``LaueError`` subclass.
    """


class InputError(LaueError, ValueError):
    """Invalid geometry selection, parameters, frame data, or metadata.

    This exception is both a :class:`LaueError` and a :class:`ValueError`, so
    callers may catch it either as a package-specific error or as invalid input.
    """


class IndexingError(LaueError, RuntimeError):
    """Numerical or internal failure in a native indexing stage.

    This exception is both a :class:`LaueError` and a :class:`RuntimeError`.
    Its message identifies the failed stage and includes the native diagnostic.
    """


class NumericalIndexingError(IndexingError):
    """Expected numerical failure for one frame in a native indexing stage.

    A subclass of :class:`IndexingError`, so existing callers catching that
    base class still handle it. Incremental indexing continues after this
    exception; other ``IndexingError`` failures stop the worker run.
    """


class ReconstructionError(LaueError, RuntimeError):
    """Runtime failure in the in-process reconstruction pipeline."""


class InvalidResultsFile(LaueError, ValueError):
    """An indexing-results HDF5 file is structurally inconsistent or incomplete.

    Raised by :func:`~lauelab.indexing.validate_results_file` when the format
    marker is present but a dataset is missing, has the wrong dtype or shape,
    row counts disagree, offsets do not partition their rows, frame identities
    repeat, or the file does not match the identities the caller expected. A
    file that cannot be opened at all raises ``OSError`` instead.
    """


class WorkerError(LaueError, RuntimeError):
    """Failure of the parallel indexing machinery rather than of one input.

    Raised by iteration over :class:`~lauelab.indexing.FrameOutcomes` when a
    worker could not initialize its indexer, a worker raised an exception that
    is not an expected input error, or the worker pool broke, for example
    because a process was killed. Expected per-input errors are reported in
    ``FrameOutcome.error`` instead and never raise this.
    """
