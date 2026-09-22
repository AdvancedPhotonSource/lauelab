# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""
Laue indexing submodule.

This submodule contains the core indexing functionality including:
- Functional indexing interface (index.py)
- Data classes for configuration and data structures
- Compiled C binaries for peak search, q-space conversion, and indexing
"""

from ._liblaue import DetectorGeometry, Geometry, WireGeometry, load_geometry
from .crystal import Atom, Cell, Crystal, load_crystal
from .errors import (
    IndexingError, InputError, InvalidResultsFile, InvalidScanFile, LaueError, NumericalIndexingError, ReconstructionError, WorkerError,
)
from ._frame import ScanFrame, load_mask
from .incremental import EXPECTED_INPUT_ERRORS, FrameInput, FrameOutcome, FrameOutcomes
from .index import IndexingResult, lauego
from .indexer import (
    FrameMetadata, FrameResult, Indexer, IndexParams, Pattern, PeakParams, index_frame,
)
from .results import ResultsFileSummary, ResultsWriter, validate_results_file
from .xml_utils import XmlResultsWriter

__all__ = [
    'lauego',
    'IndexingResult',
    'index_frame',
    'Indexer',
    'Cell',
    'Atom',
    'Crystal',
    'load_crystal',
    'DetectorGeometry',
    'WireGeometry',
    'Geometry',
    'load_geometry',
    'LaueError',
    'InputError',
    'IndexingError',
    'NumericalIndexingError',
    'ReconstructionError',
    'PeakParams',
    'IndexParams',
    'Pattern',
    'FrameMetadata',
    'FrameResult',
    'ResultsWriter',
    'load_mask',
    'ScanFrame',
    'WorkerError',
    'FrameInput',
    'FrameOutcome',
    'FrameOutcomes',
    'EXPECTED_INPUT_ERRORS',
    'InvalidResultsFile',
    'InvalidScanFile',
    'ResultsFileSummary',
    'validate_results_file',
    'XmlResultsWriter',
]
