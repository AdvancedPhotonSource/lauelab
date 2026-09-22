# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Wire scan reconstruction for Laue analysis."""

from lauelab.indexing.errors import InvalidScanFile, ReconstructionError

from lauelab.indexing._frame import ScanFrame

from ._export import export_per_depth
from . import inspection
from ._per_depth_reader import PerDepthReader
from ._scan_reader import PointEntry, PointReader, ScanFileSummary, ScanReader, validate_scan_file
from .batch import reconstruct_points
from .scan import PointOutcome, ScanResult, reconstruct_scan
from .reconstructor import ImageGeometry, Reconstructor, StripeTiming
from .reconstruct import (
    reconstruct,
    find_executable,
    ReconstructionResult,
    # GPU functions
    reconstruct_gpu,
    find_gpu_executable,
    gpu_available
)

__all__ = [
    # CPU functions
    'reconstruct',
    'find_executable',
    'Reconstructor',
    'reconstruct_points',
    'ImageGeometry',
    'StripeTiming',
    # Single-file scan output
    'reconstruct_scan',
    'ScanResult',
    'PointOutcome',
    'ScanReader',
    'PointReader',
    'PerDepthReader',
    'PointEntry',
    'validate_scan_file',
    'ScanFileSummary',
    'ScanFrame',
    'export_per_depth',
    'inspection',
    # GPU functions
    'reconstruct_gpu',
    'find_gpu_executable',
    'gpu_available',
    # Common types
    'ReconstructionResult',
    'ReconstructionError',
    'InvalidScanFile',
]
