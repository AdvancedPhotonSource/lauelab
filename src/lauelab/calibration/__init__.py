# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Detector geometry calibration with a crystal of known lattice.

This is the detector calibration of LaueGo (``microN/Calibration.ipf``): index
a frame of a calibration crystal, add measured energies of some reflections,
and fit the detector rotation and translation, together with the sample
rotation for the first detector. The result is written into a copy of the
geometry file.

Positions are full-detector pixels, lengths are in µm (geometry files store
mm), energies in keV, and reciprocal rows in 1/nm including 2π. Rotation
vectors are axis times angle in radians, as in the geometry file; they are not
the Rodrigues vectors (axis times tan(angle/2)) of
:func:`lauelab.analysis.orientation_to_rodrigues`. Sample rotations take the
crystal reference basis to the laboratory frame.
"""

from ._core import STATUSES, DetectorCalibration, calibrate_detector
from ._geometry_file import write_geometry
from ._model import (
    HC_KEV_NM,
    detector_azimuth_deg,
    matrix_from_igor,
    peak_directions,
    rotation_matrix,
    rotation_vector,
)
from ._peak import fit_peak
from ._surface import DEFAULT_SURFACE_NORMAL, SurfaceCheck, surface_check
from ._table import CalibrationTable, calibration_table

__all__ = [
    "CalibrationTable",
    "DEFAULT_SURFACE_NORMAL",
    "DetectorCalibration",
    "HC_KEV_NM",
    "STATUSES",
    "SurfaceCheck",
    "calibrate_detector",
    "calibration_table",
    "detector_azimuth_deg",
    "fit_peak",
    "matrix_from_igor",
    "peak_directions",
    "rotation_matrix",
    "rotation_vector",
    "surface_check",
    "write_geometry",
]
