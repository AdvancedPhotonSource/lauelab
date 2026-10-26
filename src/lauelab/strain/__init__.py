# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Deviatoric lattice strain from assigned Laue reflections.

Use StrainRefiner for indexing results/files or the functional stages for
arrays. Reciprocal rows include 2π in 1/nm. Strain is reference-volume
normalized deviatoric Biot stretch, with tensor shear. Hydrostatic strain,
calibrated error bars, peak refitting and reassignment are outside this API.
"""

from ._core import StrainRefiner, refine_deviatoric
from ._inputs import StrainReflections, prepare_reflections
from ._models import StrainResult
from ._tensor import strain_from_reciprocal, transform_strain
from ._io import export_strain_csv, load_strain_results, save_strain_results

__all__ = [
    "StrainRefiner",
    "StrainReflections",
    "StrainResult",
    "prepare_reflections",
    "refine_deviatoric",
    "strain_from_reciprocal",
    "transform_strain",
    "save_strain_results",
    "load_strain_results",
    "export_strain_csv",
]
