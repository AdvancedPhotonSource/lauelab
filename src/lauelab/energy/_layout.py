# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Complete required dataset layout of Q-distribution format version 1.

An incompatible path, meaning or dtype change requires a new version. Optional
additions may retain version 1. Counts and matrices are dimensionless. The
by_intensity and by_normalization markers resolve units from recorded metadata
and correction mode; these markers are not literal file attributes.
"""

FORMAT = "lauelab-q-distribution"
VERSION = 1

# name -> (dtype, units); every profile has the common shape (n,).
PROFILE = {
    "q_nm_inv": ("float64", "1/nm"),
    "intensity_mean": ("float64", "by_intensity"),
    "intensity_sum": ("float64", "by_intensity"),
    "pixel_count": ("int64", None),
    "nonfinite_pixel_count": ("int64", None),
}
# path -> (dtype, shape, units). UTF-8 scalar text is denoted by 'text'.
SETTINGS = {
    "parameters/energy_kev": ("float64", (), "keV"),
    "parameters/depth_um": ("float64", (), "um"),
    "parameters/start": ("int64", (2,), "pixel"),
    "parameters/group": ("int64", (2,), None),
    "parameters/roi": ("int64", (4,), "pixel"),
    "parameters/hc_kev_nm": ("float64", (), "keV nm"),
    "parameters/normalization_factor": ("float64", (), "by_normalization"),
    "parameters/exposure_seconds": ("float64", (), "s"),
    "parameters/i0_counts": ("float64", (), "unspecified"),
    "parameters/scaler_count_time_seconds": ("float64", (), "s"),
    "parameters/i0_gain": ("float64", (), "unspecified"),
    "parameters/normalization": ("text", (), None),
    "geometry/pixels": ("int64", (2,), "pixel"),
    "geometry/size_um": ("float64", (2,), "um"),
    "geometry/translation_um": ("float64", (3,), "um"),
    "geometry/rotation_vector_rad": ("float64", (3,), "rad"),
    "geometry/rotation": ("float64", (3, 3), None),
    "geometry/detector_id": ("text", (), None),
    "geometry/detector_index": ("int64", (), None),
    "geometry/xml": ("text", (), None),
    "provenance/json": ("text", (), None),
}
