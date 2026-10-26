# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Version-one standalone strain-result layout."""

import h5py

FORMAT = "lauelab-strain-results"
VERSION = 1
STATUSES = (
    "converged",
    "insufficient_support",
    "ill_conditioned",
    "nonconverged",
    "invalid_input",
)
STATUS_CODES = {name: code for code, name in enumerate(STATUSES)}
STATUS_TYPE = h5py.enum_dtype(STATUS_CODES, basetype="i1")
CELL_UNITS = ("nm", "nm", "nm", "deg", "deg", "deg")
ARRAYS = {
    "tensor": ((3, 3), None),
    "reciprocal": ((3, 3), "1/nm"),
    "reference_reciprocal": ((3, 3), "1/nm"),
    "rotation": ((3, 3), None),
    "cell_parameters": ((6,), CELL_UNITS),
    "working_cell_parameters": ((6,), CELL_UNITS),
    "predicted_qhat": ((None, 3), None),
    "predicted_xy": ((None, 2), "pixel"),
    "angular_residual_deg": ((None,), "deg"),
}
TEXT = ("message", "coordinates", "fit_mask", "provenance_json")
DIAGNOSTICS = ("n_parameters", "rank", "condition_number", "nfev")
