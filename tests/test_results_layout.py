# Copyright 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Tests for the indexing-results HDF5 layout table."""

import h5py
import numpy as np
import pytest

from lauelab._results_layout import (
    DATASETS,
    FORMAT,
    FRAME_IDS_STRING_SPEC,
    SUPPORTED_VERSIONS,
    VERSION,
)

EXPECTED_PATHS = {
    "/frames/exposure_seconds", "/frames/beam_bad", "/frames/light_on",
    "/frames/hutch_temperature", "/frames/sample_distance",
    "/crystal/lattice_parameters", "/crystal/atom_symbols", "/crystal/atom_labels",
    "/crystal/atom_positions", "/crystal/atom_occupancies", "/geometry/xml",
    "/frames/frame_ids", "/frames/sample_positions", "/frames/depths",
    "/frames/scan_numbers", "/frames/energies_kev", "/frames/detector_ids",
    "/frames/input_images", "/frames/titles", "/frames/sample_names",
    "/frames/user_names", "/frames/beamlines", "/frames/dates_exposed",
    "/frames/ccd_shutters", "/frames/mono_modes", "/frames/image_shapes", "/frames/roi_starts",
    "/frames/roi_groups", "/frames/n_peaks", "/frames/n_patterns",
    "/frames/threshold_used", "/frames/threshold_ratio", "/frames/total_sum",
    "/frames/sum_above_threshold", "/frames/num_above_threshold",
    "/frames/peak_minwidth", "/frames/peak_maxwidth",
    "/frames/peak_max_cent_to_fit", "/frames/peak_boxsize",
    "/frames/peaksearch_seconds", "/frames/indexing_seconds",
    "/frames/peak_offsets", "/frames/pattern_offsets", "/peaks/fit_x",
    "/peaks/fit_y", "/peaks/intens", "/peaks/integral", "/peaks/hwhm_x",
    "/peaks/hwhm_y", "/peaks/tilt", "/peaks/chisq", "/peaks/background",
    "/peaks/qhat", "/patterns/rank", "/patterns/reciprocal",
    "/patterns/goodness", "/patterns/rms_error_deg", "/patterns/n_indexed",
    "/patterns/assignment_offsets", "/assignments/peak_index",
    "/assignments/hkl", "/assignments/error_deg", "/assignments/energy_kev",
    "/assignments/pred_intens",
}


def test_format_identity_and_complete_dataset_table():
    assert FORMAT == "lauelab-indexing-results"
    assert VERSION == 1
    assert SUPPORTED_VERSIONS == {1}
    assert set(DATASETS) == EXPECTED_PATHS


def test_numeric_precision_policy_and_string_alternative():
    assert DATASETS["/patterns/reciprocal"].dtype == np.dtype("<f8")
    assert DATASETS["/frames/total_sum"].dtype == np.dtype("<f8")
    assert DATASETS["/peaks/fit_x"].dtype == np.dtype("<f4")
    assert DATASETS["/frames/n_peaks"].dtype == np.dtype("<i4")
    assert DATASETS["/patterns/rank"].dtype == np.dtype("<i2")
    assert DATASETS["/assignments/hkl"].dtype == np.dtype("<i2")
    assert h5py.check_string_dtype(DATASETS["/frames/detector_ids"].dtype).encoding == "utf-8"
    assert h5py.check_string_dtype(FRAME_IDS_STRING_SPEC.dtype).encoding == "utf-8"


def test_shapes_units_chunks_and_scientific_attributes():
    assert DATASETS["/frames/sample_positions"].shape == (3,)
    assert DATASETS["/peaks/qhat"].shape == (3,)
    assert DATASETS["/patterns/reciprocal"].shape == (3, 3)
    assert DATASETS["/assignments/hkl"].shape == (3,)

    assert DATASETS["/frames/depths"].units == "um"
    assert DATASETS["/peaks/fit_x"].units == "pixel"
    assert DATASETS["/assignments/energy_kev"].units == "keV"
    assert DATASETS["/frames/n_peaks"].units is None

    assert DATASETS["/frames/frame_ids"].chunk_rows == 1024
    assert DATASETS["/patterns/rank"].chunk_rows == 1024
    assert DATASETS["/peaks/fit_x"].chunk_rows == 4096
    assert DATASETS["/assignments/hkl"].chunk_rows == 4096
    assert not DATASETS["/crystal/lattice_parameters"].resizable
    assert DATASETS["/crystal/lattice_parameters"].chunk_rows is None

    assert DATASETS["/crystal/lattice_parameters"].attrs == {"angle_units": "deg"}
    assert DATASETS["/patterns/reciprocal"].attrs == {
        "rows": "a*,b*,c*",
        "includes_two_pi": True,
    }


def test_layout_is_immutable():
    with pytest.raises(TypeError):
        DATASETS["/new"] = DATASETS["/frames/depths"]
    with pytest.raises(TypeError):
        DATASETS["/patterns/reciprocal"].attrs["rows"] = "columns"
