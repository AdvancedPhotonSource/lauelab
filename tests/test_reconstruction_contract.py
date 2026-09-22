# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Value semantics of the reconstruction storage and inspection contract.

These tests need no native library. They pin three things the scan writer,
the reductions, and the ROI tools are built on: what HDF5 stores for a
computed float64 pixel, what the recorded goldens contain, and the
hand-worked fixtures under ``tests/data/reconstruction_contract/``.
"""

import itertools
import json
from numbers import Integral
from pathlib import Path

import h5py
import numpy as np
import pytest

from lauelab.reconstruct._scan_layout import accumulator_dtype
from lauelab.reconstruct._writer import PIXEL_DTYPES, normalization_rescale
from tests.data.reconstruction_contract import fixtures

REFERENCE_DIR = Path(__file__).parent / "data/reconstruction"


def _stored_by_hdf5(tmp_path, values, dtype):
    """Write float64 values as the per-depth writer does and reopen them."""
    path = tmp_path / "conversion.h5"
    values = np.asarray(values, dtype=np.float64)
    with h5py.File(path, "w") as target:
        target.create_dataset("data", shape=values.shape, dtype=dtype)[...] = values
    with h5py.File(path, "r") as source:
        return np.asarray(source["data"])


# --- Stored values ---------------------------------------------------------

@pytest.mark.parametrize("dtype", sorted(fixtures.CONVERSION))
def test_hdf5_truncates_toward_zero_and_saturates(tmp_path, dtype):
    computed, expected = zip(*fixtures.CONVERSION[dtype])
    stored = _stored_by_hdf5(tmp_path, computed, dtype)
    np.testing.assert_array_equal(stored, np.asarray(expected, dtype=dtype))


@pytest.mark.parametrize("dtype", ["<f4", "<f8"])
def test_hdf5_float_output_equals_a_numpy_cast(tmp_path, dtype):
    stored = _stored_by_hdf5(tmp_path, fixtures.COMPUTED, dtype)
    np.testing.assert_array_equal(stored, fixtures.COMPUTED.astype(dtype))


@pytest.mark.parametrize("name", sorted(fixtures.STORED))
def test_stored_fixture_is_what_hdf5_writes(tmp_path, name):
    case = fixtures.STORED[name]
    dtype = PIXEL_DTYPES[case["pixel_type"]]
    assert case["data"].dtype == dtype
    stored = _stored_by_hdf5(tmp_path, fixtures.COMPUTED * case["rescale"], dtype)
    np.testing.assert_array_equal(stored, case["data"])


def test_rescaled_fixture_uses_the_writer_rescale():
    case = fixtures.STORED["int16_rescaled"]
    assert normalization_rescale(case["pixel_type"]) == case["rescale"]


@pytest.mark.parametrize("name", sorted(fixtures.STORED))
def test_stored_reductions_are_exact_in_the_accumulator_dtype(name):
    case = fixtures.STORED[name]
    accumulator = accumulator_dtype(case["data"].dtype)
    assert accumulator == np.dtype("<i8")
    assert case["depth_intensity"].dtype == accumulator
    np.testing.assert_array_equal(
        case["data"].sum(axis=(1, 2), dtype=accumulator), case["depth_intensity"]
    )
    np.testing.assert_array_equal(
        case["data"].sum(axis=0, dtype=accumulator), case["sum_reconstructed"]
    )
    assert case["sum_reconstructed"].sum() == case["depth_intensity"].sum()


def test_integer_reduction_would_overflow_in_the_stored_dtype():
    data = fixtures.STORED["int16_rescaled"]["data"]
    wrapped = data.sum(axis=(1, 2), dtype=data.dtype)
    assert wrapped[2] != fixtures.STORED["int16_rescaled"]["depth_intensity"][2]


def test_computed_totals_are_not_stored_totals():
    # rtol covers float64 summation order, which the contract leaves open.
    np.testing.assert_allclose(
        fixtures.COMPUTED.sum(axis=(1, 2)), fixtures.COMPUTED_DEPTH_INTENSITY,
        rtol=1e-12, atol=0,
    )
    stored = fixtures.STORED["uint16"]["depth_intensity"]
    assert stored[3] == 0 and fixtures.COMPUTED_DEPTH_INTENSITY[3] < 0
    assert stored[0] > fixtures.COMPUTED_DEPTH_INTENSITY[0]


def test_fixture_covers_the_cases_the_goldens_lack():
    computed = fixtures.COMPUTED
    assert not computed[1].any()
    assert (computed[3] <= 0).all()
    unsigned = fixtures.STORED["uint16"]["data"]
    rescaled = fixtures.STORED["int16_rescaled"]["data"]
    assert (unsigned == np.iinfo(np.uint16).max).any()
    assert (rescaled == np.iinfo(np.int16).max).any()
    assert (rescaled == np.iinfo(np.int16).min).any()


# --- Recorded goldens ------------------------------------------------------

# variant -> (dtype, negative pixels, pixels at dtype minimum, pixels at dtype
# maximum, nonpositive planes, brightest plane, stored total). Read from the
# recorded files; a change here means a golden changed.
GOLDEN_FACTS = {
    "": ("float64", 82844, None, None, 9, 25, None),
    "_trailing": ("float64", 85522, None, None, 23, 40, None),
    "_cosmic": ("float64", 82858, None, None, 9, 25, None),
    "_norm_vector": ("float64", 95876, None, None, 27, 38, None),
    "_norm_exponent": ("float64", 82844, None, None, 16, 26, None),
    "_both": ("int32", 61768, 0, 0, 8, 25, 510655),
    "_out_int16": ("int16", 61768, 0, 0, 8, 25, 510655),
    "_out_uint16": ("uint16", 0, 795047, 0, 0, 25, 573801),
}


def _golden(suffix):
    return np.load(REFERENCE_DIR / f"cpu_reference{suffix}.npz")


@pytest.mark.parametrize("suffix", sorted(GOLDEN_FACTS))
def test_golden_value_characterization(suffix):
    dtype, negative, at_minimum, at_maximum, nonpositive, brightest, total = GOLDEN_FACTS[suffix]
    golden = _golden(suffix)
    images = golden["images"]
    metadata = json.loads((REFERENCE_DIR / f"cpu_reference{suffix}.json").read_text())

    assert images.dtype == np.dtype(dtype)
    assert images.shape == (51, 128, 128)
    assert metadata.get("output_dtype", "float64") == dtype
    np.testing.assert_array_equal(golden["depth_um"], np.arange(-25.0, 26.0))
    assert int((images < 0).sum()) == negative
    # No golden has a zero image; the contract fixture supplies one.
    assert images.any(axis=(1, 2)).all()

    totals = images.sum(axis=(1, 2), dtype=accumulator_dtype(images.dtype))
    assert int((totals <= 0).sum()) == nonpositive
    assert int(totals.argmax()) == brightest
    if images.dtype.kind in "iu":
        limits = np.iinfo(images.dtype)
        assert int((images == limits.min).sum()) == at_minimum
        assert int((images == limits.max).sum()) == at_maximum
        assert totals.dtype == np.dtype("<i8")
        assert int(totals.sum()) == total


def test_leading_edge_output_is_signed_and_unsigned_storage_clamps_it():
    computed = _golden("")["images"]
    stored = _golden("_out_uint16")["images"]
    np.testing.assert_array_equal(
        np.clip(np.trunc(computed), 0, 65535).astype(np.uint16), stored
    )
    # Clamping removes every negative pixel, so the stored total is larger.
    assert stored.sum(dtype=np.int64) > computed.sum()


def test_both_edge_goldens_store_the_same_values_in_two_dtypes():
    np.testing.assert_array_equal(
        _golden("_both")["images"], _golden("_out_int16")["images"]
    )


# --- ROI placement ---------------------------------------------------------

def _nearest_lower(click, size):
    """Brute-force oracle: nearest legal centre, ties to the lower coordinate.

    Legal centres lie on the size's lattice whether or not the square fits,
    so the search brackets the click, not the image.
    """
    starts = range(int(np.floor(click)) - size - 2, int(np.ceil(click)) + size + 3)
    centres = [start + (size - 1) / 2 for start in starts]
    best = min(centres, key=lambda centre: (abs(centre - click), centre))
    return int(round(best - (size - 1) / 2)), best


def _place(size, click, shape):
    if isinstance(size, (bool, np.bool_)) or not isinstance(size, Integral) or size < 1:
        raise ValueError("size")
    if not np.isfinite(click).all():
        return None
    (x0, centre_x), (y0, centre_y) = (
        _nearest_lower(click[0], size), _nearest_lower(click[1], size)
    )
    if x0 < 0 or y0 < 0 or x0 + size > shape[1] or y0 + size > shape[0]:
        return None
    return (y0, y0 + size, x0, x0 + size), (centre_x, centre_y)


@pytest.mark.parametrize(("size", "click", "bounds", "centre"), fixtures.PLACEMENTS)
def test_placement_is_the_nearest_legal_centre_with_ties_low(size, click, bounds, centre):
    assert _place(size, click, fixtures.IMAGE_SHAPE) == (bounds, centre)
    y0, y1, x0, x1 = bounds
    assert (y1 - y0, x1 - x0) == (size, size)
    # The closed form recorded in the format page.
    assert x0 == int(np.ceil(click[0] - size / 2))
    assert y0 == int(np.ceil(click[1] - size / 2))
    assert centre == (x0 + (size - 1) / 2, y0 + (size - 1) / 2)


@pytest.mark.parametrize(("size", "click"), fixtures.REJECTED_PLACEMENTS)
def test_out_of_image_placement_is_rejected(size, click):
    assert _place(size, click, fixtures.IMAGE_SHAPE) is None


@pytest.mark.parametrize("size", fixtures.INVALID_SIZES, ids=repr)
def test_size_must_be_a_positive_integer(size):
    with pytest.raises(ValueError):
        _place(size, (1.0, 1.0), fixtures.IMAGE_SHAPE)


def test_closed_form_agrees_with_the_oracle_on_a_dense_grid():
    for size, step in itertools.product(range(1, 6), range(-40, 121)):
        click = step / 8
        start, _ = _nearest_lower(click, size)
        assert start == int(np.ceil(click - size / 2)), (size, click)


# --- ROI traces, normalization, logarithmic display -------------------------

@pytest.mark.parametrize("name", sorted(fixtures.ROI_TRACES))
def test_roi_trace_is_the_sum_of_stored_pixels_in_half_open_bounds(name):
    (y0, y1, x0, x1), expected = fixtures.ROI_TRACES[name]
    data = fixtures.STORED["int32"]["data"]
    trace = [
        sum(int(data[depth, y, x]) for y in range(y0, y1) for x in range(x0, x1))
        for depth in range(len(data))
    ]
    assert trace == expected.tolist()
    assert expected.dtype == np.dtype("<i8")
    assert len(expected) == len(fixtures.DEPTH_UM)


def test_full_frame_roi_trace_is_the_embedded_reduction():
    np.testing.assert_array_equal(
        fixtures.ROI_TRACES["full_frame"][1],
        fixtures.STORED["int32"]["depth_intensity"],
    )


@pytest.mark.parametrize(("trace", "expected"), fixtures.NORMALIZED)
def test_normalization_needs_a_positive_maximum(trace, expected):
    maximum = max(trace.tolist())
    if expected is None:
        assert maximum <= 0
        return
    assert maximum > 0
    np.testing.assert_array_equal([value / maximum for value in trace.tolist()], expected)
    assert (np.sign(expected) == np.sign(trace)).all()


@pytest.mark.parametrize(("trace", "kept", "omitted"), fixtures.LOG_SAMPLES)
def test_logarithmic_axis_omits_nonpositive_samples(trace, kept, omitted):
    positive = [index for index, value in enumerate(trace.tolist()) if value > 0]
    assert positive == kept.tolist()
    assert len(trace) - len(positive) == omitted
