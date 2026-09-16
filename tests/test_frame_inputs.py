# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Frame dtype, physical depth, mask loading, and peak-limit input contracts."""

from pathlib import Path
from xml.etree import ElementTree

import h5py
import numpy as np
import pytest

from conftest import requires_liblaue

from lauelab.indexing import (
    Indexer, IndexParams, InputError, PeakParams, lauego, load_mask,
)
from lauelab.indexing.index import _find_executables
from lauelab.indexing.indexer import SUPPORTED_FRAME_DTYPES
from lauelab.reconstruct import Reconstructor
from tests.data.reconstruction.generate_reference import (
    BINNING, DEPTH_RANGE_UM, GEOMETRY_FILE as RECON_GEOMETRY, write_input_file,
)

ROOT = Path(__file__).resolve().parents[1]
pytestmark = requires_liblaue
GEOMETRY = ROOT / "tests/data/geo/geoN_2022-03-29_14-15-05.xml"
CRYSTAL = ROOT / "tests/config/Ni.xml"
FRAMES = ROOT / "tests/data/synthetic/frames"
TWO_GRAINS = FRAMES / "synthetic_ni_two_grains.h5"

# Settings shared with the synthetic LaueGo baseline (tests/data/synthetic/provenance.json).
PEAKS = PeakParams(
    boxsize=18, max_rfactor=0.5, min_size=3, min_separation=20,
    threshold=None, threshold_ratio=4.0, max_peaks=200,
)
INDEXING = IndexParams(
    kev_max_calc=17.2, kev_max_test=35.0, angle_tolerance_deg=0.1, cone_deg=72.0,
)


def _two_grains_image() -> np.ndarray:
    with h5py.File(TWO_GRAINS) as source:
        return source["entry1/data/data"][...]


def _write_frame(path, image, **datasets):
    with h5py.File(path, "w") as output:
        output.create_dataset("entry1/data/data", data=image)
        for name, value in datasets.items():
            output.create_dataset(f"entry1/{name}", data=value)


def _copy_frame_with_dtype(source: Path, target: Path, dtype, scale=1.0):
    with h5py.File(source) as origin, h5py.File(target, "w") as output:
        for name in origin:
            if name != "entry1":
                origin.copy(origin[name], output, name=name)
        for key, value in origin.attrs.items():
            output.attrs[key] = value
        entry = output.create_group("entry1")
        for name in origin["entry1"]:
            if name == "data":
                image = np.asarray(origin["entry1/data/data"][...], dtype=np.float64) * scale
                entry.create_dataset("data/data", data=image.astype(dtype))
            else:
                origin.copy(origin["entry1"][name], entry, name=name)


def _peaks_table(path: Path) -> np.ndarray:
    lines = Path(path).read_text().splitlines()
    start = next(index for index, line in enumerate(lines) if "$peakList" in line) + 1
    return np.loadtxt(lines[start:], ndmin=2)


# --- frame dtypes -----------------------------------------------------------


def test_supported_frame_dtypes_are_the_documented_set():
    assert {dtype.name for dtype in SUPPORTED_FRAME_DTYPES} == {
        "uint16", "uint8", "int8", "int16", "int32", "float32", "float64",
    }


@pytest.mark.parametrize("dtype", [np.int32, np.float32, np.float64])
def test_wide_dtypes_give_identical_results_to_uint16(dtype):
    image = _two_grains_image()
    indexer = Indexer(GEOMETRY, CRYSTAL, peak_params=PEAKS, index_params=INDEXING)

    expected = indexer.index(image, keep_image=False)
    actual = indexer.index(image.astype(dtype), keep_image=True)

    assert actual.image.dtype == np.dtype(dtype)
    assert actual.n_peaks == expected.n_peaks == 48
    np.testing.assert_array_equal(actual.peaks, expected.peaks)
    assert actual.total_sum == expected.total_sum
    assert actual.num_above_threshold == expected.num_above_threshold
    assert actual.n_patterns == expected.n_patterns == 2
    for left, right in zip(actual.patterns, expected.patterns):
        np.testing.assert_array_equal(left.reciprocal, right.reciprocal)
        np.testing.assert_array_equal(left.pk_index, right.pk_index)


@pytest.mark.parametrize("dtype", [np.uint8, np.int8, np.int16])
def test_narrow_dtypes_match_uint16_on_small_frames(dtype):
    image = np.full((64, 64), 5, dtype=np.uint16)
    image[30:34, 30:34] = 120
    indexer = Indexer(GEOMETRY, peak_params=PeakParams(threshold=50))

    expected = indexer.index(image)
    actual = indexer.index(image.astype(dtype))

    assert actual.n_peaks == expected.n_peaks == 1
    np.testing.assert_array_equal(actual.peaks, expected.peaks)
    assert actual.total_sum == expected.total_sum


def test_big_endian_frame_is_byte_swapped_not_reinterpreted():
    image = _two_grains_image()
    indexer = Indexer(GEOMETRY, peak_params=PEAKS)

    expected = indexer.index(image)
    actual = indexer.index(image.astype(">u2"))

    np.testing.assert_array_equal(actual.peaks, expected.peaks)
    assert actual.image.dtype == np.dtype(np.uint16)


@pytest.mark.parametrize("dtype", [np.uint32, np.int64, np.bool_, np.uint64])
def test_unsupported_dtypes_are_rejected_not_cast(dtype):
    frame = np.zeros((8, 8), dtype=dtype)

    with pytest.raises(InputError, match="dtype in .*uint16.*int32.*int8"):
        Indexer(GEOMETRY).index(frame)


def test_non_finite_float_frame_is_rejected():
    frame = np.zeros((8, 8), dtype=np.float64)
    frame[2, 2] = np.nan

    with pytest.raises(InputError, match="non-finite"):
        Indexer(GEOMETRY).index(frame)


def test_hdf5_frame_dtype_follows_the_same_contract(tmp_path):
    accepted = tmp_path / "int32.h5"
    rejected = tmp_path / "uint32.h5"
    _copy_frame_with_dtype(TWO_GRAINS, accepted, np.int32)
    _copy_frame_with_dtype(TWO_GRAINS, rejected, np.uint32)
    indexer = Indexer(GEOMETRY, peak_params=PEAKS)

    expected = indexer.index(TWO_GRAINS)
    actual = indexer.index(accepted)
    np.testing.assert_array_equal(actual.peaks, expected.peaks)
    with pytest.raises(InputError, match="uint32"):
        indexer.index(rejected)


@pytest.mark.parametrize(
    ("dtype", "scale"), [(np.int32, 1.0), (np.float64, 0.75)]
)
def test_typed_frames_agree_with_lauego_peaksearch(tmp_path, dtype, scale):
    # The LaueGo peaksearch program reads any numeric HDF5 image as double,
    # so it is an independent check of the typed native path.
    try:
        _find_executables()
    except FileNotFoundError:
        pytest.skip("LaueGo executables are not installed")
    frame = tmp_path / f"frame_{np.dtype(dtype).name}.h5"
    _copy_frame_with_dtype(TWO_GRAINS, frame, dtype, scale=scale)

    reference = lauego(
        str(frame), str(tmp_path / "lauego"), str(GEOMETRY), str(CRYSTAL),
        boxsize=PEAKS.boxsize, max_rfactor=PEAKS.max_rfactor, min_size=PEAKS.min_size,
        min_separation=PEAKS.min_separation, threshold=None,
        threshold_ratio=PEAKS.threshold_ratio, peak_shape="L",
        max_peaks=PEAKS.max_peaks, generate_xml=False,
    )
    assert reference.success, reference.error
    expected = _peaks_table(reference.output_files["peaks"])

    actual = Indexer(GEOMETRY, peak_params=PEAKS).index(frame)

    assert actual.image.dtype == np.dtype(dtype)
    assert actual.n_peaks == len(expected) == 48
    np.testing.assert_allclose(actual.peaks["fit_x"], expected[:, 0], atol=5e-4, rtol=0)
    np.testing.assert_allclose(actual.peaks["fit_y"], expected[:, 1], atol=5e-4, rtol=0)
    np.testing.assert_allclose(actual.peaks["intens"], expected[:, 2], atol=5e-4, rtol=1e-8)
    np.testing.assert_allclose(actual.peaks["integral"], expected[:, 3], atol=5e-6, rtol=1e-8)


# --- physical depth -----------------------------------------------------------


def _q_of_first_peak(result):
    return result.peaks["qhat"][0].copy()


def test_hdf5_depth_is_used_when_no_explicit_depth_is_given(tmp_path):
    image = _two_grains_image()
    with_depth = tmp_path / "depth.h5"
    without_depth = tmp_path / "no-depth.h5"
    _write_frame(with_depth, image, depth=np.asarray([40.0]))
    _write_frame(without_depth, image)
    indexer = Indexer(GEOMETRY, peak_params=PEAKS)

    from_file = indexer.index(with_depth)
    baseline = indexer.index(without_depth)
    explicit = indexer.index(without_depth, depth=40.0)

    assert from_file.depth == 40.0
    assert baseline.depth is None
    assert from_file.to_step().depth == 40.0
    np.testing.assert_array_equal(_q_of_first_peak(from_file), _q_of_first_peak(explicit))
    assert not np.array_equal(_q_of_first_peak(from_file), _q_of_first_peak(baseline))


def test_explicit_depth_overrides_hdf5_depth_including_zero(tmp_path):
    image = _two_grains_image()
    path = tmp_path / "depth.h5"
    _write_frame(path, image, depth=np.asarray([40.0]))
    plain = tmp_path / "plain.h5"
    _write_frame(plain, image)
    indexer = Indexer(GEOMETRY, peak_params=PEAKS)

    overridden = indexer.index(path, depth=12.5)
    zero = indexer.index(path, depth=0.0)
    no_depth = indexer.index(plain)

    assert overridden.depth == 12.5
    assert zero.depth == 0.0
    # A zero depth is a real depth: the geometry conversion result equals the
    # no-depth conversion, but the record says 0.0 rather than None.
    np.testing.assert_array_equal(_q_of_first_peak(zero), _q_of_first_peak(no_depth))
    assert no_depth.depth is None


@pytest.mark.parametrize("stored", [np.asarray([np.nan]), np.asarray([]), np.float32(np.inf)])
def test_non_finite_or_empty_hdf5_depth_means_no_depth(tmp_path, stored):
    path = tmp_path / "depth.h5"
    _write_frame(path, np.zeros((4, 5), dtype=np.uint16), depth=stored)

    result = Indexer(GEOMETRY).index(path)

    assert result.depth is None


def test_hdf5_depth_accepts_integer_and_scalar_storage(tmp_path):
    for name, stored in (("int", np.int32(7)), ("scalar", 7.0), ("vector", np.asarray([7.0, 9.0]))):
        path = tmp_path / f"{name}.h5"
        _write_frame(path, np.zeros((4, 5), dtype=np.uint16), depth=stored)
        assert Indexer(GEOMETRY).index(path).depth == 7.0


def test_non_numeric_hdf5_depth_is_an_error(tmp_path):
    path = tmp_path / "depth.h5"
    _write_frame(path, np.zeros((4, 5), dtype=np.uint16), depth=np.asarray([b"deep"]))

    with pytest.raises(ValueError, match="entry1/depth"):
        Indexer(GEOMETRY).index(path)


def test_reconstructed_per_depth_files_index_with_their_depth_roi_and_dtype(tmp_path):
    # Existing-style output: one int32 file per depth from a both-edge
    # reconstruction of the synthetic wire scan, on the same geometry.
    source = tmp_path / "wire-scan.h5"
    write_input_file(source)
    reconstruction = Reconstructor(
        RECON_GEOMETRY, 0, depth_range=DEPTH_RANGE_UM, wire_edge="both",
        num_threads=1, rows_per_stripe=32,
    ).reconstruct(source, tmp_path / "recon_")
    assert reconstruction.success, reconstruction.error
    depth_files = reconstruction.output_files[:-1]
    assert len(depth_files) == len(reconstruction.depth_um) == 51

    indexer = Indexer(RECON_GEOMETRY, peak_params=PeakParams(threshold=100.0, boxsize=4))
    chosen = 30
    result = indexer.index(depth_files[chosen])
    overridden = indexer.index(depth_files[chosen], depth=0.0)

    with h5py.File(depth_files[chosen]) as stored:
        assert stored["entry1/data/data"].dtype == np.dtype(np.int32)
    assert result.image.dtype == np.dtype(np.int32)
    assert result.depth == reconstruction.depth_um[chosen] == 5.0
    assert result.start == (0, 0)
    assert result.group == (BINNING, BINNING)
    assert result.image_shape == (128, 128)
    assert result.metadata["detector_id"] == "PE1621 723-3335"
    assert overridden.depth == 0.0
    assert result.to_step().detector.roi.endx == 2047


# --- masks --------------------------------------------------------------------


def test_load_mask_follows_nonzero_excludes_convention(tmp_path):
    image = np.arange(1, 8 * 12 + 1, dtype=np.uint16).reshape(8, 12)
    stored = np.zeros(image.shape, dtype=np.int32)
    stored[0, 0] = 7
    stored[3, 4] = -2
    path = tmp_path / "mask.h5"
    _write_frame(path, stored)

    mask = load_mask(path)

    assert mask.dtype == np.bool_
    assert mask.shape == image.shape
    assert mask.sum() == 2 and mask[0, 0] and mask[3, 4]
    result = Indexer(GEOMETRY, peak_params=PeakParams(threshold=1000)).index(image, mask=mask)
    assert result.total_sum == image.sum() - image[0, 0] - image[3, 4]


def test_load_mask_rejects_wrong_rank_and_shape_is_checked_at_index(tmp_path):
    bad = tmp_path / "bad.h5"
    _write_frame(bad, np.zeros((3, 4, 5), dtype=np.uint8))
    with pytest.raises(ValueError, match="two-dimensional"):
        load_mask(bad)

    small = tmp_path / "small.h5"
    _write_frame(small, np.zeros((4, 4), dtype=np.uint8))
    with pytest.raises(InputError, match="mask shape"):
        Indexer(GEOMETRY).index(np.zeros((8, 8), dtype=np.uint16), mask=load_mask(small))

    with pytest.raises(KeyError):
        empty = tmp_path / "empty.h5"
        h5py.File(empty, "w").close()
        load_mask(empty)


# --- peak limit ----------------------------------------------------------------


def test_max_peaks_none_fits_every_blob_and_finite_limits_still_cap(tmp_path):
    image = _two_grains_image()

    def count(max_peaks):
        indexer = Indexer(
            GEOMETRY, CRYSTAL,
            peak_params=PeakParams(
                boxsize=18, max_rfactor=0.5, min_size=3, min_separation=20,
                threshold=None, threshold_ratio=4.0, max_peaks=max_peaks,
            ),
            index_params=INDEXING,
        )
        return indexer, indexer.index(image, keep_image=False)

    _, capped = count(10)
    generous_indexer, generous = count(10_000)
    unlimited_indexer, unlimited = count(None)

    assert capped.n_peaks == 10
    assert unlimited.n_peaks == generous.n_peaks == 48
    np.testing.assert_array_equal(unlimited.peaks, generous.peaks)
    assert unlimited.n_patterns == generous.n_patterns == 2
    assert unlimited_indexer.peak_params.max_peaks is None
    assert generous_indexer.peak_params.max_peaks == 10_000

    step = unlimited.to_step()
    assert step.detector.peaksXY.NpeakMax is None
    xml_path = tmp_path / "unlimited.xml"
    unlimited.write_xml(xml_path)
    peaks_xml = ElementTree.parse(xml_path).getroot().find(".//peaksXY")
    assert "max_number" not in peaks_xml.attrib
    assert peaks_xml.get("Npeaks") == "48"

    results_path = tmp_path / "unlimited.h5"
    unlimited_indexer.write_results([unlimited], results_path)
    with h5py.File(results_path) as stored:
        assert np.isnan(stored["run"].attrs["max_peaks"])
        assert stored["frames/n_peaks"][0] == 48


def test_unlimited_peak_search_finds_more_than_the_default_cap():
    image = np.full((256, 256), 10, dtype=np.uint16)
    ys, xs = np.mgrid[8:256:16, 8:256:16]
    for y, x in zip(ys.ravel(), xs.ravel()):
        image[y - 1:y + 2, x - 1:x + 2] = 900
    params = PeakParams(threshold=100.0, boxsize=4, min_size=2, min_separation=3)

    default = Indexer(GEOMETRY, peak_params=params).index(image)
    unlimited = Indexer(GEOMETRY, peak_params=PeakParams(**{**params.__dict__, "max_peaks": None})).index(image)

    assert default.n_peaks == 50
    assert unlimited.n_peaks == 256


# --- whole-number parameters -----------------------------------------------------


def test_integral_floats_are_normalized_to_int():
    indexer = Indexer(
        GEOMETRY, CRYSTAL,
        peak_params=PeakParams(boxsize=5.0, min_size=3.0, min_separation=10.0, max_peaks=50.0),
        index_params=IndexParams(hkl_prefer=(0.0, 0, 1.0), max_data=200.0),
    )

    for name in ("boxsize", "min_size", "min_separation", "max_peaks"):
        assert type(getattr(indexer.peak_params, name)) is int
    assert indexer.peak_params.min_size == 3
    assert indexer.index_params.hkl_prefer == (0, 0, 1)
    assert all(type(value) is int for value in indexer.index_params.hkl_prefer)
    assert type(indexer.index_params.max_data) is int
    assert indexer.index(np.zeros((4, 4), dtype=np.uint16)).n_peaks == 0


@pytest.mark.parametrize(
    ("peak_params", "index_params", "name", "shown"),
    [
        (PeakParams(min_size=3.5), None, "min_size", "3.5"),
        (PeakParams(min_size=1.13), None, "min_size", "1.13"),
        (PeakParams(boxsize=18.2), None, "boxsize", "18.2"),
        (PeakParams(min_separation=np.float64(20.5)), None, "min_separation", "20.5"),
        (PeakParams(max_peaks=200.5), None, "max_peaks", "200.5"),
        (PeakParams(max_peaks=True), None, "max_peaks", "True"),
        (PeakParams(min_size="3"), None, "min_size", "'3'"),
        (None, IndexParams(max_data=2.5), "max_data", "2.5"),
        (None, IndexParams(hkl_prefer=(0, 0.5, 1)), "hkl_prefer", "0.5"),
    ],
)
def test_fractional_whole_number_parameters_are_rejected_not_rounded(
    peak_params, index_params, name, shown
):
    with pytest.raises(InputError) as error:
        Indexer(GEOMETRY, peak_params=peak_params, index_params=index_params)

    message = str(error.value)
    assert name in message and "whole number" in message and shown in message


def test_zero_max_peaks_is_rejected_in_favor_of_none():
    with pytest.raises(InputError, match="positive.*None"):
        Indexer(GEOMETRY, peak_params=PeakParams(max_peaks=0))
