# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
from pathlib import Path

import h5py
import numpy as np
import pytest

from conftest import requires_liblaue

from lauelab.indexing import (
    Indexer,
    IndexingError,
    IndexParams,
    InputError,
    PeakParams,
)
from lauelab.indexing import indexer as indexer_module
from lauelab.indexing._frame import (
    detector_to_roi_pixels,
    roi_inclusive_end,
    roi_to_detector_pixels,
)
from lauelab.indexing._liblaue import get_library


ROOT = Path(__file__).resolve().parents[1]
GEOMETRY = ROOT / "tests/data/geo/geoN_2022-03-29_14-15-05.xml"
pytestmark = requires_liblaue


@pytest.mark.parametrize(
    "peak_params, message",
    [
        (PeakParams(boxsize=0), "positive"),
        (PeakParams(min_size=0), "positive"),
        (PeakParams(min_separation=0), "positive"),
        (PeakParams(max_peaks=0), "positive"),
        (PeakParams(max_rfactor=0), "max_rfactor"),
        (PeakParams(threshold_ratio=-1), "threshold_ratio"),
        (PeakParams(peak_shape="Voigt"), "peak_shape"),
        (PeakParams(peak_shape="Lemon"), "peak_shape"),
        (PeakParams(peak_shape="G"), "peak_shape"),
    ],
)
def test_indexer_rejects_invalid_peak_parameters(peak_params, message):
    with pytest.raises(InputError, match=message):
        Indexer(GEOMETRY, peak_params=peak_params)


@pytest.mark.parametrize(
    "index_params, message",
    [
        (IndexParams(kev_max_calc=0), "positive"),
        (IndexParams(kev_max_test=0), "positive"),
        (IndexParams(angle_tolerance_deg=0), "positive"),
        (IndexParams(cone_deg=0), "positive"),
        (IndexParams(hkl_prefer=(0, 1)), "exactly three"),
        (IndexParams(max_data=1), "at least 2"),
    ],
)
def test_indexer_rejects_invalid_index_parameters(index_params, message):
    with pytest.raises(InputError, match=message):
        Indexer(GEOMETRY, index_params=index_params)


@pytest.mark.parametrize(
    "frame",
    [
        np.zeros((2, 3, 4), dtype=np.uint16),
        np.zeros((3, 4), dtype=np.uint32),
        np.zeros((3, 4), dtype=np.int64),
    ],
)
def test_index_rejects_invalid_frame_arrays(frame):
    with pytest.raises(InputError, match="2D array with dtype in"):
        Indexer(GEOMETRY).index(frame)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"start": (-1, 0)}, "nonnegative"),
        ({"start": (0.5, 0)}, "nonnegative"),
        ({"group": (0, 1)}, "positive"),
        ({"group": (1, 1.5)}, "positive"),
        ({"depth": np.inf}, "finite"),
        ({"mask": np.zeros((3, 3))}, "mask shape"),
    ],
)
def test_index_rejects_invalid_frame_options(kwargs, message):
    with pytest.raises(InputError, match=message):
        Indexer(GEOMETRY).index(np.zeros((4, 5), dtype=np.uint16), **kwargs)


def test_hdf5_metadata_override_and_processing_values(tmp_path):
    path = tmp_path / "frame.h5"
    with h5py.File(path, "w") as output:
        output.create_dataset("entry1/data/data", data=np.zeros((4, 5), dtype=np.uint16))
        output.create_dataset("entry1/sample/name", data=np.asarray([b"from file"]))
        output.create_dataset("entry1/detector/startx", data=10)
        output.create_dataset("entry1/detector/starty", data=20)
        output.create_dataset("entry1/detector/binx", data=2)
        output.create_dataset("entry1/detector/biny", data=3)

    result = Indexer(GEOMETRY).index(
        path, start=(100, 100), group=(1, 1), metadata={"sample_name": "supplied"}
    )

    assert result.metadata["sample_name"] == "supplied"
    assert result.start == (10, 20)
    assert result.group == (2, 3)
    assert result.to_step().sampleName == "supplied"


def test_hdf5_integral_float_processing_values_are_coerced(tmp_path):
    path = tmp_path / "integral-floats.h5"
    with h5py.File(path, "w") as output:
        output.create_dataset("entry1/data/data", data=np.zeros((4, 5), dtype=np.uint16))
        output.create_dataset("entry1/detector/startx", data=10.0)
        output.create_dataset("entry1/detector/starty", data=20.0)
        output.create_dataset("entry1/detector/binx", data=2.0)
        output.create_dataset("entry1/detector/biny", data=3.0)

    result = Indexer(GEOMETRY).index(path)

    assert result.start == (10, 20)
    assert result.group == (2, 3)
    assert all(type(value) is int for value in result.start + result.group)


def test_hdf5_rejects_nonintegral_processing_value_with_field_name(tmp_path):
    path = tmp_path / "nonintegral.h5"
    with h5py.File(path, "w") as output:
        output.create_dataset("entry1/data/data", data=np.zeros((4, 5), dtype=np.uint16))
        output.create_dataset("entry1/detector/startx", data=10.5)
        output.create_dataset("entry1/detector/starty", data=20)

    with pytest.raises(ValueError, match="entry1/detector/startx"):
        Indexer(GEOMETRY).index(path)


def test_index_normalizes_numpy_start_and_group_to_integer_tuples():
    result = Indexer(GEOMETRY).index(
        np.zeros((4, 5), dtype=np.uint16),
        start=np.array([10, 20], dtype=np.int64),
        group=np.array([2, 3], dtype=np.int64),
    )

    assert result.start == (10, 20)
    assert result.group == (2, 3)
    assert all(type(value) is int for value in result.start + result.group)


def test_hdf5_detector_mismatch_is_reported_before_roi_bounds(tmp_path):
    path = tmp_path / "wrong-detector.h5"
    with h5py.File(path, "w") as output:
        output.create_dataset("entry1/data/data", data=np.zeros((4096, 4096), dtype=np.uint16))
        output.create_dataset("entry1/detector/ID", data="WRONG")

    with pytest.raises(InputError, match="does not match selected detector"):
        Indexer(GEOMETRY).index(path)


def test_hdf5_without_roi_metadata_preserves_explicit_processing_values(tmp_path):
    path = tmp_path / "frame.h5"
    with h5py.File(path, "w") as output:
        output.create_dataset("entry1/data/data", data=np.zeros((4, 5), dtype=np.uint16))

    result = Indexer(GEOMETRY).index(path, start=(100, 200), group=(2, 3))

    assert result.start == (100, 200)
    assert result.group == (2, 3)
    roi = result.to_step().detector.roi
    assert (roi.startx, roi.endx, roi.groupx) == (100, 109, 2)
    assert (roi.starty, roi.endy, roi.groupy) == (200, 211, 3)


def test_roi_coordinate_conversions_and_inclusive_end():
    roi_pixels = np.array([[0.0, 0.0], [4.0, 3.0]])
    detector_pixels = roi_to_detector_pixels(
        roi_pixels, start=(100, 200), group=(2, 3)
    )

    np.testing.assert_allclose(detector_pixels, [[100.5, 201.0], [108.5, 210.0]])
    np.testing.assert_allclose(
        detector_to_roi_pixels(detector_pixels, start=(100, 200), group=(2, 3)),
        roi_pixels,
    )
    assert roi_inclusive_end((4, 5), (100, 200), (2, 3)) == (109, 211)


def test_hdf5_requires_image_dataset(tmp_path):
    path = tmp_path / "empty.h5"
    with h5py.File(path, "w"):
        pass

    with pytest.raises(InputError, match="cannot read HDF5") as caught:
        Indexer(GEOMETRY).index(path)
    assert isinstance(caught.value.__cause__, KeyError)


def test_batch_order_and_image_retention():
    frames = [np.zeros((3, width), dtype=np.uint16) for width in (4, 5)]
    indexer = Indexer(GEOMETRY)

    retained = indexer.index_many(frames, keep_images=True)
    discarded = indexer.index_many(frames)

    assert [result.image_shape for result in retained] == [(3, 4), (3, 5)]
    assert all(result.image is frame for result, frame in zip(retained, frames))
    assert all(result.image is None for result in discarded)


def test_index_many_continues_after_blank_auto_threshold_frame():
    frames = [np.full((3, 4), value, dtype=np.uint16) for value in (1, 0, 2)]
    indexer = Indexer(GEOMETRY, peak_params=PeakParams(threshold=None))

    results = indexer.index_many(frames)

    assert len(results) == 3
    assert [result.n_peaks for result in results] == [0, 0, 0]
    assert np.isnan(results[1].threshold_used)
    assert [result.total_sum for result in results] == [12, 0, 24]


def _short_peak_frame():
    y, x = np.indices((2, 2))
    return 10 + (2000 * np.exp(-(x * x + y * y) / 0.4)).astype(np.uint16)


def test_short_peak_fit_raises_indexing_error():
    frame = _short_peak_frame()
    indexer = Indexer(
        GEOMETRY,
        peak_params=PeakParams(
            boxsize=2,
            min_size=1,
            min_separation=1,
            threshold=20,
            max_rfactor=100,
        ),
    )

    with pytest.raises(IndexingError, match="peak search failed: peak fitting failed"):
        indexer.index(frame)


def test_index_many_stops_on_first_failed_frame():
    good = np.zeros((2, 2), dtype=np.uint16)
    bad = _short_peak_frame()
    frames_started = 0

    def frames():
        nonlocal frames_started
        for frame in (good, bad, good):
            frames_started += 1
            yield frame

    indexer = Indexer(
        GEOMETRY,
        peak_params=PeakParams(
            boxsize=2,
            min_size=1,
            min_separation=1,
            threshold=20,
            max_rfactor=100,
        ),
    )

    with pytest.raises(IndexingError, match="peak fitting failed"):
        indexer.index_many(frames())
    assert frames_started == 2


class _FailingLibrary:
    def __init__(self, stage, status):
        self.real = get_library()
        self.stage = stage
        self.status = status
        self.free_calls = 0

    def __getattr__(self, name):
        return getattr(self.real, name)

    def laue_find_peaks_typed(self, pixels, pixel_type, nx, ny, params, result):
        if self.stage == "peak search":
            result.status = self.status
            result.message = b"injected failure"
            return self.status
        return self.real.laue_find_peaks_typed(pixels, pixel_type, nx, ny, params, result)

    def laue_pixels_to_q(self, geometry, detector_index, result):
        if self.stage == "pixel-to-q conversion":
            result.status = self.status
            result.message = b"injected failure"
            return self.status
        return self.real.laue_pixels_to_q(geometry, detector_index, result)

    def laue_frame_result_free(self, result):
        self.free_calls += 1
        self.real.laue_frame_result_free(result)


@pytest.mark.parametrize(
    ("status", "error"),
    [(1, InputError), (2, MemoryError), (3, IndexingError), (4, IndexingError)],
)
@pytest.mark.parametrize("stage", ["peak search", "pixel-to-q conversion"])
def test_native_failures_map_to_exceptions_and_release_results(
    monkeypatch, stage, status, error
):
    indexer = Indexer(GEOMETRY)
    library = _FailingLibrary(stage, status)
    monkeypatch.setattr(indexer_module, "get_library", lambda: library)

    with pytest.raises(error, match=f"{stage} failed: injected failure"):
        indexer.index(np.zeros((4, 5), dtype=np.uint16))

    assert library.free_calls == 1
