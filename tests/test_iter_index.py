# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Bounded incremental indexing: ordering, bounds, errors, cancellation, cleanup."""

import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import threading
import time

import h5py
import numpy as np
import pytest

from conftest import requires_liblaue

from lauelab.indexing import (
    EXPECTED_INPUT_ERRORS, FrameInput, FrameOutcome, FrameOutcomes, Indexer, InputError,
    PeakParams, WorkerError,
)

ROOT = Path(__file__).resolve().parents[1]
pytestmark = requires_liblaue
GEOMETRY = ROOT / "tests/data/geo/geoN_2022-03-29_14-15-05.xml"
CRYSTAL = ROOT / "tests/config/Ni.xml"
FRAMES = sorted((ROOT / "tests/data/synthetic/frames").glob("*.h5"))
TWO_GRAINS = ROOT / "tests/data/synthetic/frames/synthetic_ni_two_grains.h5"
PEAKS = PeakParams(
    boxsize=18, max_rfactor=0.5, min_size=3, min_separation=20,
    threshold=None, threshold_ratio=4.0, max_peaks=200,
)


def _indexer(**changes):
    values = dict(peak_params=PEAKS)
    values.update(changes)
    return Indexer(GEOMETRY, CRYSTAL, **values)


def _small_frame(seed=0):
    image = np.full((64, 64), 5, dtype=np.uint16)
    image[20 + seed % 3:24 + seed % 3, 30:34] = 500
    return image


def _large_image():
    with h5py.File(TWO_GRAINS) as source:
        return source["entry1/data/data"][...]


def _no_children():
    for child in multiprocessing.active_children():
        child.join(timeout=5)
    return multiprocessing.active_children() == []


def _same_result(left, right):
    np.testing.assert_array_equal(left.peaks, right.peaks)
    assert left.n_patterns == right.n_patterns
    for a, b in zip(left.patterns, right.patterns):
        np.testing.assert_array_equal(a.reciprocal, b.reciprocal)
        np.testing.assert_array_equal(a.pk_index, b.pk_index)
    assert left.depth == right.depth and left.start == right.start and left.group == right.group
    assert left.metadata == right.metadata
    assert left.to_step().indexing.Nindexed == right.to_step().indexing.Nindexed


# --- agreement and ordering -------------------------------------------------------


def test_parallel_outcomes_match_serial_results_in_input_order():
    indexer = _indexer()
    inputs = [FrameInput(path, input_id=path.stem) for path in FRAMES]

    serial = indexer.index_many(FRAMES)
    with indexer.iter_index(inputs, workers=2) as outcomes:
        parallel = list(outcomes)

    assert [o.input_index for o in parallel] == list(range(len(FRAMES)))
    assert [o.input_id for o in parallel] == [path.stem for path in FRAMES]
    assert all(o.ok and o.error is None and o.seconds > 0 for o in parallel)
    for expected, outcome in zip(serial, parallel):
        _same_result(expected, outcome.result)
        assert outcome.result.image is None
    assert outcomes.n_submitted == outcomes.n_yielded == len(FRAMES)
    assert outcomes.stopped is False and outcomes.n_cancelled == 0
    assert _no_children()


def test_plain_frames_and_arrays_are_accepted_and_ids_default_to_none():
    indexer = _indexer()
    inputs = [TWO_GRAINS, _large_image(), str(TWO_GRAINS)]

    with indexer.iter_index(inputs, workers=2) as outcomes:
        results = [o for o in outcomes]

    assert [o.input_id for o in results] == [None, None, None]
    np.testing.assert_array_equal(results[0].result.peaks, results[1].result.peaks)
    np.testing.assert_array_equal(results[0].result.peaks, results[2].result.peaks)
    assert results[0].result.input_image == str(TWO_GRAINS)
    assert results[1].result.input_image is None


def test_slow_first_input_holds_order_and_the_in_flight_bound():
    indexer = _indexer()
    inputs = [FrameInput(_large_image(), input_id="slow")] + [
        FrameInput(_small_frame(i), input_id=("fast", i)) for i in range(30)
    ]

    with indexer.iter_index(inputs, workers=2, max_in_flight=4) as outcomes:
        seen = [(o.input_index, o.input_id) for o in outcomes]
        # The first (slow) outcome is consumed before any later one can be.
        assert seen[0] == (0, "slow")

    assert seen == [(index, item.input_id) for index, item in enumerate(inputs)]
    assert outcomes.peak_in_flight == 4
    assert outcomes.n_yielded == 31


def test_zero_pattern_and_zero_peak_frames_are_successful_outcomes():
    indexer = _indexer()
    empty = np.zeros((32, 32), dtype=np.uint16)

    with indexer.iter_index([empty, FRAMES[0]], workers=1) as outcomes:
        results = list(outcomes)

    assert all(o.ok for o in results)
    assert results[0].result.n_peaks == 0 and results[0].result.n_patterns == 0
    assert isinstance(results[0], FrameOutcome)


# --- per-input values and shared configuration ------------------------------------


def test_per_input_depth_metadata_and_shared_mask_reach_the_worker():
    indexer = _indexer()
    image = _large_image()
    mask = np.zeros(image.shape, dtype=np.int32)
    mask[:512, :] = 1
    reference_masked = indexer.index(image, mask=mask, depth=40.0, keep_image=False)
    reference_plain = indexer.index(image, keep_image=False)
    inputs = [
        FrameInput(image, input_id="a", depth=40.0, metadata={"sample_name": "shared mask"}),
        FrameInput(image, input_id="b", depth=None),
    ]

    with indexer.iter_index(inputs, mask=mask, workers=2) as outcomes:
        first, second = list(outcomes)

    assert first.result.depth == 40.0 and second.result.depth is None
    assert first.result.metadata["sample_name"] == "shared mask"
    assert first.result.total_sum == reference_masked.total_sum < reference_plain.total_sum
    np.testing.assert_array_equal(first.result.peaks, reference_masked.peaks)


def test_keep_images_is_off_by_default_and_can_be_enabled():
    indexer = _indexer()

    with indexer.iter_index([FRAMES[0]], workers=1) as outcomes:
        (without,) = list(outcomes)
    with indexer.iter_index([FRAMES[0]], workers=1, keep_images=True) as outcomes:
        (with_image,) = list(outcomes)

    assert without.result.image is None
    assert with_image.result.image is not None and with_image.result.image.dtype == np.uint16


# --- expected input errors --------------------------------------------------------


def test_expected_input_errors_are_outcomes_and_iteration_continues(tmp_path):
    indexer = _indexer()
    wrong_detector = tmp_path / "wrong-detector.h5"
    with h5py.File(wrong_detector, "w") as output:
        output.create_dataset("entry1/data/data", data=np.zeros((4, 4), dtype=np.uint16))
        output.create_dataset("entry1/detector/ID", data=np.asarray([b"other"]))
    no_image = tmp_path / "no-image.h5"
    with h5py.File(no_image, "w") as output:
        output.create_dataset("entry1/other", data=1)
    inputs = [
        FrameInput(tmp_path / "missing.h5", input_id="missing"),
        FrameInput(np.zeros((4, 4), dtype=np.uint32), input_id="dtype"),
        FrameInput(wrong_detector, input_id="detector"),
        FrameInput(no_image, input_id="no-image"),
        FrameInput(_small_frame(), input_id="fine", start=(5000, 0)),
        FrameInput(FRAMES[0], input_id="ok"),
    ]

    with indexer.iter_index(inputs, workers=2) as outcomes:
        results = list(outcomes)

    assert [o.ok for o in results] == [False, False, False, False, False, True]
    assert isinstance(results[0].error, InputError) and "missing.h5" in str(results[0].error)
    assert isinstance(results[1].error, InputError) and "dtype" in str(results[1].error)
    assert isinstance(results[2].error, InputError) and "detector" in str(results[2].error)
    assert isinstance(results[3].error, InputError) and "cannot read HDF5" in str(results[3].error)
    assert isinstance(results[4].error, InputError) and "bounds" in str(results[4].error)
    assert all(isinstance(o.error, EXPECTED_INPUT_ERRORS) for o in results[:5])
    assert all(o.result is None for o in results[:5])
    assert results[5].result.n_patterns == 0
    assert outcomes.n_yielded == 6


# --- failures that stop the iteration ---------------------------------------------


def test_unexpected_worker_exception_raises_worker_error_and_cleans_up():
    indexer = _indexer()
    inputs = [FrameInput(FRAMES[0]), FrameInput(_small_frame(), metadata=5), FrameInput(FRAMES[0])]

    with pytest.raises(WorkerError, match="failed on input 1: TypeError"):
        with indexer.iter_index(inputs, workers=1) as outcomes:
            for _ in outcomes:
                pass

    assert _no_children()


def test_worker_initialization_failure_is_reported_as_worker_error(tmp_path):
    geometry = tmp_path / "geometry.xml"
    shutil.copy(GEOMETRY, geometry)
    indexer = Indexer(geometry, CRYSTAL, peak_params=PEAKS)
    geometry.unlink()

    with pytest.raises(WorkerError, match="initialization failed"):
        with indexer.iter_index([FRAMES[0]], workers=1) as outcomes:
            list(outcomes)

    assert _no_children()


def test_killed_worker_breaks_the_pool_into_worker_error():
    indexer = _indexer()
    image = _large_image()
    inputs = [FrameInput(image, input_id=i) for i in range(6)]

    with pytest.raises(WorkerError, match="broken"):
        with indexer.iter_index(inputs, workers=2, max_in_flight=4) as outcomes:
            next(outcomes)
            for child in multiprocessing.active_children():
                os.kill(child.pid, signal.SIGKILL)
            for _ in outcomes:
                pass

    assert _no_children()


def test_worker_death_after_the_head_completed_is_still_a_worker_error():
    indexer = _indexer()
    inputs = [FrameInput(_small_frame(i), input_id=i) for i in range(12)]

    with pytest.raises(WorkerError, match="broken"):
        with indexer.iter_index(inputs, workers=2, max_in_flight=2) as outcomes:
            next(outcomes)
            # Let the whole in-flight window finish, so the next step is a
            # submit() on a pool whose workers are gone.
            time.sleep(1.0)
            for child in multiprocessing.active_children():
                os.kill(child.pid, signal.SIGKILL)
            for child in multiprocessing.active_children():
                child.join(timeout=5)
            for _ in outcomes:
                pass

    assert _no_children()


# --- cooperative cancellation -----------------------------------------------------


def test_stop_before_admission_starts_nothing():
    indexer = _indexer()

    with indexer.iter_index([FRAMES[0]] * 5, workers=1, should_stop=lambda: True) as outcomes:
        results = list(outcomes)

    assert results == []
    assert outcomes.stopped is True
    assert outcomes.n_submitted == 0
    assert _no_children()


def test_stop_after_an_outcome_withdraws_unstarted_and_drains_running_work():
    indexer = _indexer()
    image = _large_image()
    inputs = [FrameInput(image, input_id=i) for i in range(12)]
    stop = threading.Event()

    with indexer.iter_index(
        inputs, workers=1, max_in_flight=4, should_stop=stop.is_set, poll_seconds=0.05
    ) as outcomes:
        results = []
        for outcome in outcomes:
            results.append(outcome)
            stop.set()

    assert outcomes.stopped is True
    assert [o.input_index for o in results] == list(range(len(results)))
    # One yielded, at most the prefetched call-queue items drained, the rest withdrawn.
    assert 1 <= len(results) <= 3
    # The window refills to max_in_flight right after the first outcome is taken,
    # before the consumer sets the stop flag.
    assert outcomes.n_submitted <= 1 + 4
    assert outcomes.n_cancelled >= 1
    assert outcomes.n_yielded + outcomes.n_cancelled == outcomes.n_submitted
    assert len(results) + outcomes.n_cancelled < len(inputs)
    assert _no_children()


def test_stop_is_observed_while_waiting_for_a_slow_head():
    indexer = _indexer()
    image = _large_image()
    inputs = [FrameInput(image, input_id=i) for i in range(12)]
    stop = threading.Event()
    threading.Timer(0.2, stop.set).start()
    started = time.perf_counter()

    with indexer.iter_index(
        inputs, workers=1, max_in_flight=6, should_stop=stop.is_set, poll_seconds=0.05
    ) as outcomes:
        results = list(outcomes)
    elapsed = time.perf_counter() - started

    assert outcomes.stopped is True
    assert 1 <= len(results) <= 2
    assert outcomes.n_cancelled >= 4
    single = results[0].seconds
    # The stop took effect during the head's computation, not after the window drained.
    assert elapsed < 3.5 * single + 2.0
    assert _no_children()


def test_should_stop_exception_propagates_and_cleans_up():
    indexer = _indexer()

    def broken():
        raise RuntimeError("stop check failed")

    with pytest.raises(RuntimeError, match="stop check failed"):
        with indexer.iter_index([FRAMES[0]] * 3, workers=1, should_stop=broken) as outcomes:
            list(outcomes)
    assert _no_children()


# --- lifecycle -----------------------------------------------------------------------


def test_consumer_exception_inside_with_shuts_workers_down():
    indexer = _indexer()

    with pytest.raises(ValueError, match="writer failed"):
        with indexer.iter_index([FrameInput(_large_image())] * 4, workers=2) as outcomes:
            for _ in outcomes:
                raise ValueError("writer failed")

    assert _no_children()


def test_breaking_out_of_the_loop_shuts_workers_down():
    indexer = _indexer()

    with indexer.iter_index([FrameInput(_large_image())] * 4, workers=2) as outcomes:
        for _ in outcomes:
            break

    assert outcomes.n_yielded == 1
    assert _no_children()
    assert list(outcomes) == []


def test_completed_iteration_without_with_closes_workers():
    indexer = _indexer()

    results = [o for o in indexer.iter_index([FRAMES[0], FRAMES[1]], workers=1)]

    assert len(results) == 2 and all(o.ok for o in results)
    assert _no_children()


def test_close_is_idempotent_and_ends_iteration():
    indexer = _indexer()
    outcomes = indexer.iter_index([FRAMES[0]] * 3, workers=1)
    assert isinstance(outcomes, FrameOutcomes)

    with outcomes:
        first = next(outcomes)
        outcomes.close()
        outcomes.close()
        assert list(outcomes) == []

    assert first.ok
    assert _no_children()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"workers": 0}, "workers"),
        ({"workers": 2.0}, "workers"),
        ({"workers": 2, "max_in_flight": 1}, "max_in_flight"),
        ({"max_in_flight": 1.5}, "max_in_flight"),
        ({"should_stop": 5}, "should_stop"),
        ({"poll_seconds": 0}, "poll_seconds"),
        ({"mask": np.zeros((2, 2, 2))}, "mask"),
    ],
)
def test_invalid_iteration_settings_are_rejected_before_starting_workers(kwargs, message):
    with pytest.raises(InputError, match=message):
        _indexer().iter_index([FRAMES[0]], **kwargs)
    assert _no_children()


def test_default_in_flight_window_is_twice_the_workers():
    indexer = _indexer()

    with indexer.iter_index([_small_frame(i) for i in range(20)], workers=3) as outcomes:
        list(outcomes)

    assert outcomes.peak_in_flight == 6


@pytest.mark.parametrize("error_type", [ValueError, KeyError, OSError])
def test_programming_errors_are_fatal_worker_failures(monkeypatch, error_type):
    from lauelab.indexing import incremental

    class BrokenIndexer:
        def index(self, *args, **kwargs):
            raise error_type("injected internal failure")

    monkeypatch.setattr(incremental, "_worker_indexer", BrokenIndexer())
    monkeypatch.setattr(incremental, "_worker_init_failure", None)
    payload = incremental._index_task((0, _small_frame(), (0, 0), (1, 1), None, None, False))
    with _indexer().iter_index([]) as outcomes:
        with pytest.raises(WorkerError, match="injected internal failure"):
            outcomes._to_outcome(payload)


@pytest.mark.parametrize("status", [3, 4])
def test_native_numerical_and_internal_failures_have_distinct_outcomes(monkeypatch, status):
    import pickle
    from lauelab.indexing import incremental, NumericalIndexingError
    from lauelab.indexing.indexer import _raise_native_error

    class FailingIndexer:
        def index(self, *args, **kwargs):
            _raise_native_error(status, "indexing", "injected native failure")

    monkeypatch.setattr(incremental, "_worker_indexer", FailingIndexer())
    monkeypatch.setattr(incremental, "_worker_init_failure", None)
    payload = incremental._index_task((0, _small_frame(), (0, 0), (1, 1), None, None, False))
    payload = pickle.loads(pickle.dumps(payload))  # crosses a process boundary
    with _indexer().iter_index([]) as outcomes:
        if status == 3:
            assert isinstance(outcomes._to_outcome(payload).error, NumericalIndexingError)
        else:
            with pytest.raises(WorkerError, match="injected native failure"):
                outcomes._to_outcome(payload)


def test_malformed_hdf5_metadata_does_not_stop_iteration(tmp_path):
    path = tmp_path / "bad-roi.h5"
    with h5py.File(path, "w") as output:
        output.create_dataset("entry1/data/data", data=_small_frame())
        output.create_dataset("entry1/detector/startx", data="not an integer")
    with _indexer().iter_index([path, _small_frame()]) as outcomes:
        bad, good = list(outcomes)
    assert isinstance(bad.error, InputError)
    assert "startx" in str(bad.error)
    assert good.ok
