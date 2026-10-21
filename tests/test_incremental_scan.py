"""Incremental task admission, publication, and durable receipt recovery."""

from dataclasses import asdict, replace
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
import pytest

from conftest import requires_liblaue
from lauelab.indexing import InputError, InvalidScanFile
from lauelab.reconstruct import (
    PointOutcome, PointReader, PreparedScan, ScanReader, prepare_scan,
    reconstruct_point, reconstruct_scan, validate_scan_file,
)
from lauelab.reconstruct import scan as scan_module
from tests.data.reconstruction.generate_reference import (
    DEPTH_RANGE_UM, GEOMETRY_FILE, write_input_file,
)

pytestmark = requires_liblaue


def source(directory, name="point.h5"):
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    write_input_file(path)
    return path


def prepare(directory, paths=(), **changes):
    options = dict(geometry=GEOMETRY_FILE, detector=0, depth_range=DEPTH_RANGE_UM)
    options.update(changes)
    return prepare_scan(paths, directory, **options)


def run(scan, task):
    scan.record_dispatch(task)
    outcome = reconstruct_point(task, num_threads=1)
    assert outcome.status == "complete", outcome.error
    scan.record(outcome)
    return outcome


def checkpoint(scan):
    return json.loads(json.dumps(scan.export_state(), allow_nan=False))


def test_delayed_out_of_order_arrivals_publish_without_mutating_points(tmp_path):
    first = source(tmp_path, "point10.h5")
    second = source(tmp_path, "point9.h5")
    with prepare(tmp_path / "out") as scan:
        old_reader = ScanReader(scan.path)
        before = scan.path.read_bytes()
        ten = scan.add(first, point_id="10")
        assert scan.path.read_bytes() == before  # Admission is not publication.
        assert (ten.point_id, ten.index) == ("10", 0)
        prior_tasks, prior_outcomes = scan.tasks, scan.outcomes
        run(scan, ten)
        scan.snapshot(force=True)
        closed_bytes = Path(ten.output).read_bytes()
        nine = scan.add(second, point_id="9")
        assert (nine.point_id, nine.index) == ("9", 1)
        scan.snapshot(force=True)
        assert [v.status for v in ScanReader(scan.path).points] == ["complete", "pending"]
        assert old_reader.point_ids == ()
        assert len(prior_tasks) == 1 and prior_outcomes[0].status == "pending"
        with ScanReader(scan.path) as reader, reader.point("10") as point:
            assert point.manifest_index == 0
        run(scan, nine)
        assert Path(ten.output).read_bytes() == closed_bytes
        assert scan.finish().complete
        old_reader.close()


@pytest.mark.parametrize("case", ["id", "source", "casefold", "invalid_id", "missing", "bad_input", "output"])
def test_failed_admission_is_atomic_and_does_not_consume_indices(tmp_path, case):
    first = source(tmp_path / "raw", "one.h5")
    next_path = source(tmp_path / "raw", "two.h5")
    with prepare(tmp_path / "out") as scan:
        scan.add(first, point_id="one")
        kwargs, candidate = {}, next_path
        if case == "id":
            kwargs = {"point_id": "one"}
        elif case == "source":
            candidate = first
        elif case == "casefold":
            candidate = source(tmp_path / "elsewhere", "ONE.h5")
        elif case == "invalid_id":
            kwargs = {"point_id": ""}
        elif case == "missing":
            candidate = tmp_path / "absent.h5"
        elif case == "bad_input":
            candidate = tmp_path / "invalid.h5"
            candidate.write_text("not HDF5")
        elif case == "output":
            (scan.directory / "points/two.h5.partial").write_text("keep")
        before = checkpoint(scan)
        with pytest.raises((InputError, OSError)):
            scan.add(candidate, **kwargs)
        assert checkpoint(scan) == before
        task = scan.add(source(tmp_path / "raw", "three.h5"))
        assert task.index == 1
        scan.finish(cancelled=True)


def test_batch_failures_keep_their_indices_when_inputs_are_added(tmp_path):
    first, third = source(tmp_path, "first.h5"), source(tmp_path, "third.h5")
    with prepare(tmp_path / "out", [first, tmp_path / "missing.h5"]) as scan:
        task = scan.add(third)
        assert task.index == 2
        assert [v.status for v in scan.outcomes] == ["pending", "failed", "pending"]
        assert [v.index for v in scan.tasks] == [0, 2]
        restored = PreparedScan.restore(checkpoint(scan))
        assert restored.outcomes == scan.outcomes
        restored.finish(cancelled=True)


def test_geometry_is_frozen_for_later_admissions(tmp_path):
    geometry = tmp_path / "geometry.xml"
    shutil.copyfile(GEOMETRY_FILE, geometry)
    with prepare(tmp_path / "out", geometry=geometry) as scan:
        xml = geometry.read_text()
        geometry.unlink()
        task = scan.add(source(tmp_path))
        assert task.geometry_xml == xml and task.geometry_path == str(geometry)
        run(scan, task)
        scan.finish()


def test_admission_and_restore_do_not_read_detector_pixels(tmp_path):
    path = source(tmp_path)
    raw = tmp_path / "external.raw"
    with h5py.File(path, "r+") as f:
        shape, dtype = f["entry1/data/data"].shape, f["entry1/data/data"].dtype
        del f["entry1/data/data"]
        f["entry1/data"].create_dataset("data", shape=shape, dtype=dtype,
                                       external=[(str(raw), 0, int(np.prod(shape) * dtype.itemsize))])
    assert not raw.exists()
    with prepare(tmp_path / "out") as scan:
        scan.add(path)
        state = checkpoint(scan)
        path.unlink()
        restored = PreparedScan.restore(state)
        assert restored.tasks == scan.tasks
        restored.finish(cancelled=True)


def test_batch_and_incremental_scientific_outputs_are_identical(tmp_path):
    paths = [source(tmp_path / "raw", f"p{i}.h5") for i in range(2)]
    batch = reconstruct_scan(paths, tmp_path / "batch", geometry=GEOMETRY_FILE,
                             detector=0, depth_range=DEPTH_RANGE_UM, threads_per_worker=1)
    assert batch.complete
    with prepare(tmp_path / "live") as scan:
        tasks = [scan.add(path) for path in paths]
        for task in reversed(tasks):
            run(scan, task)
        scan.finish()
    for task in tasks:
        with h5py.File(task.output, "r") as live, h5py.File(tmp_path / "batch/points" / Path(task.output).name, "r") as old:
            for name in ["data/data", "depth", "reconstruction/depth_intensity",
                         "reconstruction/computed_depth_intensity", "reconstruction/sum_raw",
                         "reconstruction/sum_reconstructed", "reconstruction/first_raw"]:
                np.testing.assert_array_equal(live["entry1/" + name][...], old["entry1/" + name][...])


def test_checkpoint_is_detached_from_tasks_and_coordinator(tmp_path):
    with prepare(tmp_path / "out") as scan:
        task = scan.add(source(tmp_path))
        state = checkpoint(scan)
        state["configuration"]["settings"]["depth_range"][0] = 100
        state["points"][0]["task"]["settings"]["depth_range"][0] = 200
        task.settings["wire_edge"] = "trailing"
        clean = checkpoint(scan)
        assert clean["configuration"]["settings"]["depth_range"] == list(DEPTH_RANGE_UM)
        assert clean["points"][0]["task"]["settings"]["wire_edge"] == "leading"
        scan.finish(cancelled=True)


def test_restore_replays_durable_receipts_without_reading_raw_or_geometry_or_writing(tmp_path):
    raw = source(tmp_path)
    geometry = tmp_path / "geometry.xml"
    shutil.copyfile(GEOMETRY_FILE, geometry)
    scan = prepare(tmp_path / "out", geometry=geometry)
    task = scan.add(raw)
    state = checkpoint(scan)
    scan.record_dispatch(task)
    receipt = reconstruct_point(task, num_threads=1)
    raw.unlink(); geometry.unlink()
    before = {p.name: p.read_bytes() for p in scan.directory.iterdir() if p.is_file()}
    restored = PreparedScan.restore(state, outcomes=[receipt, receipt])
    assert restored.outcomes == (receipt,)
    assert {p.name: p.read_bytes() for p in scan.directory.iterdir() if p.is_file()} == before
    assert restored.snapshot(force=True)
    assert restored.finish().complete


def test_restore_allows_a_saved_completion_to_replay_exactly(tmp_path):
    scan = prepare(tmp_path / "out")
    task = scan.add(source(tmp_path))
    receipt = run(scan, task)
    state = checkpoint(scan)
    restored = PreparedScan.restore(state, outcomes=[receipt])
    assert restored.outcomes == scan.outcomes
    with pytest.raises(InputError, match="conflicting"):
        PreparedScan.restore(state, outcomes=[replace(receipt, seconds=receipt.seconds + 1)])
    restored.finish()


@pytest.mark.parametrize("case", ["version", "native", "package", "unknown", "index", "shape", "path", "settings", "nan"])
def test_malformed_or_incompatible_state_is_rejected_without_writing(tmp_path, case):
    with prepare(tmp_path / "out") as scan:
        scan.add(source(tmp_path))
        state = checkpoint(scan)
        if case == "version": state["version"] = True
        elif case == "native": state["native_version"] = "other"
        elif case == "package": state["lauelab_version"] = "other"
        elif case == "unknown": state["surprise"] = 1
        elif case == "index": state["points"][0]["task"]["index"] = True
        elif case == "shape": state["points"][0]["task"]["image_shape"] = [0, 128]
        elif case == "path": state["points"][0]["output"] = str(tmp_path / "elsewhere.h5")
        elif case == "settings": state["points"][0]["task"]["settings"]["wire_edge"] = "trailing"
        elif case == "nan": state["points"][0]["outcome"]["seconds"] = float("nan")
        before = scan.path.read_bytes()
        with pytest.raises(InputError): PreparedScan.restore(state)
        assert scan.path.read_bytes() == before
        scan.finish(cancelled=True)


@pytest.mark.parametrize("case", ["ahead", "orphan", "missing", "terminal", "identity"])
def test_restore_preserves_unreconciled_or_unavailable_outputs(tmp_path, case):
    scan = prepare(tmp_path / "out")
    task = scan.add(source(tmp_path))
    state = checkpoint(scan)
    receipt = reconstruct_point(task, num_threads=1)
    if case != "orphan": scan.record(receipt)
    if case == "ahead": scan.snapshot(force=True)
    elif case == "missing":
        state = checkpoint(scan)
        Path(task.output).unlink()
    elif case == "terminal": scan.finish()
    elif case == "identity":
        state = checkpoint(scan)
        state["created"] = "2000-01-01T00:00:00+00:00"
    before = scan.path.read_bytes()
    with pytest.raises((InputError, InvalidScanFile, OSError)):
        PreparedScan.restore(state)
    assert scan.path.read_bytes() == before
    if case != "missing": assert Path(task.output).exists()


@pytest.mark.parametrize("name", ["sampleX", "incident_energy", "scanNum"])
def test_acceptance_checks_acquisition_metadata(tmp_path, name):
    with prepare(tmp_path / "out") as scan:
        task = scan.add(source(tmp_path))
        outcome = reconstruct_point(task, num_threads=1)
        path = "entry1/scanNum" if name == "scanNum" else f"entry1/sample/{name}"
        with h5py.File(task.output, "r+") as f: f[path][...] += 1
        with pytest.raises(InvalidScanFile, match="disagrees"):
            scan.record(outcome)
        assert scan.outcomes[0].status == "pending"
        scan.finish(cancelled=True)


def test_snapshot_failure_preserves_previous_view_and_can_be_retried(tmp_path, monkeypatch):
    with prepare(tmp_path / "out") as scan:
        task = scan.add(source(tmp_path))
        scan.snapshot(force=True)
        prior = scan.path.read_bytes()
        outcome = reconstruct_point(task, num_threads=1)
        scan.record(outcome)
        writer = scan_module.write_catalog
        def fail(*args, **kwargs): raise OSError("publication unavailable")
        monkeypatch.setattr(scan_module, "write_catalog", fail)
        with pytest.raises(OSError): scan.snapshot(force=True)
        assert scan.path.read_bytes() == prior
        monkeypatch.setattr(scan_module, "write_catalog", writer)
        assert scan.snapshot(force=True)
        assert validate_scan_file(scan.path).n_complete == 1
        assert not scan.snapshot(force=True)
        scan.finish()


def test_empty_success_and_batch_execution_are_rejected_but_cancellation_is_valid(tmp_path):
    with prepare(tmp_path / "out") as scan:
        with pytest.raises(InputError, match="empty"): scan.finish()
        summary = validate_scan_file(scan.path)
        assert summary.run_status == "running" and not summary.complete
        assert not scan.finish(cancelled=True).complete
        with pytest.raises(InputError): scan.add(source(tmp_path))
    with pytest.raises(InputError, match="at least one input"):
        reconstruct_scan([], tmp_path / "batch", geometry=GEOMETRY_FILE,
                         detector=0, depth_range=DEPTH_RANGE_UM)
    assert not (tmp_path / "batch").exists()


@pytest.mark.parametrize("changes", [dict(index=True), dict(seconds=-1), dict(seconds=float("inf")),
                                     dict(error=42), dict(output="unexpected.h5")])
def test_invalid_receipts_do_not_change_coordinator_state(tmp_path, changes):
    with prepare(tmp_path / "out") as scan:
        task = scan.add(source(tmp_path))
        receipt = PointOutcome(task.index, task.point_id, "failed", "worker lost")
        before = checkpoint(scan)
        with pytest.raises(InputError):
            scan.record(replace(receipt, **changes))
        assert checkpoint(scan) == before
        scan.finish(cancelled=True)


def test_restoration_does_not_publish_even_when_snapshot_interval_elapsed(tmp_path, monkeypatch):
    scan = prepare(tmp_path / "out")
    task = scan.add(source(tmp_path))
    state = checkpoint(scan)
    receipt = reconstruct_point(task, num_threads=1)
    def forbid(*args, **kwargs):
        raise AssertionError("restore must never publish")
    monkeypatch.setattr(scan_module, "perf_counter", lambda: 1e20)
    monkeypatch.setattr(scan_module, "write_catalog", forbid)
    restored = PreparedScan.restore(state, outcomes=[receipt])
    assert restored.outcomes == (receipt,)


def test_completed_point_recovery_validates_metadata_without_reading_pixels(tmp_path):
    scan = prepare(tmp_path / "out")
    task = scan.add(source(tmp_path))
    state = checkpoint(scan)
    receipt = reconstruct_point(task, num_threads=1)
    with h5py.File(task.output, "r+") as f:
        shape, dtype = f["entry1/data/data"].shape, f["entry1/data/data"].dtype
        del f["entry1/data/data"]
        f["entry1/data"].create_dataset("data", shape=shape, dtype=dtype,
            external=[(str(tmp_path / "missing.raw"), 0, int(np.prod(shape) * dtype.itemsize))])
        f["entry1/data/data"].attrs["signal"] = 1
    restored = PreparedScan.restore(state, outcomes=[receipt])
    assert restored.finish().complete
