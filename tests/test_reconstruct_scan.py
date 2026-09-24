# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Tests for scan coordination: local worker processes and external schedulers."""

import dataclasses
import multiprocessing
import os
from pathlib import Path
import pickle
import re
import signal
import subprocess
import sys
import textwrap
import time

import h5py
import numpy as np
import pytest

from conftest import requires_liblaue
from lauelab.indexing import InputError, ReconstructionError
from lauelab.reconstruct import (
    PointOutcome, PointReader, PointTask, ScanReader, prepare_scan, reconstruct_point, reconstruct_scan,
    validate_scan_file,
)
from lauelab.reconstruct import scan as scan_module
from tests.data.reconstruction.generate_reference import (
    DEPTH_RANGE_UM, GEOMETRY_FILE, write_input_file,
)

pytestmark = requires_liblaue
LIBRARY = Path(__file__).resolve().parents[1]
# About 3 s per point with one thread, so that a point is still running when
# a test stops or interrupts the scan.
SLOW = dict(depth_range=(-200.0, 200.0), resolution=0.25, output_pixel_type=3, rows_per_stripe=1)


def _sources(directory, count, prefix="p"):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory / f"{prefix}{index}.h5" for index in range(count)]
    for path in paths:
        write_input_file(path)
    return paths


def _scan(paths, directory, **changes):
    options = dict(geometry=GEOMETRY_FILE, detector=0, depth_range=DEPTH_RANGE_UM,
                   threads_per_worker=1)
    options.update(changes)
    return reconstruct_scan(paths, directory, **options)


def _statuses(path):
    return [entry.status for entry in ScanReader(path).points]


def _datasets(path):
    values = {}
    with h5py.File(path, "r") as file:
        file.visititems(lambda name, node: values.__setitem__(name, node[()])
                        if isinstance(node, h5py.Dataset) else None)
        attrs = {key: value for key, value in file.attrs.items() if key != "created"}
    return values, attrs


def _assert_same_files(left, right):
    (left_values, left_attrs), (right_values, right_attrs) = _datasets(left), _datasets(right)
    assert left_attrs == right_attrs
    assert left_values.keys() == right_values.keys()
    for name in left_values:
        if name == "entry1/reconstruction/date":
            continue  # Each reconstruction has its own processing timestamp.
        np.testing.assert_array_equal(left_values[name], right_values[name], err_msg=name)


# --- Equivalent execution paths ----------------------------------------------

def test_sequential_pool_and_external_coordination_write_equivalent_scans(tmp_path):
    inputs = [*_sources(tmp_path / "in", 3), tmp_path / "in/missing.h5"]
    sequential = _scan(inputs, tmp_path / "sequential", workers=1)
    pooled = _scan(inputs, tmp_path / "pooled", workers=3)

    # An external coordinator dispatches everything and records in reverse.
    with prepare_scan(inputs, tmp_path / "external", geometry=GEOMETRY_FILE, detector=0,
                      depth_range=DEPTH_RANGE_UM) as scan:
        for task in scan.tasks:
            scan.record_dispatch(task)
        outcomes = [reconstruct_point(task, num_threads=1) for task in reversed(scan.tasks)]
        for outcome in outcomes:
            scan.record(outcome)
        external = scan.finish()

    for result in (sequential, pooled, external):
        assert not result.cancelled and not result.complete
        assert [(outcome.index, outcome.status) for outcome in result.outcomes] == [
            (0, "complete"), (1, "complete"), (2, "complete"), (3, "failed"),
        ]
        assert validate_scan_file(result.path).run_status == "finished"
    for result in (pooled, external):
        _assert_same_files(sequential.path, result.path)
        for name in ("p0.h5", "p1.h5", "p2.h5"):
            _assert_same_files(sequential.path.parent / "points" / name,
                               result.path.parent / "points" / name)
    assert sorted(path.name for path in (tmp_path / "pooled/points").iterdir()) == [
        "p0.h5", "p1.h5", "p2.h5",
    ]


def test_local_runs_submit_exactly_the_prepared_tasks_and_no_pixels(tmp_path, monkeypatch):
    inputs = _sources(tmp_path / "in", 3)
    submitted = []
    real = scan_module.ProcessPoolExecutor.submit

    def recording(self, function, *args):
        submitted.append((function, args))
        return real(self, function, *args)

    monkeypatch.setattr(scan_module.ProcessPoolExecutor, "submit", recording)
    result = _scan(inputs, tmp_path / "local", workers=2, threads_per_worker=3)
    monkeypatch.undo()
    with prepare_scan(inputs, tmp_path / "external", geometry=GEOMETRY_FILE, detector=0,
                      depth_range=DEPTH_RANGE_UM) as scan:
        prepared = scan.tasks

    def relative(task, directory):
        return dataclasses.replace(task, output=os.path.relpath(task.output, directory))

    assert [function for function, _ in submitted] == [scan_module._run_task] * 3
    assert [args[1] for _, args in submitted] == [3, 3, 3]
    assert [relative(args[0], tmp_path / "local") for _, args in submitted] == [
        relative(task, tmp_path / "external") for task in prepared
    ]
    assert all(type(args[0]) is PointTask for _, args in submitted)
    assert all(len(pickle.dumps(outcome)) < 1024 for outcome in result.outcomes)


def test_the_coordinator_cannot_be_sent_to_a_worker(tmp_path):
    with prepare_scan(_sources(tmp_path / "in", 1), tmp_path / "scan", geometry=GEOMETRY_FILE,
                      detector=0, depth_range=DEPTH_RANGE_UM) as scan:
        with pytest.raises(TypeError, match="send its tasks"):
            pickle.dumps(scan)


# --- External coordination ---------------------------------------------------

def test_finish_requires_settled_dispatches_and_explicit_cancellation(tmp_path):
    inputs = _sources(tmp_path / "in", 3)
    with prepare_scan(inputs, tmp_path / "scan", geometry=GEOMETRY_FILE, detector=0,
                      depth_range=DEPTH_RANGE_UM) as scan:
        first, second, _ = scan.tasks
        scan.record_dispatch(first)
        scan.snapshot(force=True)
        assert _statuses(scan.path) == ["writing", "pending", "pending"]
        with pytest.raises(InputError, match="already writing"):
            scan.record_dispatch(first)
        with pytest.raises(InputError, match="differs from the prepared task"):
            scan.record_dispatch(dataclasses.replace(second, n_depths=2))
        with pytest.raises(InputError, match="1 dispatched point"):
            scan.finish()
        scan.record(reconstruct_point(first, num_threads=1))
        with pytest.raises(InputError, match="2 point.* were never dispatched"):
            scan.finish()
        result = scan.finish(cancelled=True)
        with pytest.raises(InputError, match="the scan is cancelled"):
            scan.record_dispatch(second)
    assert result.cancelled
    assert [outcome.status for outcome in result.outcomes] == ["complete", "unattempted", "unattempted"]
    assert validate_scan_file(scan.path).run_status == "cancelled"


def test_a_point_published_but_not_recorded_stays_usable_and_is_not_claimed(tmp_path):
    inputs = _sources(tmp_path / "in", 2)
    with prepare_scan(inputs, tmp_path / "scan", geometry=GEOMETRY_FILE, detector=0,
                      depth_range=DEPTH_RANGE_UM) as scan:
        scan.record_dispatch(scan.tasks[0])
        outcome = reconstruct_point(scan.tasks[0], num_threads=1)
        # The coordinator stops before it records the published point.
    assert _statuses(scan.path) == ["interrupted", "unattempted"]
    assert validate_scan_file(scan.path).run_status == "failed"
    with ScanReader(scan.path) as catalog, pytest.raises(InputError, match="is interrupted"):
        catalog.point("p0")
    with PointReader(outcome.output) as point:
        assert point.point_id == "p0" and point.frame(25).any()


# --- Catalog snapshots -------------------------------------------------------

def test_snapshots_batch_changes_and_skip_unchanged_catalogs(tmp_path, monkeypatch):
    clock = [10.0]
    monkeypatch.setattr(scan_module, "perf_counter", lambda: clock[0])
    writes = []
    real_write = scan_module.write_catalog

    def write(*args, **kwargs):
        real_write(*args, **kwargs)
        writes.append(clock[0])

    monkeypatch.setattr(scan_module, "write_catalog", write)
    with prepare_scan(_sources(tmp_path / "in", 3), tmp_path / "scan",
                      geometry=GEOMETRY_FILE, detector=0, depth_range=DEPTH_RANGE_UM) as scan:
        first, second, _ = scan.tasks
        scan.record_dispatch(first)
        scan.record_dispatch(second)
        scan.record(PointOutcome(first.index, first.point_id, "failed", "input unavailable"))
        assert _statuses(scan.path) == ["pending", "pending", "pending"]
        # Duplicate dispatch checks use live memory, even before publication.
        with pytest.raises(InputError, match="already writing"):
            scan.record_dispatch(second)
        clock[0] = 14.99
        assert not scan.snapshot()
        clock[0] = 15.0
        assert scan.snapshot()
        assert _statuses(scan.path) == ["failed", "writing", "pending"]
        clock[0] = 100.0
        assert not scan.snapshot()
        assert not scan.snapshot(force=True)
        assert writes == [10.0, 15.0]
    assert writes == [10.0, 15.0, 100.0]
    assert validate_scan_file(scan.path).run_status == "failed"
    assert _statuses(scan.path) == ["failed", "interrupted", "unattempted"]


@pytest.mark.parametrize("cancelled", [False, True])
def test_finalization_publishes_without_waiting_for_snapshot_interval(tmp_path, monkeypatch, cancelled):
    monkeypatch.setattr(scan_module, "perf_counter", lambda: 10.0)
    with prepare_scan(_sources(tmp_path / "in", 2), tmp_path / "scan",
                      geometry=GEOMETRY_FILE, detector=0, depth_range=DEPTH_RANGE_UM) as scan:
        tasks = scan.tasks[:1] if cancelled else scan.tasks
        for task in tasks:
            scan.record_dispatch(task)
            scan.record(PointOutcome(task.index, task.point_id, "failed", "input unavailable"))
        assert _statuses(scan.path) == ["pending", "pending"]
        result = scan.finish(cancelled=cancelled)
        assert _statuses(scan.path) == [outcome.status for outcome in result.outcomes]
        assert validate_scan_file(scan.path).run_status == ("cancelled" if cancelled else "finished")
        assert not scan.snapshot(force=True)


def test_failed_snapshot_keeps_pending_changes_and_last_published_file(tmp_path, monkeypatch):
    monkeypatch.setattr(scan_module, "perf_counter", lambda: 10.0)
    with prepare_scan(_sources(tmp_path / "in", 1), tmp_path / "scan",
                      geometry=GEOMETRY_FILE, detector=0, depth_range=DEPTH_RANGE_UM) as scan:
        scan.record_dispatch(scan.tasks[0])
        real_write = scan_module.write_catalog

        def fail(*args, **kwargs):
            raise OSError("catalog storage is full")

        monkeypatch.setattr(scan_module, "write_catalog", fail)
        with pytest.raises(OSError, match="catalog storage is full"):
            scan.snapshot(force=True)
        assert _statuses(scan.path) == ["pending"]
        monkeypatch.setattr(scan_module, "write_catalog", real_write)
        assert scan.snapshot(force=True)
        assert _statuses(scan.path) == ["writing"]


def test_local_scheduler_snapshots_while_waiting_for_workers(tmp_path, monkeypatch):
    monkeypatch.setattr(scan_module, "_SNAPSHOT_SECONDS", 0.1)
    observed = []
    catalog = tmp_path / "scan/scan.h5"

    def should_stop():
        statuses = _statuses(catalog)
        observed.append(statuses)
        return "writing" in statuses

    result = _scan(_sources(tmp_path / "in", 2), catalog.parent,
                   workers=1, should_stop=should_stop, **SLOW)
    assert any("writing" in statuses for statuses in observed)
    assert result.cancelled
    assert [outcome.status for outcome in result.outcomes] == ["complete", "unattempted"]


# --- Local runs --------------------------------------------------------------

def test_invalid_scheduling_arguments_are_rejected_before_writing(tmp_path):
    inputs = _sources(tmp_path / "in", 1)
    for changes, message in [({"workers": 0}, "workers"), ({"workers": True}, "workers"),
                             ({"workers": 1.5}, "workers"), ({"threads_per_worker": 0}, "threads"),
                             ({"progress": 3}, "progress"), ({"should_stop": 3}, "should_stop"),
                             ({"num_threads": 2}, "num_threads")]:
        with pytest.raises(InputError, match=message):
            _scan(inputs, tmp_path / "scan", **changes)
        assert not (tmp_path / "scan").exists()


def test_progress_reports_inspection_failures_and_each_recorded_point(tmp_path):
    inputs = [tmp_path / "in/gone.h5", *_sources(tmp_path / "in", 2)]
    seen = []
    result = _scan(inputs, tmp_path / "scan", workers=2,
                   progress=lambda outcome: seen.append((outcome.point_id, outcome.status)))
    assert seen[0] == ("gone", "failed")
    assert sorted(seen[1:]) == [("p0", "complete"), ("p1", "complete")]
    assert [outcome.point_id for outcome in result.outcomes] == ["gone", "p0", "p1"]
    assert "input file does not exist" in result.outcomes[0].error


def test_a_point_that_fails_during_reconstruction_does_not_stop_the_run(tmp_path):
    good, bad = _sources(tmp_path / "in", 2)
    # Metadata stays readable, so the point passes inspection; its frames do not.
    raw = tmp_path / "in/frames.raw"
    with h5py.File(bad, "r+") as file:
        images = file["entry1/data/data"][...]
        del file["entry1/data/data"]
        file["entry1/data"].create_dataset("data", shape=images.shape, dtype=images.dtype,
                                           external=[(os.fspath(raw), 0, images.nbytes)])
    result = _scan([bad, good], tmp_path / "scan", workers=2)
    assert [outcome.status for outcome in result.outcomes] == ["failed", "complete"]
    assert result.outcomes[0].error
    assert sorted(path.name for path in (tmp_path / "scan/points").iterdir()) == ["p0.h5"]


def test_a_stop_request_before_any_work_leaves_every_point_unattempted(tmp_path):
    inputs = [*_sources(tmp_path / "in", 2), tmp_path / "in/gone.h5"]
    result = _scan(inputs, tmp_path / "scan", workers=2, should_stop=lambda: True)
    assert result.cancelled
    assert [outcome.status for outcome in result.outcomes] == ["unattempted", "unattempted", "failed"]
    assert validate_scan_file(result.path).run_status == "cancelled"
    assert not any((tmp_path / "scan/points").iterdir())


def test_a_stop_request_during_work_lets_running_points_finish(tmp_path):
    inputs = _sources(tmp_path / "in", 5)
    stop = {"now": False}

    def progress(outcome):
        stop["now"] = True

    result = _scan(inputs, tmp_path / "scan", workers=2, progress=progress,
                   should_stop=lambda: stop["now"])
    statuses = [outcome.status for outcome in result.outcomes]
    assert result.cancelled and statuses[0] == "complete"
    assert set(statuses) == {"complete", "unattempted"}
    assert "unattempted" in statuses
    published = sorted(path.name for path in (tmp_path / "scan/points").iterdir())
    assert published == [f"p{index}.h5" for index, status in enumerate(statuses) if status == "complete"]
    assert validate_scan_file(result.path).run_status == "cancelled"


def test_an_output_failure_stops_admission_and_fails_the_run(tmp_path):
    inputs = _sources(tmp_path / "in", 4)
    points = tmp_path / "scan/points"

    def progress(outcome):
        points.chmod(0o555)

    try:
        with pytest.raises(ReconstructionError, match="creating the private file of"):
            _scan(inputs, tmp_path / "scan", workers=1, progress=progress)
    finally:
        points.chmod(0o755)
    assert _statuses(tmp_path / "scan/scan.h5") == ["complete", "interrupted", "unattempted", "unattempted"]
    assert validate_scan_file(tmp_path / "scan/scan.h5").run_status == "failed"
    with PointReader(points / "p0.h5") as point:
        assert point.frame(25).any()


def test_a_lost_worker_fails_the_run_and_keeps_completed_points(tmp_path):
    inputs = _sources(tmp_path / "in", 6)

    def progress(outcome):
        if outcome.index == 0 or outcome.status == "complete":
            for child in multiprocessing.active_children():
                child.kill()

    with pytest.raises(ReconstructionError, match="worker process ended unexpectedly"):
        _scan(inputs, tmp_path / "scan", workers=2, progress=progress)
    statuses = _statuses(tmp_path / "scan/scan.h5")
    assert validate_scan_file(tmp_path / "scan/scan.h5").run_status == "failed"
    assert "complete" in statuses and not {"pending", "writing"} & set(statuses)
    for index, status in enumerate(statuses):
        if status == "complete":
            with PointReader(tmp_path / f"scan/points/p{index}.h5") as point:
                assert point.frame(25).any()
    assert not multiprocessing.active_children()


def test_a_catalog_failure_propagates_and_keeps_the_last_snapshot(tmp_path, monkeypatch):
    inputs = _sources(tmp_path / "in", 3)
    # Make every change due so this test can fail a specific publication.
    monkeypatch.setattr(scan_module, "_SNAPSHOT_SECONDS", 0.0)
    real = scan_module.write_catalog
    calls = {"n": 0}

    def fail_after_four(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] > 4:
            raise OSError("catalog storage is full")
        real(*args, **kwargs)

    monkeypatch.setattr(scan_module, "write_catalog", fail_after_four)
    with pytest.raises(OSError, match="catalog storage is full"):
        _scan(inputs, tmp_path / "scan", workers=1)
    monkeypatch.undo()
    # Initial, dispatch 0, complete 0, dispatch 1 were published; recording
    # point 1 failed and so did recording the failed run.
    summary = validate_scan_file(tmp_path / "scan/scan.h5")
    assert summary.run_status == "running"
    assert _statuses(tmp_path / "scan/scan.h5") == ["complete", "writing", "pending"]
    assert not multiprocessing.active_children()


# --- Interrupts --------------------------------------------------------------

SCRIPT = """
import importlib.machinery, sys, types
sys.path.insert(0, {library!r})
tests = types.ModuleType("tests"); tests.__path__ = [{tests!r}]
tests.__spec__ = importlib.machinery.ModuleSpec("tests", loader=None, is_package=True)
tests.__spec__.submodule_search_locations = tests.__path__
sys.modules["tests"] = tests
from lauelab.reconstruct import reconstruct_scan
from lauelab.reconstruct import scan as scan_module
# Exercise interrupts without waiting five seconds for dispatch visibility.
scan_module._SNAPSHOT_SECONDS = 0.05
from tests.data.reconstruction.generate_reference import GEOMETRY_FILE

def run():
    return reconstruct_scan({inputs!r}, {output!r}, geometry=GEOMETRY_FILE, detector=0,
                            workers=2, threads_per_worker=1, **{slow!r})
"""


def _script(inputs, output):
    return SCRIPT.format(library=os.fspath(LIBRARY), tests=os.fspath(LIBRARY / "tests"),
                         inputs=[os.fspath(path) for path in inputs], output=os.fspath(output),
                         slow=SLOW)


def _wait_for_running_points(catalog, count, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            if _statuses(catalog).count("writing") >= count:
                return
        except (OSError, ValueError):
            pass
        time.sleep(0.05)
    raise AssertionError(f"{count} points never started")


@pytest.mark.parametrize("interrupts", [1, 2])
def test_interrupts_in_a_script_drain_then_abandon_running_points(tmp_path, interrupts):
    inputs = _sources(tmp_path / "in", 4)
    output = tmp_path / "scan"
    script = tmp_path / "run.py"
    script.write_text(_script(inputs, output) + textwrap.dedent("""
        if __name__ == "__main__":
            result = run()
            print("RESULT", result.cancelled, [outcome.status for outcome in result.outcomes])
    """))
    process = subprocess.Popen([sys.executable, os.fspath(script)], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        _wait_for_running_points(output / "scan.h5", 2)
        # A terminal Ctrl-C reaches the whole foreground process group.
        os.killpg(process.pid, signal.SIGINT)
        if interrupts == 2:
            time.sleep(0.5)
            os.killpg(process.pid, signal.SIGINT)
        stdout, stderr = process.communicate(timeout=120)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
    assert "running point(s) will finish" in stderr
    if interrupts == 1:
        assert process.returncode == 0, stderr
        assert "RESULT True ['complete', 'complete', 'unattempted', 'unattempted']" in stdout
        assert validate_scan_file(output / "scan.h5").run_status == "cancelled"
    else:
        assert process.returncode != 0 and "KeyboardInterrupt" in stderr
        assert _statuses(output / "scan.h5") == ["interrupted", "interrupted",
                                                 "unattempted", "unattempted"]
        assert validate_scan_file(output / "scan.h5").run_status == "failed"
        assert not any(path.suffix == ".h5" for path in (output / "points").iterdir())


def test_a_first_interrupt_in_a_notebook_kernel_drains_running_points(tmp_path):
    jupyter_client = pytest.importorskip("jupyter_client")
    inputs = _sources(tmp_path / "in", 4)
    output = tmp_path / "scan"
    manager = jupyter_client.KernelManager(kernel_name="python3")
    manager.start_kernel(cwd=os.fspath(tmp_path))
    client = manager.client()
    client.start_channels()
    try:
        client.wait_for_ready(timeout=60)
        client.execute_interactive(_script(inputs, output), timeout=60)
        request = client.execute("result = run()")
        _wait_for_running_points(output / "scan.h5", 2)
        manager.interrupt_kernel()
        reply = client.get_shell_msg(timeout=120)
        while reply["parent_header"].get("msg_id") != request:
            reply = client.get_shell_msg(timeout=120)
        assert reply["content"]["status"] == "ok", reply["content"]
        lines = []
        client.execute_interactive(
            "print(result.cancelled, [outcome.status for outcome in result.outcomes])",
            output_hook=lambda message: lines.append(message["content"].get("text", "")),
            timeout=60,
        )
    finally:
        client.stop_channels()
        manager.shutdown_kernel(now=True)
    assert "True ['complete', 'complete', 'unattempted', 'unattempted']" in "".join(lines)
    assert validate_scan_file(output / "scan.h5").run_status == "cancelled"


# --- Guide -------------------------------------------------------------------

def _guide_blocks(first_heading, last_heading):
    text = (LIBRARY / "docs/guides/reconstruction.md").read_text()
    section = text[text.index(first_heading):text.index(last_heading)]
    return re.findall(r"```python\n(.*?)```", section, flags=re.S)


def test_guide_scan_examples_run_as_documented(tmp_path, monkeypatch, capsys):
    """Every code block of the scan sections of docs/guides/reconstruction.md."""
    for index in (1, 2, 3):
        write_input_file(tmp_path / f"point_{index}.h5")
    (tmp_path / "tests").symlink_to(LIBRARY / "tests")
    monkeypatch.chdir(tmp_path)
    blocks = _guide_blocks("## Reconstruct a scan\n", "## Handle failures")
    assert len(blocks) == 7
    namespace = {}
    for block in blocks:
        exec(compile(block, "docs/guides/reconstruction.md", "exec"), namespace)

    assert namespace["failed"] == []
    assert [outcome.point_id for outcome in namespace["result"].outcomes] == [
        "point_1", "point_2", "point_3",
    ]
    assert validate_scan_file("run/scan.h5").run_status == "finished"
    assert validate_scan_file("scheduled/scan.h5").run_status == "finished"
    assert namespace["outcome"].status == "complete"
    assert Path("single/point_1.h5").is_file() and Path("run/indexed.h5").is_file()
    assert len(namespace["paths"]) == 52
    printed = capsys.readouterr().out
    assert "scan12_p1 complete" in printed and "scan12_p1 complete (51, 128, 128)" in printed
