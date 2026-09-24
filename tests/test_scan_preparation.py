# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Tests for scan preparation: tasks, destinations, and the initial catalog."""

import dataclasses
import json
import os
from pathlib import Path
import pickle

import h5py
import numpy as np
import pytest

from conftest import requires_liblaue
from lauelab.indexing import InputError, InvalidScanFile
from lauelab.reconstruct import (
    PointTask, PreparedScan, ScanReader, prepare_scan, validate_scan_file,
)
from lauelab.reconstruct import _scan_layout as layout
from lauelab.reconstruct.scan import _point_task
from tests.data.reconstruction.generate_reference import (
    DEPTH_RANGE_UM, GEOMETRY_FILE, write_input_file,
)

pytestmark = requires_liblaue


def _input(path: Path, **sample) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_input_file(path)
    with h5py.File(path, "r+") as source:
        for name, value in sample.items():
            source[f"entry1/sample/{name}"][...] = value
    return path


def _prepare(paths, directory, **changes):
    options = dict(geometry=GEOMETRY_FILE, detector=0, depth_range=DEPTH_RANGE_UM)
    options.update(changes)
    return prepare_scan(paths, directory, **options)


def _catalog(path):
    with h5py.File(path, "r") as source:
        return {name: source[name][...] for name in layout.CATALOG_DATASETS}


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    """A three-point scan: readable, missing, and readable with the same content."""
    work = tmp_path_factory.mktemp("prepared")
    first = _input(work / "in/a/Twin2_wire_1.h5", sampleX=1.5, sampleY=-2.0, sampleZ=3.25)
    second = _input(work / "in/b/Twin2_wire_2.h5")
    scan = _prepare([first, work / "in/Twin2_wire_9.h5", second], work / "out",
                    wire_edge="both", percent_brightest=50.0)
    yield scan, work
    scan.close()


# --- Initial catalog ---------------------------------------------------------

def test_initial_catalog_lists_every_input_in_order_before_any_point_runs(prepared):
    scan, work = prepared
    summary = validate_scan_file(scan.path)
    assert (summary.run_status, summary.n_points, summary.n_pending, summary.n_failed) == (
        "running", 3, 2, 1,
    )
    entries = ScanReader(scan.path).points
    assert [entry.point_id for entry in entries] == ["Twin2_wire_1", "Twin2_wire_9", "Twin2_wire_2"]
    assert [entry.status for entry in entries] == ["pending", "failed", "pending"]
    assert [entry.path for entry in entries] == [
        "points/Twin2_wire_1.h5", "points/Twin2_wire_9.h5", "points/Twin2_wire_2.h5",
    ]
    assert "input file does not exist" in entries[1].error
    assert entries[1].shape == (0, 0, 0) and entries[1].dtype is None
    assert sorted(path.name for path in scan.directory.iterdir()) == ["points", "scan.h5"]
    assert not any((scan.directory / "points").iterdir())


def test_catalog_records_provenance_acquisition_and_output_description(prepared):
    scan, work = prepared
    catalog = _catalog(scan.path)
    source = work / "in/a/Twin2_wire_1.h5"
    status = os.stat(source)
    assert catalog["/catalog/source_paths"][0].decode() == os.fspath(source)
    assert catalog["/catalog/source_sizes"][0] == status.st_size
    assert catalog["/catalog/source_mtimes_ns"][0] == status.st_mtime_ns
    assert catalog["/catalog/source_sizes"][1] == -1
    with h5py.File(source, "r") as file:
        n_stored = file["entry1/data/data"].shape[0]
    assert catalog["/catalog/raw_slices"].tolist() == [[1, n_stored - 1], [-1, -1], [1, n_stored - 1]]
    assert catalog["/catalog/scan_numbers"].tolist() == [1, -1, 1]
    np.testing.assert_array_equal(catalog["/catalog/sample_positions"][0], [1.5, -2.0, 3.25])
    assert np.isnan(catalog["/catalog/sample_positions"][1]).all()
    assert catalog["/catalog/energies_kev"][0] == 20.0
    assert catalog["/catalog/image_shapes"].tolist() == [[128, 128], [0, 0], [128, 128]]
    assert catalog["/catalog/n_depths"].tolist() == [51, 0, 51]
    assert catalog["/catalog/depth_bounds"][0].tolist() == [-25.0, 25.0]
    # wire_edge="both" stores int32 unless a type is requested.
    assert catalog["/catalog/pixel_types"].tolist() == [1, -1, 1]


def test_catalog_embeds_settings_geometry_and_versions(prepared):
    scan, _ = prepared
    with h5py.File(scan.path, "r") as source:
        assert source.attrs["format"] == layout.SCAN_FORMAT
        assert source.attrs["version"] == layout.SCAN_VERSION
        assert source["settings/wire_edge"][()].decode() == "both"
        assert source["settings/percent_brightest"][()] == 50.0
        assert source["settings/output_pixel_type"][()] == -1
        assert source["settings/rows_per_stripe"][()] == -1
        assert source["settings/depth_range"][...].tolist() == list(DEPTH_RANGE_UM)
        assert source["geometry/xml"][()].decode() == Path(GEOMETRY_FILE).read_text()
        assert source["run/native_version"][()].decode()
        assert source.attrs["lauelab_version"]


def test_inspection_reads_metadata_but_no_frames(tmp_path):
    source = _input(tmp_path / "in/external.h5")
    raw = tmp_path / "in/frames.raw"
    with h5py.File(source, "r+") as file:
        images = file["entry1/data/data"][...]
        del file["entry1/data/data"]
        file["entry1/data"].create_dataset(
            "data", shape=images.shape, dtype=images.dtype,
            external=[(os.fspath(raw), 0, images.nbytes)],
        )[...] = images
    raw.unlink()
    with h5py.File(source, "r") as file, pytest.raises(OSError):
        file["entry1/data/data"][1]

    with _prepare([source], tmp_path / "out") as scan:
        assert ScanReader(scan.path).points[0].status == "pending"
    assert len(scan.tasks) == 1


@pytest.mark.parametrize(("field", "value"), [
    ("entry1/detector/Nx", np.array([np.nan])),
    ("entry1/detector/Nx", np.array([b"invalid"])),
    ("entry1/sample/sampleX", np.array([b"invalid"])),
    ("entry1/sample/incident_energy", np.array([b"invalid"])),
    ("entry1/scanNum", np.array([2**31], dtype=np.int64)),
    ("entry1/scanNum", np.array([-2], dtype=np.int32)),
])
def test_malformed_metadata_is_a_failed_point_not_a_failed_scan(tmp_path, field, value):
    good = _input(tmp_path / "in/good.h5")
    bad = _input(tmp_path / "in/bad.h5")
    with h5py.File(bad, "r+") as file:
        del file[field]
        file[field] = value
    with _prepare([good, bad], tmp_path / "out") as scan:
        assert [task.point_id for task in scan.tasks] == ["good"]
        entries = ScanReader(scan.path).points
    assert [entry.status for entry in entries] == ["pending", "failed"]
    assert "invalid metadata" in entries[1].error


# --- Tasks -------------------------------------------------------------------

def test_tasks_cover_the_readable_points_in_manifest_order(prepared):
    scan, work = prepared
    assert [(task.index, task.point_id) for task in scan.tasks] == [
        (0, "Twin2_wire_1"), (2, "Twin2_wire_2"),
    ]
    first = scan.tasks[0]
    assert first.source == os.fspath(work / "in/a/Twin2_wire_1.h5")
    assert first.output == os.fspath(scan.directory / "points/Twin2_wire_1.h5")
    assert first.geometry_xml == Path(GEOMETRY_FILE).read_text()
    assert first.settings["wire_edge"] == "both" and first.settings["percent_brightest"] == 50.0
    assert (first.image_shape, first.n_depths, first.depth_bounds_um, first.pixel_type) == (
        (128, 128), 51, (-25.0, 25.0), 1,
    )


def test_tasks_survive_pickle_and_json_unchanged(prepared):
    scan, _ = prepared
    for task in scan.tasks:
        assert pickle.loads(pickle.dumps(task)) == task
        values = json.loads(json.dumps(dataclasses.asdict(task)))
        assert PointTask(**values) == task


def test_tasks_hold_only_plain_values(prepared):
    scan, _ = prepared
    plain = (str, int, float, bool, type(None), tuple, dict)

    def check(value):
        assert isinstance(value, plain), type(value)
        if isinstance(value, dict):
            for item in value.values():
                check(item)
        elif isinstance(value, tuple):
            for item in value:
                check(item)

    for task in scan.tasks:
        for field in dataclasses.fields(task):
            check(getattr(task, field.name))


def test_standalone_and_scan_preparation_build_the_same_task(prepared):
    scan, work = prepared
    for task in scan.tasks:
        standalone = _point_task(
            task.source, task.output, geometry=GEOMETRY_FILE, detector=0,
            depth_range=DEPTH_RANGE_UM, wire_edge="both", percent_brightest=50.0,
        )
        assert dataclasses.replace(standalone, index=task.index) == task


def test_task_rejects_settings_that_are_not_reconstructor_options(prepared):
    scan, _ = prepared
    values = dataclasses.asdict(scan.tasks[0])
    values["settings"] = {**values["settings"], "num_threads": 4}
    with pytest.raises(ValueError, match="settings must have exactly the keys"):
        PointTask(**values)


def test_point_ids_are_independent_of_filenames(tmp_path):
    paths = [_input(tmp_path / f"in/{name}.h5") for name in ("x", "y")]
    with _prepare(paths, tmp_path / "out", point_ids=["7", "8"]) as scan:
        assert [(task.point_id, Path(task.output).name) for task in scan.tasks] == [
            ("7", "x.h5"), ("8", "y.h5"),
        ]


# --- Rejection before any work -----------------------------------------------

@pytest.mark.parametrize(("names", "message"), [
    (["a/p.h5", "b/p.h5"], "would both write points/p.h5"),
    (["a/P.h5", "b/p.h5"], "ignoring case"),
    (["a/p.h5", "a/p.h5"], "would both write"),
    (["a/.p.h5"], "starts with '.'"),
    (["a/p\x01.h5"], "control character"),
])
def test_unsafe_or_colliding_point_files_are_rejected_before_writing(tmp_path, names, message):
    paths = [_input(tmp_path / "in" / name) for name in names]
    with pytest.raises(InputError, match=message):
        _prepare(paths, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize(("changes", "error", "message"), [
    ({"depth_range": (5.0, -5.0)}, InputError, "depth_range"),
    ({"detector": 7}, InputError, "detector"),
    ({"num_threads": 2}, InputError, "num_threads"),
    ({"compression": "lz4"}, InputError, "compression"),
    ({"point_ids": ["a", "a"]}, InputError, "point_ids"),
    ({"point_ids": ["a", ""]}, InputError, "point_ids"),
    ({"point_ids": ["a"]}, InputError, "one entry for each path"),
])
def test_invalid_shared_configuration_is_rejected_before_writing(tmp_path, changes, error, message):
    paths = [_input(tmp_path / f"in/{name}.h5") for name in ("x", "y")]
    with pytest.raises(error, match=message):
        _prepare(paths, tmp_path / "out", **changes)
    assert not (tmp_path / "out").exists()


def test_empty_input_list_is_rejected(tmp_path):
    with pytest.raises(InputError, match="at least one input"):
        _prepare([], tmp_path / "out")


def test_nonempty_or_non_directory_destinations_are_rejected(tmp_path):
    source = _input(tmp_path / "in/x.h5")
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "notes.txt").write_text("keep")
    with pytest.raises(FileExistsError, match="is not empty"):
        _prepare([source], occupied)
    assert [path.name for path in occupied.iterdir()] == ["notes.txt"]
    with pytest.raises(FileExistsError, match="not a directory"):
        _prepare([source], occupied / "notes.txt")

    empty = tmp_path / "empty"
    empty.mkdir()
    with _prepare([source], empty) as scan:
        assert scan.path == empty / "scan.h5"


# --- Coordinator lifetime ----------------------------------------------------

def test_leaving_the_context_without_finishing_records_a_failed_run(tmp_path):
    source = _input(tmp_path / "in/x.h5")
    with _prepare([source], tmp_path / "out") as scan:
        assert isinstance(scan, PreparedScan)
        assert validate_scan_file(scan.path).run_status == "running"
    assert validate_scan_file(scan.path).run_status == "failed"
    scan.close()
    assert sorted(path.name for path in scan.directory.iterdir()) == ["points", "scan.h5"]


def test_a_reader_keeps_the_snapshot_it_opened(tmp_path):
    source = _input(tmp_path / "in/x.h5")
    scan = _prepare([source], tmp_path / "out")
    before = ScanReader(scan.path)
    with h5py.File(scan.path, "r") as held:
        scan.close()
        assert held["run/status"][()] == layout.RunStatus.RUNNING
    assert before.run_status == "running"
    assert ScanReader(scan.path).run_status == "failed"
    assert before.point_path("x") == scan.directory / "points/x.h5"


# --- Unsupported files -------------------------------------------------------

def _marked_file(path, **attrs):
    with h5py.File(path, "w") as file:
        file.attrs.update(attrs)
    return path


def test_retired_single_file_scans_are_named_and_refused(tmp_path):
    old = _marked_file(tmp_path / "reconstruction.h5", format=layout.SCAN_FORMAT, version=1)
    for read in (validate_scan_file, ScanReader):
        with pytest.raises(InvalidScanFile, match="retired single-file reconstruction-scan layout"):
            read(old)


@pytest.mark.parametrize(("attrs", "message"), [
    ({"format": "lauelab-results", "version": 1}, "not a 'lauelab-reconstruction-scan' file"),
    ({"format": layout.SCAN_FORMAT, "version": 3}, "unsupported"),
    ({"format": layout.POINT_FORMAT, "version": 1}, "not a 'lauelab-reconstruction-scan' file"),
])
def test_other_files_are_not_catalogs(tmp_path, attrs, message):
    other = _marked_file(tmp_path / "other.h5", **attrs)
    with pytest.raises(InvalidScanFile, match=message):
        ScanReader(other)


# --- Validation --------------------------------------------------------------

def _damage(path, name, value):
    with h5py.File(path, "r+") as file:
        del file[name]
        file[name] = value


@pytest.mark.parametrize(("name", "value", "message"), [
    ("catalog/point_ids", np.array(["a", "a"], dtype=object), "point IDs"),
    ("catalog/point_paths", np.array(["points/a.h5", "../b.h5"], dtype=object), "not a point-file path"),
    ("catalog/point_paths", np.array(["points/a.h5", "points/.b.h5"], dtype=object), "not a point-file path"),
    ("catalog/point_paths", np.array(["points/a.h5", "points/A.h5"], dtype=object), "unique, ignoring case"),
    ("catalog/status", np.array([0, 9], dtype="u1"), "unknown"),
    ("catalog/status", np.array([3, 0], dtype="u1"), "no error text"),
    ("catalog/n_depths", np.array([51, 0], dtype="<i4"), "only a failed point"),
    ("catalog/pixel_types", np.array([4, 3], dtype="<i4"), "pixel type 4"),
    ("catalog/depth_bounds", np.array([[25.0, -25.0], [-25.0, 25.0]]), "finite and ordered"),
    ("catalog/errors", np.array([b"x", b"y"], dtype="S8"), "dtype"),
    ("catalog/source_sizes", np.array([1, 2, 3], dtype="<i8"), "shape"),
    ("settings/detector", np.array(0.0), "dtype"),
])
def test_validator_rejects_structural_damage(tmp_path, name, value, message):
    paths = [_input(tmp_path / f"in/{stem}.h5") for stem in ("a", "b")]
    with _prepare(paths, tmp_path / "out") as scan:
        pass
    _damage(scan.path, name, value)
    with pytest.raises(InvalidScanFile, match=message):
        validate_scan_file(scan.path)


def test_a_settled_run_cannot_hold_pending_points(tmp_path):
    source = _input(tmp_path / "in/x.h5")
    with _prepare([source], tmp_path / "out") as scan:
        # The published running snapshot lists the point as pending.
        _damage(scan.path, "run/status", np.array(layout.RunStatus.FINISHED, dtype="u1"))
        with pytest.raises(InvalidScanFile, match="finished run has unsettled"):
            validate_scan_file(scan.path)
