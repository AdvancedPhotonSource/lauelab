# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Tests for standalone point files: stored values, publication, and reading."""

import dataclasses
import json
import os
from pathlib import Path
import pickle
import shutil

import h5py
import numpy as np
import pytest

from conftest import requires_liblaue
from lauelab import partial_path
from lauelab.indexing import InputError, InvalidScanFile, ReconstructionError
from lauelab.reconstruct import (
    PointReader, Reconstructor, ScanReader, prepare_scan, reconstruct_point,
    validate_scan_file,
)
from lauelab.reconstruct import _scan_layout as layout
from lauelab.reconstruct import _scan_writer
from lauelab.reconstruct import scan as scan_module
from lauelab.reconstruct._scan_writer import store_stripe
from lauelab.reconstruct._writer import PIXEL_DTYPES
from tests.data.reconstruction.generate_reference import (
    DEPTH_RANGE_UM, GEOMETRY_FILE, VARIANTS, write_input_file,
)
from tests.data.reconstruction_contract import fixtures

pytestmark = requires_liblaue
REFERENCE_DIR = Path(__file__).parent / "data/reconstruction"


def _options(variant=None, **changes):
    options = VARIANTS.get(variant, {})
    values = dict(
        depth_range=DEPTH_RANGE_UM,
        wire_edge=options.get("wire_edge", "leading"),
        normalization=options.get("normalization"),
        norm_exponent=options.get("norm_exponent"),
        cosmic_filter=options.get("cosmic_filter", False),
        output_pixel_type=options.get("output_pixel_type", 5),
        rows_per_stripe=31,
    )
    values.update(changes)
    return values


def _source(directory, variant=None, name="synthetic.h5"):
    options = VARIANTS.get(variant, {})
    path = Path(directory) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    write_input_file(path, write_mA=options.get("write_mA", False),
                     write_microdiffraction=options.get("write_microdiffraction", False))
    return path


def _prepare(paths, directory, variant=None, **changes):
    keywords = {key: changes.pop(key) for key in ("point_ids", "compression") if key in changes}
    return prepare_scan(paths, directory, geometry=GEOMETRY_FILE, detector=0, **keywords,
                        **_options(variant, **changes))


def _point(directory, source, variant=None, **changes):
    """Reconstruct one input through a prepared scan; return the point file."""
    with _prepare([source], directory, variant, **changes) as scan:
        outcome = reconstruct_point(scan.tasks[0], num_threads=1)
        assert outcome.status == "complete", outcome.error
        scan.record(outcome)
    return Path(outcome.output)


def _standalone(source, output, variant=None, **changes):
    keywords = {key: changes.pop(key) for key in ("point_id", "compression") if key in changes}
    return reconstruct_point(source, output, geometry=GEOMETRY_FILE, detector=0, num_threads=1,
                             **keywords, **_options(variant, **changes))


# --- Stored values -----------------------------------------------------------

@pytest.mark.parametrize("variant", [None, *VARIANTS])
def test_stored_frames_equal_per_depth_files_and_goldens(tmp_path, variant):
    source = _source(tmp_path, variant)
    per_depth = Reconstructor(GEOMETRY_FILE, 0, num_threads=1, **_options(variant)).reconstruct(
        source, tmp_path / "depth" / "point_"
    )
    suffix = f"_{variant}" if variant else ""
    golden = np.load(REFERENCE_DIR / f"cpu_reference{suffix}.npz")
    tolerance = json.loads((REFERENCE_DIR / f"cpu_reference{suffix}.json").read_text())["comparison"]

    assert per_depth.success
    with PointReader(_point(tmp_path / "scan", source, variant)) as point:
        np.testing.assert_array_equal(point.depth_um, golden["depth_um"])
        assert point.dtype == golden["images"].dtype
        for index, path in enumerate(per_depth.output_files[:-1]):
            with h5py.File(path) as handle:
                np.testing.assert_array_equal(point.frame(index), handle["entry1/data/data"])
                assert point.depth_um[index] == handle["entry1/depth"][0]
        stored = point.region((0, 128, 0, 128))
        np.testing.assert_allclose(stored, golden["images"],
                                   rtol=tolerance["rtol"], atol=tolerance["atol"])
        np.testing.assert_array_equal(
            point.computed_depth_intensity(), per_depth.depth_intensity
        )


@pytest.mark.parametrize(("pixel_type", "rescale"), [(3, 255.0), (2, 128.0), (1, 32768.0)])
def test_rescaled_integer_reductions_equal_reopened_pixels(tmp_path, pixel_type, rescale):
    source = _source(tmp_path, "norm_exponent")
    unscaled = Reconstructor(GEOMETRY_FILE, 0, num_threads=1,
                             **_options("norm_exponent")).reconstruct(source, return_images=True)
    path = _point(tmp_path / "scan", source, "norm_exponent", output_pixel_type=pixel_type)
    with PointReader(path) as point:
        stored = np.concatenate([block for _, block in point.iter_blocks(10**9)])
        limits = np.iinfo(PIXEL_DTYPES[pixel_type])
        # The rescale is applied once, then HDF5 truncates and saturates.
        np.testing.assert_array_equal(
            stored, np.clip(np.trunc(unscaled.images * rescale), limits.min, limits.max)
        )
        assert point.norm_rescale == rescale
        assert point.norm_threshold is not None
        totals = point.depth_intensity()
        assert totals.dtype == np.int64
        np.testing.assert_array_equal(totals, stored.sum(axis=(1, 2), dtype=np.int64))
        np.testing.assert_array_equal(
            point.reference("sum_reconstructed"), stored.sum(axis=0, dtype=np.int64)
        )
        # The computed totals stay unscaled and are a different quantity.
        np.testing.assert_array_equal(point.computed_depth_intensity(), unscaled.depth_intensity)


def test_float_reductions_match_reopened_pixels_within_the_contract_bound(tmp_path):
    with PointReader(_point(tmp_path / "scan", _source(tmp_path), output_pixel_type=0)) as point:
        stored = point.region((0, 128, 0, 128))
        assert stored.dtype == np.float32
        totals = point.depth_intensity()
        assert totals.dtype == np.float64
        # Signed pixels cancel, so the bound is relative to the absolute sum.
        magnitude = np.abs(stored, dtype=np.float64)
        np.testing.assert_array_less(
            np.abs(totals - stored.sum(axis=(1, 2), dtype=np.float64)),
            1e-12 * magnitude.sum(axis=(1, 2)),
        )
        np.testing.assert_array_less(
            np.abs(point.reference("sum_reconstructed") - stored.sum(axis=0, dtype=np.float64)),
            1e-12 * magnitude.sum(axis=0) + np.finfo(np.float64).tiny,
        )


@pytest.mark.parametrize("name", sorted(fixtures.STORED))
def test_native_store_reproduces_the_hand_worked_values(name):
    case = fixtures.STORED[name]
    stored, depth_sums, pixel_sums = store_stripe(
        fixtures.COMPUTED, case["data"].dtype, case["rescale"], n_threads=2
    )
    assert stored.dtype == case["data"].dtype
    np.testing.assert_array_equal(stored, case["data"])
    np.testing.assert_array_equal(depth_sums, case["depth_intensity"])
    np.testing.assert_array_equal(pixel_sums, case["sum_reconstructed"])
    assert depth_sums.dtype == pixel_sums.dtype == np.int64


@pytest.mark.parametrize("dtype", sorted(fixtures.CONVERSION))
def test_native_store_truncates_and_saturates_like_a_dataset_write(dtype):
    computed, expected = zip(*fixtures.CONVERSION[dtype])
    stored, _, _ = store_stripe(np.array(computed)[None, None, :], dtype)
    np.testing.assert_array_equal(stored[0, 0], np.asarray(expected, dtype=dtype))


@pytest.mark.parametrize("dtype", sorted(PIXEL_DTYPES.values(), key=str))
def test_native_store_equals_a_dataset_write_for_finite_values_and_infinity(tmp_path, dtype):
    rng = np.random.default_rng(0)
    values = rng.normal(0.0, 3e4, (5, 7, 11))
    values.ravel()[::13] = np.inf
    values.ravel()[::17] = -np.inf
    values.ravel()[::19] = 1e300
    with h5py.File(tmp_path / "written.h5", "w") as target:
        target.create_dataset("data", shape=values.shape, dtype=dtype)[...] = values * 3.0
        written = target["data"][...]
    stored, _, _ = store_stripe(values, dtype, 3.0)
    np.testing.assert_array_equal(stored, written)


def test_native_store_stores_zero_for_nan_in_every_integer_type():
    values = np.array([[[np.nan, 1.5, -1.5]]])
    for dtype in PIXEL_DTYPES.values():
        stored, depth_sums, _ = store_stripe(values, dtype)
        if dtype.kind in "iu":
            assert stored[0, 0, 0] == 0
            assert depth_sums[0] == stored[0, 0].sum(dtype=np.int64)
        else:
            assert np.isnan(stored[0, 0, 0]) and np.isnan(depth_sums[0])


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32])
def test_native_store_sums_do_not_depend_on_thread_count(dtype):
    rng = np.random.default_rng(1)
    values = rng.normal(0.0, 100.0, (9, 40, 3000))
    one = store_stripe(values, dtype, 1.0, n_threads=1)
    many = store_stripe(values, dtype, 1.0, n_threads=7)
    for left, right in zip(one, many):
        np.testing.assert_array_equal(left, right)
    stored, depth_sums, pixel_sums = one
    magnitude = np.abs(stored, dtype=np.float64)
    if np.dtype(dtype).kind == "f":
        assert np.abs(depth_sums - stored.sum(axis=(1, 2), dtype=np.float64)).max() <= (
            1e-12 * magnitude.sum(axis=(1, 2))
        ).min()
        np.testing.assert_array_less(
            np.abs(pixel_sums - stored.sum(axis=0, dtype=np.float64)),
            1e-12 * magnitude.sum(axis=0) + np.finfo(np.float64).tiny,
        )
    else:
        np.testing.assert_array_equal(depth_sums, stored.sum(axis=(1, 2), dtype=np.int64))
        np.testing.assert_array_equal(pixel_sums, stored.sum(axis=0, dtype=np.int64))


def test_native_store_rejects_bad_shapes_and_thread_counts():
    with pytest.raises(ValueError, match="shape"):
        store_stripe(np.zeros((2, 2)), np.int16)
    with pytest.raises(ValueError, match="rejected"):
        store_stripe(np.zeros((1, 1, 1)), np.int16, n_threads=0)


def test_gzip_stores_the_same_pixels_in_a_smaller_file(tmp_path):
    source = _source(tmp_path)
    plain = _point(tmp_path / "plain", source, output_pixel_type=3)
    packed = _point(tmp_path / "packed", source, output_pixel_type=3, compression="gzip")
    assert packed.stat().st_size < plain.stat().st_size
    with PointReader(plain) as left, PointReader(packed) as right:
        np.testing.assert_array_equal(left.region((0, 128, 0, 128)), right.region((0, 128, 0, 128)))
        np.testing.assert_array_equal(left.depth_intensity(), right.depth_intensity())
    with h5py.File(packed) as handle:
        data = handle["entry1/data/data"]
        assert (data.compression, data.compression_opts, data.shuffle) == ("gzip", 1, True)
        assert data.chunks == (4, 128, 128)


def test_points_keep_their_own_dtype_and_equal_their_standalone_reconstruction(tmp_path):
    counts = _source(tmp_path / "in", name="counts.h5")
    scaled = _source(tmp_path / "in", name="scaled.h5")
    with h5py.File(scaled, "r+") as handle:
        raw = np.asarray(handle["entry1/data/data"], dtype=np.float32) * 2
        del handle["entry1/data/data"]
        handle["entry1/data"].create_dataset("data", data=raw)
    with _prepare([counts, scaled], tmp_path / "scan", output_pixel_type=None) as scan:
        for task in scan.tasks:
            scan.record(reconstruct_point(task, num_threads=1))
    with ScanReader(scan.path) as catalog:
        first, second = catalog.point("counts"), catalog.point("scaled")
        assert (first.dtype, second.dtype) == (np.uint16, np.float32)
        assert first.reference("sum_raw").dtype == np.int64
        assert second.reference("sum_raw").dtype == np.float64
        assert first.depth_intensity().dtype == np.int64
        assert second.depth_intensity().dtype == np.float64
        np.testing.assert_array_equal(second.reference("first_raw"), 2 * first.reference("first_raw"))
        for point, source in ((first, counts), (second, scaled)):
            alone = _standalone(source, tmp_path / f"alone_{source.stem}.h5", output_pixel_type=None)
            assert alone.status == "complete" and alone.index is None
            with PointReader(alone.output) as single:
                assert single.manifest_index is None and point.manifest_index is not None
                np.testing.assert_array_equal(point.region((0, 128, 0, 128)),
                                              single.region((0, 128, 0, 128)))
                np.testing.assert_array_equal(point.depth_intensity(), single.depth_intensity())


# --- Reference images --------------------------------------------------------

def test_raw_references_exclude_bookkeeping_slices_and_precede_processing(tmp_path):
    source = _source(tmp_path, "cosmic")
    with h5py.File(source, "r+") as handle:
        data = handle["entry1/data/data"]
        data[0] = 60000
        data[-1] = 50000
        handle["entry1"].create_dataset("mA", data=np.linspace(51.0, 153.0, data.shape[0]))
        raw = np.asarray(data)
    path = _point(tmp_path / "scan", source, "cosmic", normalization="mA", norm_exponent=0.5)
    with PointReader(path) as point:
        assert point.raw_slices == (1, len(raw) - 1)
        first = point.reference("first_raw")
        assert first.dtype == raw.dtype
        np.testing.assert_array_equal(first, raw[1])
        total = point.reference("sum_raw")
        assert total.dtype == np.int64
        np.testing.assert_array_equal(total, raw[1:-1].sum(axis=0, dtype=np.int64))
        with pytest.raises(InputError, match="reference image must be one of"):
            point.reference("sum")


def test_float_input_sums_raw_frames_in_float64(tmp_path):
    source = _source(tmp_path)
    with h5py.File(source, "r+") as handle:
        raw = np.asarray(handle["entry1/data/data"], dtype=np.float32) + 0.25
        del handle["entry1/data/data"]
        handle["entry1/data"].create_dataset("data", data=raw)
    with PointReader(_point(tmp_path / "scan", source)) as point:
        assert point.reference("first_raw").dtype == np.float32
        total = point.reference("sum_raw")
        assert total.dtype == np.float64
        np.testing.assert_allclose(total, raw[1:-1].sum(axis=0, dtype=np.float64), rtol=1e-12)


# --- Point file layout -------------------------------------------------------

@pytest.fixture(scope="module")
def published(tmp_path_factory):
    """A two-point scan whose points were recorded complete."""
    work = tmp_path_factory.mktemp("published")
    first = _source(work / "in/a", name="Twin2_wire_1.h5")
    second = _source(work / "in/b", name="Twin2_wire_2.h5")
    with h5py.File(first, "r+") as handle:
        handle["entry1/sample/sampleX"][...] = 1.5
    with _prepare([first, second], work / "scan", wire_edge="both",
                  output_pixel_type=None) as scan:
        for task in scan.tasks:
            scan.record(reconstruct_point(task, num_threads=2))
    return scan, work


def test_point_file_follows_the_layout_table(published):
    scan, work = published
    source = work / "in/a/Twin2_wire_1.h5"
    with h5py.File(scan.directory / "points/Twin2_wire_1.h5") as handle:
        assert handle.attrs["format"] == layout.POINT_FORMAT
        assert handle.attrs["version"] == layout.POINT_VERSION
        tables = {**layout.POINT_SETTINGS_DATASETS, **layout.POINT_DATASETS}
        found = {"/entry1/data/data", "/entry1/depth"}
        handle["entry1/reconstruction"].visititems(
            lambda name, node: found.add("/entry1/reconstruction/" + name)
            if isinstance(node, h5py.Dataset) and not name.endswith("/depth") else None
        )
        assert found == set(tables)
        for path, spec in tables.items():
            dataset = handle[path]
            assert dataset.attrs.get("units") == spec.units, path
            for name, value in spec.attrs.items():
                assert dataset.attrs[name] == value, path
        assert handle["entry1/data/data"].chunks == (4, 128, 128)
        assert "entry1/detector/Nx" in handle
        assert "entry1/wire" not in handle
        assert "source" not in handle
        assert handle["entry1/reconstruction/point/id"][()].decode() == "Twin2_wire_1"
        assert handle["entry1/reconstruction/point/manifest_index"][()] == 0
        assert handle["entry1/reconstruction/point/complete"][()] == 1
        assert handle["entry1/reconstruction/execution/num_threads"][()] == 2
        assert handle["entry1/reconstruction/execution/rows_per_stripe"][()] == 31
        assert handle["entry1/reconstruction/acquisition/source_path"][()].decode() == os.fspath(source)
        assert handle["entry1/reconstruction/acquisition/source_size"][()] == source.stat().st_size
        np.testing.assert_array_equal(handle["entry1/reconstruction/acquisition/sample_position"], [1.5, 0.0, 0.0])
        assert handle["entry1/reconstruction/geometry/xml"][()].decode() == GEOMETRY_FILE.read_text()
        assert handle["entry1/reconstruction/settings/wire_edge"][()].decode() == "both"
        assert handle["entry1/reconstruction/settings/output_pixel_type"][()] == -1
        assert handle["entry1/reconstruction/settings/rows_per_stripe"][()] == 31
    with h5py.File(scan.path) as catalog, h5py.File(scan.directory / "points/Twin2_wire_2.h5") as point:
        for path in layout.SETTINGS_DATASETS:
            np.testing.assert_array_equal(catalog[path][()], point["/entry1/reconstruction" + path][()], err_msg=path)


def test_recorded_points_are_complete_in_the_catalog(published):
    scan, _ = published
    summary = validate_scan_file(scan.path)
    # The fixture leaves the coordinator without finishing the run.
    assert (summary.run_status, summary.n_complete, summary.n_pending) == ("failed", 2, 0)
    assert sorted(path.name for path in (scan.directory / "points").iterdir()) == [
        "Twin2_wire_1.h5", "Twin2_wire_2.h5",
    ]


# --- Reading -----------------------------------------------------------------

def test_reader_returns_owned_arrays_for_exactly_the_selection(published):
    scan, _ = published
    with PointReader(scan.directory / "points/Twin2_wire_1.h5") as point:
        assert point.detector_id == "PE1621 723-3335"
        assert point.detector_size == (2048, 2048)
        assert point.start == (0, 0) and point.group == (16, 16)
        frame = point.frame(25)
        region = point.region((60, 70, 50, 54), slice(20, 30))
        assert frame.shape == (128, 128) and frame.flags.owndata
        assert region.shape == (10, 10, 4) and region.flags.owndata
        np.testing.assert_array_equal(region[5], frame[60:70, 50:54])
        frame[:] = 0
        assert point.frame(25).any()
        assert point.depth_um[25] == 0.0 and point.depth_um[0] == -25.0

        blocks = list(point.iter_blocks(10 * 128 * 128 * 4))
        assert [first for first, _ in blocks] == [0, 10, 20, 30, 40, 50]
        assert all(block.nbytes <= 10 * 128 * 128 * 4 for _, block in blocks)
        np.testing.assert_array_equal(np.concatenate([block for _, block in blocks])[25],
                                      point.frame(25))


def test_reader_rejects_selections_it_would_have_to_clip(published):
    scan, _ = published
    with PointReader(scan.directory / "points/Twin2_wire_1.h5") as point:
        for bounds in [(0, 129, 0, 1), (-1, 1, 0, 1), (5, 5, 0, 1), (0, 1, 3, 2),
                       (0, 1, 0), (0.0, 1, 0, 1), (True, 2, 0, 1)]:
            with pytest.raises(InputError, match="bounds"):
                point.region(bounds)
        with pytest.raises(InputError, match="step of 1"):
            point.region((0, 1, 0, 1), slice(0, 10, 2))
        for index in (-1, 51):
            with pytest.raises(IndexError):
                point.frame(index)
        with pytest.raises(TypeError):
            point.frame(1.0)
        with pytest.raises(InputError, match="at least one frame"):
            next(point.iter_blocks(100))


def test_a_copied_point_needs_no_catalog_raw_input_or_geometry_file(tmp_path):
    geometry = tmp_path / "geometry.xml"
    shutil.copy(GEOMETRY_FILE, geometry)
    source = _source(tmp_path / "in", name="Twin2_wire_1.h5")
    with prepare_scan([source], tmp_path / "scan", geometry=geometry, detector=0,
                      depth_range=DEPTH_RANGE_UM) as scan:
        scan.record(reconstruct_point(scan.tasks[0], num_threads=1))
    with PointReader(scan.directory / "points/Twin2_wire_1.h5") as original:
        expected = (original.region((0, 128, 0, 128)), original.reference("sum_raw"),
                    original.depth_intensity(), original.depth_um)
    copied = shutil.copy(scan.directory / "points/Twin2_wire_1.h5", tmp_path / "elsewhere.h5")
    shutil.rmtree(tmp_path / "scan")
    shutil.rmtree(tmp_path / "in")
    geometry.unlink()

    with PointReader(copied) as point:
        assert (point.point_id, point.manifest_index) == ("Twin2_wire_1", 0)
        assert point.source_path == os.fspath(source)
        assert (point.scan_number, point.energy_kev, point.sample_position) == (1, 20.0, (0.0, 0.0, 0.0))
        assert point.settings["depth_range"] == DEPTH_RANGE_UM
        assert point.settings["geometry_path"] == os.fspath(geometry)
        assert point.settings["norm_exponent"] is None and point.settings["cosmic_filter"] is False
        assert point.geometry_xml() == GEOMETRY_FILE.read_text()
        for value, read in zip(expected, (point.region((0, 128, 0, 128)), point.reference("sum_raw"),
                                          point.depth_intensity(), point.depth_um)):
            np.testing.assert_array_equal(read, value)


def test_scan_reader_opens_only_the_requested_point_and_closes_it(published, monkeypatch):
    scan, _ = published
    opened = []
    real = h5py.File

    def recording(path, *args, **kwargs):
        opened.append(Path(path).name)
        return real(path, *args, **kwargs)

    with ScanReader(scan.path) as catalog:
        monkeypatch.setattr(h5py, "File", recording)
        point = catalog.point("Twin2_wire_2")
        monkeypatch.undo()
        assert opened == ["Twin2_wire_2.h5"]
        assert point.frame(0).shape == (128, 128)
        with pytest.raises(KeyError):
            catalog.point("absent")
    with pytest.raises(Exception):
        point.frame(0)


def test_scan_reader_refuses_incomplete_points_and_mismatched_files(tmp_path):
    sources = [_source(tmp_path / "in", name=f"{name}.h5") for name in ("a", "b")]
    with _prepare([*sources, tmp_path / "in/gone.h5"], tmp_path / "scan") as scan:
        scan.record(reconstruct_point(scan.tasks[0], num_threads=1))
        with ScanReader(scan.path) as catalog, pytest.raises(InputError, match="'b' is pending"):
            catalog.point("b")
    with ScanReader(scan.path) as catalog:
        with pytest.raises(InputError, match="'b' is unattempted"):
            catalog.point("b")
        with pytest.raises(InputError, match="'gone' is failed: input file does not exist"):
            catalog.point("gone")
    # Reject a file whose point ID differs from the requested ID.
    points = scan.directory / "points"
    _standalone(sources[1], tmp_path / "b.h5", point_id="b")
    os.replace(tmp_path / "b.h5", points / "a.h5")
    with ScanReader(scan.path) as catalog, pytest.raises(InvalidScanFile, match="does not hold point 'a'"):
        catalog.point("a")


def test_point_reader_refuses_catalogs_retired_files_and_incomplete_points(published, tmp_path):
    scan, _ = published
    with pytest.raises(InvalidScanFile, match="is a reconstruction-scan catalog, not a point file"):
        PointReader(scan.path)
    retired = tmp_path / "retired.h5"
    with h5py.File(retired, "w") as handle:
        handle.attrs.update(format=layout.SCAN_FORMAT, version=1)
    with pytest.raises(InvalidScanFile, match="retired single-file"):
        PointReader(retired)
    with pytest.raises(InvalidScanFile, match="not a 'lauelab-reconstruction-point' file"):
        PointReader(_source(tmp_path / "in"))
    incomplete = shutil.copy(scan.directory / "points/Twin2_wire_1.h5", tmp_path / "incomplete.h5")
    with h5py.File(incomplete, "r+") as handle:
        handle["entry1/reconstruction/point/complete"][()] = 0
    with pytest.raises(InvalidScanFile, match="not complete"):
        PointReader(incomplete)
    damaged = shutil.copy(scan.directory / "points/Twin2_wire_1.h5", tmp_path / "damaged.h5")
    with h5py.File(damaged, "r+") as handle:
        del handle["entry1/reconstruction/stored_depth_intensity/data"]
    with pytest.raises(InvalidScanFile, match="entry1/reconstruction/stored_depth_intensity/data"):
        PointReader(damaged)


# --- Execution and outcomes --------------------------------------------------

def test_outcomes_are_small_and_carry_no_pixels(tmp_path):
    source = _source(tmp_path)
    outcome = _standalone(source, tmp_path / "point.h5")
    assert outcome.status == "complete"
    assert len(pickle.dumps(outcome)) < 1024
    assert {field.name for field in dataclasses.fields(outcome)} == {
        "index", "point_id", "status", "error", "seconds", "output",
    }


def test_standalone_arguments_must_be_consistent(tmp_path):
    source = _source(tmp_path)
    with _prepare([source], tmp_path / "scan") as scan:
        task = scan.tasks[0]
        with pytest.raises(InputError, match="pass a PointTask alone"):
            reconstruct_point(task, tmp_path / "other.h5")
        with pytest.raises(InputError, match="pass a PointTask alone"):
            reconstruct_point(task, depth_range=(0.0, 1.0))
    with pytest.raises(InputError, match="output is required"):
        reconstruct_point(source, geometry=GEOMETRY_FILE, detector=0, depth_range=DEPTH_RANGE_UM)
    with pytest.raises(InputError, match="input file does not exist"):
        _standalone(tmp_path / "missing.h5", tmp_path / "point.h5")


def test_input_failures_are_failed_points_that_publish_nothing(tmp_path):
    sources = [_source(tmp_path / "in", name=f"{name}.h5") for name in ("gone", "changed", "ok")]
    with _prepare(sources, tmp_path / "scan") as scan:
        sources[0].unlink()
        with h5py.File(sources[1], "r+") as handle:
            data = handle["entry1/data/data"][:, :64, :]
            del handle["entry1/data/data"]
            handle["entry1/data"].create_dataset("data", data=data)
        outcomes = [reconstruct_point(task, num_threads=1) for task in scan.tasks]
        for outcome in outcomes:
            scan.record(outcome)
    assert [outcome.status for outcome in outcomes] == ["failed", "failed", "complete"]
    assert "input file does not exist" in outcomes[0].error
    assert "changed after the point was prepared" in outcomes[1].error
    assert sorted(path.name for path in (scan.directory / "points").iterdir()) == ["ok.h5"]
    assert [entry.status for entry in ScanReader(scan.path).points] == ["failed", "failed", "complete"]


def test_a_read_failure_after_the_first_stripe_is_a_failed_point(tmp_path, monkeypatch):
    source = _source(tmp_path)
    real = _scan_writer.PointSink.raw
    calls = {"n": 0}

    def raise_on_second(self, row0, stripe):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("input became unavailable")
        real(self, row0, stripe)

    monkeypatch.setattr(_scan_writer.PointSink, "raw", raise_on_second)
    outcome = _standalone(source, tmp_path / "point.h5")
    assert outcome.status == "failed" and "input became unavailable" in outcome.error
    assert sorted(path.name for path in tmp_path.iterdir()) == ["synthetic.h5"]


def test_a_write_failure_is_an_output_error_not_a_failed_point(tmp_path, monkeypatch):
    source = _source(tmp_path)
    real = _scan_writer.PointSink._write
    calls = {"n": 0}

    def fail_second_write(self, row0, values):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("no space left on device")
        real(self, row0, values)

    monkeypatch.setattr(_scan_writer.PointSink, "_write", fail_second_write)
    with pytest.raises(ReconstructionError, match="writing .*point.h5 failed: no space left"):
        _standalone(source, tmp_path / "point.h5")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["synthetic.h5"]


@pytest.mark.parametrize("stage", ["_check_point_file", "publish_file"])
def test_validation_and_publication_failures_are_output_errors(tmp_path, monkeypatch, stage):
    source = _source(tmp_path)

    def fail(*args, **kwargs):
        raise OSError(f"{stage} failed on purpose")

    monkeypatch.setattr(scan_module, stage, fail)
    with pytest.raises(ReconstructionError, match=f"publishing .*{stage} failed on purpose"):
        _standalone(source, tmp_path / "point.h5")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["synthetic.h5"]


def test_an_unwritable_destination_is_an_output_error(tmp_path):
    source = _source(tmp_path / "in")
    outcome = _standalone(source, tmp_path / "new/dir/point.h5")
    assert outcome.status == "complete"
    locked = tmp_path / "locked"
    locked.mkdir(mode=0o555)
    try:
        with pytest.raises(ReconstructionError, match="creating the private file"):
            _standalone(source, locked / "point.h5")
    finally:
        locked.chmod(0o755)
    assert not any(locked.iterdir())


def test_published_and_private_files_of_others_are_never_replaced(tmp_path):
    source = _source(tmp_path / "in")
    output = tmp_path / "point.h5"
    output.write_text("published")
    with pytest.raises(ReconstructionError, match="already exists"):
        _standalone(source, output)
    assert output.read_text() == "published"
    output.unlink()
    partial_path(output).write_text("another worker")
    with pytest.raises(ReconstructionError, match="creating the private file"):
        _standalone(source, output)
    assert partial_path(output).read_text() == "another worker"
    assert not output.exists()


# --- Recording ---------------------------------------------------------------

def test_record_rejects_duplicate_foreign_and_conflicting_outcomes(tmp_path):
    sources = [_source(tmp_path / "in", name=f"{name}.h5") for name in ("a", "b")]
    with _prepare(sources, tmp_path / "scan") as scan:
        first, second = scan.tasks
        outcome = reconstruct_point(first, num_threads=1)
        scan.record(outcome)
        with pytest.raises(InputError, match="already recorded as complete"):
            scan.record(outcome)
        with pytest.raises(InputError, match="matches no task"):
            scan.record(dataclasses.replace(outcome, index=1))
        with pytest.raises(InputError, match="matches no task"):
            scan.record(dataclasses.replace(outcome, index=None))
        with pytest.raises(InputError, match="was published at"):
            scan.record(dataclasses.replace(outcome, index=1, point_id="b"))
        # The first point's file is not the second point's file.
        with pytest.raises(InputError, match="only a complete outcome or a failed outcome"):
            scan.record(dataclasses.replace(outcome, index=1, point_id="b", status="failed"))
        shutil.copy(first.output, second.output)
        with pytest.raises(InvalidScanFile, match="holds"):
            scan.record(dataclasses.replace(outcome, index=1, point_id="b", output=second.output))
        assert ScanReader(scan.path).points[1].status == "pending"
    with pytest.raises(InputError, match="the scan is failed"):
        scan.record(dataclasses.replace(outcome, index=1, point_id="b", status="failed", error="x"))


# --- Prepared request and metadata consistency ------------------------------

def test_editing_an_exported_task_does_not_change_the_prepared_request(tmp_path):
    source = _source(tmp_path / "in")
    with _prepare([source], tmp_path / "scan") as scan:
        task = scan.tasks[0]
        task.settings["percent_brightest"] = 10.0
        with pytest.raises(InputError, match="differs from the prepared task"):
            scan.record_dispatch(task)
        outcome = reconstruct_point(task, num_threads=1)
        assert outcome.status == "complete"
        with pytest.raises(InvalidScanFile, match="percent_brightest.*prepared task"):
            scan.record(outcome)
        assert ScanReader(scan.path).points[0].status == "pending"


@pytest.mark.parametrize("change", ["cosmic_filter", "geometry", "source"])
def test_external_worker_cannot_substitute_a_different_request(tmp_path, change):
    source = _source(tmp_path / "in")
    with _prepare([source], tmp_path / "scan") as scan:
        task = scan.tasks[0]
        scan.record_dispatch(task)
        if change == "cosmic_filter":
            altered = dataclasses.replace(task, settings={**task.settings, "cosmic_filter": True})
        elif change == "geometry":
            altered = dataclasses.replace(task, geometry_xml=task.geometry_xml + "\n")
        else:
            other = tmp_path / "other.h5"
            shutil.copyfile(source, other)
            altered = dataclasses.replace(task, source=os.fspath(other))
        outcome = reconstruct_point(altered, num_threads=1)
        assert outcome.status == "complete"
        with pytest.raises(InvalidScanFile, match="disagrees with the prepared task"):
            scan.record(outcome)
        scan.snapshot(force=True)
        assert ScanReader(scan.path).points[0].status == "writing"


@pytest.mark.parametrize("change", ["scanNum", "sampleX", "incident_energy", "raw_slices"])
def test_changed_input_selection_or_acquisition_is_rejected_before_writing(tmp_path, change):
    source = _source(tmp_path / "in")
    with _prepare([source], tmp_path / "scan") as scan:
        task = scan.tasks[0]
        with h5py.File(source, "r+") as file:
            if change == "raw_slices":
                data = file["entry1/data/data"][...]
                del file["entry1/data/data"]
                file["entry1/data/data"] = data[:-2]
            else:
                path = "entry1/scanNum" if change == "scanNum" else f"entry1/sample/{change}"
                file[path][...] = 987
        outcome = reconstruct_point(task, num_threads=1)
        assert outcome.status == "failed"
        assert "changed after the point was prepared" in outcome.error
        assert not Path(task.output).exists()
        assert not partial_path(task.output).exists()
        scan.record(outcome)
        scan.snapshot(force=True)
        assert ScanReader(scan.path).points[0].status == "failed"


def test_missing_acquisition_values_survive_task_transport_and_recording(tmp_path):
    source = _source(tmp_path / "in")
    with h5py.File(source, "r+") as file:
        del file["entry1/sample/sampleX"]
        file["entry1/sample/incident_energy"][...] = np.nan
        del file["entry1/scanNum"]
    with _prepare([source], tmp_path / "scan") as scan:
        task = scan.tasks[0]
        transported = type(task)(**json.loads(json.dumps(dataclasses.asdict(task), allow_nan=False)))
        assert transported == task
        assert pickle.loads(pickle.dumps(task)) == task
        scan.record_dispatch(transported)
        outcome = reconstruct_point(transported, num_threads=1)
        assert outcome.status == "complete", outcome.error
        scan.record(outcome)
        scan.finish()
        with ScanReader(scan.path) as catalog, catalog.point(task.point_id) as point:
            assert np.isnan(point.sample_position[0])
            assert point.energy_kev is None
            assert point.scan_number is None


@pytest.mark.parametrize("field", ["entry1/depth", "entry1/reconstruction/acquisition/sample_position",
                                    "entry1/reconstruction/acquisition/energy_kev", "entry1/reconstruction/acquisition/scan_number"])
def test_scan_reader_rejects_scientific_metadata_that_disagrees_with_catalog(tmp_path, field):
    source = _source(tmp_path / "in")
    point_path = _point(tmp_path / "scan", source)
    with h5py.File(point_path, "r+") as file:
        file[field][...] = file[field][...] + 100
    with ScanReader(tmp_path / "scan/scan.h5") as catalog:
        with pytest.raises(InvalidScanFile, match="as the catalog describes"):
            catalog.point(source.stem)




# --- Memory budget -----------------------------------------------------------

@pytest.mark.parametrize("budget_mb", [2, 5])
@pytest.mark.parametrize("output_type", [3, 5])
def test_live_stripe_arrays_fit_the_point_budget(tmp_path, monkeypatch, budget_mb, output_type):
    import weakref

    source = _source(tmp_path)
    live = {}
    peak = 0

    def tracked(function):
        def allocate(*args, **kwargs):
            nonlocal peak
            result = function(*args, **kwargs)
            if isinstance(result, np.ndarray) and result.ndim == 3 and result.flags.owndata:
                live[id(result)] = weakref.ref(result)
                arrays = [reference() for reference in list(live.values())]
                peak = max(peak, sum(array.nbytes for array in arrays if array is not None))
            return result
        return allocate

    for name in ("empty", "zeros", "ascontiguousarray"):
        monkeypatch.setattr(np, name, tracked(getattr(np, name)))
    outcome = _standalone(source, tmp_path / "point.h5", depth_range=(-200, 200),
                          rows_per_stripe=None, memory_limit_mb=budget_mb,
                          output_pixel_type=output_type)
    monkeypatch.undo()
    assert outcome.status == "complete"
    assert peak > 0
    assert peak <= budget_mb * 2**20
    with PointReader(outcome.output) as point:
        # Stored-pixel reductions must agree across stripe sizes.
        totals = np.concatenate([frames.sum(axis=(1, 2), dtype=np.float64)
                                 for _, frames in point.iter_blocks(2**20)])
        np.testing.assert_allclose(point.depth_intensity(), totals, rtol=1e-12)


def test_point_budget_must_hold_at_least_one_row(tmp_path):
    outcome = _standalone(_source(tmp_path), tmp_path / "point.h5", depth_range=(-200, 200),
                          rows_per_stripe=None, memory_limit_mb=1, output_pixel_type=5)
    assert outcome.status == "failed"
    assert "memory_limit_mb cannot hold 1 stripe row" in outcome.error
    assert not (tmp_path / "point.h5").exists() and not partial_path(tmp_path / "point.h5").exists()


def test_explicit_stripe_size_cannot_bypass_budget(tmp_path):
    outcome = _standalone(_source(tmp_path), tmp_path / "point.h5", depth_range=(-200, 200),
                          rows_per_stripe=128, memory_limit_mb=2)
    assert outcome.status == "failed"
    assert "memory_limit_mb cannot hold 128 stripe row" in outcome.error


# --- NeXus structure and acquisition metadata ---------------------------------


def test_nexus_default_signal_axes_and_reference_groups(published):
    scan, _ = published
    with h5py.File(scan.directory / "points/Twin2_wire_1.h5") as file:
        # Discover the stack through NeXus attributes rather than lauelab paths.
        entry = file[file.attrs["default"]]
        assert entry.attrs["NX_class"] == "NXentry"
        data = entry[entry.attrs["default"]]
        assert data.attrs["NX_class"] == "NXdata"
        signal = data[data.attrs["signal"]]
        assert signal.shape == (51, 128, 128)
        assert list(data.attrs["axes"]) == ["depth", ".", "."]
        np.testing.assert_array_equal(data["depth"], np.arange(-25, 26))
        assert data["depth"].attrs["units"] == "um"
        assert data["depth"].id == entry["depth"].id
        process = entry["reconstruction"]
        assert process.attrs["NX_class"] == "NXprocess"
        assert process["program"].asstr()[()] == "lauelab"
        assert process["version"].asstr()[()] == file.attrs["lauelab_version"]
        assert process["date"].asstr()[()] == file.attrs["created"]
        for name in ("first_raw", "sum_raw", "sum_reconstructed",
                     "stored_depth_intensity", "computed_depth_intensity"):
            group = process[name]
            assert group.attrs["NX_class"] == "NXdata"
            values = group[group.attrs["signal"]]
            if values.ndim == 1:
                assert list(group.attrs["axes"]) == ["depth"]
                assert group["depth"].id == entry["depth"].id
                assert values.shape == entry["depth"].shape
            else:
                assert list(group.attrs["axes"]) == [".", "."]
                assert values.shape == signal.shape[1:]


def test_point_preserves_metadata_paths_and_replaces_raw_axes(tmp_path):
    source = _source(tmp_path / "in")
    with h5py.File(source, "r+") as file:
        file.attrs["operator_note"] = "keep root metadata"
        file["entry1"].attrs["default"] = "old_data"
        file["entry1/depth"][...] = [999.0]
        group = file["entry1/data"]
        group.attrs.update(signal="old_signal", axes=["wire", ".", "."],
                           wire_indices=[0], auxiliary_signals=["old_signal"])
        group.create_dataset("depth", data=[123.0])
        group.attrs["acquisition_note"] = "keep group metadata"
        file.create_dataset("entry1/user_annotation", data="science note")
    path = _point(tmp_path / "scan", source)
    with h5py.File(source) as raw, h5py.File(path) as point:
        assert point.attrs["operator_note"] == raw.attrs["operator_note"]
        assert point["entry1/user_annotation"].asstr()[()] == "science note"
        np.testing.assert_array_equal(point["entry1/detector/Nx"], raw["entry1/detector/Nx"])
        assert "entry1/wire" not in point
        assert "source" not in point
        assert point["entry1/data"].attrs["acquisition_note"] == "keep group metadata"
        assert "wire_indices" not in point["entry1/data"].attrs
        assert "auxiliary_signals" not in point["entry1/data"].attrs
        assert point["entry1"].attrs["default"] == "data"
        np.testing.assert_array_equal(point["entry1/depth"], np.arange(-25, 26))
        assert point["entry1/data/depth"].id == point["entry1/depth"].id


@pytest.mark.parametrize("path,name,value", [
    ("entry1", "NX_class", "NXcollection"),
    ("entry1", "default", "missing"),
    ("entry1/data", "signal", "missing"),
    ("entry1/data", "axes", [".", ".", "depth"]),
    ("entry1/depth", "units", "mm"),
    ("entry1/reconstruction/sum_raw", "signal", "missing"),
])
def test_point_validator_rejects_incorrect_nexus_metadata(published, tmp_path, path, name, value):
    scan, _ = published
    copy = tmp_path / "point.h5"
    shutil.copyfile(scan.directory / "points/Twin2_wire_1.h5", copy)
    with h5py.File(copy, "r+") as file:
        file[path].attrs[name] = value
    with pytest.raises(InvalidScanFile, match="attribute"):
        PointReader(copy)


def test_point_validator_requires_shared_depth_coordinates(published, tmp_path):
    scan, _ = published
    copy = tmp_path / "point.h5"
    shutil.copyfile(scan.directory / "points/Twin2_wire_1.h5", copy)
    with h5py.File(copy, "r+") as file:
        del file["entry1/data/depth"]
        file["entry1/data/depth"] = file["entry1/depth"][...]
    with pytest.raises(InvalidScanFile, match="must link to /entry1/depth"):
        PointReader(copy)


def test_preparation_rejects_reserved_reconstruction_path(tmp_path):
    source = _source(tmp_path / "in")
    with h5py.File(source, "r+") as file:
        file.create_group("entry1/reconstruction")
    with _prepare([source], tmp_path / "scan") as scan:
        assert not scan.tasks
        assert "reserved group" in ScanReader(scan.path).points[0].error
        scan.finish()
    with pytest.raises(InputError, match="reserved group"):
        _standalone(source, tmp_path / "point.h5")
    assert not (tmp_path / "point.h5").exists()


def test_old_point_version_is_rejected(published, tmp_path):
    scan, _ = published
    copy = tmp_path / "point.h5"
    shutil.copyfile(scan.directory / "points/Twin2_wire_1.h5", copy)
    with h5py.File(copy, "r+") as file:
        file.attrs["version"] = 1
    with pytest.raises(InvalidScanFile, match="unsupported.*version"):
        PointReader(copy)
