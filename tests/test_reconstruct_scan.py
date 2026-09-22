# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Tests for single-file scan reconstruction: writer, reader, and validator."""

import json
from pathlib import Path
import shutil

import h5py
import numpy as np
import pytest

from conftest import requires_liblaue
from lauelab import partial_path
from lauelab.indexing import InputError, InvalidScanFile, ReconstructionError
from lauelab.reconstruct import (
    Reconstructor, ScanReader, reconstruct_scan, validate_scan_file,
)
from lauelab.reconstruct import _scan_layout as layout
from lauelab.reconstruct import _scan_writer
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
        num_threads=1,
        rows_per_stripe=31,
    )
    values.update(changes)
    return values


def _source(tmp_path, variant=None, name="synthetic.h5"):
    options = VARIANTS.get(variant, {})
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    write_input_file(path, write_mA=options.get("write_mA", False),
                     write_microdiffraction=options.get("write_microdiffraction", False))
    return path


def _scan(paths, output, **changes):
    variant = changes.pop("variant", None)
    keywords = {key: changes.pop(key) for key in
                ("point_ids", "overwrite", "compression", "progress", "should_stop")
                if key in changes}
    return reconstruct_scan(paths, output, geometry=GEOMETRY_FILE, detector=0,
                            **keywords, **_options(variant, **changes))


@pytest.fixture(scope="module")
def published(tmp_path_factory):
    """A three-point run: complete, unreadable input, complete with the same stem."""
    work = tmp_path_factory.mktemp("published")
    first = _source(work / "a")
    second = _source(work / "b")
    result = _scan([first, work / "missing.h5", second], work / "run.h5",
                   point_ids=["first", "gone", "second"], wire_edge="both",
                   output_pixel_type=None)
    return result, first


# --- Stored values -----------------------------------------------------------

@pytest.mark.parametrize("variant", [None, *VARIANTS])
def test_stored_frames_equal_per_depth_files_and_goldens(tmp_path, variant):
    source = _source(tmp_path, variant)
    per_depth = Reconstructor(GEOMETRY_FILE, 0, **_options(variant)).reconstruct(
        source, tmp_path / "depth" / "point_"
    )
    result = _scan([source], tmp_path / "run.h5", variant=variant)
    suffix = f"_{variant}" if variant else ""
    golden = np.load(REFERENCE_DIR / f"cpu_reference{suffix}.npz")
    tolerance = json.loads((REFERENCE_DIR / f"cpu_reference{suffix}.json").read_text())["comparison"]

    assert per_depth.success and result.complete
    with ScanReader(result.path) as scan:
        point = scan.point("0")
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
    unscaled = Reconstructor(GEOMETRY_FILE, 0, **_options("norm_exponent")).reconstruct(
        source, return_images=True
    )
    result = _scan([source], tmp_path / "run.h5", variant="norm_exponent",
                   output_pixel_type=pixel_type)
    with ScanReader(result.path) as scan:
        point = scan.point("0")
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
    result = _scan([_source(tmp_path)], tmp_path / "run.h5", output_pixel_type=0)
    with ScanReader(result.path) as scan:
        point = scan.point("0")
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
    plain = _scan([source], tmp_path / "plain.h5", output_pixel_type=3)
    packed = _scan([source], tmp_path / "packed.h5", output_pixel_type=3, compression="gzip")
    assert packed.path.stat().st_size < plain.path.stat().st_size
    assert validate_scan_file(packed.path).complete
    with ScanReader(plain.path) as left, ScanReader(packed.path) as right:
        np.testing.assert_array_equal(
            left.point("0").region((0, 128, 0, 128)), right.point("0").region((0, 128, 0, 128))
        )
        np.testing.assert_array_equal(
            left.point("0").depth_intensity(), right.point("0").depth_intensity()
        )
    with h5py.File(packed.path) as handle:
        data = handle["points/000000/data"]
        assert (data.compression, data.compression_opts, data.shuffle) == ("gzip", 1, True)
        assert data.chunks == (4, 128, 128)
    with pytest.raises(InputError, match="compression must be None or 'gzip'"):
        _scan([source], tmp_path / "other.h5", compression="lzf")
    assert not partial_path(tmp_path / "other.h5").exists()


# --- Reference images --------------------------------------------------------

def test_raw_references_exclude_bookkeeping_slices_and_precede_processing(tmp_path):
    source = _source(tmp_path, "cosmic")
    with h5py.File(source, "r+") as handle:
        data = handle["entry1/data/data"]
        data[0] = 60000
        data[-1] = 50000
        handle["entry1"].create_dataset("mA", data=np.linspace(51.0, 153.0, data.shape[0]))
        raw = np.asarray(data)
    result = _scan([source], tmp_path / "run.h5", variant="cosmic", normalization="mA",
                   norm_exponent=0.5)
    with ScanReader(result.path) as scan:
        point = scan.point("0")
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
    result = _scan([source], tmp_path / "run.h5")
    with ScanReader(result.path) as scan:
        point = scan.point("0")
        assert point.reference("first_raw").dtype == np.float32
        total = point.reference("sum_raw")
        assert total.dtype == np.float64
        np.testing.assert_allclose(total, raw[1:-1].sum(axis=0, dtype=np.float64), rtol=1e-12)


# --- File layout -------------------------------------------------------------

def test_published_file_follows_the_layout_table(published):
    result, _ = published
    with h5py.File(result.path) as handle:
        assert handle.attrs["format"] == layout.FORMAT
        assert handle.attrs["version"] == layout.VERSION
        for path, spec in {**layout.RUN_DATASETS, **layout.CATALOG_DATASETS}.items():
            dataset = handle[path]
            assert dataset.attrs.get("units") == spec.units, path
            for name, value in spec.attrs.items():
                assert dataset.attrs[name] == value, path
        assert set(handle["points"]) == {"000000", "000002"}
        group = handle["points/000002"]
        found = []
        group.visititems(
            lambda name, node: found.append(name)
            if isinstance(node, h5py.Dataset) and not name.startswith("source/") else None
        )
        assert set(found) == set(layout.POINT_DATASETS)
        assert group["data"].chunks is not None
        assert group["depth_um"].attrs["units"] == "um"
        assert "entry1/detector/Nx" in group["source"]
        assert "entry1/wire" not in group["source"]
        assert "entry1/data/data" not in group["source"]
        assert handle["geometry/xml"][()].decode() == GEOMETRY_FILE.read_text()
        assert handle["settings/wire_edge"][()].decode() == "both"
        assert handle["settings/output_pixel_type"][()] == -1
        assert np.isnan(handle["settings/norm_exponent"][()])
        np.testing.assert_array_equal(handle["settings/depth_range"], DEPTH_RANGE_UM)


def test_catalog_lists_every_point_without_opening_a_group(published):
    result, first = published
    assert [outcome.status for outcome in result.outcomes] == ["complete", "failed", "complete"]
    assert not result.complete and not result.cancelled
    with ScanReader(result.path) as scan:
        assert scan.run_status == "finished"
        assert scan.point_ids == ("first", "gone", "second")
        complete, failed, _ = scan.points
        assert complete.shape == (51, 128, 128)
        assert complete.dtype == np.int32
        assert complete.depth_bounds_um == DEPTH_RANGE_UM
        assert complete.scan_number == 1
        assert complete.energy_kev == 20.0
        assert complete.sample_position == (0.0, 0.0, 0.0)
        assert complete.source_path == str(first)
        assert failed.status == "failed" and "does not exist" in failed.error
        assert failed.shape == (0, 0, 0) and failed.dtype is None
    with h5py.File(result.path) as handle:
        assert handle["catalog/source_sizes"][0] == first.stat().st_size
        assert handle["catalog/source_mtimes_ns"][0] == first.stat().st_mtime_ns
        assert handle["catalog/source_sizes"][1] == -1


def test_points_with_one_file_stem_stay_distinct(published):
    result, _ = published
    with ScanReader(result.path) as scan:
        first, second = scan.point("first"), scan.point("second")
        assert Path(first.entry.source_path).stem == Path(second.entry.source_path).stem
        assert first.entry.index == 0 and second.entry.index == 2
        np.testing.assert_array_equal(first.frame(25), second.frame(25))


# --- Reader ------------------------------------------------------------------

def test_reader_returns_owned_arrays_for_exactly_the_selection(published):
    result, _ = published
    with ScanReader(result.path) as scan:
        point = scan.point("first")
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
        depth_um = point.depth_um
        assert depth_um[25] == 0.0 and depth_um[0] == -25.0

        blocks = list(point.iter_blocks(10 * 128 * 128 * 4))
        assert [first for first, _ in blocks] == [0, 10, 20, 30, 40, 50]
        assert all(block.nbytes <= 10 * 128 * 128 * 4 for _, block in blocks)
        np.testing.assert_array_equal(np.concatenate([block for _, block in blocks])[25],
                                      point.frame(25))


def test_reader_rejects_selections_it_would_have_to_clip(published):
    result, _ = published
    with ScanReader(result.path) as scan:
        point = scan.point("first")
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


def test_reader_refuses_points_that_are_not_complete(published):
    result, _ = published
    with ScanReader(result.path) as scan:
        with pytest.raises(InputError, match="'gone' is failed: input file does not exist"):
            scan.point("gone")
        with pytest.raises(KeyError):
            scan.point("absent")


def test_reader_rejects_other_hdf5_files(tmp_path):
    with pytest.raises(ValueError, match="not a 'lauelab-reconstruction-scan' file"):
        ScanReader(_source(tmp_path))


# --- Run behavior ------------------------------------------------------------

def test_stop_request_publishes_completed_points_and_marks_the_rest(tmp_path):
    source = _source(tmp_path)
    seen = []
    result = _scan([source, source, source], tmp_path / "run.h5",
                   progress=seen.append, should_stop=lambda: len(seen) >= 1)
    assert result.cancelled and not result.complete
    assert [outcome.status for outcome in result.outcomes] == [
        "complete", "unattempted", "unattempted",
    ]
    assert [outcome.index for outcome in seen] == [0]
    summary = validate_scan_file(result.path)
    assert (summary.run_status, summary.n_complete, summary.n_unattempted) == ("cancelled", 1, 2)
    assert not summary.complete
    with ScanReader(result.path) as scan:
        assert scan.point("0").frame(25).any()
        with pytest.raises(InputError, match="'1' is unattempted"):
            scan.point("1")


def test_point_failure_after_its_first_stripe_does_not_stop_the_run(tmp_path, monkeypatch):
    source = _source(tmp_path)
    broken = _source(tmp_path, name="broken.h5")
    real = Reconstructor._run_file

    def fail_broken(self, handle, info, sink, return_images):
        if Path(handle.filename) == broken:
            calls = {"n": 0}
            raw = sink.raw

            def raise_on_second(row0, stripe):
                calls["n"] += 1
                if calls["n"] == 2:
                    raise OSError("input became unavailable")
                raw(row0, stripe)

            sink.raw = raise_on_second
        return real(self, handle, info, sink, return_images)

    monkeypatch.setattr(Reconstructor, "_run_file", fail_broken)
    result = _scan([source, broken, source], tmp_path / "run.h5")
    assert [outcome.status for outcome in result.outcomes] == ["complete", "failed", "complete"]
    assert "input became unavailable" in result.outcomes[1].error
    assert validate_scan_file(result.path).n_failed == 1
    with ScanReader(result.path) as scan:
        with pytest.raises(InputError, match="is failed: input became unavailable"):
            scan.point("1")
        np.testing.assert_array_equal(scan.point("0").frame(25), scan.point("2").frame(25))


def test_output_file_failure_stops_the_run_and_publishes_nothing(tmp_path, monkeypatch):
    source = _source(tmp_path)
    real = _scan_writer._ScanSink._write
    calls = {"n": 0}

    def fail_second_write(self, row0, values):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("output became unavailable")
        real(self, row0, values)

    monkeypatch.setattr(_scan_writer._ScanSink, "_write", fail_second_write)
    output = tmp_path / "run.h5"
    with pytest.raises(ReconstructionError, match="output became unavailable"):
        _scan([source, source], output)
    assert not output.exists()
    assert partial_path(output).exists()
    with pytest.raises(InvalidScanFile, match="run status is failed"):
        validate_scan_file(partial_path(output))


def test_existing_output_is_refused_before_any_work(tmp_path):
    output = tmp_path / "run.h5"
    output.write_text("keep")
    with pytest.raises(FileExistsError):
        _scan([tmp_path / "missing.h5"], output)
    assert output.read_text() == "keep"
    assert not partial_path(output).exists()
    result = _scan([_source(tmp_path)], output, overwrite=True)
    assert validate_scan_file(result.path).complete


@pytest.mark.parametrize("point_ids", [["a"], ["a", "a"], ["a", ""], ["a", 1]])
def test_point_ids_must_be_unique_non_empty_strings(tmp_path, point_ids):
    source = tmp_path / "unused.h5"
    with pytest.raises(InputError, match="point_ids"):
        _scan([source, source], tmp_path / "run.h5", point_ids=point_ids)
    assert not partial_path(tmp_path / "run.h5").exists()


def test_invalid_shared_configuration_raises_before_the_file_exists(tmp_path):
    with pytest.raises(InputError, match="resolution"):
        _scan([_source(tmp_path)], tmp_path / "run.h5", resolution=0)
    assert not partial_path(tmp_path / "run.h5").exists()


def test_unreadable_point_metadata_is_a_catalog_failure(tmp_path):
    short = tmp_path / "short.h5"
    with h5py.File(short, "w") as handle:
        handle.create_dataset("entry1/data/data", shape=(4, 2, 2), dtype=np.uint16)
    text = tmp_path / "text.h5"
    text.write_text("not hdf5")
    result = _scan([short, text, _source(tmp_path)], tmp_path / "run.h5")
    assert [outcome.status for outcome in result.outcomes] == ["failed", "failed", "complete"]
    assert "at least 5 stored slices" in result.outcomes[0].error
    assert validate_scan_file(result.path).n_failed == 2


# --- Validation --------------------------------------------------------------

def _tampered(published, tmp_path, change):
    path = tmp_path / "tampered.h5"
    shutil.copy(published[0].path, path)
    with h5py.File(path, "r+") as handle:
        change(handle)
    return path


def _set(path, value):
    def change(handle):
        handle[path][...] = value
    return change


def _delete(path):
    def change(handle):
        del handle[path]
    return change


def _retype(handle):
    del handle["points/000000/reductions/depth_intensity"]
    handle["points/000000/reductions"].create_dataset("depth_intensity", shape=(51,), dtype="<f8")


@pytest.mark.parametrize(("change", "message"), [
    (_set("run/status", int(layout.RunStatus.RUNNING)), "run status is running"),
    (_set("catalog/status", int(layout.PointStatus.WRITING)), "not terminal"),
    (_set("catalog/status", 9), "not terminal"),
    (_set("catalog/point_ids", "same"), "non-empty and unique"),
    (_delete("catalog/errors"), "'/catalog/errors' is missing"),
    (_delete("points/000000"), "complete point has no group"),
    (_delete("points/000002/reference/sum_raw"), "'reference/sum_raw' is missing"),
    (_delete("points/000002/source"), "group 'source' is missing"),
    (_retype, "has dtype float64, expected int64"),
    (_set("catalog/n_depths", 50), "has shape"),
    (_set("catalog/depth_bounds", 0.0), "depth_bounds disagree"),
    (_set("catalog/pixel_types", 4), "pixel type 4 is unknown"),
    (lambda handle: handle.attrs.__setitem__("version", 2), "unsupported"),
    (lambda handle: handle.attrs.__setitem__("format", "other"), "not a"),
])
def test_validator_rejects_structural_damage(published, tmp_path, change, message):
    with pytest.raises(InvalidScanFile, match=message):
        validate_scan_file(_tampered(published, tmp_path, change))


def test_validator_accepts_a_valid_incomplete_run_and_reads_no_pixels(published):
    summary = validate_scan_file(published[0].path)
    assert (summary.n_points, summary.n_complete, summary.n_failed) == (3, 2, 1)
    assert summary.run_status == "finished"
    assert not summary.complete


# --- Guide example -----------------------------------------------------------

def test_guide_example_runs_as_documented(tmp_path):
    """The flow shown in docs/guides/reconstruction.md, on the synthetic point."""
    paths = [_source(tmp_path / str(index)) for index in range(2)]
    seen = []
    result = reconstruct_scan(
        paths, tmp_path / "run" / "scan.h5", geometry=GEOMETRY_FILE, detector=0,
        point_ids=["scan12_p1", "scan12_p2"], depth_range=(-25.0, 25.0), num_threads=1,
        progress=lambda outcome: seen.append((outcome.point_id, outcome.status)),
    )
    assert [outcome for outcome in result.outcomes if outcome.status != "complete"] == []
    assert seen == [("scan12_p1", "complete"), ("scan12_p2", "complete")]

    with ScanReader(tmp_path / "run" / "scan.h5") as scan:
        assert [(entry.point_id, entry.status, entry.shape, entry.error)
                for entry in scan.points][0] == ("scan12_p1", "complete", (51, 128, 128), "")
        point = scan.point("scan12_p1")
        depth_index = int(point.depth_intensity().argmax())
        assert point.frame(depth_index).shape == (128, 128)
        assert point.depth_um[depth_index] == 0.0
        assert point.region((60, 70, 50, 54)).shape == (51, 10, 4)


def test_points_in_one_run_keep_their_own_dtype_and_values(tmp_path):
    """One file holds one stack per point; points need not share a stored dtype."""
    counts = _source(tmp_path, name="counts.h5")
    scaled = _source(tmp_path, name="scaled.h5")
    with h5py.File(scaled, "r+") as handle:
        raw = np.asarray(handle["entry1/data/data"], dtype=np.float32) * 2
        del handle["entry1/data/data"]
        handle["entry1/data"].create_dataset("data", data=raw)
    result = _scan([counts, scaled], tmp_path / "run.h5", point_ids=["counts", "scaled"],
                   output_pixel_type=None)
    assert result.complete
    with ScanReader(result.path) as scan:
        first, second = scan.point("counts"), scan.point("scaled")
        assert (first.dtype, second.dtype) == (np.uint16, np.float32)
        assert first.reference("sum_raw").dtype == np.int64
        assert second.reference("sum_raw").dtype == np.float64
        assert first.depth_intensity().dtype == np.int64
        assert second.depth_intensity().dtype == np.float64
        np.testing.assert_array_equal(second.reference("first_raw"), 2 * first.reference("first_raw"))
        # Each point equals what the same input gives when reconstructed alone.
        for name, source in (("counts", counts), ("scaled", scaled)):
            alone = _scan([source], tmp_path / f"alone_{name}.h5", output_pixel_type=None)
            with ScanReader(alone.path) as single:
                np.testing.assert_array_equal(
                    scan.point(name).region((0, 128, 0, 128)),
                    single.point("0").region((0, 128, 0, 128)),
                )


@pytest.mark.parametrize(("field", "value"), [
    ("entry1/detector/Nx", np.nan),
    ("entry1/detector/Nx", b"invalid"),
    ("entry1/sample/sampleX", b"invalid"),
    ("entry1/sample/incident_energy", b"invalid"),
])
def test_malformed_metadata_does_not_discard_other_points(tmp_path, field, value):
    bad = _source(tmp_path, name="bad.h5")
    good = _source(tmp_path, name="good.h5")
    with h5py.File(bad, "r+") as handle:
        del handle[field]
        handle.create_dataset(field, data=[value])
    result = _scan([bad, good], tmp_path / "run.h5")
    assert [outcome.status for outcome in result.outcomes] == ["failed", "complete"]
    assert "invalid metadata" in result.outcomes[0].error
    assert validate_scan_file(result.path).n_complete == 1


@pytest.mark.parametrize("budget_mb", [2, 5])
@pytest.mark.parametrize("output_type", [3, 5])
def test_live_stripe_arrays_fit_the_scan_budget(tmp_path, monkeypatch, budget_mb, output_type):
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
    result = _scan([source], tmp_path / "run.h5", depth_range=(-200, 200),
                   rows_per_stripe=None, memory_limit_mb=budget_mb, output_pixel_type=output_type)
    assert result.complete
    assert peak > 0
    assert peak <= budget_mb * 2**20
    with ScanReader(result.path) as scan:
        point = scan.point("0")
        # A budget changes chunking of computation, not the meaning of the reductions.
        totals = np.concatenate([frames.sum(axis=(1, 2), dtype=np.float64)
                                 for _, frames in point.iter_blocks(2**20)])
        np.testing.assert_allclose(point.depth_intensity(), totals, rtol=1e-12)


def test_scan_budget_must_hold_at_least_one_row(tmp_path):
    source = _source(tmp_path)
    result = _scan([source], tmp_path / "run.h5", depth_range=(-200, 200),
                   rows_per_stripe=None, memory_limit_mb=1, output_pixel_type=5)
    assert result.outcomes[0].status == "failed"
    assert "memory_limit_mb cannot hold 1 stripe row" in result.outcomes[0].error


def test_explicit_stripe_size_cannot_bypass_budget(tmp_path):
    result = _scan([_source(tmp_path)], tmp_path / "run.h5", depth_range=(-200, 200),
                   rows_per_stripe=128, memory_limit_mb=2)
    assert result.outcomes[0].status == "failed"
    assert "memory_limit_mb cannot hold 128 stripe row" in result.outcomes[0].error
