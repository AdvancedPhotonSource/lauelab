# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Scan frames as indexing input, their provenance in results, and per-depth export."""

import pickle
from pathlib import Path
import re
import shutil

import h5py
import numpy as np
import pytest

from conftest import requires_liblaue
from lauelab.indexing import (
    FrameInput, Indexer, InputError, InvalidScanFile, ScanFrame, validate_results_file,
)
from lauelab.indexing._frame import read_h5_frame
from lauelab.reconstruct import (
    PointReader, Reconstructor, ScanReader, export_per_depth, reconstruct_scan,
)
from lauelab.visualization import load_results, prepare_detector_view
from tests.data.reconstruction.generate_reference import (
    DEPTH_RANGE_UM, GEOMETRY_FILE, VARIANTS, write_input_file,
)

pytestmark = requires_liblaue
ROOT = Path(__file__).resolve().parents[1]
CRYSTAL = ROOT / "tests/config/Ni.xml"


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


def _frame(scan_path, point_id, depth_index):
    """Resolve a catalog point to a frame reference, as a scan selection does."""
    return ScanFrame(ScanReader(scan_path).point_path(point_id), point_id, depth_index)


def _source(directory, variant=None, name="synthetic.h5"):
    options = VARIANTS.get(variant, {})
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    write_input_file(path, write_mA=options.get("write_mA", False),
                     write_microdiffraction=options.get("write_microdiffraction", False))
    return path


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """Three points, both-edge signed int32 output: complete, failed input, complete.

    Point IDs differ from the input stems so that nothing can select a frame
    by filename, and the failed point sits between the complete ones so that
    nothing can select one by completion order.
    """
    work = tmp_path_factory.mktemp("run")
    first = _source(work / "a", name="synthetic_b.h5")
    second = _source(work / "b", name="synthetic_a.h5")
    with h5py.File(second, "r+") as handle:
        handle["entry1/sample/sampleX"][0] = 12.5
        handle["entry1/scanNum"][0] = 7
    result = reconstruct_scan(
        [first, work / "missing.h5", second], work / "run", geometry=GEOMETRY_FILE,
        detector=0, point_ids=["first", "gone", "second"], workers=2, threads_per_worker=1,
        **_options(wire_edge="both", output_pixel_type=None),
    )
    assert [outcome.status for outcome in result.outcomes] == ["complete", "failed", "complete"]
    return result.path


@pytest.fixture(scope="module")
def indexer():
    return Indexer(GEOMETRY_FILE, CRYSTAL)


# --- ScanFrame ---------------------------------------------------------------

def test_scan_frame_holds_only_plain_values_and_pickles():
    frame = ScanFrame(Path("/tmp/run.h5"), "p1", np.int64(3))
    assert frame == ScanFrame("/tmp/run.h5", "p1", 3)
    assert isinstance(frame.path, str) and type(frame.depth_index) is int
    assert pickle.loads(pickle.dumps(frame)) == frame
    with pytest.raises(ValueError, match="point_id"):
        ScanFrame("run.h5", "", 0)
    with pytest.raises(TypeError, match="depth_index"):
        ScanFrame("run.h5", "p1", 1.0)
    with pytest.raises(TypeError, match="depth_index"):
        ScanFrame("run.h5", "p1", True)
    with pytest.raises(ValueError, match="nonnegative"):
        ScanFrame("run.h5", "p1", -1)


# --- Indexing one frame -------------------------------------------------------

def test_indexing_a_scan_frame_reads_that_frame_and_its_provenance(run, indexer):
    frame = _frame(run, "second", 30)
    assert Path(frame.path).name == "synthetic_a.h5"
    result = indexer.index(frame)
    with ScanReader(run) as scan:
        point = scan.point("second")
        expected = point.frame(30)
        depth_um = float(point.depth_um[30])
        start, group = point.start, point.group
    assert result.image.dtype == np.int32
    np.testing.assert_array_equal(result.image, expected)
    assert result.depth == depth_um == 5.0
    assert (result.start, result.group) == (start, group) == ((0, 0), (16, 16))
    assert result.input_image == frame.path
    assert result.source == frame
    assert result.metadata["scan_number"] == 7
    assert result.metadata["energy_kev"] == 20.0
    assert result.metadata["detector_id"] == "PE1621 723-3335"
    assert result.metadata["sample_position"] == (12.5, 0.0, 0.0)

    # The same frame given as an array with the same processing values is the same result.
    as_array = indexer.index(expected, start=start, group=group, depth=depth_um)
    np.testing.assert_array_equal(result.peaks, as_array.peaks)
    assert result.n_patterns == as_array.n_patterns
    assert as_array.source is None


def test_explicit_depth_overrides_the_stored_depth_including_zero(run, indexer):
    frame = _frame(run, "first", 0)
    stored = indexer.index(frame, keep_image=False)
    assert stored.depth == -25.0
    assert indexer.index(frame, depth=0.0, keep_image=False).depth == 0.0
    assert indexer.index(frame, depth=3.5, keep_image=False).depth == 3.5


def test_scan_frames_that_cannot_be_read_raise_input_error(run, indexer, tmp_path):
    # A failed point has no point file.
    with pytest.raises(InputError, match="cannot read .*No such file"):
        indexer.index(_frame(run, "gone", 0))
    with pytest.raises(InputError, match="holds point 'second', not 'first'"):
        indexer.index(ScanFrame(_frame(run, "second", 0).path, "first", 0))
    with pytest.raises(InputError, match="depth_index 51 is outside 0 to 50"):
        indexer.index(_frame(run, "first", 51))
    with pytest.raises(InputError, match="not a 'lauelab-reconstruction-point' file"):
        indexer.index(ScanFrame(_source(tmp_path), "first", 0))
    with pytest.raises(InputError, match="is a reconstruction-scan catalog, not a point file"):
        indexer.index(ScanFrame(run, "first", 0))
    with pytest.raises(InputError, match="cannot read"):
        indexer.index(ScanFrame(tmp_path / "nowhere.h5", "first", 0))


# --- Incremental indexing -----------------------------------------------------

def test_scan_frames_index_in_spawned_workers(run, indexer):
    inputs = [
        FrameInput(_frame(run, "second", 30), input_id="second/30"),
        FrameInput(_frame(run, "gone", 0), input_id="gone/0"),
        FrameInput(_frame(run, "first", 20), input_id="first/20", depth=0.0),
        FrameInput(_frame(run, "first", 99), input_id="first/99"),
    ]
    with indexer.iter_index(inputs, workers=2) as outcomes:
        results = list(outcomes)
    assert [outcome.ok for outcome in results] == [True, False, True, False]
    assert isinstance(results[1].error, InputError) and "cannot read" in str(results[1].error)
    assert "depth_index 99" in str(results[3].error)
    direct = indexer.index(_frame(run, "second", 30), keep_image=False)
    np.testing.assert_array_equal(results[0].result.peaks, direct.peaks)
    assert results[0].result.source == _frame(run, "second", 30)
    assert results[2].result.depth == 0.0


# --- Results provenance and source images ------------------------------------

@pytest.fixture(scope="module")
def results_file(run, indexer, tmp_path_factory):
    """Index nonconsecutive depths of both complete points into one results file."""
    frames = [
        _frame(run, "second", 30), _frame(run, "first", 25),
        _frame(run, "second", 10), _frame(run, "first", 40),
    ]
    results = indexer.index_many(frames)
    path = tmp_path_factory.mktemp("results") / "indexed.h5"
    indexer.write_results(results, path, frame_ids=["s30", "f25", "s10", "f40"])
    return path, frames


def test_results_file_records_the_scan_source_of_each_frame(results_file):
    path, frames = results_file
    validate_results_file(path)
    with h5py.File(path) as handle:
        assert list(handle["frames/source_point_ids"].asstr()[...]) == [
            "second", "first", "second", "first",
        ]
        assert handle["frames/source_depth_indices"][...].tolist() == [30, 25, 10, 40]
        assert list(handle["frames/input_images"].asstr()[...]) == [frame.path for frame in frames]
        np.testing.assert_array_equal(handle["frames/depths"], [5.0, 0.0, -15.0, 15.0])
        assert handle["frames/source_depth_indices"].attrs.keys() == set()


def test_reopened_results_show_each_frame_from_its_exact_source_plane(results_file, run):
    path, frames = results_file
    dataset = load_results(path)
    assert dataset.sources == tuple(frames)
    with ScanReader(run) as scan:
        for frame_id, frame in zip(dataset.frame_ids, frames):
            view = prepare_detector_view(dataset, frame_id=frame_id, image=True)
            point = scan.point(frame.point_id)
            np.testing.assert_array_equal(view.image, point.frame(frame.depth_index))
            assert view.image.dtype == np.int32
            index = dataset.frame_ids.index(frame_id)
            assert dataset.depths[index] == point.depth_um[frame.depth_index]
            assert tuple(dataset.starts[index]) == point.start
            assert tuple(dataset.groups[index]) == point.group
            assert dataset.detector_ids[index] == point.detector_id
    # The two points differ, so the planes cannot have been swapped.
    first = prepare_detector_view(dataset, frame_id="f25", image=True).image
    second = prepare_detector_view(dataset, frame_id="s30", image=True).image
    assert not np.array_equal(first, second)


def test_results_written_before_the_source_datasets_still_validate_and_load(results_file, tmp_path):
    path, _ = results_file
    older = tmp_path / "older.h5"
    with h5py.File(path) as source, h5py.File(older, "w") as target:
        for key, value in source.attrs.items():
            target.attrs[key] = value
        for name in source:
            source.copy(source[name], target, name=name)
        del target["frames/source_point_ids"]
        del target["frames/source_depth_indices"]
    assert validate_results_file(older).n_frames == 4
    dataset = load_results(older)
    assert dataset.sources == (None,) * 4
    assert dataset.input_images[0].endswith("points/synthetic_a.h5")


def test_converted_xml_results_have_no_scan_sources(results_file, indexer, tmp_path):
    from lauelab.visualization import convert_xml

    _, frames = results_file
    xml_path = tmp_path / "indexed.xml"
    indexer.write_many_xml(indexer.index_many(frames[:2]), xml_path)
    converted = convert_xml(xml_path, tmp_path / "converted.h5", geometry=GEOMETRY_FILE)
    dataset = load_results(converted)
    assert dataset.sources == (None, None)
    assert dataset.input_images == (frames[0].path, frames[1].path)


def test_a_copied_point_indexes_and_reopens_without_its_scan(indexer, tmp_path):
    source = _source(tmp_path / "in", name="Twin2_wire_1.h5")
    result = reconstruct_scan([source], tmp_path / "scan", geometry=GEOMETRY_FILE, detector=0,
                              threads_per_worker=1, **_options(wire_edge="both",
                                                               output_pixel_type=None))
    copied = Path(shutil.copy(result.outcomes[0].output, tmp_path / "copied.h5"))
    shutil.rmtree(tmp_path / "scan")
    shutil.rmtree(tmp_path / "in")

    frames = [ScanFrame(copied, "Twin2_wire_1", index) for index in (31, 20)]
    indexer.write_results(indexer.index_many(frames), tmp_path / "indexed.h5",
                          frame_ids=["d31", "d20"])
    dataset = load_results(tmp_path / "indexed.h5")
    assert dataset.sources == tuple(frames)
    with PointReader(copied) as point:
        for frame_id, frame in zip(("d31", "d20"), frames):
            view = prepare_detector_view(dataset, frame_id=frame_id, image=True)
            np.testing.assert_array_equal(view.image, point.frame(frame.depth_index))
            index = dataset.frame_ids.index(frame_id)
            assert dataset.depths[index] == point.depth_um[frame.depth_index]


# --- Per-depth export ---------------------------------------------------------

def _objects(handle):
    objects = {}
    handle.visititems(lambda name, obj: objects.setdefault(name, obj))
    return objects


def _summary(path):
    tags = {}
    lines = Path(path).read_text().splitlines()
    array_start = None
    for index, line in enumerate(lines):
        match = re.match(r"^\$(\w+)\s+(.*?)(\s*//.*)?$", line)
        if match:
            tags[match.group(1)] = match.group(2).rstrip()
            if match.group(1) == "array0":
                array_start = index + 1
    return tags, "\n".join(lines[array_start:])


@pytest.mark.parametrize(("variant", "changes"), [
    (None, {}),
    ("both", {"output_pixel_type": None}),
    ("norm_exponent", {"output_pixel_type": 3}),
    ("cosmic", {}),
])
def test_export_equals_the_per_depth_writer(tmp_path, variant, changes):
    source = _source(tmp_path, variant)
    options = _options(variant, **changes)
    direct = Reconstructor(GEOMETRY_FILE, 0, num_threads=1, **options).reconstruct(
        source, tmp_path / "direct_"
    )
    assert direct.success, direct.error
    scan = reconstruct_scan([source], tmp_path / "run", geometry=GEOMETRY_FILE, detector=0,
                            point_ids=["p"], threads_per_worker=1, **options)
    exported = export_per_depth(scan.outcomes[0].output, tmp_path / "export_")

    assert len(exported) == len(direct.output_files)
    assert exported[-1] == str(tmp_path / "export_summary.txt")
    for left_path, right_path in zip(direct.output_files[:-1], exported[:-1]):
        with h5py.File(left_path) as left, h5py.File(right_path) as right:
            left_objects, right_objects = _objects(left), _objects(right)
            assert left_objects.keys() == right_objects.keys()
            assert left.attrs.keys() == right.attrs.keys()
            for name in left_objects:
                a, b = left_objects[name], right_objects[name]
                assert type(a) is type(b), name
                # Export adds NeXus group annotations; original metadata and pixels agree.
                extra = set(b.attrs) - set(a.attrs)
                allowed = {"NX_class", "default"} if name == "entry1" else (
                    {"NX_class", "signal", "axes"} if name == "entry1/data" else set()
                )
                assert extra <= allowed, name
                assert set(a.attrs) <= set(b.attrs), name
                for attribute in a.attrs:
                    np.testing.assert_array_equal(a.attrs[attribute], b.attrs[attribute])
                if isinstance(a, h5py.Dataset):
                    assert a.dtype == b.dtype, name
                    assert a.shape == b.shape, name
                    np.testing.assert_array_equal(a, b, err_msg=name)
    left_tags, left_array = _summary(direct.output_files[-1])
    right_tags, right_array = _summary(exported[-1])
    excluded = {"ws_outfile", "executionTime"}
    assert {tag: value for tag, value in left_tags.items() if tag not in excluded} == {
        tag: value for tag, value in right_tags.items() if tag not in excluded
    }
    assert left_array == right_array

    from lauelab.reconstruct import PerDepthReader
    from lauelab.reconstruct.inspection import depth_trace

    with PointReader(scan.outcomes[0].output) as point, \
            PerDepthReader(exported[:-1], "p") as exported_point:
        for name in ("shape", "dtype", "detector_id", "detector_size", "start", "group",
                     "norm_rescale", "norm_threshold"):
            assert getattr(exported_point, name) == getattr(point, name)
        np.testing.assert_array_equal(exported_point.depth_um, point.depth_um)
        np.testing.assert_allclose(exported_point.depth_intensity(), point.depth_intensity(), rtol=1e-12)
        np.testing.assert_array_equal(exported_point.reference("sum_reconstructed"),
                                      point.reference("sum_reconstructed"))
        np.testing.assert_array_equal(depth_trace(exported_point, (60, 65, 50, 55)).values,
                                      depth_trace(point, (60, 65, 50, 55)).values)


def test_exported_files_index_like_the_scan_frame(run, indexer, tmp_path):
    exported = export_per_depth(ScanReader(run).point_path("second"), tmp_path / "second_")
    depth_file = exported[30]
    _, _, processing = read_h5_frame(depth_file)
    assert processing == {"start": (0, 0), "group": (16, 16), "depth": 5.0}
    from_file = indexer.index(depth_file)
    from_scan = indexer.index(_frame(run, "second", 30))
    np.testing.assert_array_equal(from_file.image, from_scan.image)
    np.testing.assert_array_equal(from_file.peaks, from_scan.peaks)
    assert from_file.depth == from_scan.depth == 5.0
    assert from_file.metadata["scan_number"] == from_scan.metadata["scan_number"] == 7


def test_export_refuses_anything_but_a_complete_point_file(run, tmp_path):
    with pytest.raises(FileNotFoundError):
        export_per_depth(ScanReader(run).point_path("gone"), tmp_path / "gone_")
    with pytest.raises(InvalidScanFile, match="is a reconstruction-scan catalog"):
        export_per_depth(run, tmp_path / "catalog_")
    incomplete = shutil.copy(ScanReader(run).point_path("first"), tmp_path / "incomplete.h5")
    with h5py.File(incomplete, "r+") as handle:
        handle["entry1/reconstruction/point/complete"][()] = 0
    with pytest.raises(InvalidScanFile, match="not complete"):
        export_per_depth(incomplete, tmp_path / "incomplete_")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["incomplete.h5"]


def test_guide_example_indexes_around_the_brightest_depth(run, indexer, tmp_path):
    """The flow shown in docs/guides/reconstruction.md."""
    with ScanReader(run) as scan:
        point = scan.point("second")
        brightest = int(point.depth_intensity().argmax())
    frames = [_frame(run, "second", index) for index in range(brightest - 2, brightest + 3)]
    results = indexer.index_many(frames)
    indexer.write_results(results, tmp_path / "indexed.h5")
    dataset = load_results(tmp_path / "indexed.h5")
    assert dataset.sources == tuple(frames)
    assert brightest == 25
    np.testing.assert_array_equal(dataset.depths, [-2.0, -1.0, 0.0, 1.0, 2.0])


@pytest.mark.parametrize("missing", ["source_point_ids", "source_depth_indices"])
def test_source_reference_fields_must_be_present_together(results_file, tmp_path, missing):
    from lauelab.indexing import InvalidResultsFile

    path = tmp_path / "incomplete-source.h5"
    shutil.copyfile(results_file[0], path)
    with h5py.File(path, "r+") as target:
        del target[f"frames/{missing}"]
    with pytest.raises(InvalidResultsFile, match="both be present or both absent"):
        validate_results_file(path)
    with pytest.raises(InvalidResultsFile, match="both be present or both absent"):
        load_results(path)


@pytest.mark.parametrize(("field", "value"), [
    ("source_depth_indices", -1), ("source_point_ids", ""), ("input_images", ""),
])
def test_source_reference_values_must_identify_a_frame(results_file, tmp_path, field, value):
    from lauelab.indexing import InvalidResultsFile

    path = tmp_path / "invalid-source.h5"
    shutil.copyfile(results_file[0], path)
    with h5py.File(path, "r+") as target:
        target[f"frames/{field}"][0] = value
    with pytest.raises(InvalidResultsFile, match="scan source needs"):
        validate_results_file(path)
