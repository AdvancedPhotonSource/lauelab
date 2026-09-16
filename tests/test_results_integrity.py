# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Streaming XML output, writer failure state, structural validation, and safe publication."""

from pathlib import Path
from xml.etree import ElementTree

import h5py
import numpy as np
import pytest

from conftest import requires_liblaue

import lauelab
from lauelab import partial_path, publish_file
from lauelab.indexing import (
    Indexer, InvalidResultsFile, PeakParams, ResultsFileSummary, XmlResultsWriter,
    validate_results_file,
)
from lauelab.indexing.xml_utils import write_combined_xml
from lauelab.visualization import convert_xml
import lauelab.visualization.results as conversion

ROOT = Path(__file__).resolve().parents[1]
pytestmark = requires_liblaue
GEOMETRY = ROOT / "tests/data/geo/geoN_2022-03-29_14-15-05.xml"
CRYSTAL = ROOT / "tests/config/Ni.xml"
FRAMES = sorted((ROOT / "tests/data/synthetic/frames").glob("*.h5"))
PEAKS = PeakParams(
    boxsize=18, max_rfactor=0.5, min_size=3, min_separation=20,
    threshold=None, threshold_ratio=4.0, max_peaks=200,
)


@pytest.fixture(scope="module")
def indexer():
    return Indexer(GEOMETRY, CRYSTAL, peak_params=PEAKS)


@pytest.fixture(scope="module")
def results(indexer):
    return indexer.index_many(FRAMES)


def _steps_in(path: Path) -> int:
    return len(ElementTree.parse(path).getroot().findall("step"))


def _write_valid(indexer, results, path, ids=("a", "b", "c", "d")):
    with indexer.results_writer(path) as writer:
        for frame_id, result in zip(ids, results):
            writer.append(result, frame_id=frame_id)
    return path


# --- streaming XML writer ------------------------------------------------------------


@pytest.mark.parametrize("count", [0, 1, 4])
def test_streaming_xml_is_byte_identical_to_combined_xml(tmp_path, results, count):
    chosen = results[:count]
    combined = tmp_path / "combined.xml"
    streamed = tmp_path / "streamed.xml"
    write_combined_xml([result.to_step() for result in chosen], str(combined))

    with XmlResultsWriter(streamed) as writer:
        for position, result in enumerate(chosen):
            writer.append(result if position % 2 == 0 else result.to_step())

    assert streamed.read_bytes() == combined.read_bytes()
    assert writer.count == count and not writer.failed
    assert _steps_in(streamed) == count


def test_write_many_xml_matches_combined_and_replaces_existing(tmp_path, indexer, results):
    combined = tmp_path / "combined.xml"
    many = tmp_path / "many.xml"
    write_combined_xml([result.to_step() for result in results], str(combined))
    many.write_text("stale")

    indexer.write_many_xml(results, many)

    assert many.read_bytes() == combined.read_bytes()


def test_xml_writer_refuses_existing_unless_overwrite(tmp_path, results):
    path = tmp_path / "out.xml"
    path.write_text("keep")

    with pytest.raises(FileExistsError):
        XmlResultsWriter(path).__enter__()
    assert path.read_text() == "keep"
    with XmlResultsWriter(path, overwrite=True) as writer:
        writer.append(results[0])
    assert _steps_in(path) == 1


def test_xml_writer_rejects_misuse(tmp_path, results):
    writer = XmlResultsWriter(tmp_path / "out.xml")
    with pytest.raises(RuntimeError, match="context manager"):
        writer.append(results[0])
    with writer:
        with pytest.raises(TypeError, match="FrameResult or a Step"):
            writer.append("not a result")
    assert _steps_in(tmp_path / "out.xml") == 0


def test_xml_write_failure_marks_writer_and_still_terminates_document(tmp_path, results):
    path = tmp_path / "out.xml"
    with XmlResultsWriter(path) as writer:
        writer.append(results[0])
        real_write = writer._handle.write
        calls = {"n": 0}

        def flaky(text):
            calls["n"] += 1
            if calls["n"] == 1:
                raise OSError("disk full")
            return real_write(text)

        writer._handle.write = flaky
        with pytest.raises(OSError, match="disk full"):
            writer.append(results[1])
        assert writer.failed and isinstance(writer.error, OSError)
        with pytest.raises(RuntimeError, match="failed earlier"):
            writer.append(results[2])
        assert writer.count == 1

    # The closing tag was still written, so the partial document parses.
    assert _steps_in(path) == 1


def test_exception_in_xml_body_still_closes_the_document(tmp_path, results):
    path = tmp_path / "out.xml"
    with pytest.raises(KeyError):
        with XmlResultsWriter(path) as writer:
            writer.append(results[0])
            writer.append(results[1])
            raise KeyError("consumer")

    assert not writer.failed
    assert _steps_in(path) == 2


def test_xml_partial_write_failure_is_truncated_to_the_last_complete_step(tmp_path, results):
    path = tmp_path / "out.xml"
    with XmlResultsWriter(path) as writer:
        writer.append(results[0])
        real_write = writer._handle.write

        def half_then_fail(text):
            real_write(text[: len(text) // 2])  # a partial chunk reaches the file
            raise OSError("disk full")

        writer._handle.write = half_then_fail
        with pytest.raises(OSError, match="disk full"):
            writer.append(results[1])
        writer._handle.write = real_write

    assert writer.failed and writer.count == 1
    assert _steps_in(path) == 1  # the half step was truncated away; the document parses


def test_write_many_xml_failure_leaves_existing_destination_untouched(tmp_path, indexer, results):
    path = tmp_path / "many.xml"
    indexer.write_many_xml(results[:1], path)
    before = path.read_bytes()
    from dataclasses import replace as dc_replace
    without_snapshot = dc_replace(results[1], _step=None)

    with pytest.raises(RuntimeError, match="no XML step snapshot"):
        indexer.write_many_xml([results[0], without_snapshot], path)

    assert path.read_bytes() == before
    assert sorted(item.name for item in tmp_path.iterdir()) == ["many.xml"]


# --- HDF5 writer failure state ----------------------------------------------------------


def test_append_failure_marks_writer_and_file_fails_validation(tmp_path, indexer, results):
    path = tmp_path / "results.h5"
    with indexer.results_writer(path) as writer:
        writer.append(results[0], frame_id="first")
        real_append = writer._append

        def failing(name, values):
            if name == "/frames/pattern_offsets":
                raise OSError("simulated write failure")
            return real_append(name, values)

        writer._append = failing
        with pytest.raises(OSError, match="simulated"):
            writer.append(results[3], frame_id="second")
        assert writer.failed and writer.count == 1
        with pytest.raises(RuntimeError, match="failed earlier"):
            writer.append(results[1], frame_id="third")

    with pytest.raises(InvalidResultsFile, match="pattern_offsets"):
        validate_results_file(path)
    assert lauelab.is_results_file(path)  # the marker alone says nothing about validity


def test_rejected_arguments_leave_writer_usable(tmp_path, indexer, results):
    path = tmp_path / "results.h5"
    with indexer.results_writer(path) as writer:
        writer.append(results[0], frame_id="a")
        with pytest.raises(ValueError, match="unique"):
            writer.append(results[1], frame_id="a")
        with pytest.raises(TypeError, match="mix"):
            writer.append(results[1], frame_id=7)
        with pytest.raises(TypeError, match="FrameResult"):
            writer.append("nope", frame_id="b")
        assert not writer.failed
        writer.append(results[1], frame_id="b")

    summary = validate_results_file(path, frame_ids=["a", "b"])
    assert summary.n_frames == 2


# --- structural validation -------------------------------------------------------------


def test_validation_summary_and_manifest_checks(tmp_path, indexer, results):
    path = _write_valid(indexer, results, tmp_path / "results.h5")

    summary = validate_results_file(path, frame_ids=["a", "b", "c", "d"], n_frames=4)

    assert isinstance(summary, ResultsFileSummary)
    assert summary.version == 1 and summary.n_frames == 4
    assert summary.n_peaks == sum(result.n_peaks for result in results)
    assert summary.n_patterns == sum(result.n_patterns for result in results)
    assert summary.n_assignments == sum(result.n_indexed for result in results)
    assert summary.frame_ids == ("a", "b", "c", "d")
    assert summary.has_crystal and summary.has_geometry_text
    assert summary.source is None and summary.lauelab_version
    with pytest.raises(InvalidResultsFile, match="manifest has 3"):
        validate_results_file(path, frame_ids=["a", "b", "c"])
    with pytest.raises(InvalidResultsFile, match="frame 2 has identity 'c', manifest has 'x'"):
        validate_results_file(path, frame_ids=["a", "b", "x", "d"])
    with pytest.raises(InvalidResultsFile, match="expected 5"):
        validate_results_file(path, n_frames=5)


def test_empty_and_pattern_free_runs_are_valid(tmp_path, indexer):
    empty = tmp_path / "empty.h5"
    indexer.write_results([], empty)
    assert validate_results_file(empty).n_frames == 0

    no_crystal = Indexer(GEOMETRY, peak_params=PEAKS)
    peaks_only = tmp_path / "peaks-only.h5"
    no_crystal.write_results(no_crystal.index_many(FRAMES[:2]), peaks_only, frame_ids=[10, 11])
    summary = validate_results_file(peaks_only, frame_ids=[10, 11])
    assert summary.n_patterns == 0 and summary.n_assignments == 0
    assert summary.n_peaks > 0 and not summary.has_crystal
    assert summary.frame_ids == (10, 11)


def _tamper(path, action):
    with h5py.File(path, "r+") as target:
        action(target)


@pytest.mark.parametrize(
    ("action", "message"),
    [
        (lambda f: f["/peaks/fit_x"].resize((3,)), "'/peaks/fit_y' has 97 rows, expected 3"),
        (lambda f: f["/frames/peak_offsets"].__setitem__(-1, 1), "does not partition"),
        (lambda f: f["/frames/n_peaks"].__setitem__(0, 99), "n_peaks' disagrees"),
        (lambda f: f["/patterns/n_indexed"].__setitem__(0, 1), "n_indexed' disagrees"),
        (lambda f: f["/patterns/rank"].__setitem__(0, 5), "restart at zero"),
        (lambda f: f.__delitem__("/patterns/goodness"), "missing dataset '/patterns/goodness'"),
        (lambda f: f.__delitem__("/assignments"), "missing group 'assignments'"),
        (lambda f: f.attrs.__setitem__("version", 99), "unsupported"),
        (lambda f: f.attrs.__setitem__("format", "other"), "not a"),
        (lambda f: f["/frames/frame_ids"].__setitem__(1, "a"), "identities repeat"),
        (lambda f: (f.__delitem__("/frames/depths"), f.create_dataset(
            "/frames/depths", data=np.zeros(4, dtype=np.float32))), "has dtype float32"),
        (lambda f: f["/crystal/atom_symbols"].__setitem__(0, "X") or f.__delitem__("/crystal/atom_labels"),
         "missing dataset '/crystal/atom_labels'"),
    ],
)
def test_structural_defects_are_named(tmp_path, indexer, results, action, message):
    path = _write_valid(indexer, results, tmp_path / "results.h5")
    _tamper(path, action)

    with pytest.raises(InvalidResultsFile, match=message):
        validate_results_file(path)


def test_two_dimensional_frame_ids_are_a_named_defect(tmp_path, indexer, results):
    path = _write_valid(indexer, results, tmp_path / "results.h5")
    with h5py.File(path, "r+") as target:
        del target["/frames/frame_ids"]
        target.create_dataset("/frames/frame_ids", data=np.zeros((4, 2), dtype=np.int32))

    with pytest.raises(InvalidResultsFile, match="one-dimensional"):
        validate_results_file(path)


def test_marker_only_and_non_hdf5_files(tmp_path):
    marker_only = tmp_path / "marker.h5"
    with h5py.File(marker_only, "w") as target:
        target.attrs["format"] = "lauelab-indexing-results"
        target.attrs["version"] = 1
    assert lauelab.is_results_file(marker_only)
    with pytest.raises(InvalidResultsFile, match="missing group"):
        validate_results_file(marker_only)

    text = tmp_path / "text.h5"
    text.write_text("not hdf5")
    with pytest.raises(OSError):
        validate_results_file(text)
    with pytest.raises(OSError):
        validate_results_file(tmp_path / "absent.h5")


# --- publication primitive -------------------------------------------------------------


def test_partial_path_and_publish_file(tmp_path):
    final = tmp_path / "out.h5"
    partial = partial_path(final)
    assert partial == tmp_path / "out.h5.partial"
    partial.write_text("new")

    assert publish_file(partial, final) == final
    assert final.read_text() == "new" and not partial.exists()

    partial.write_text("newer")
    with pytest.raises(FileExistsError):
        publish_file(partial, final)
    assert final.read_text() == "new" and partial.read_text() == "newer"
    publish_file(partial, final, overwrite=True)
    assert final.read_text() == "newer" and not partial.exists()

    with pytest.raises(FileNotFoundError):
        publish_file(partial, final, overwrite=True)
    other = tmp_path / "sub"
    other.mkdir()
    partial.write_text("x")
    with pytest.raises(ValueError, match="same directory"):
        publish_file(partial, other / "out.h5")


def test_publish_file_falls_back_to_a_checked_rename_without_hard_links(tmp_path, monkeypatch):
    import errno
    import os

    def no_links(source, target):
        raise OSError(errno.EPERM, "hard links are not permitted here")

    monkeypatch.setattr(os, "link", no_links)
    final = tmp_path / "out.h5"
    partial = partial_path(final)
    partial.write_text("new")
    assert publish_file(partial, final) == final
    assert final.read_text() == "new" and not partial.exists()

    partial.write_text("newer")
    with pytest.raises(FileExistsError):
        publish_file(partial, final)
    assert final.read_text() == "new" and partial.exists()


# --- conversion publication ---------------------------------------------------------------


def test_convert_xml_publishes_only_a_validated_file(tmp_path, indexer, results):
    xml_path = tmp_path / "run.xml"
    indexer.write_many_xml(results, xml_path)

    output = convert_xml(xml_path)

    assert output == tmp_path / "run.h5"
    assert sorted(item.name for item in tmp_path.iterdir()) == ["run.h5", "run.xml"]
    summary = validate_results_file(output, n_frames=4)
    assert summary.source == str(xml_path)
    assert summary.n_peaks == sum(result.n_peaks for result in results)


def test_convert_xml_failure_leaves_no_partial_or_destination(tmp_path, indexer, results, monkeypatch):
    xml_path = tmp_path / "run.xml"
    indexer.write_many_xml(results, xml_path)

    def reject(path):
        raise InvalidResultsFile(f"{path}: injected")

    monkeypatch.setattr(conversion, "validate_results_file", reject)
    with pytest.raises(InvalidResultsFile, match="injected"):
        convert_xml(xml_path)

    assert sorted(item.name for item in tmp_path.iterdir()) == ["run.xml"]


def test_convert_xml_does_not_clobber_a_destination_that_appeared_meanwhile(
    tmp_path, indexer, results, monkeypatch
):
    xml_path = tmp_path / "run.xml"
    indexer.write_many_xml(results, xml_path)
    destination = tmp_path / "run.h5"
    real_write = conversion._write_converted

    def write_then_race(output_path, *args):
        real_write(output_path, *args)
        destination.write_text("someone else's valid file")

    monkeypatch.setattr(conversion, "_write_converted", write_then_race)
    with pytest.raises(FileExistsError):
        convert_xml(xml_path)

    assert destination.read_text() == "someone else's valid file"
    assert not partial_path(destination).exists()
    assert convert_xml(xml_path, overwrite=True) == destination
    validate_results_file(destination)


def test_concurrent_conversions_of_one_destination_cannot_corrupt_each_other(
    tmp_path, indexer, results, monkeypatch
):
    xml_path = tmp_path / "run.xml"
    indexer.write_many_xml(results, xml_path)
    destination = tmp_path / "run.h5"
    real_write = conversion._write_converted
    state = {"nested": False}

    def write_and_race(output_path, *args):
        real_write(output_path, *args)
        if not state["nested"]:
            state["nested"] = True
            # A second conversion starts and finishes while the first is
            # between writing and publishing.
            assert convert_xml(xml_path, destination) == destination
            validate_results_file(destination)

    monkeypatch.setattr(conversion, "_write_converted", write_and_race)
    with pytest.raises(FileExistsError):
        convert_xml(xml_path, destination)

    validate_results_file(destination, n_frames=4)
    assert sorted(item.name for item in tmp_path.iterdir()) == ["run.h5", "run.xml"]


def test_convert_xml_classifies_missing_and_malformed_documents(tmp_path):
    with pytest.raises(FileNotFoundError):
        convert_xml(tmp_path / "absent.xml")
    malformed = tmp_path / "malformed.xml"
    malformed.write_text('<?xml version="1.0" ?>\n<AllSteps><step>')
    with pytest.raises(ElementTree.ParseError):
        convert_xml(malformed)
    assert sorted(item.name for item in tmp_path.iterdir()) == ["malformed.xml"]


def test_convert_xml_without_geometry_context_is_valid_and_reports_it(tmp_path, indexer, results):
    xml_path = tmp_path / "run.xml"
    indexer.write_many_xml(results, xml_path)
    text = xml_path.read_text().replace(str(GEOMETRY), str(tmp_path / "gone.xml"))
    xml_path.write_text(text)

    output = convert_xml(xml_path)

    summary = validate_results_file(output)
    assert summary.has_crystal and not summary.has_geometry_text


@pytest.mark.parametrize("bad_index", [-1, 100000])
def test_validator_rejects_invalid_assignment_indices(tmp_path, indexer, results, bad_index):
    path = _write_valid(indexer, results, tmp_path / "bad-index.h5")
    with h5py.File(path, "r+") as target:
        target["/assignments/peak_index"][0] = bad_index
    with pytest.raises(InvalidResultsFile, match="assignment 0.*outside frame"):
        validate_results_file(path)


def test_assignment_indices_are_checked_against_the_owning_frame(tmp_path, indexer, results):
    # Put the smaller frame after the larger one, so the bad index is within
    # the file's total peak count and another frame's local peak range.
    indexed = sorted((r for r in results if r.n_patterns), key=lambda r: r.n_peaks, reverse=True)
    large, small = indexed[0], indexed[-1]
    assert large.n_peaks > small.n_peaks
    path = tmp_path / "wrong-owner.h5"
    indexer.write_results([large, small], path, frame_ids=["large", "small"])
    with h5py.File(path, "r+") as target:
        pattern = target["/frames/pattern_offsets"][1]
        assignment = target["/patterns/assignment_offsets"][pattern]
        target["/assignments/peak_index"][assignment] = small.n_peaks
    with pytest.raises(InvalidResultsFile, match="outside frame 'small'"):
        validate_results_file(path)


def test_assignment_validation_checks_beyond_the_first_chunk(tmp_path, indexer, results):
    from dataclasses import replace

    result = next(r for r in results if r.n_patterns)
    original = result.patterns[0]
    count = 65537
    pattern = replace(
        original, pk_index=np.zeros(count, dtype=np.int32),
        hkl=np.repeat(original.hkl[:1], count, axis=0),
        err_deg=np.repeat(original.err_deg[:1], count),
        energy_kev=np.repeat(original.energy_kev[:1], count),
        pred_intens=np.repeat(original.pred_intens[:1], count),
    )
    path = tmp_path / "chunk-boundary.h5"
    indexer.write_results([replace(result, patterns=(pattern,))], path)
    assert validate_results_file(path).n_assignments == count
    with h5py.File(path, "r+") as target:
        target["/assignments/peak_index"][65536] = result.n_peaks
    with pytest.raises(InvalidResultsFile, match="assignment 65536.*outside frame"):
        validate_results_file(path)
