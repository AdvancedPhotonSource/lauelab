# Copyright 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Merging must preserve scientific arrays, identities, and existing outputs."""

from pathlib import Path
import gc
from xml.etree import ElementTree as ET
from xml.parsers import expat

import h5py
import numpy as np
import pytest

from conftest import requires_liblaue
from lauelab.indexing import (
    Indexer,
    InvalidResultsFile,
    PeakParams,
    XmlResultsWriter,
    merge_results_files,
    merge_xml_files,
    validate_results_file,
)
from lauelab.visualization import load_results
import lauelab.indexing.merge as merging

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def indexed():
    indexer = Indexer(
        ROOT / "tests/data/geo/geoN_2022-03-29_14-15-05.xml",
        ROOT / "tests/config/Ni.xml",
        peak_params=PeakParams(
            boxsize=18,
            max_rfactor=0.5,
            min_size=3,
            min_separation=20,
            threshold=None,
            threshold_ratio=4,
            max_peaks=200,
        ),
    )
    return indexer, indexer.index_many(
        sorted((ROOT / "tests/data/synthetic/frames").glob("*.h5"))
    )


def _write(indexed, path, positions, ids=None):
    indexer, results = indexed
    indexer.write_results(
        [results[i] for i in positions],
        path,
        frame_ids=positions if ids is None else ids,
    )
    return path


def _assert_columns_equal(actual, expected):
    with h5py.File(actual, "r") as a, h5py.File(expected, "r") as b:
        names = []
        b.visititems(
            lambda name, item: (
                names.append(name) if isinstance(item, h5py.Dataset) else None
            )
        )
        for name in names:
            np.testing.assert_array_equal(a[name][()], b[name][()], err_msg=name)
            assert dict(a[name].attrs) == dict(b[name].attrs)


@requires_liblaue
@pytest.mark.parametrize("ids", [(8, 2, 41, 19), ("point/α", "empty", "b", "point/β")])
def test_merge_matches_one_writer_and_owns_all_arrays(
    tmp_path, indexed, ids, monkeypatch
):
    # Force copies across several boundaries, including within ragged records.
    monkeypatch.setattr(merging, "_COPY_ROWS", 3)
    paths = [
        _write(indexed, tmp_path / "a.h5", [0, 1], ids[:2]),
        _write(indexed, tmp_path / "empty.h5", []),
        _write(indexed, tmp_path / "b.h5", [2, 3], ids[2:]),
    ]
    expected = _write(indexed, tmp_path / "expected.h5", range(4), ids)
    destination = tmp_path / "merged.h5"
    summary = merge_results_files(iter(paths), destination, frame_ids=iter(ids))
    assert summary.path == destination and summary.frame_ids == ids
    assert (
        summary.n_frames == 4 and summary.n_patterns > 0 and summary.n_assignments > 0
    )
    _assert_columns_equal(destination, expected)
    with h5py.File(destination) as output:
        assert output.attrs["merged_sources"].tolist() == [str(p) for p in paths]
    for path in paths:
        path.unlink()
    assert load_results(destination).frame_ids == ids


@requires_liblaue
def test_empty_file_before_string_id_file_and_all_empty(tmp_path, indexed):
    empty = _write(indexed, tmp_path / "empty.h5", [])
    populated = _write(indexed, tmp_path / "one.h5", [0], ["α"])
    assert merge_results_files([empty, populated], tmp_path / "out.h5").frame_ids == (
        "α",
    )
    summary = merge_results_files([empty], tmp_path / "zero.h5", frame_ids=[])
    assert (
        summary.n_frames
        == summary.n_peaks
        == summary.n_patterns
        == summary.n_assignments
        == 0
    )


@requires_liblaue
def test_older_optional_scan_columns_are_filled(tmp_path, indexed):
    a = _write(indexed, tmp_path / "a.h5", [0])
    b = _write(indexed, tmp_path / "b.h5", [1])
    with h5py.File(a, "r+") as source:
        del source["frames/source_point_ids"], source["frames/source_depth_indices"]
    destination = tmp_path / "out.h5"
    merge_results_files([a, b], destination)
    with h5py.File(destination) as output:
        assert output["frames/source_point_ids"].asstr()[:].tolist() == ["", ""]
        assert output["frames/source_depth_indices"][:].tolist() == [-1, -1]


@requires_liblaue
@pytest.mark.parametrize(
    "fault",
    [
        "duplicate",
        "mixed_ids",
        "order",
        "invalid_offsets",
        "run",
        "geometry",
        "crystal",
        "units",
        "unknown_column",
    ],
)
def test_rejects_bad_or_incompatible_inputs_without_replacing_destination(
    tmp_path, indexed, fault
):
    a = _write(indexed, tmp_path / "a.h5", [0], [0])
    b = _write(
        indexed,
        tmp_path / "b.h5",
        [1],
        [0 if fault == "duplicate" else "one" if fault == "mixed_ids" else 1],
    )
    with h5py.File(b, "r+") as source:
        if fault == "invalid_offsets":
            source["frames/peak_offsets"][0] = 1
        elif fault == "run":
            source["run"].attrs["kev_max_calc"] = 123.0
        elif fault == "geometry":
            source["geometry/xml"][()] = "<different/>"
        elif fault == "crystal":
            source["crystal/lattice_parameters"][0] += 1
        elif fault == "units":
            source["patterns/reciprocal"].attrs["units"] = "wrong"
        elif fault == "unknown_column":
            source["frames/unknown"] = [5]
    destination = tmp_path / "out.h5"
    destination.write_bytes(b"previous result")
    with pytest.raises(InvalidResultsFile):
        merge_results_files(
            [a, b],
            destination,
            overwrite=True,
            frame_ids=[1, 0] if fault == "order" else None,
        )
    assert destination.read_bytes() == b"previous result"
    assert not list(tmp_path.glob("*.partial-*"))


@requires_liblaue
def test_copy_failure_preserves_destination_and_cleans_temporary(
    tmp_path, indexed, monkeypatch
):
    source = _write(indexed, tmp_path / "a.h5", [0])
    destination = tmp_path / "out.h5"
    destination.write_bytes(b"keep")
    original = h5py.Dataset.__getitem__

    def broken(data, key, *args, **kwargs):
        if data.name == "/peaks/fit_y":
            raise OSError("simulated read failure")
        return original(data, key, *args, **kwargs)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", broken)
    with pytest.raises(OSError, match="simulated read failure"):
        merge_results_files([source], destination, overwrite=True)
    assert destination.read_bytes() == b"keep"
    assert not list(tmp_path.glob("*.partial-*"))


@requires_liblaue
def test_contiguous_source_and_provenance(tmp_path, indexed):
    source = _write(indexed, tmp_path / "a.h5", [0, 1])
    with h5py.File(source, "r+") as data:
        names = list(data["frames"])
        for name in names:
            column = data["frames/" + name]
            values, attrs = column[()], dict(column.attrs)
            del data["frames/" + name]
            data.create_dataset("frames/" + name, data=values).attrs.update(attrs)
        data.attrs["created"] = "old"
        data.attrs["source"] = "single-converted.xml"
    destination = tmp_path / "out.h5"
    summary = merge_results_files([source], destination)
    assert summary.created != "old" and summary.source is None
    _assert_columns_equal(destination, source)
    with pytest.raises(FileExistsError):
        merge_results_files([source], destination)
    merge_results_files([source], destination, overwrite=True)
    assert validate_results_file(destination).n_frames == 2


@requires_liblaue
def test_merge_real_writer_xml_preserves_steps(tmp_path, indexed):
    _, results = indexed
    paths = []
    for number, chosen in enumerate([results[:2], [], results[2:]]):
        path = tmp_path / f"{number}.xml"
        with XmlResultsWriter(path) as writer:
            for result in chosen:
                writer.append(result)
        paths.append(path)
    destination = tmp_path / "merged.xml"
    assert merge_xml_files(paths, destination, n_steps=4) == 4
    expected = [
        ET.tostring(step).strip() for p in paths for step in ET.parse(p).getroot()
    ]
    assert [
        ET.tostring(step).strip() for step in ET.parse(destination).getroot()
    ] == expected


@pytest.mark.parametrize("block_bytes", [1, 7, 1024 * 1024])
def test_xml_token_boundaries_and_numeric_text(tmp_path, monkeypatch, block_bytes):
    monkeypatch.setattr(merging, "_COPY_BYTES", block_bytes)
    body = '<step original_xmlns="anything"><title>α &amp; β</title><x>1.000000000000001e-9</x></step>'
    a, b = tmp_path / "a.xml", tmp_path / "b.xml"
    a.write_text(
        '<?xml version="1.0" encoding="UTF-8"?><!--before--><AllSteps >'
        + body
        + "</AllSteps><!--after-->"
    )
    b.write_text("<AllSteps/>")
    destination = tmp_path / "out.xml"
    assert merge_xml_files([a, b], destination, n_steps=1) == 1
    assert body in destination.read_text()
    assert len(ET.parse(destination).getroot()) == 1


@pytest.mark.parametrize("encoding", ["ASCII", "US-ASCII", "UTF-8"])
def test_xml_supported_encoding_declarations(tmp_path, encoding):
    source = tmp_path / "input.xml"
    source.write_text(
        f'<?xml version="1.0" encoding="{encoding}"?><AllSteps><step/></AllSteps>'
    )
    assert merge_xml_files([source], tmp_path / "out.xml") == 1


def test_xml_parsers_are_released_between_files(tmp_path):
    paths = []
    for i in range(16):
        path = tmp_path / f"{i}.xml"
        path.write_text("<AllSteps><step/></AllSteps>")
        paths.append(path)
    parser_type = type(expat.ParserCreate())
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    try:
        before = sum(isinstance(obj, parser_type) for obj in gc.get_objects())
        assert merge_xml_files(paths, tmp_path / "out.xml") == 16
        assert sum(isinstance(obj, parser_type) for obj in gc.get_objects()) == before
    finally:
        if enabled:
            gc.enable()


@pytest.mark.parametrize(
    "text",
    [
        "<AllSteps><step>",
        "<Other/>",
        "<AllSteps><other/></AllSteps>",
        "<AllSteps>unexpected<step/></AllSteps>",
        '<AllSteps xmlns="namespace"/>',
        '<!DOCTYPE AllSteps [<!ENTITY value "hello">]><AllSteps><step>&value;</step></AllSteps>',
        '<?xml version="1.0" encoding="ISO-8859-1"?><AllSteps/>',
    ],
)
def test_bad_xml_preserves_existing_destination(tmp_path, text):
    a, b = tmp_path / "a.xml", tmp_path / "b.xml"
    a.write_text("<AllSteps><step/></AllSteps>")
    b.write_text(text)
    destination = tmp_path / "out.xml"
    destination.write_bytes(b"keep")
    with pytest.raises(ValueError):
        merge_xml_files([a, b], destination, overwrite=True)
    assert destination.read_bytes() == b"keep"
    assert not list(tmp_path.glob("*.partial-*"))


@pytest.mark.parametrize("n_steps", [-1, True, 1.5, 1])
def test_xml_expected_step_count(tmp_path, n_steps):
    source = tmp_path / "empty.xml"
    source.write_text("<AllSteps/>")
    destination = tmp_path / "out.xml"
    with pytest.raises(ValueError):
        merge_xml_files([source], destination, n_steps=n_steps)
    assert not destination.exists() and not list(tmp_path.glob("*.partial-*"))


@pytest.mark.parametrize("merge", [merge_results_files, merge_xml_files])
def test_path_errors_and_aliases(tmp_path, merge):
    source = tmp_path / "input"
    source.write_bytes(b"keep")
    destination = tmp_path / "out"
    with pytest.raises(ValueError, match="at least one"):
        merge([], destination)
    with pytest.raises(TypeError, match="iterable"):
        merge(source, destination)
    with pytest.raises(ValueError, match="repeat"):
        merge([source, source], destination)
    for alias in [source, destination]:
        if alias == destination:
            alias.hardlink_to(source)
        with pytest.raises(ValueError, match="input file"):
            merge([source], alias, overwrite=True)
    assert source.read_bytes() == b"keep"
