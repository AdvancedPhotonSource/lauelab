# Copyright 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Round-trip tests for indexing-results HDF5 files."""

from dataclasses import fields, replace
from pathlib import Path
import time

import h5py
import numpy as np
import pytest

from conftest import requires_liblaue

from lauelab.indexing import FrameMetadata, Indexer, PeakParams, ResultsWriter
from lauelab.visualization import ResultSet, VisualizationDataset, convert_xml, load_results

ROOT = Path(__file__).resolve().parents[1]
GEOMETRY = ROOT / "tests/data/geo/geoN_2022-03-29_14-15-05.xml"
CRYSTAL = ROOT / "tests/config/Ni.xml"
FRAMES = ROOT / "tests/data/synthetic/frames"


def _assert_visualization_equal(actual, expected):
    assert actual.frame_ids == expected.frame_ids
    assert actual.detector_ids == expected.detector_ids
    assert actual.input_images == expected.input_images
    for item in fields(VisualizationDataset):
        name = item.name
        if name in {"frame_ids", "detector_ids", "input_images", "images", "crystal", "geometry"}:
            continue
        actual_value = getattr(actual, name)
        expected_value = getattr(expected, name)
        if name == "peaks":
            for field_name in actual_value.dtype.names:
                np.testing.assert_allclose(
                    actual_value[field_name], expected_value[field_name],
                    rtol=2e-6, atol=2e-6, equal_nan=True,
                )
        else:
            np.testing.assert_allclose(
                actual_value, expected_value, rtol=2e-6, atol=2e-6,
                equal_nan=True,
            )
    assert actual.crystal == expected.crystal
    assert actual.geometry.detector_count == expected.geometry.detector_count
    for slot in range(actual.geometry.detector_count):
        actual_detector = actual.geometry.detector(slot)
        expected_detector = expected.geometry.detector(slot)
        np.testing.assert_array_equal(actual_detector.translation, expected_detector.translation)
        np.testing.assert_array_equal(actual_detector.rotation, expected_detector.rotation)
    assert actual.images == (None,) * actual.n_frames


@requires_liblaue
def test_streaming_writer_round_trips_synthetic_results(tmp_path):
    indexer = Indexer(
        GEOMETRY,
        CRYSTAL,
        peak_params=PeakParams(threshold=None, max_peaks=200),
    )
    paths = [
        FRAMES / "synthetic_ni_grain_a.h5",
        FRAMES / "synthetic_ni_empty.h5",
        FRAMES / "synthetic_ni_two_grains.h5",
    ]
    results = [
        indexer.index(
            path,
            depth=index,
            metadata=FrameMetadata(sample_position=(index, index + 1, index + 2)),
            keep_image=False,
        )
        for index, path in enumerate(paths)
    ]
    results.insert(2, replace(results[1], peaks=results[1].peaks[:0], patterns=()))
    expected = ResultSet.from_indexer(
        indexer, results, frame_ids=("grain-a", "no-patterns", "empty", "two-grains")
    ).to_visualization()
    output = tmp_path / "results.h5"

    with indexer.results_writer(output) as writer:
        for frame_id, result in zip(expected.frame_ids, results, strict=True):
            writer.append(result, frame_id=frame_id)

    actual = load_results(output)
    _assert_visualization_equal(actual, expected)
    assert actual.frame_n_peaks[2] == 0
    assert not np.any(np.isin(actual.pattern_frame_indices, (1, 2)))


@requires_liblaue
def test_one_call_live_writer_matches_result_set(tmp_path):
    indexer = Indexer(GEOMETRY, CRYSTAL)
    paths = sorted(FRAMES.glob("*.h5"))
    results = indexer.index_many(paths)
    expected = ResultSet.from_indexer(indexer, results, frame_ids=(10, 20, 30, 40)).to_visualization()
    output = tmp_path / "results.h5"

    indexer.write_results(results, output, frame_ids=expected.frame_ids)

    _assert_visualization_equal(load_results(output), expected)


@requires_liblaue
def test_writer_records_layout_and_run_configuration(tmp_path):
    indexer = Indexer(GEOMETRY, CRYSTAL)
    output = tmp_path / "results.h5"
    with ResultsWriter(
        output,
        crystal=indexer.crystal,
        geometry=indexer.geometry,
        peak_params=indexer.peak_params,
        index_params=indexer.index_params,
        detector_index=indexer.detector_index,
        detector_id=indexer.detector_id,
        cosmic_filter=indexer.cosmic_filter,
    ):
        pass

    with h5py.File(output) as source:
        assert source["frames/frame_ids"].shape == (0,)
        assert source["frames/peak_offsets"][...].tolist() == [0]
        assert source["frames/pattern_offsets"][...].tolist() == [0]
        assert source["patterns/assignment_offsets"][...].tolist() == [0]
        assert source["patterns/reciprocal"].attrs["units"] == "1/nm"
        assert bool(source["patterns/reciprocal"].attrs["includes_two_pi"])
        assert source["run"].attrs["program"] == "liblaue"
        assert np.isnan(source["run"].attrs["threshold_ratio"])


@requires_liblaue
def test_load_results_rejects_unknown_version(tmp_path):
    indexer = Indexer(GEOMETRY, CRYSTAL)
    output = tmp_path / "results.h5"
    indexer.write_results([], output)
    with h5py.File(output, "r+") as source:
        source.attrs["version"] = 2
    with pytest.raises(ValueError, match="unsupported.*version 2"):
        load_results(output)


@requires_liblaue
def test_writer_validates_path_and_frame_ids(tmp_path):
    indexer = Indexer(GEOMETRY, CRYSTAL)
    result = indexer.index(np.zeros((2, 2), dtype=np.uint16), keep_image=False)
    output = tmp_path / "results.h5"
    indexer.write_results([result], output)
    with pytest.raises(FileExistsError):
        indexer.write_results([result], output)
    with pytest.raises(ValueError, match="fewer values"):
        indexer.write_results([result], tmp_path / "few.h5", frame_ids=())
    with pytest.raises(ValueError, match="contain 1 values"):
        indexer.write_results([result], tmp_path / "many.h5", frame_ids=(1, 2))


@requires_liblaue
def test_load_results_accepts_frame_id_and_geometry_overrides(tmp_path):
    indexer = Indexer(GEOMETRY, CRYSTAL)
    result = indexer.index(np.zeros((2, 2), dtype=np.uint16), keep_image=False)
    output = tmp_path / "results.h5"
    indexer.write_results([result], output)

    loaded = load_results(output, frame_ids=("replacement",), geometry=GEOMETRY)

    assert loaded.frame_ids == ("replacement",)
    assert loaded.geometry.path == GEOMETRY


@requires_liblaue
def test_load_results_uses_embedded_geometry_and_ignores_unknown_data(tmp_path):
    geometry_path = tmp_path / "geometry.xml"
    geometry_path.write_bytes(GEOMETRY.read_bytes())
    indexer = Indexer(geometry_path, CRYSTAL)
    output = tmp_path / "results.h5"
    indexer.write_results([], output)
    geometry_path.unlink()
    with h5py.File(output, "r+") as source:
        source.create_dataset("future_addition", data=[1])

    loaded = load_results(output)

    assert loaded.geometry.detector_count == indexer.geometry.detector_count
    assert loaded.n_frames == 0


@requires_liblaue
def test_geometry_snapshot_survives_external_changes_and_explicit_override_wins(tmp_path):
    geometry_path = tmp_path / "geometry.xml"
    original = GEOMETRY.read_text()
    geometry_path.write_text(original)
    indexer = Indexer(geometry_path, CRYSTAL)
    output = tmp_path / "results.h5"
    indexer.write_results([], output)
    geometry_path.write_text(original.replace("28.720 3.010 513.097", "28.720 3.010 613.097"))

    loaded = load_results(output)
    overridden = load_results(output, geometry=geometry_path)
    np.testing.assert_array_equal(
        loaded.geometry.detector().translation, indexer.geometry.detector().translation
    )
    assert not np.array_equal(
        overridden.geometry.detector().translation, loaded.geometry.detector().translation
    )
    assert loaded.geometry.path.read_text() == original


@requires_liblaue
def test_geometry_path_is_used_when_no_snapshot_exists(tmp_path):
    indexer = Indexer(GEOMETRY, CRYSTAL)
    output = tmp_path / "results.h5"
    indexer.write_results([], output)
    with h5py.File(output, "r+") as source:
        del source["geometry/xml"]
    assert load_results(output).geometry.path == GEOMETRY


@requires_liblaue
def test_native_and_converted_files_preserve_acquisition_metadata(tmp_path):
    indexer = Indexer(GEOMETRY, CRYSTAL)
    metadata = FrameMetadata(
        exposure_seconds=1.25, beam_bad=1, light_on=0,
        hutch_temperature=25.5, sample_distance=3.5,
    )
    frame = np.zeros((2, 2), dtype=np.uint16)
    results = [
        indexer.index(frame, metadata=metadata, keep_image=False),
        indexer.index(frame, keep_image=False),
    ]
    output = tmp_path / "native.h5"
    indexer.write_results(results, output)
    xml_path = tmp_path / "converted.xml"
    indexer.write_many_xml(results, xml_path)
    converted = convert_xml(xml_path)
    for path in (output, converted):
        with h5py.File(path) as source:
            for name, value in metadata.as_dict().items():
                assert source[f"frames/{name}"][0] == value
                missing = source[f"frames/{name}"][1]
                if name in ("beam_bad", "light_on"):
                    assert missing == -1
                else:
                    assert np.isnan(missing)
            assert source["frames/exposure_seconds"].attrs["units"] == "s"
            for name in ("hutch_temperature", "sample_distance"):
                assert source[f"frames/{name}"].attrs["units"] == "unspecified"


@requires_liblaue
def test_converted_file_preserves_available_native_run_parameters(tmp_path):
    indexer = Indexer(GEOMETRY, CRYSTAL, cosmic_filter=True)
    result = indexer.index(np.zeros((2, 2), dtype=np.uint16), keep_image=False)
    xml_path = tmp_path / "results.xml"
    indexer.write_many_xml([result], xml_path)
    with h5py.File(convert_xml(xml_path)) as source:
        run = source["run"].attrs
        assert run["program"] == "liblaue"
        assert run["peak_program"] == "liblaue"
        assert bool(run["cosmic_filter"])
        for name in ("max_rfactor", "max_peaks", "min_separation", "peak_shape"):
            assert run[name] == getattr(indexer.peak_params, name)
        for name in ("kev_max_calc", "kev_max_test", "angle_tolerance_deg", "cone_deg", "hkl_prefer"):
            np.testing.assert_array_equal(run[name], getattr(indexer.index_params, name))
        assert "smooth" not in run and "min_size" not in run


@requires_liblaue
def test_streaming_and_one_call_writers_are_byte_identical(tmp_path, frozen_results_clock):
    indexer = Indexer(GEOMETRY, CRYSTAL)
    results = [indexer.index(FRAMES / "synthetic_ni_grain_a.h5", keep_image=False)]
    results.append(indexer.index(np.zeros((2, 2), dtype=np.uint16), keep_image=False))
    first, second = tmp_path / "first.h5", tmp_path / "second.h5"
    indexer.write_results(results, first, frame_ids=("grain", "empty"))
    time.sleep(1.1)  # Expose HDF5 wall-clock timestamps as well as /created.
    with indexer.results_writer(second) as writer:
        for frame_id, result in zip(("grain", "empty"), results, strict=True):
            writer.append(result, frame_id)
    assert first.read_bytes() == second.read_bytes()
