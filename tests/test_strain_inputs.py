# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
from dataclasses import replace

import numpy as np
import pytest

from lauelab.indexing import IndexParams, InputError, PeakParams, ResultsWriter
from lauelab.strain import StrainRefiner, prepare_reflections
from lauelab.visualization import ResultSet, load_visualization_xml
from strain_cases import frame_case, write_xml


def test_nonzero_pattern_uses_explicit_associations_and_chosen_geometry(tmp_path):
    geometry, crystal, frame, b, expected = frame_case(tmp_path)
    refiner = StrainRefiner(crystal, geometry=geometry, detector_index=2)
    result = refiner.refine(frame, pattern=1)
    assert result.success, result.message
    np.testing.assert_allclose(result.tensor_in("crystal"), expected, atol=2e-10)
    assert not np.allclose(result.measurements.qhat, frame.peaks["qhat"])
    np.testing.assert_allclose(result.predicted_xy, result.measurements.xy, atol=1e-7)
    assert result.pattern_index == 1
    assert result.measurements.provenance["detector_index"] == 2
    assert len(result.reflections) == len(frame.peaks)


def test_grouped_centroids_and_physical_depth_retain_their_geometry(tmp_path):
    geometry, crystal, frame, _, expected = frame_case(
        tmp_path, group=(2, 2), start=(10, 20), depth=125
    )
    result = StrainRefiner(crystal, geometry=geometry).refine(frame, pattern=1)
    assert result.success
    np.testing.assert_allclose(result.tensor_in("crystal"), expected, atol=2e-10)
    np.testing.assert_allclose(result.predicted_xy, result.measurements.xy, atol=1e-7)
    assert result.measurements.provenance["group"] == [2, 2]
    changed = prepare_reflections(frame, pattern=1, geometry=geometry, depth=0)
    assert changed.provenance["depth_um"] == 0
    assert not np.allclose(changed.qhat, result.measurements.qhat, atol=1e-10, rtol=0)


def test_source_selection_preserves_original_peak_ids(tmp_path):
    geometry, _, frame, _, _ = frame_case(tmp_path)
    selected = [0, 2, 4, 6, 8, 10]
    data = prepare_reflections(
        frame, pattern=1, geometry=geometry, peak_indices=selected
    )
    assert data.peak_indices.tolist() == selected
    np.testing.assert_array_equal(data.hkl, frame.patterns[1].hkl[selected])
    with pytest.raises(InputError, match="not assigned"):
        prepare_reflections(frame, pattern=1, geometry=geometry, peak_indices=[100])


def test_xml_and_hdf5_reuse_readers_and_recover_saved_measurements(tmp_path):
    geometry, crystal, frame, _, expected = frame_case(tmp_path)
    xml = write_xml(tmp_path / "index.xml", frame)
    hdf5 = tmp_path / "index.h5"
    with ResultsWriter(
        hdf5,
        crystal=crystal,
        geometry=geometry,
        peak_params=PeakParams(),
        index_params=IndexParams(),
        detector_index=2,
        detector_id="STRAIN-SYNTHETIC",
        cosmic_filter=False,
    ) as writer:
        writer.append(frame, frame_id="point-A")
    refiner = StrainRefiner(crystal, geometry=geometry)
    for source, frame_id, tolerance in ((xml, 0, 2e-10), (hdf5, "point-A", 2e-7)):
        result = refiner.refine(source, frame_id=frame_id, pattern=1)
        assert result.success, result.message
        np.testing.assert_allclose(
            result.tensor_in("crystal"), expected, atol=tolerance, rtol=0
        )
        assert result.frame_id == frame_id
        assert result.measurements.provenance["source"]["path"] == str(source.resolve())


def test_live_resultset_needs_no_xml_snapshot_or_image_copy(tmp_path):
    geometry, crystal, frame, _, _ = frame_case(tmp_path)
    assert frame._step is None
    source = ResultSet((frame,), frame_ids=("custom",), geometry=geometry)
    result = StrainRefiner(crystal).refine(source, frame_id="custom", pattern=1)
    assert result.success and result.frame_id == "custom"


@pytest.mark.parametrize(
    "corruption", ["missing", "out_of_bounds", "length", "fractional", "numeric_prefix"]
)
def test_strict_xml_preserves_per_pattern_failures_before_display_truncation(
    tmp_path, corruption
):
    geometry, crystal, frame, _, _ = frame_case(tmp_path)
    xml = write_xml(tmp_path / "bad.xml", frame)
    text = xml.read_text()
    if corruption == "missing":
        text = text.replace("<PkIndex>0 1</PkIndex>", "<PkIndex></PkIndex>", 1)
    elif corruption == "out_of_bounds":
        text = text.replace("<PkIndex>0 1</PkIndex>", "<PkIndex>0 10000</PkIndex>", 1)
    elif corruption == "length":
        text = text.replace("<PkIndex>0 1</PkIndex>", "<PkIndex>0</PkIndex>", 1)
    elif corruption == "fractional":
        text = text.replace("<PkIndex>0 1</PkIndex>", "<PkIndex>0 0.5</PkIndex>", 1)
    else:
        text = text.replace(
            "<PkIndex>0 1</PkIndex>", "<PkIndex>0 1garbage</PkIndex>", 1
        )
    xml.write_text(text)
    refiner = StrainRefiner(crystal, geometry=geometry)
    with pytest.raises(InputError):
        refiner.refine(xml, frame_id=0, pattern=0)
    results = refiner.refine_many(xml)
    assert [r.pattern_index for r in results] == [0, 1]
    assert results[0].status == "invalid_input"
    assert results[1].success
    if corruption not in ("fractional", "numeric_prefix"):
        assert (
            load_visualization_xml(xml).n_patterns == 2
        )  # Existing display behavior retained.


def test_all_patterns_default_and_selection_do_not_inherit_visualization_filters(
    tmp_path,
):
    geometry, crystal, frame, _, _ = frame_case(tmp_path)
    source = ResultSet((frame, frame), frame_ids=("b", "a"), geometry=geometry)
    refiner = StrainRefiner(crystal)
    results = refiner.refine_many(source)
    assert [(r.frame_id, r.pattern_index) for r in results] == [
        ("b", 0),
        ("b", 1),
        ("a", 0),
        ("a", 1),
    ]
    assert [r.status for r in results] == ["insufficient_support", "converged"] * 2
    assert len(refiner.refine_many(source, patterns="best")) == 2
    selected = refiner.refine_many(source, frame_ids=["a"], patterns=(1,))
    assert len(selected) == 1 and selected[0].success


def test_unexpected_solver_errors_stop_batch(tmp_path, monkeypatch):
    import lauelab.strain._core as core

    geometry, crystal, frame, _, _ = frame_case(tmp_path)

    def broken(*args, **kwargs):
        raise RuntimeError("unexpected internal failure")

    monkeypatch.setattr(core, "least_squares", broken)
    with pytest.raises(RuntimeError, match="unexpected"):
        StrainRefiner(crystal, geometry=geometry).refine_many(frame)


def test_detector_conflict_and_missing_calibration_are_input_errors(tmp_path):
    geometry, _, frame, _, _ = frame_case(tmp_path)
    with pytest.raises(InputError, match="geometry"):
        prepare_reflections(frame, pattern=1)
    wrong = replace(frame, metadata={"detector_id": "OTHER"})
    with pytest.raises((InputError, ValueError)):
        prepare_reflections(wrong, pattern=1, geometry=geometry, detector_index=2)


def test_preparation_and_configured_refinement_compose_without_losing_identity(
    tmp_path,
):
    geometry, crystal, frame, _, _ = frame_case(tmp_path)
    prepared = prepare_reflections(frame, pattern=1, geometry=geometry)
    result = StrainRefiner(crystal, geometry=geometry, detector_index=2).refine(
        prepared
    )
    assert result.success and result.pattern_index == 1
    with pytest.raises(InputError, match="identity"):
        prepare_reflections(prepared, pattern=999)


def test_geometry_snapshot_remains_accurate_after_source_file_changes(tmp_path):
    geometry, _, frame, _, _ = frame_case(tmp_path)
    snapshot = geometry._source_xml
    geometry.path.unlink()
    prepared = prepare_reflections(frame, pattern=1, geometry=geometry)
    assert prepared.provenance["geometry_xml"] == snapshot
    assert prepared.provenance["detector_calibration"]["translation"][2] == -100000


@pytest.mark.parametrize("error", [ValueError, TypeError])
def test_internal_solver_validation_errors_are_not_bad_measurements(
    tmp_path, monkeypatch, error
):
    import lauelab.strain._core as core

    geometry, crystal, frame, _, _ = frame_case(tmp_path)

    def broken(*args, **kwargs):
        raise error("unexpected internal solver failure")

    monkeypatch.setattr(core, "least_squares", broken)
    with pytest.raises(error, match="internal solver"):
        StrainRefiner(crystal, geometry=geometry).refine_many(frame)


def test_unselected_invalid_xml_count_does_not_hide_supported_pattern(tmp_path):
    geometry, crystal, frame, _, _ = frame_case(tmp_path)
    path = write_xml(tmp_path / "count.xml", frame)
    path.write_text(path.read_text().replace('Nindexed="2"', 'Nindexed="nan"', 1))
    refiner = StrainRefiner(crystal, geometry=geometry)
    assert refiner.refine(path, frame_id=0, pattern=1).success
    assert refiner.refine_many(path)[0].status == "invalid_input"
