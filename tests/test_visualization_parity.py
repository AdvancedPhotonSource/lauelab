# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Rodrigues symmetry and reference options, and frame-only records for unindexed frames."""

from dataclasses import replace

import numpy as np
import pytest

from lauelab.analysis import (
    lattice_params_to_reciprocal, orientation_to_rodrigues, rodrigues_colors,
    symmetry_operations, symmetry_reduce_orientation,
)
from lauelab.analysis.orientation import _rotation_matrix
from lauelab.indexing import Cell, Crystal, FrameResult, Pattern
from lauelab.indexing.indexer import PEAK_DTYPE
from lauelab.visualization import (
    NO_PATTERN, Axis, DataScope, MapData, ResultSet, ScalarColor, peak_table, plot_map,
    prepare_map, selection_from_plotly,
)

CUBIC = Crystal("Ni", 225, Cell(0.35238, 0.35238, 0.35238), (), source=None)
HEXAGONAL = Crystal("Ti", 194, Cell(0.295, 0.295, 0.468, 90.0, 90.0, 120.0), (), source=None)
TETRAGONAL = Crystal("tet", 139, Cell(0.4, 0.4, 0.6), (), source=None)


def _reference_reciprocal(crystal):
    cell = crystal.cell
    return lattice_params_to_reciprocal(
        cell.a, cell.b, cell.c, cell.alpha, cell.beta, cell.gamma, space_group=crystal.space_group
    )


def _pattern(rotation, crystal, count=4):
    # Rows a*, b*, c*: measured = reference @ R.T so that reciprocal_to_orientation gives R.
    reciprocal = _reference_reciprocal(crystal) @ rotation.T
    return Pattern(
        euler_deg=np.zeros(3), rotation=rotation, reciprocal=reciprocal, goodness=10.0,
        rms_error_deg=0.1, hkl=np.tile([1, 1, 1], (count, 1)),
        pk_index=np.arange(count, dtype=np.int32), err_deg=np.zeros(count),
        energy_kev=np.full(count, 12.0), pred_intens=np.ones(count),
    )


def _frame(patterns, position, n_peaks=6):
    peaks = np.zeros(n_peaks, dtype=PEAK_DTYPE)
    peaks["fit_x"] = np.arange(n_peaks) * 10
    peaks["fit_y"] = np.arange(n_peaks) * 10 + 1
    peaks["intens"] = 100
    return FrameResult(
        peaks=peaks, patterns=tuple(patterns), threshold_used=10, total_sum=100,
        sum_above_threshold=50, num_above_threshold=n_peaks, peaksearch_seconds=0.1,
        indexing_seconds=0.1, metadata={"sample_position": position}, image_shape=(8, 10),
    )


def _dataset(crystal, rotations, *, empty_frames=(), ids=None):
    """Frames with one pattern each, then frames with no patterns."""
    frames = [_frame([_pattern(rotation, crystal)], (index, 0, 0)) for index, rotation in enumerate(rotations)]
    frames += [_frame([], (len(rotations) + offset, 0, 0), n_peaks=count) for offset, count in enumerate(empty_frames)]
    ids = tuple(f"f{index}" for index in range(len(frames))) if ids is None else ids
    return ResultSet(tuple(frames), frame_ids=ids, crystal=crystal).to_visualization()


ALL = DataScope(patterns="all", min_indexed=0)
WITH_FRAMES = DataScope(patterns="all", min_indexed=0, unindexed_frames=True)


# --- symmetry ------------------------------------------------------------------------


def test_auto_symmetry_uses_hexagonal_operations_for_a_hexagonal_crystal():
    sixty = _rotation_matrix([0, 0, 1], 60.0)
    dataset = _dataset(HEXAGONAL, [sixty])

    reduced = prepare_map(dataset, color="rodrigues", scope=ALL)
    unreduced = prepare_map(dataset, color="rodrigues", scope=ALL, orientation_symmetry="none")

    assert reduced.symmetry == "hexagonal" and unreduced.symmetry == "none"
    np.testing.assert_array_equal(reduced.colors[0], 0.0)  # a 60-degree turn about c is identity
    expected = rodrigues_colors([orientation_to_rodrigues(sixty)])[0]
    np.testing.assert_allclose(unreduced.colors[0], expected)


def test_explicit_cubic_symmetry_reduces_a_sixty_degree_turn_to_thirty():
    sixty = _rotation_matrix([0, 0, 1], 60.0)
    dataset = _dataset(HEXAGONAL, [sixty])

    forced = prepare_map(dataset, color="rodrigues", scope=ALL, orientation_symmetry="cubic")

    assert forced.symmetry == "cubic"
    reduced = symmetry_reduce_orientation(sixty, operations=symmetry_operations("cubic"))
    angle = np.degrees(np.arccos((np.trace(reduced) - 1) / 2))
    assert angle == pytest.approx(30.0)
    np.testing.assert_allclose(forced.colors[0], rodrigues_colors([orientation_to_rodrigues(reduced)])[0])


def test_unsupported_crystal_system_reports_no_reduction_instead_of_cubic():
    thirty = _rotation_matrix([0, 0, 1], 30.0)
    dataset = _dataset(TETRAGONAL, [thirty])

    prepared = prepare_map(dataset, color="rodrigues", scope=ALL)

    assert prepared.symmetry == "none"
    np.testing.assert_allclose(prepared.colors[0], rodrigues_colors([orientation_to_rodrigues(thirty)])[0])
    without_crystal = prepare_map(replace(dataset, crystal=None), color="rodrigues", scope=ALL)
    assert without_crystal.symmetry == "none"


def test_unknown_symmetry_choice_is_rejected():
    dataset = _dataset(CUBIC, [np.eye(3)])
    with pytest.raises(ValueError, match="orientation_symmetry 'trigonal'"):
        prepare_map(dataset, color="rodrigues", scope=ALL, orientation_symmetry="trigonal")


def test_misorientation_uses_the_same_symmetry_choice():
    ninety = _rotation_matrix([0, 0, 1], 90.0)
    dataset = _dataset(CUBIC, [np.eye(3), ninety])

    auto = prepare_map(dataset, color="misorientation", scope=ALL, misorientation_reference=("f0", 0))
    none = prepare_map(
        dataset, color="misorientation", scope=ALL, misorientation_reference=("f0", 0),
        orientation_symmetry="none",
    )

    assert auto.symmetry == "cubic" and none.symmetry == "none"
    np.testing.assert_array_equal(auto.colors[1], 0.0)  # 90 degrees about a cube axis is a symmetry
    assert np.linalg.norm(none.colors[1]) > 0.5


# --- references ------------------------------------------------------------------------


def test_pattern_reference_maps_that_pattern_to_zero_and_matches_misorientation():
    rotations = [_rotation_matrix([1, 0.3, -0.2], 37.0), _rotation_matrix([0.2, -0.7, 1.0], 64.0), np.eye(3)]
    dataset = _dataset(CUBIC, rotations)

    relative = prepare_map(dataset, color="rodrigues", scope=ALL, rodrigues_reference=("f1", 0))
    misorientation = prepare_map(
        dataset, color="misorientation", scope=ALL, misorientation_reference=("f1", 0)
    )
    lab = prepare_map(dataset, color="rodrigues", scope=ALL)

    np.testing.assert_array_equal(relative.colors[1], 0.0)
    np.testing.assert_allclose(relative.colors, misorientation.colors)
    assert relative.symmetry == "cubic"
    assert not np.allclose(relative.colors, lab.colors)


def test_reciprocal_reference_uses_the_native_reference_basis_in_inverse_nanometres():
    rotation = _rotation_matrix([1, 0.3, -0.2], 37.0)
    other = _rotation_matrix([0.2, -0.7, 1.0], 64.0)
    dataset = _dataset(CUBIC, [rotation, other])
    g_ref = dataset.pattern_reciprocals[0]

    relative = prepare_map(dataset, color="rodrigues", scope=ALL, rodrigues_reference_reciprocal=g_ref)
    by_pattern = prepare_map(dataset, color="rodrigues", scope=ALL, rodrigues_reference=("f0", 0))

    np.testing.assert_allclose(relative.colors[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(relative.colors, by_pattern.colors, atol=1e-9)
    # The same lattice in 1/angstrom does not have the crystal's metric and is refused.
    with pytest.raises(ValueError, match="1/nm"):
        prepare_map(dataset, color="rodrigues", scope=ALL, rodrigues_reference_reciprocal=g_ref / 10.0)


def test_reciprocal_reference_on_a_hexagonal_crystal_uses_the_native_setting():
    rotation = _rotation_matrix([0.3, 1.0, 0.2], 25.0)
    other = _rotation_matrix([1.0, 0.0, 0.4], 70.0)
    dataset = _dataset(HEXAGONAL, [rotation, other])
    g_ref = dataset.pattern_reciprocals[0]

    relative = prepare_map(dataset, color="rodrigues", scope=ALL, rodrigues_reference_reciprocal=g_ref)
    by_pattern = prepare_map(dataset, color="rodrigues", scope=ALL, rodrigues_reference=("f0", 0))
    lab = prepare_map(dataset, color="rodrigues", scope=ALL)

    assert relative.symmetry == "hexagonal"
    np.testing.assert_allclose(relative.colors[0], 0.0, atol=1e-12)
    np.testing.assert_allclose(relative.colors, by_pattern.colors, atol=1e-9)
    assert not np.allclose(relative.colors[1], lab.colors[1])
    # Wrong units, or rows in an order that does not match the hexagonal cell
    # (|c*| differs from |a*|), do not carry the crystal's metric and are refused.
    with pytest.raises(ValueError, match="1/nm"):
        prepare_map(dataset, color="rodrigues", scope=ALL, rodrigues_reference_reciprocal=g_ref * 10.0)
    with pytest.raises(ValueError, match="crystal's lattice"):
        prepare_map(dataset, color="rodrigues", scope=ALL, rodrigues_reference_reciprocal=g_ref[[2, 0, 1]])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"rodrigues_reference": ("missing", 0)}, "rodrigues_reference must identify"),
        ({"rodrigues_reference": ("f0", 7)}, "rodrigues_reference must identify"),
        ({"rodrigues_reference": "f0"}, "rodrigues_reference must identify"),
        ({"rodrigues_reference_reciprocal": np.zeros((3, 3))}, "singular"),
        ({"rodrigues_reference_reciprocal": np.full((3, 3), np.nan)}, "finite"),
        ({"rodrigues_reference_reciprocal": np.eye(2)}, "3x3"),
        ({"rodrigues_reference": ("f0", 0), "rodrigues_reference_reciprocal": np.eye(3)}, "mutually exclusive"),
    ],
)
def test_invalid_references_are_rejected(kwargs, message):
    dataset = _dataset(CUBIC, [np.eye(3)])
    with pytest.raises(ValueError, match=message):
        prepare_map(dataset, color="rodrigues", scope=ALL, **kwargs)


def test_reciprocal_reference_requires_crystal_and_invalid_reference_orientation_is_rejected():
    dataset = _dataset(CUBIC, [np.eye(3), np.eye(3)])
    with pytest.raises(ValueError, match="crystal context"):
        prepare_map(replace(dataset, crystal=None), color="rodrigues", scope=ALL, rodrigues_reference_reciprocal=np.eye(3))

    rotations = dataset.pattern_rotations.copy()
    rotations[1] = np.nan
    broken = replace(dataset, pattern_rotations=rotations)
    with pytest.raises(ValueError, match="no finite orientation"):
        prepare_map(broken, color="rodrigues", scope=ALL, rodrigues_reference=("f1", 0))
    with pytest.raises(ValueError, match="no finite orientation"):
        prepare_map(broken, color="misorientation", scope=ALL, misorientation_reference=("f1", 0))


# --- frame-only records ---------------------------------------------------------------


def test_unindexed_frames_are_frame_only_records_with_no_pattern_identity():
    dataset = _dataset(CUBIC, [np.eye(3)], empty_frames=(6, 2))

    default = prepare_map(dataset, color="rodrigues", scope=ALL)
    prepared = prepare_map(dataset, color="rodrigues", scope=WITH_FRAMES)

    assert default.frame_ids == ("f0",)
    assert prepared.frame_ids == ("f0", "f1", "f2")
    assert prepared.pattern_indices.tolist() == [0, NO_PATTERN, NO_PATTERN]
    assert prepared.has_pattern.tolist() == [True, False, False]
    assert prepared.indexed.tolist() == [True, False, False]
    np.testing.assert_array_equal(prepared.coordinates[:, 0], [0, 1, 2])
    assert np.isnan(prepared.colors[1:]).all() and np.isfinite(prepared.colors[0]).all()


def test_all_frames_scope_and_min_detected_govern_frame_only_records():
    dataset = _dataset(CUBIC, [np.eye(3)], empty_frames=(6, 2))

    all_frames = prepare_map(dataset, color="n_indexed", scope=DataScope(patterns="all_frames", min_indexed=0))
    detected = prepare_map(
        dataset, color="n_indexed",
        scope=DataScope(patterns="all", min_indexed=0, min_detected=4, unindexed_frames=True),
    )

    assert all_frames.frame_ids == ("f0", "f1", "f2")
    assert detected.frame_ids == ("f0", "f1")  # f2 has only 2 detected peaks


def test_frames_whose_patterns_are_filtered_become_frame_only_records():
    dataset = _dataset(CUBIC, [np.eye(3), np.eye(3)])
    n_indexed = dataset.pattern_n_indexed.copy()
    n_indexed[1] = 2
    dataset = replace(dataset, pattern_n_indexed=n_indexed)

    prepared = prepare_map(dataset, color="goodness", scope=DataScope(patterns="all", min_indexed=3, unindexed_frames=True))

    assert prepared.frame_ids == ("f0", "f1")
    assert prepared.pattern_indices.tolist() == [0, NO_PATTERN]


def test_frame_only_records_are_distinct_from_patterns_with_invalid_orientation():
    dataset = _dataset(CUBIC, [np.eye(3), np.eye(3)], empty_frames=(5,))
    rotations = dataset.pattern_rotations.copy()
    rotations[1] = np.nan
    dataset = replace(dataset, pattern_rotations=rotations)

    prepared = prepare_map(dataset, color="rodrigues", scope=WITH_FRAMES)

    assert prepared.indexed.tolist() == [True, False, False]
    assert prepared.has_pattern.tolist() == [True, True, False]
    assert prepared.pattern_indices.tolist() == [0, 0, NO_PATTERN]


def test_scalar_and_frame_aligned_values_for_frame_only_records():
    dataset = _dataset(CUBIC, [np.eye(3)], empty_frames=(6,))

    goodness = prepare_map(dataset, color="goodness", scope=WITH_FRAMES)
    n_patterns = prepare_map(dataset, color="n_patterns", scope=WITH_FRAMES)
    per_frame = prepare_map(dataset, color=ScalarColor([5.0, 7.0], label="load", alignment="frame"), scope=WITH_FRAMES)
    per_record = prepare_map(dataset, color=ScalarColor([1.0, 2.0], alignment="selected"), scope=WITH_FRAMES)
    frame_axis = prepare_map(dataset, axes=(Axis([10.0, 20.0], "Load", "N"), "Y"), color="n_indexed", scope=WITH_FRAMES)

    assert np.isfinite(goodness.colors[0]) and np.isnan(goodness.colors[1])
    assert n_patterns.colors.tolist() == [1.0, 0.0]
    assert per_frame.colors.tolist() == [5.0, 7.0]
    assert per_record.colors.tolist() == [1.0, 2.0]
    assert frame_axis.coordinates[:, 0].tolist() == [10.0, 20.0]
    with pytest.raises(ValueError, match="non-finite coordinates for frames \\('f1',\\)"):
        prepare_map(dataset, axes=(Axis([1.0], "Strain", alignment="pattern"), "Y"), scope=WITH_FRAMES)
    with pytest.raises(ValueError, match="one value per selected record"):
        prepare_map(dataset, color=ScalarColor([1.0], alignment="selected"), scope=WITH_FRAMES)


def test_map_data_rejects_indexed_frame_only_records():
    with pytest.raises(ValueError, match="frame-only record cannot be indexed"):
        MapData(np.zeros((1, 2)), ("x", "y"), ("f",), [NO_PATTERN], np.zeros(1), "scalar", "v", indexed=[True])
    with pytest.raises(ValueError, match="NO_PATTERN"):
        MapData(np.zeros((1, 2)), ("x", "y"), ("f",), [-2], np.zeros(1), "scalar", "v")


def test_plot_map_draws_frame_only_records_gray_with_absent_pattern_identity():
    dataset = _dataset(CUBIC, [np.eye(3)], empty_frames=(6,))

    rgb = plot_map(dataset, color="cubic_ipf", scope=WITH_FRAMES)
    scalar = plot_map(dataset, color="goodness", scope=WITH_FRAMES)

    roles = {trace.meta["role"]: trace for trace in rgb.data}
    assert set(roles) == {"data", "unindexed"}
    assert list(roles["unindexed"].customdata[0]) == ["f1", None, None]
    assert roles["unindexed"].text[0] == "none"
    assert roles["unindexed"].marker.color == "rgb(150,150,150)"
    selection = selection_from_plotly({"points": [{"customdata": list(roles["unindexed"].customdata[0])}]})
    assert selection.frame_ids == ("f1",) and selection.pattern_ids == ()

    scalar_roles = {trace.meta["role"]: trace for trace in scalar.data}
    np.testing.assert_array_equal(scalar_roles["data"].marker.color, [10.0])  # NaN never enters the range
    assert list(scalar_roles["unindexed"].customdata[0]) == ["f1", None, None]
    assert scalar_roles["data"].marker.showscale is True


def test_plot_map_integer_frame_ids_encode_absent_pattern_as_nan():
    dataset = _dataset(CUBIC, [np.eye(3)], empty_frames=(6,), ids=(10, 11))

    figure = plot_map(dataset, color="rodrigues", scope=WITH_FRAMES)

    unindexed = next(trace for trace in figure.data if trace.meta["role"] == "unindexed")
    row = np.asarray(unindexed.customdata[0], dtype=float)
    assert row[0] == 11 and np.isnan(row[1])
    selection = selection_from_plotly({"points": [{"customdata": row.tolist()}]})
    assert selection.frame_ids == (11,) and selection.pattern_ids == ()


def test_peak_table_honors_unindexed_frames_flag():
    dataset = _dataset(CUBIC, [np.eye(3)], empty_frames=(6, 2))

    without = peak_table(dataset, scope=DataScope(patterns=(0,), min_indexed=0))
    with_frames = peak_table(dataset, scope=DataScope(patterns=(0,), min_indexed=0, unindexed_frames=True))
    detected = peak_table(dataset, scope=DataScope(patterns=(0,), min_indexed=0, min_detected=4, unindexed_frames=True))

    assert sorted(set(without["frame_id"].tolist())) == ["f0"]
    assert sorted(set(with_frames["frame_id"].tolist())) == ["f0", "f1", "f2"]
    assert sorted(set(detected["frame_id"].tolist())) == ["f0", "f1"]


def test_data_scope_validates_unindexed_frames_flag():
    with pytest.raises(ValueError, match="unindexed_frames"):
        DataScope(unindexed_frames="yes")
    assert DataScope(patterns="all_frames").includes_unindexed_frames
    assert not DataScope().includes_unindexed_frames


@pytest.mark.parametrize("reflection", [np.diag([-1, 1, 1]), -np.eye(3)])
def test_custom_reference_rejects_improper_rotations(reflection):
    dataset = _dataset(CUBIC, [np.eye(3)])
    reference = dataset.pattern_reciprocals[0] @ reflection
    with pytest.raises(ValueError, match="proper rotation"):
        prepare_map(dataset, color="rodrigues", scope=ALL,
                    rodrigues_reference_reciprocal=reference)
