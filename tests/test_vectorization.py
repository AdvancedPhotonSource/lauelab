# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Equivalence of vectorized library code with the per-row forms it replaced.

The reference forms live in ``tests/reference_loops.py``. Every test builds
inputs with the shapes and edge cases the library sees on real scans: many
frames, several patterns per frame, missing rows, identity and 180-degree
rotations, and non-orthonormal orientation matrices from strained lattices.
"""

from dataclasses import replace

import numpy as np
import pytest

import reference_loops as reference

from lauelab.visualization import DataScope
from lauelab.visualization.data import VisualizationDataset
from lauelab.visualization.tables import _dataset  # noqa: F401  (import check)


def _patterns_dataset(rng, n_frames=400, n_patterns=900):
    """Return a minimal dataset with unsorted frames and duplicate ranks."""
    from lauelab.indexing.indexer import PEAK_DTYPE

    frames = rng.integers(0, n_frames, n_patterns)
    ranks = rng.integers(0, 4, n_patterns)
    return VisualizationDataset(
        frame_ids=tuple(range(n_frames)),
        sample_positions=np.zeros((n_frames, 3)),
        depths=np.zeros(n_frames),
        frame_n_peaks=np.zeros(n_frames, dtype=int),
        scan_numbers=np.zeros(n_frames),
        energies_kev=np.zeros(n_frames),
        detector_ids=(None,) * n_frames,
        image_shapes=np.ones((n_frames, 2), dtype=int),
        starts=np.zeros((n_frames, 2), dtype=int),
        groups=np.ones((n_frames, 2), dtype=int),
        input_images=(None,) * n_frames,
        images=(None,) * n_frames,
        peak_frame_indices=np.empty(0, dtype=int),
        peak_indices=np.empty(0, dtype=int),
        peaks=np.empty(0, dtype=PEAK_DTYPE),
        pattern_frame_indices=frames,
        pattern_indices=ranks,
        pattern_rotations=np.tile(np.eye(3), (n_patterns, 1, 1)),
        pattern_reciprocals=np.tile(np.eye(3), (n_patterns, 1, 1)),
        pattern_goodness=np.zeros(n_patterns),
        pattern_rms_error_deg=np.zeros(n_patterns),
        pattern_n_indexed=rng.integers(0, 10, n_patterns),
        assignment_pattern_rows=np.empty(0, dtype=int),
        assignment_peak_indices=np.empty(0, dtype=int),
        assignment_hkl=np.empty((0, 3), dtype=int),
        assignment_error_deg=np.empty(0),
        assignment_energy_kev=np.empty(0),
        assignment_predicted_intensity=np.empty(0),
    )


def test_best_pattern_mask_matches_per_frame_argmin():
    rng = np.random.default_rng(51)
    dataset = _patterns_dataset(rng)
    expected = reference.best_pattern_mask(
        dataset.pattern_frame_indices, dataset.pattern_indices
    )
    expected &= dataset.pattern_n_indexed >= 3

    actual = DataScope(patterns="best", min_indexed=3).pattern_mask(dataset)

    np.testing.assert_array_equal(actual, expected)
    assert actual.sum() <= len(np.unique(dataset.pattern_frame_indices))


def test_best_pattern_mask_keeps_first_of_tied_ranks_and_handles_empty():
    rng = np.random.default_rng(52)
    dataset = _patterns_dataset(rng, n_frames=5, n_patterns=12)
    tied = replace(dataset, pattern_indices=np.zeros(12, dtype=int))
    expected = reference.best_pattern_mask(tied.pattern_frame_indices, tied.pattern_indices)
    np.testing.assert_array_equal(
        DataScope(patterns="best", min_indexed=0).pattern_mask(tied), expected
    )

    empty = _patterns_dataset(rng, n_frames=3, n_patterns=0)
    assert DataScope(patterns="best").pattern_mask(empty).shape == (0,)


def _random_rotations(rng, count, *, strain=0.0):
    """Return rotation matrices, optionally multiplied by a small symmetric strain."""
    from lauelab.analysis.orientation import _rotation_matrix

    axes = rng.normal(size=(count, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    angles = rng.uniform(0.0, 180.0, count)
    rotations = np.asarray([
        _rotation_matrix(axis, angle) for axis, angle in zip(axes, angles, strict=True)
    ])
    if strain:
        symmetric = rng.normal(scale=strain, size=(count, 3, 3))
        symmetric = (symmetric + np.swapaxes(symmetric, 1, 2)) / 2
        rotations = rotations @ (np.eye(3) + symmetric)
    return rotations


def _orientation_stack(rng, count=500):
    """Strained orientations with identity, 180-degree, NaN, and singular rows."""
    from lauelab.analysis.orientation import _rotation_matrix

    rotations = _random_rotations(rng, count, strain=0.02)
    rotations[0] = np.eye(3)
    rotations[1] = _rotation_matrix([1, 0, 0], 180)
    rotations[2] = _rotation_matrix([1, 1, 0], 180)
    rotations[3] = np.nan
    rotations[4] = 0.0
    rotations[5] = np.ones((3, 3))
    return rotations


def test_crystal_directions_and_cubic_ipf_colors_match_row_loops():
    from lauelab.analysis import cubic_ipf_colors
    from lauelab.visualization.preparation import _crystal_directions

    rng = np.random.default_rng(72)
    rotations = _orientation_stack(rng)
    normal = np.array([0.0, 1.0, -1.0]) / np.sqrt(2.0)

    directions = _crystal_directions(rotations, normal)
    expected = reference.crystal_directions(rotations, normal)
    np.testing.assert_allclose(directions, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
    assert np.isnan(directions[3]).all() and np.isnan(directions[4]).all()

    directions[6] = 0.0
    directions[7] = [np.nan, 0.0, 1.0]
    colors = cubic_ipf_colors(directions)
    np.testing.assert_allclose(
        colors, reference.cubic_ipf_colors(directions), rtol=1e-12, atol=1e-12
    )
    np.testing.assert_array_equal(colors[6], 0.0)
    np.testing.assert_array_equal(colors[7], 0.5)
    np.testing.assert_array_equal(colors[3], 0.5)
    assert cubic_ipf_colors(directions[8]).shape == (3,)
    assert cubic_ipf_colors(directions.reshape(50, 10, 3)).shape == (50, 10, 3)


def test_batched_orientation_math_matches_row_loops():
    from lauelab.analysis import (
        CUBIC_SYMMETRY,
        misorientation_from_reference,
        misorientation_matrix,
        orientation_to_rodrigues,
        symmetry_reduce_orientation,
    )

    rng = np.random.default_rng(73)
    rotations = _orientation_stack(rng)[:3]
    rotations = np.concatenate([rotations, _random_rotations(rng, 300)])
    rotations[10] = np.nan
    finite = np.isfinite(rotations).all(axis=(1, 2))

    reduced = symmetry_reduce_orientation(rotations[finite], operations=CUBIC_SYMMETRY)
    expected = np.asarray([
        reference.symmetry_reduce_orientation(rotation, CUBIC_SYMMETRY)
        for rotation in rotations[finite]
    ])
    np.testing.assert_allclose(reduced, expected, rtol=0, atol=1e-14)

    anchor = rotations[5]  # before the NaN row, so the finite subset keeps its index
    relative = misorientation_matrix(rotations[finite], anchor, operations=CUBIC_SYMMETRY)
    expected = np.asarray([
        reference.misorientation_matrix(rotation, anchor, CUBIC_SYMMETRY)
        for rotation in rotations[finite]
    ])
    np.testing.assert_allclose(relative, expected, rtol=0, atol=1e-14)

    vectors = orientation_to_rodrigues(rotations)
    expected = reference.rodrigues_vectors(rotations)
    np.testing.assert_allclose(vectors, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
    assert np.isnan(vectors[10]).all()
    np.testing.assert_allclose(orientation_to_rodrigues(rotations[1]), expected[1])
    assert orientation_to_rodrigues(rotations[:6].reshape(2, 3, 3, 3)).shape == (2, 3, 3)

    stacked_vectors, angles = misorientation_from_reference(
        rotations[finite], 5, operations=CUBIC_SYMMETRY
    )
    np.testing.assert_allclose(
        stacked_vectors, reference.rodrigues_vectors(relative), rtol=1e-12, atol=1e-12
    )
    assert angles[5] == pytest.approx(0.0, abs=1e-6)
    assert angles.shape == (finite.sum(),) and np.all(angles <= 62.8)


def test_map_orientation_colors_keep_nan_rows_out_of_the_batch():
    from lauelab.analysis import CUBIC_SYMMETRY, rodrigues_colors
    from lauelab.visualization.preparation import _finite_rows
    from lauelab.analysis import orientation_to_rodrigues, symmetry_reduce_orientation

    rng = np.random.default_rng(74)
    rotations = _random_rotations(rng, 40)
    rotations[[3, 17]] = np.nan
    vectors = _finite_rows(
        rotations,
        lambda finite: orientation_to_rodrigues(
            symmetry_reduce_orientation(finite, operations=CUBIC_SYMMETRY)
        ),
    )
    expected = np.full((40, 3), np.nan)
    for index, rotation in enumerate(rotations):
        if np.isfinite(rotation).all():
            expected[index] = orientation_to_rodrigues(
                reference.symmetry_reduce_orientation(rotation, CUBIC_SYMMETRY)
            )
    np.testing.assert_allclose(vectors, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
    assert np.isnan(rodrigues_colors(vectors)[[3, 17]]).sum() == 0
    assert _finite_rows(np.full((2, 3, 3), np.nan), lambda finite: finite).shape == (2, 3)


def test_rodrigues_colors_match_row_loop():
    from lauelab.analysis import rodrigues_colors

    rng = np.random.default_rng(75)
    vectors = rng.normal(scale=0.3, size=(600, 3))
    vectors[0] = 0.0
    vectors[1] = [np.nan, 0.1, 0.2]
    vectors[2] = 5.0  # beyond the color range: every channel clips
    for max_angle in (12.0, 45.0):
        np.testing.assert_allclose(
            rodrigues_colors(vectors, max_angle_deg=max_angle),
            reference.rodrigues_colors(vectors, max_angle),
            rtol=0,
            atol=1e-15,
        )
    automatic = rodrigues_colors(vectors)
    assert automatic.shape == (600, 3) and np.all((automatic >= 0) & (automatic <= 1))
    np.testing.assert_array_equal(automatic[[0, 1]], 0.0)


def test_hex_colors_encode_the_same_bytes_as_the_rgb_loop():
    from lauelab.visualization.rendering import _rgb

    rng = np.random.default_rng(76)
    values = rng.uniform(-0.1, 1.1, (5000, 3))
    values[7] = np.nan
    colors = _rgb(values)
    expected = reference.rgb_strings(values)
    assert len(colors) == 5000
    for color, triple in zip(colors, expected, strict=True):
        if triple is None:
            assert color == "#969696"
        else:
            assert color.startswith("#") and len(color) == 7
            assert tuple(int(color[index:index + 2], 16) for index in (1, 3, 5)) == triple
    assert _rgb([0.0, 0.5, 1.0]) == ["#0080ff"]


def test_numeric_customdata_round_trips_through_selection():
    from lauelab.visualization import selection_from_plotly
    from lauelab.visualization.rendering import _customdata

    rows = _customdata([3, 5, 5], [0, 1, None])
    assert isinstance(rows, np.ndarray) and rows.shape == (3, 3) and rows.dtype == np.float32
    assert _customdata([2**24, 1], [0, 0]).dtype == np.float64
    assert np.isnan(rows[:, 2]).all() and np.isnan(rows[2, 1])

    selection = selection_from_plotly({"points": [{"customdata": list(row)} for row in rows]})
    assert selection.frame_ids == (3, 5)
    assert selection.pattern_ids == ((3, 0), (5, 1))
    assert selection.peak_ids == ()

    assert _customdata(["a", "b"], [0, 1]) == [["a", 0, None], ["b", 1, None]]
    assert _customdata([3, 5], [0, 1], array=False) == [[3, 0, None], [5, 1, None]]
    assert _customdata([], None) == []
    with pytest.raises(ValueError, match="pattern indices"):
        selection_from_plotly({"points": [{"customdata": [3, 0.5, None]}]})


@pytest.mark.parametrize("array", [True, False])
def test_customdata_does_not_round_large_frame_ids(array):
    from lauelab.visualization import selection_from_plotly
    from lauelab.visualization.rendering import _customdata

    for value in (2**53, 2**53 + 1, -(2**53 + 1), np.uint64(2**63)):
        with pytest.raises(ValueError, match="use string frame IDs"):
            _customdata([value], [0], array=array)
    ids = [-(2**53 - 1), 2**53 - 1]
    rows = _customdata(ids, [0, 1], array=array)
    selection = selection_from_plotly({"points": [{"customdata": list(row)} for row in rows]})
    assert selection.frame_ids == tuple(ids)
    strings = [str(2**53), str(2**53 + 1)]
    rows = _customdata(strings, [0, 1], array=array)
    selection = selection_from_plotly({"points": [{"customdata": row} for row in rows]})
    assert selection.frame_ids == tuple(strings)


def test_indexed_peak_table_matches_dictionary_join():
    from lauelab.visualization import DataScope, indexed_peak_table
    from lauelab.indexing.indexer import PEAK_DTYPE

    rng = np.random.default_rng(77)
    n_frames, n_peaks_per_frame = 30, 7
    peaks = np.zeros(n_frames * n_peaks_per_frame, dtype=PEAK_DTYPE)
    peaks["fit_x"] = rng.uniform(0, 100, len(peaks))
    peaks["intens"] = np.arange(len(peaks), dtype=float)
    peak_frames = np.repeat(np.arange(n_frames), n_peaks_per_frame)
    peak_indices = np.tile(np.arange(n_peaks_per_frame), n_frames)
    pattern_frames = np.repeat(np.arange(n_frames), 2)
    ranks = np.tile([1, 0], n_frames)
    n_patterns = len(ranks)
    assignment_rows = np.repeat(np.arange(n_patterns), 3)
    assignment_peaks = rng.integers(0, n_peaks_per_frame, len(assignment_rows))
    # Shuffle the peak rows so the lookup cannot assume ordering.
    shuffle = rng.permutation(len(peaks))
    dataset = VisualizationDataset(
        frame_ids=tuple(f"frame-{index}" for index in range(n_frames)),
        sample_positions=np.zeros((n_frames, 3)),
        depths=np.zeros(n_frames),
        frame_n_peaks=np.full(n_frames, n_peaks_per_frame),
        scan_numbers=np.zeros(n_frames),
        energies_kev=np.zeros(n_frames),
        detector_ids=(None,) * n_frames,
        image_shapes=np.ones((n_frames, 2), dtype=int),
        starts=np.zeros((n_frames, 2), dtype=int),
        groups=np.ones((n_frames, 2), dtype=int),
        input_images=(None,) * n_frames,
        images=(None,) * n_frames,
        peak_frame_indices=peak_frames[shuffle],
        peak_indices=peak_indices[shuffle],
        peaks=peaks[shuffle],
        pattern_frame_indices=pattern_frames,
        pattern_indices=ranks,
        pattern_rotations=np.tile(np.eye(3), (n_patterns, 1, 1)),
        pattern_reciprocals=np.tile(np.eye(3), (n_patterns, 1, 1)),
        pattern_goodness=np.arange(n_patterns, dtype=float),
        pattern_rms_error_deg=np.zeros(n_patterns),
        pattern_n_indexed=np.full(n_patterns, 3),
        assignment_pattern_rows=assignment_rows,
        assignment_peak_indices=assignment_peaks,
        assignment_hkl=np.zeros((len(assignment_rows), 3), dtype=int),
        assignment_error_deg=np.zeros(len(assignment_rows)),
        assignment_energy_kev=np.zeros(len(assignment_rows)),
        assignment_predicted_intensity=np.zeros(len(assignment_rows)),
    )

    table = indexed_peak_table(dataset, scope=DataScope(patterns="all", min_indexed=0))

    lookup = {
        (int(frame), int(peak)): row
        for row, (frame, peak) in enumerate(zip(dataset.peak_frame_indices, dataset.peak_indices))
    }
    frame_lookup = {frame_id: index for index, frame_id in enumerate(dataset.frame_ids)}
    expected_rows = [
        lookup[(frame_lookup[frame_id], int(peak))]
        for frame_id, peak in zip(table["frame_id"], table["peak_index"])
    ]
    np.testing.assert_array_equal(table["intens"], dataset.peaks["intens"][expected_rows])
    np.testing.assert_array_equal(table["fit_x"], dataset.peaks["fit_x"][expected_rows])
    pattern_lookup = {
        (dataset.frame_ids[frame], int(rank)): row
        for row, (frame, rank) in enumerate(zip(dataset.pattern_frame_indices, dataset.pattern_indices))
    }
    expected_patterns = [
        pattern_lookup[(frame_id, int(rank))]
        for frame_id, rank in zip(table["frame_id"], table["pattern_index"])
    ]
    np.testing.assert_array_equal(table["goodness"], dataset.pattern_goodness[expected_patterns])
    assert len(table) == len(assignment_rows)

    broken = replace(dataset, assignment_peak_indices=np.full(len(assignment_rows), 99))
    with pytest.raises(KeyError):
        indexed_peak_table(broken, scope=DataScope(patterns="all", min_indexed=0))


def test_batched_orientation_derivation_matches_per_lattice_form():
    from lauelab.analysis import lattice_params_to_reciprocal
    from lauelab.visualization.xml import orientations_from_reciprocals

    rng = np.random.default_rng(78)
    reference_lattice = lattice_params_to_reciprocal(0.352, 0.352, 0.352, 90, 90, 90, space_group=225)
    reciprocals = np.einsum("nij,jk->nik", _random_rotations(rng, 200), reference_lattice)
    reciprocals[4] = np.nan
    reciprocals[9, 1, 1] = np.inf

    rotations = orientations_from_reciprocals(reciprocals, reference_lattice)
    expected = reference.reciprocal_to_orientations(reciprocals, reference_lattice)
    np.testing.assert_allclose(rotations, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
    assert np.isnan(rotations[[4, 9]]).all()
    assert np.isnan(orientations_from_reciprocals(reciprocals, None)).all()
    assert np.isnan(orientations_from_reciprocals(reciprocals, np.zeros((3, 3)))).all()
    assert orientations_from_reciprocals(np.empty((0, 3, 3)), reference_lattice).shape == (0, 3, 3)
