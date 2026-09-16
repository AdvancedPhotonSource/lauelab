# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
import numpy as np
import pytest

from conftest import requires_liblaue

from lauelab.analysis import (
    CUBIC_SYMMETRY,
    HEXAGONAL_SYMMETRY,
    SurfaceFrame,
    closest_pole_colors,
    cubic_hkl_family,
    cubic_ipf_colors,
    cubic_ipf_key,
    hsv_key,
    hsv_position_colors,
    lattice_params_to_reciprocal,
    misorientation_angle,
    misorientation_from_reference,
    misorientation_matrix,
    orientation_to_rodrigues,
    pole_color_radius,
    pole_figure_points,
    rodrigues_colors,
    symmetry_operations,
    symmetry_reduce_orientation,
)
from lauelab.indexing import Cell, Crystal
from lauelab.indexing._liblaue import NativeCrystal


def _rotation_about_z(angle_deg):
    angle = np.radians(angle_deg)
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0],
        [0, 0, 1],
    ])


@requires_liblaue
@pytest.mark.parametrize(
    ("space_group", "cell"),
    [
        (1, (0.4, 0.5, 0.6, 70, 80, 75)),
        (3, (0.4, 0.5, 0.6, 88, 105, 92)),
        (16, (0.4, 0.5, 0.6, 88, 91, 92)),
        (75, (0.4, 0.5, 0.6, 88, 91, 92)),
        pytest.param(146, (0.4, 0.5, 0.6, 70, 80, 75), id="trigonal-rhombohedral"),
        pytest.param(146, (0.4, 0.5, 0.6, 90, 90, 120), id="trigonal-hexagonal"),
        (168, (0.4, 0.5, 0.6, 88, 91, 115)),
        (195, (0.4, 0.5, 0.6, 88, 91, 92)),
    ],
)
def test_reciprocal_lattice_matches_native_crystal(space_group, cell):
    crystal = Crystal("test", space_group, Cell(*cell))
    expected = NativeCrystal.create(crystal).reciprocal()

    actual = lattice_params_to_reciprocal(*cell, space_group=space_group)

    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-14)


def test_jzt_basis_has_equivalent_reciprocal_metric():
    from lauelab.analysis._vendor.jzt.Lattice import Lattice3D

    cell = (0.4, 0.5, 0.6, 70, 80, 75)
    native_basis = lattice_params_to_reciprocal(*cell, space_group=1)
    jzt_basis = np.asarray(Lattice3D(1, cell).recip).T

    np.testing.assert_allclose(
        native_basis @ native_basis.T,
        jzt_basis @ jzt_basis.T,
        atol=1e-12,
    )


def test_aps_surface_frames_are_right_handed_and_read_only():
    for name in ("normal", "F", "X", "H", "Y", "Z"):
        frame = SurfaceFrame.aps_34ide(name)
        np.testing.assert_allclose(np.cross(frame.tilt, frame.roll), frame.normal)
        np.testing.assert_allclose(
            np.asarray([frame.tilt, frame.roll, frame.normal])
            @ np.asarray([frame.tilt, frame.roll, frame.normal]).T,
            np.eye(3),
            atol=1e-15,
        )
        assert not frame.normal.flags.writeable
    with pytest.raises(ValueError, match="unknown"):
        SurfaceFrame.aps_34ide("missing")


def test_surface_frame_normalizes_and_validates_handedness():
    frame = SurfaceFrame.from_vectors(tilt=[2, 0, 0], roll=[0, 3, 0], normal=[0, 0, 4])
    np.testing.assert_allclose(frame.normal, [0, 0, 1])
    with pytest.raises(ValueError, match="right-handed"):
        SurfaceFrame.from_vectors(tilt=[1, 0, 0], roll=[0, 1, 0], normal=[0, 0, -1])


@pytest.mark.parametrize("hkl,count", [((1, 0, 0), 6), ((1, 1, 0), 12), ((1, 1, 1), 8), ((2, 1, 0), 24), ((3, 2, 1), 48)])
def test_cubic_hkl_family_counts(hkl, count):
    family = cubic_hkl_family(hkl)
    assert family.shape == (count, 3)
    np.testing.assert_allclose(np.linalg.norm(family, axis=1), 1)


def test_pole_figure_points_are_in_unit_circle_and_keep_pattern_ids():
    lattices = np.tile(np.eye(3), (3, 1, 1))
    points, indices = pole_figure_points(lattices, cubic_hkl_family((1, 0, 0)))
    assert points.shape[1] == 2
    assert set(indices) == {0, 1, 2}
    assert np.all(np.linalg.norm(points, axis=1) <= 1 + 1e-12)


def test_symmetry_and_misorientation_primitives():
    assert CUBIC_SYMMETRY.shape == (24, 3, 3)
    assert HEXAGONAL_SYMMETRY.shape == (12, 3, 3)
    assert not CUBIC_SYMMETRY.flags.writeable
    assert not HEXAGONAL_SYMMETRY.flags.writeable
    np.testing.assert_allclose(symmetry_operations(225), CUBIC_SYMMETRY)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    np.testing.assert_allclose(orientation_to_rodrigues(rotation), [0, 0, 1], atol=1e-12)
    assert misorientation_angle(rotation, rotation, operations=CUBIC_SYMMETRY) == pytest.approx(0)
    with pytest.raises(ValueError, match="supported"):
        symmetry_operations(1)


def test_rodrigues_round_trips_generic_rotations():
    from lauelab.analysis.orientation import _rotation_matrix

    rng = np.random.default_rng(3851)
    for angle in rng.uniform(0.01, 179.0, 50):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        vector = orientation_to_rodrigues(_rotation_matrix(axis, angle))

        np.testing.assert_allclose(vector / np.linalg.norm(vector), axis, atol=1e-12)
        assert 2.0 * np.arctan(np.linalg.norm(vector)) == pytest.approx(
            np.radians(angle), abs=1e-12
        )


def test_rodrigues_and_misorientation_handle_random_180_degree_rotations():
    from lauelab.analysis.orientation import _rotation_matrix

    rng = np.random.default_rng(180)
    for _ in range(50):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        first = np.flatnonzero(np.abs(axis) > 1e-12)[0]
        if axis[first] < 0:
            axis = -axis
        rotation = _rotation_matrix(axis, 180)

        vector = orientation_to_rodrigues(rotation)
        np.testing.assert_allclose(vector / np.linalg.norm(vector), axis, atol=1e-12)
        assert np.degrees(2.0 * np.arctan(np.linalg.norm(vector))) == pytest.approx(
            180.0, abs=1e-5
        )
        vectors, angles = misorientation_from_reference(
            np.asarray([np.eye(3), rotation]), 0
        )
        np.testing.assert_allclose(vectors[1], vector, rtol=1e-12)
        assert angles[1] == pytest.approx(180.0, abs=2e-6)
        assert angles[1] == pytest.approx(misorientation_angle(rotation, np.eye(3)))


def test_rotation_matrix_does_not_mutate_input_axis():
    from lauelab.analysis.orientation import _rotation_matrix

    axis = np.array([2.0, 0.0, 0.0])
    _rotation_matrix(axis, 30)
    np.testing.assert_array_equal(axis, [2.0, 0.0, 0.0])


def test_orientation_reduction_and_misorientation_are_distinct():
    rotation_30 = _rotation_about_z(30)
    rotation_20 = _rotation_about_z(20)
    operation_90 = _rotation_about_z(90)

    np.testing.assert_allclose(
        symmetry_reduce_orientation(rotation_30, operations=[np.eye(3), operation_90]),
        rotation_30,
        atol=1e-15,
    )
    relative = misorientation_matrix(rotation_30, rotation_20)
    np.testing.assert_allclose(relative, _rotation_about_z(10), atol=1e-15)

    vectors, angles = misorientation_from_reference(
        np.asarray([rotation_20, rotation_30]), 0
    )
    np.testing.assert_allclose(vectors[0], 0, atol=1e-15)
    np.testing.assert_allclose(vectors[1], [0, 0, np.tan(np.radians(5))], atol=1e-15)
    np.testing.assert_allclose(angles, [0, 10], atol=1e-12)


def test_color_primitives_have_expected_shapes_and_ranges():
    directions = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=float)
    ipf = cubic_ipf_colors(directions)
    rods = rodrigues_colors(np.array([[0, 0, 0], [1, 0, 0]], dtype=float))
    np.testing.assert_array_equal(rodrigues_colors(np.zeros((2, 3))), 0)
    hsv = hsv_position_colors([0, 1], [0, 0])
    assert ipf.shape == (3, 3)
    assert rods.shape == hsv.shape == (2, 3)
    assert np.all((ipf >= 0) & (ipf <= 1))
    np.testing.assert_allclose(hsv[0], [1, 1, 1])
    np.testing.assert_allclose(hsv[1], [1, 0, 0])


def test_closest_pole_colors_and_keys():
    colors = closest_pole_colors(
        np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5]]),
        np.array([0, 0, 1]),
        2,
    )
    np.testing.assert_allclose(colors[0], [1, 1, 1])
    assert cubic_ipf_key(16).shape == (16, 16, 4)
    assert hsv_key(16).shape == (16, 16, 4)
    assert pole_color_radius((0, 0), 45) == pytest.approx(np.tan(np.radians(22.5)))


def test_rodrigues_of_near_identity_noise_is_zero_not_180_degrees():
    # A rotation multiplied by its own inverse is identity up to 1e-16 noise;
    # arccos amplifies that to an angle near 1e-8, which must not be taken for
    # a 180-degree rotation with a tiny axis.
    from lauelab.analysis.orientation import _rotation_matrix

    rotation = _rotation_matrix([0.3, -0.7, 1.0], 37.0)
    noisy = rotation @ np.linalg.inv(rotation)
    assert not np.array_equal(noisy, np.eye(3))

    vectors = orientation_to_rodrigues(np.stack([noisy, np.eye(3), _rotation_matrix([1, 0, 0], 180.0)]))

    np.testing.assert_array_equal(vectors[0], 0.0)
    np.testing.assert_array_equal(vectors[1], 0.0)
    assert np.linalg.norm(vectors[2]) > 1e6
    np.testing.assert_array_equal(orientation_to_rodrigues(misorientation_matrix(rotation, rotation)), 0.0)
