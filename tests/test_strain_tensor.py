# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lauelab.indexing import Cell, Crystal, InputError
from lauelab.strain import strain_from_reciprocal, transform_strain
from strain_cases import case, direct


@pytest.mark.parametrize(
    "cell",
    [
        Cell(0.35, 0.35, 0.35),
        Cell(0.29, 0.29, 0.46, 90, 90, 120),
        Cell(0.4, 0.5, 0.6, 78, 82, 104),
        Cell(0.4, 0.4, 0.4, 76, 76, 76),
    ],
)
@pytest.mark.parametrize("finite", [False, True])
def test_known_stretch_in_native_reference_basis(cell, finite):
    cell, b, _, _, expected = case(cell, finite=finite)
    actual = strain_from_reciprocal(b, reference=cell)
    np.testing.assert_allclose(actual, expected, atol=2e-15, rtol=0)
    np.testing.assert_allclose(actual, actual.T, atol=1e-16, rtol=0)
    assert abs(np.trace(actual)) < 1e-15


def test_identity_rotation_and_uniform_scale_are_not_deviatoric_measurements():
    cell = Cell(0.4, 0.5, 0.6, 78, 82, 104)
    rotation = Rotation.from_rotvec([1.4, -0.2, 0.6]).as_matrix()
    b = 2 * np.pi * np.linalg.inv(1.04 * rotation @ direct(cell))
    np.testing.assert_allclose(strain_from_reciprocal(b, reference=cell), 0, atol=1e-15)


def test_conversion_normalizes_volume_before_removing_trace_and_supports_stacks():
    cell, b, _, _, expected = case(finite=True)
    stacked = strain_from_reciprocal(np.stack([b, b * 1.7]), reference=cell)
    np.testing.assert_allclose(stacked, np.stack([expected, expected]), atol=2e-15)


def test_frame_transforms_follow_fitted_cell_orientation_for_finite_shear():
    cell, b, _, _, expected = case(finite=True)
    a = 2 * np.pi * np.linalg.inv(b)
    lengths = np.linalg.norm(a, axis=0)
    angles = [
        np.degrees(np.arccos(np.dot(a[:, i], a[:, j]) / (lengths[i] * lengths[j])))
        for i, j in [(1, 2), (0, 2), (0, 1)]
    ]
    rho = a @ np.linalg.inv(direct(Cell(*lengths, *angles)))
    beamline = rho @ expected @ rho.T
    np.testing.assert_allclose(
        strain_from_reciprocal(b, reference=cell, coordinates="beamline"),
        beamline,
        atol=3e-15,
    )
    r = 1 / np.sqrt(2)
    for name, rotation in {
        "xhf": [[1, 0, 0], [0, r, r], [0, -r, r]],
        "sample": [[1, 0, 0], [0, -r, -r], [0, r, -r]],
    }.items():
        np.testing.assert_allclose(
            strain_from_reciprocal(b, reference=cell, coordinates=name),
            np.array(rotation) @ beamline @ np.array(rotation).T,
            atol=3e-15,
        )


def test_crystal_symmetry_coercion_is_separate_from_explicit_cell():
    cell = Cell(0.4, 0.42, 0.45, 89, 91, 92)
    reference = Crystal("cubic", 225, cell, ())
    b = np.eye(3) * 2 * np.pi / 0.4
    np.testing.assert_allclose(
        strain_from_reciprocal(b, reference=reference), 0, atol=1e-15
    )
    assert np.linalg.norm(strain_from_reciprocal(b, reference=cell)) > 0.01


def test_crystal_coercion_precedes_effective_metric_validation():
    cell = Cell(0.4, 0.42, 0.45, 170, 170, 170)
    b = np.eye(3) * 2 * np.pi / 0.4
    np.testing.assert_allclose(
        strain_from_reciprocal(b, reference=Crystal("cubic", 225, cell, ())),
        0,
        atol=1e-15,
    )
    with pytest.raises(InputError):
        strain_from_reciprocal(b, reference=cell)


def test_transform_broadcasts_and_does_not_double_tensor_shear():
    tensor = np.array([[0, 0.02, 0], [0.02, 0, 0], [0, 0, 0]])
    rotation = Rotation.from_rotvec([0, 0, 0.3]).as_matrix()
    out = transform_strain(np.stack([tensor, tensor]), rotation)
    np.testing.assert_allclose(out[0], rotation @ tensor @ rotation.T, atol=1e-16)
    assert not np.shares_memory(out, tensor)


@pytest.mark.parametrize(
    "b",
    [
        np.zeros((3, 3)),
        np.diag([-1, 1, 1]),
        np.full((3, 3), np.nan),
        np.eye(2),
        np.eye(3) * 1j,
    ],
)
def test_invalid_lattice_cannot_produce_plausible_strain(b):
    with pytest.raises(InputError):
        strain_from_reciprocal(b, reference=Cell(0.4, 0.4, 0.4))


@pytest.mark.parametrize("rotation", [np.ones((3, 3)), np.diag([-1, 1, 1]), np.eye(2)])
def test_coordinate_transform_rejects_nonorthonormal_or_left_handed_basis(rotation):
    with pytest.raises(InputError):
        transform_strain(np.zeros((3, 3)), rotation)
