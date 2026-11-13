# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Conventions of the calibration model: rotations, Igor matrices, surface check, peak fit."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from calibration_cases import GEOMETRY_2023, PERFECT_SI_AXIS, si_crystal, si_reference
from lauelab.calibration import (
    CalibrationTable,
    fit_peak,
    matrix_from_igor,
    peak_directions,
    rotation_matrix,
    rotation_vector,
    surface_check,
)
from lauelab.indexing import InputError, load_geometry

GEOMETRIES = [
    GEOMETRY_2023,
    Path(__file__).parent / "config" / "geoN_2023-02-07_19-19-10.xml",
    Path(__file__).parent / "data" / "geo" / "geoN_2022-03-29_14-15-05.xml",
]


@pytest.mark.parametrize("path", GEOMETRIES)
def test_rotation_vector_matches_the_native_detector_rotation(path):
    geometry = load_geometry(path)
    for slot in range(3):
        detector = geometry.detector(slot)
        np.testing.assert_allclose(rotation_matrix(detector.rotation_vector), detector.rotation, atol=1e-15)
        np.testing.assert_allclose(rotation_vector(detector.rotation), detector.rotation_vector, atol=1e-12)


@pytest.mark.parametrize("path", GEOMETRIES)
def test_peak_directions_match_native_pixels_to_q(path):
    geometry = load_geometry(path)
    rng = np.random.default_rng(3)
    for slot in range(3):
        detector = geometry.detector(slot)
        pixels = rng.uniform(0, [detector.nx - 1, detector.ny - 1], size=(20, 2))
        np.testing.assert_allclose(
            peak_directions(detector, pixels),
            geometry.pixels_to_q(pixels, detector_index=slot),
            atol=1e-12,
        )


def test_igor_matrix_text_is_read_by_columns():
    matrix = np.arange(1.0, 10.0).reshape(3, 3)
    # Igor's LaueSimulation writes m[0][0], m[1][0], m[2][0] as the first group.
    text = "{{%g,%g,%g}{%g,%g,%g}{%g,%g,%g}}" % tuple(matrix.T.ravel())
    np.testing.assert_array_equal(matrix_from_igor(text), matrix)
    np.testing.assert_array_equal(matrix_from_igor(text.replace("}{", "}, {")), matrix)
    with pytest.raises(InputError):
        matrix_from_igor("{{1,2,3}{4,5,6}}")


def test_perfect_si_rotation_puts_001_between_the_beams():
    """Igor's perfect-Si sample axis rotates the standard basis to the crystal."""
    rotation = Rotation.from_rotvec(PERFECT_SI_AXIS).as_matrix()
    np.testing.assert_allclose(rotation @ [0, 0, 1], [0.0192, 0.7043, -0.7097], atol=5e-4)
    check = surface_check(si_crystal(), si_reference() @ rotation.T)
    assert tuple(np.abs(check.hkl)) == (0, 0, 1)
    assert check.between
    assert check.angle_to_expected_deg == pytest.approx(1.12, abs=0.01)
    assert check.angle_to_incident_deg == pytest.approx(44.79, abs=0.01)
    assert check.angle_to_exit_deg == pytest.approx(45.23, abs=0.01)
    np.testing.assert_allclose(check.exit_direction, [0, 1, 0], atol=1e-12)


def test_transposed_rotation_fails_the_surface_check():
    rotation = Rotation.from_rotvec(PERFECT_SI_AXIS).as_matrix()
    check = surface_check(si_crystal(), si_reference() @ rotation)  # the inverse rotation
    # Another {001} member then lies between the beams, so only the angle to
    # the expected normal shows the error.
    assert check.angle_to_expected_deg == pytest.approx(29.89, abs=0.01)


def test_surface_check_of_a_noncubic_family_uses_plus_minus_only():
    from lauelab.indexing import Atom, Cell, Crystal

    crystal = Crystal("tetragonal", 123, Cell(0.4, 0.4, 0.6), (Atom("Fe", (0, 0, 0)),))
    from lauelab.analysis import lattice_params_to_reciprocal

    reciprocal = lattice_params_to_reciprocal(0.4, 0.4, 0.6, 90, 90, 90)
    check = surface_check(crystal, reciprocal, family=(1, 0, 0), expected_normal=(0, 1, -1))
    assert tuple(check.hkl) in {(1, 0, 0), (-1, 0, 0)}


def _gaussian(shape, x, y, sigma_x, sigma_y, amplitude=1000.0, background=50.0):
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    return background + amplitude * np.exp(-0.5 * (((xx - x) / sigma_x) ** 2 + ((yy - y) / sigma_y) ** 2))


def test_fit_peak_finds_the_nearby_peak():
    image = _gaussian((64, 80), 40.3, 21.7, 1.8, 1.2).astype(np.uint16)
    peak = fit_peak(image, 37, 24, half_width=8)
    assert peak.shape == (1,)
    assert peak["fit_x"][0] == pytest.approx(40.3, abs=0.05)
    assert peak["fit_y"][0] == pytest.approx(21.7, abs=0.05)
    assert peak["hwhm_x"][0] == pytest.approx(1.8 * np.sqrt(2 * np.log(2)), rel=0.05)
    assert peak["background"][0] == pytest.approx(50, abs=1)


def test_fit_peak_returns_none_without_a_peak():
    assert fit_peak(np.full((40, 40), 7, dtype=np.uint16), 20, 20) is None
    with pytest.raises(InputError):
        fit_peak(np.zeros((40, 40)), 50, 20)


def test_table_validation_and_copies():
    table = CalibrationTable(np.array([[0, 0, 4]]), [[1.0, 2.0]], [np.nan], ("",), [0])
    changed = table.with_energy(0, 6.6, "scan 7")
    assert table.n_energies == 0 and changed.n_energies == 1
    assert changed.energy_source == ("scan 7",)
    assert changed.with_energy(0, None).energy_source == ("",)
    with pytest.raises(InputError):
        table.with_energy(0, -1.0)
    with pytest.raises(InputError):
        table.with_energy(1, 6.0)
    with pytest.raises(InputError):
        CalibrationTable(np.array([[0, 0, 0]]), [[1.0, 2.0]], [np.nan], ("",), [0])
    with pytest.raises(ValueError):
        table.hkl[0, 0] = 1
