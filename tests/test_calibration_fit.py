# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Detector calibration fit against controlled cases from LaueGo Calibration.ipf."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from calibration_cases import (
    FAKE_SAMPLE_AXIS,
    GEOMETRY_2023,
    IGOR_PIXELS_DETECTOR0,
    IGOR_PIXELS_SIDE,
    fake_table,
    igor_error,
    reference_detector,
    si_crystal,
    si_reference,
)
from lauelab.calibration import (
    CalibrationTable,
    calibrate_detector,
    calibration_table,
    detector_azimuth_deg,
    peak_directions,
)
from lauelab.calibration._model import Residuals, detector_with
from lauelab.indexing import InputError, index_orientations, load_geometry

FAKE_ROTATION = Rotation.from_rotvec(FAKE_SAMPLE_AXIS).as_matrix()


def _perturbed(detector, rotation=(0.01, -0.008, 0.005), translation=(1500.0, -800.0, 3000.0)):
    return detector_with(
        detector, detector.rotation_vector + np.asarray(rotation), detector.translation + np.asarray(translation)
    )


def _start_reciprocal(rotation, axis=(0.004, -0.003, 0.006)):
    return si_reference() @ (Rotation.from_rotvec(axis).as_matrix() @ rotation).T


def _angle_deg(a, b):
    return np.degrees(np.linalg.norm(Rotation.from_matrix(a @ b.T).as_rotvec()))


def test_fake_data_matches_igor_selection():
    table = fake_table(reference_detector(0), FAKE_ROTATION, IGOR_PIXELS_DETECTOR0)
    # Igor keeps only reflections on the detector below 99 keV.
    assert 0 < len(table) <= 25
    assert np.all(table.energy_kev <= 99)
    assert table.n_energies == len(table)


@pytest.mark.parametrize("azimuth", [None, 90.0])
@pytest.mark.parametrize("offset", [0.0, 1.0])
def test_residuals_equal_igor_error(azimuth, offset):
    detector = reference_detector(0)
    table = fake_table(detector, FAKE_ROTATION, IGOR_PIXELS_DETECTOR0)
    # Drop some energies and some pixels so all three row kinds occur.
    table = table.with_energy(0, None).with_energy(1, None)
    pixels = np.array(table.pixels)
    pixels[2] = np.nan
    # An unused row (no pixel, no energy) still counts in Igor's energy weight.
    table = CalibrationTable(
        np.vstack([table.hkl, [[1, 1, 1]] * 5]), np.vstack([pixels, np.full((5, 2), np.nan)]),
        np.concatenate([table.energy_kev, np.full(5, np.nan)]), table.energy_source + ("",) * 5,
        np.concatenate([table.peak_index, np.full(5, -1)]),
    )
    r = detector.rotation_vector + offset * np.array([0.01, -0.02, 0.005])
    p = detector.translation + offset * np.array([900.0, -300.0, 2500.0])
    axis = FAKE_SAMPLE_AXIS + offset * np.array([0.01, 0.0, -0.02])
    residuals = Residuals(detector, si_reference(), table.hkl, table.pixels, table.energy_kev, azimuth)
    ours = np.linalg.norm(residuals(r, p, Rotation.from_rotvec(axis).as_matrix()))
    expected = igor_error(table, detector, r, p, axis, azimuth)
    assert ours == pytest.approx(expected, rel=1e-10, abs=1e-14)


def test_detector0_and_sample_rotation_are_recovered():
    truth = reference_detector(0)
    table = fake_table(truth, FAKE_ROTATION, IGOR_PIXELS_DETECTOR0)
    result = calibrate_detector(table, si_crystal(), _start_reciprocal(FAKE_ROTATION), _perturbed(truth))
    assert result.success, result.message
    # Igor rounds the rotation vector to 1e-8 rad and the translation to 1 µm here.
    np.testing.assert_allclose(result.refined.rotation_vector, truth.rotation_vector, atol=1e-8)
    np.testing.assert_allclose(result.refined.translation, truth.translation, atol=1.0)
    assert _angle_deg(result.refined_sample_rotation, FAKE_ROTATION) < 1e-8
    assert result.error_end < 1e-6 < result.error_start
    assert result.rms_pixel_error < 1e-3
    assert np.nanmax(np.abs(result.energy_error_ev)) < 1e-3
    assert detector_azimuth_deg(result.refined) == pytest.approx(90.0, abs=1e-6)


def test_noisy_fake_data_stays_within_reported_errors():
    truth = reference_detector(0)
    table = fake_table(truth, FAKE_ROTATION, IGOR_PIXELS_DETECTOR0, noise_px=0.5, seed=1)
    result = calibrate_detector(table, si_crystal(), _start_reciprocal(FAKE_ROTATION), _perturbed(truth))
    assert result.success, result.message
    assert 0.2 < result.rms_pixel_error < 1.0
    error = np.abs(result.refined.translation - truth.translation)
    assert np.all(error < 4 * result.translation_sigma_um + 1.0)
    error = np.abs(result.refined.rotation_vector - truth.rotation_vector)
    assert np.all(error < 4 * result.rotation_vector_sigma)


def test_least_squares_and_igor_bfgs_reach_the_same_minimum():
    """Igor minimizes the scalar error with BFGS over absolute rotation vectors."""
    truth = reference_detector(0)
    table = fake_table(truth, FAKE_ROTATION, IGOR_PIXELS_DETECTOR0, noise_px=0.5, seed=2)
    start = _perturbed(truth, (0.002, -0.001, 0.001), (300.0, -200.0, 500.0))
    start_rotation = Rotation.from_rotvec([0.001, -0.001, 0.002]).as_matrix() @ FAKE_ROTATION
    ours = calibrate_detector(table, si_crystal(), si_reference() @ start_rotation.T, start)
    assert ours.success, ours.message

    x0 = np.concatenate([
        start.rotation_vector, start.translation, Rotation.from_matrix(start_rotation).as_rotvec()
    ])
    scale = np.array([1e-3] * 3 + [200.0] * 3 + [1e-3] * 3)

    def igor(y):
        x = x0 + y * scale
        return igor_error(table, truth, x[0:3], x[3:6], x[6:9], 90.0)

    bfgs = minimize(igor, np.zeros(9), method="BFGS", options={"gtol": 1e-12, "maxiter": 5000})
    x = x0 + bfgs.x * scale
    assert ours.error_end < 0.1 * ours.error_start
    assert ours.error_end == pytest.approx(bfgs.fun, rel=1e-3)
    np.testing.assert_allclose(ours.refined.rotation_vector, x[0:3], atol=5e-6)
    np.testing.assert_allclose(ours.refined.translation, x[3:6], atol=5.0)
    assert _angle_deg(ours.refined_sample_rotation, Rotation.from_rotvec(x[6:9]).as_matrix()) < 1e-3


@pytest.mark.parametrize("slot", [1, 2])
def test_side_detector_with_fixed_sample_rotation(slot):
    truth = reference_detector(slot)
    table = fake_table(truth, FAKE_ROTATION, IGOR_PIXELS_SIDE)
    reciprocal = si_reference() @ FAKE_ROTATION.T
    start = _perturbed(truth, (0.005, 0.004, -0.006), (800.0, 600.0, -1500.0))
    result = calibrate_detector(table, si_crystal(), reciprocal, start, fit_sample_rotation=False)
    assert result.success, result.message
    assert result.azimuth_deg is None
    np.testing.assert_allclose(result.refined.rotation_vector, truth.rotation_vector, atol=1e-8)
    np.testing.assert_allclose(result.refined.translation, truth.translation, atol=1.0)
    np.testing.assert_array_equal(result.refined_sample_rotation, result.initial_sample_rotation)
    assert np.isnan(result.sample_rotation_sigma).all()


def test_recovers_the_2023_detector0_from_a_perturbed_start():
    truth = load_geometry(GEOMETRY_2023).detector(0)
    perfect = Rotation.from_rotvec([-2.2256407716336, -0.884037203122317, -0.38914203759791]).as_matrix()
    table = fake_table(truth, perfect, IGOR_PIXELS_DETECTOR0)
    assert len(table) >= 8
    result = calibrate_detector(
        table, si_crystal(), _start_reciprocal(perfect), _perturbed(truth), azimuth_deg=detector_azimuth_deg(truth)
    )
    assert result.success, result.message
    np.testing.assert_allclose(result.refined.rotation_vector, truth.rotation_vector, atol=1e-8)
    np.testing.assert_allclose(result.refined.translation, truth.translation, atol=1.0)


def test_indexed_frame_gives_the_table_and_fit():
    """Index peaks of a perfect-Si frame on the 2023 detector, then calibrate."""
    truth = load_geometry(GEOMETRY_2023).detector(0)
    perfect = Rotation.from_rotvec([-2.2256407716336, -0.884037203122317, -0.38914203759791]).as_matrix()
    from lauelab.analysis import simulate_reflections

    simulated = simulate_reflections(si_crystal(), si_reference() @ perfect.T, truth, energy_range_kev=(6.0, 20.0))
    peaks = simulated.detector_xy
    start = _perturbed(truth, (0.002, 0.001, -0.001), (300.0, 200.0, -400.0))
    patterns = index_orientations(peak_directions(start, peaks), si_crystal(), kev_max_calc=20, kev_max_test=25,
                                  angle_tolerance_deg=0.3)
    assert patterns
    table = calibration_table(peaks, patterns[0])
    assert len(table) >= 0.8 * len(peaks)
    row = int(np.argmin(np.linalg.norm(table.pixels - peaks[0], axis=1)))
    energy = simulated.energy_kev[0]
    table = table.with_energy(row, energy, "scan 123")
    # Indexing may label the reflection with another harmonic; use the simulated one.
    order = int(np.gcd.reduce(np.abs(simulated.hkl[0])))
    primitive = table.hkl[row] // int(np.gcd.reduce(np.abs(table.hkl[row])))
    table = table.with_hkl(row, primitive * order)
    result = calibrate_detector(table, si_crystal(), patterns[0].reciprocal, start,
                                azimuth_deg=detector_azimuth_deg(truth))
    assert result.success, result.message
    np.testing.assert_allclose(result.refined.translation, truth.translation, atol=2.0)
    np.testing.assert_allclose(result.refined.rotation_vector, truth.rotation_vector, atol=2e-6)
    assert result.rms_pixel_error < 0.01

    # Indexing again with the refined detector keeps the energy on its peak.
    again = index_orientations(peak_directions(result.refined, peaks), si_crystal(), kev_max_calc=20,
                               kev_max_test=25, angle_tolerance_deg=0.3)
    carried = calibration_table(peaks, again[0], previous=table)
    assert carried.n_energies == 1
    kept = int(np.flatnonzero(np.isfinite(carried.energy_kev))[0])
    np.testing.assert_allclose(carried.pixels[kept], table.pixels[row])
    assert carried.energy_source[kept] == "scan 123"
    assert np.gcd.reduce(np.abs(carried.hkl[kept])) == order


def test_fitting_the_sample_rotation_needs_an_energy():
    truth = reference_detector(0)
    table = fake_table(truth, FAKE_ROTATION, IGOR_PIXELS_DETECTOR0)
    for row in range(len(table)):
        table = table.with_energy(row, None)
    result = calibrate_detector(table, si_crystal(), _start_reciprocal(FAKE_ROTATION), _perturbed(truth))
    assert result.status == "insufficient_data"
    assert not result.success
    np.testing.assert_array_equal(result.refined.translation, result.initial.translation)


def test_an_energy_of_another_harmonic_is_reported():
    truth = reference_detector(0)
    table = fake_table(truth, FAKE_ROTATION, IGOR_PIXELS_DETECTOR0)
    table = table.with_energy(0, 2 * table.energy_kev[0])
    result = calibrate_detector(table, si_crystal(), si_reference() @ FAKE_ROTATION.T, truth)
    assert result.status == "energy_mismatch"
    assert "row 0" in result.message
    table = table.with_hkl(0, 2 * table.hkl[0])
    result = calibrate_detector(table, si_crystal(), si_reference() @ FAKE_ROTATION.T, truth)
    assert result.success, result.message


def test_too_few_reflections():
    truth = reference_detector(0)
    table = fake_table(truth, FAKE_ROTATION, IGOR_PIXELS_DETECTOR0[:2])
    result = calibrate_detector(table, si_crystal(), si_reference() @ FAKE_ROTATION.T, truth)
    assert result.status == "insufficient_data"


def test_reciprocal_of_another_crystal_is_refused():
    truth = reference_detector(0)
    table = fake_table(truth, FAKE_ROTATION, IGOR_PIXELS_DETECTOR0)
    with pytest.raises(InputError, match="not a rotation"):
        calibrate_detector(table, si_crystal(), 1.01 * si_reference(), truth)


def test_result_reports_per_row_values_and_provenance():
    truth = reference_detector(0)
    table = fake_table(truth, FAKE_ROTATION, IGOR_PIXELS_DETECTOR0)
    result = calibrate_detector(table, si_crystal(), _start_reciprocal(FAKE_ROTATION), _perturbed(truth))
    n = len(table)
    assert result.predicted_pixels.shape == (n, 2)
    assert result.pixel_error.shape == (n, 2)
    np.testing.assert_allclose(result.calculated_energy_kev, table.energy_kev, rtol=1e-9)
    assert not result.predicted_pixels.flags.writeable
    provenance = result.provenance()
    assert provenance["empirical_igor_parity"] == "pending"
    assert provenance["hc_kev_nm"] == 1.239841984
    note = result.summary_note()
    assert len(note) < 100 and note.isascii() and "<" not in note
    np.testing.assert_allclose(
        result.refined_reciprocal, si_reference() @ result.refined_sample_rotation.T
    )


def test_one_repeated_reflection_does_not_determine_the_detector():
    truth = reference_detector(0)
    table = fake_table(truth, FAKE_ROTATION, IGOR_PIXELS_DETECTOR0)
    n = 10
    same = CalibrationTable(
        np.tile(table.hkl[0], (n, 1)), np.tile(table.pixels[0], (n, 1)),
        table.energy_kev[0] + np.linspace(-0.01, 0.01, n), ("",) * n, np.arange(n),
    )
    result = calibrate_detector(same, si_crystal(), si_reference() @ FAKE_ROTATION.T, truth)
    assert result.status == "insufficient_data"
    assert np.isnan(result.translation_sigma_um).all()


def test_energies_alone_do_not_determine_the_detector():
    truth = reference_detector(0)
    table = fake_table(truth, FAKE_ROTATION, IGOR_PIXELS_DETECTOR0)
    table = CalibrationTable(table.hkl, np.full_like(table.pixels, np.nan), table.energy_kev,
                             table.energy_source, table.peak_index)
    result = calibrate_detector(table, si_crystal(), si_reference() @ FAKE_ROTATION.T, truth,
                                fit_sample_rotation=False)
    assert result.status == "insufficient_data"
    assert "condition number" in result.message
