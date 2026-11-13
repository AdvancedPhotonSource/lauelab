# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Controlled detector-calibration cases ported from LaueGo Calibration.ipf.

``fake_table`` follows ``MakeFakeCalibrationData`` and ``reference_detector``
follows ``GeoReferenceOrientation`` (microGeometryN.ipf). ``igor_error`` is a
line-by-line port of ``CalibrationErrorRPinternal`` and the azimuth term of
``CalibrationErrorRPrho``, kept unvectorized so that it checks the vectorized
residuals independently.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from lauelab.calibration import CalibrationTable, HC_KEV_NM, detector_azimuth_deg
from lauelab.calibration._model import detector_with
from lauelab.indexing import DetectorGeometry, load_crystal
from lauelab.indexing._liblaue import NativeCrystal

SI_XML = Path(__file__).parent / "config" / "Si.xml"
GEOMETRY_2023 = Path(__file__).parent / "config" / "geoN_2023-04-06_03-07-11.xml"

# Igor's "perfect Si" sample axis in findSampleAxis (rad).
PERFECT_SI_AXIS = np.array([-2.2256407716336, -0.884037203122317, -0.38914203759791])
# The sample axis of MakeFakeCalibrationData: -135 degrees about x.
FAKE_SAMPLE_AXIS = np.array([np.radians(-135.0), 0.0, 0.0])

# Pixel positions of MakeFakeCalibrationData for detector 0 and the side detectors.
IGOR_PIXELS_DETECTOR0 = np.array([
    (1023.5, 1023.5), (140, 1180), (1907, 1180), (1023.5, 166), (1023.5, 1881), (140, 86),
    (1907, 86), (140, 1961), (1907, 1961), (631, 544), (602, 1501), (1513, 1492), (1542, 514),
    (197, 598), (265, 1623), (612, 1852), (1494, 1824), (1793, 1580), (1827, 730), (1480, 163),
    (578, 191), (587, 1000), (1007, 1426), (1431, 1014), (1026, 580),
], dtype=float)
IGOR_PIXELS_SIDE = np.array([
    (511.5, 511.5), (70, 590), (940, 590), (512, 83), (512, 941), (70, 43), (920, 63), (70, 981),
    (950, 981), (316, 272), (301, 751), (757, 746), (771, 257), (99, 299), (133, 812), (306, 926),
    (747, 912), (897, 790), (914, 365), (740, 82), (289, 96), (294, 500), (504, 713), (716, 507),
    (513, 290),
], dtype=float)

_INCIDENT = np.array([0.0, 0.0, 1.0])


def si_crystal():
    return load_crystal(SI_XML)


def si_reference() -> np.ndarray:
    return NativeCrystal.create(si_crystal()).reciprocal()


def reference_detector(slot: int) -> DetectorGeometry:
    """Detector of Igor's GeoReferenceOrientation, lengths in µm."""
    if slot == 0:
        value = -2.0 / 3.0 * np.pi / np.sqrt(3.0)
        r, p, n, size, name = [value] * 3, (25e3, 0.0, 510e3), 2048, 409.6e3, "PE1621 723-3335"
    elif slot == 1:
        r = (-1.77549569095761, -0.735434395129632, -1.74815646188668)
        p, n, size, name = (-144e3, -1e3, 410e3), 1024, 204.8e3, "PE0822 883-4841"
    else:
        r = (-0.619944391960603, -1.49667815898843, -0.610398437087624)
        p, n, size, name = (-187e3, 0.0, 400e3), 1024, 204.8e3, "PE0822 883-4843"
    r = np.asarray(r, dtype=float)
    return DetectorGeometry(
        n, n, size, size, name, np.asarray(p, float), r, Rotation.from_rotvec(r).as_matrix()
    )


def _closest_hkl(values: np.ndarray) -> np.ndarray:
    """Igor's ClosestHKL."""
    maximum = np.max(np.abs(values))
    best, best_dot = 0, -np.inf
    for i in range(1, 26):
        hkl = np.round(values * i / maximum)
        dot = np.dot(hkl, values) / np.linalg.norm(hkl)
        if dot > best_dot:
            best, best_dot = i, dot
    return np.round(values * best / maximum).astype(np.int64) + 0


def _allowed_diamond(hkl) -> bool:
    h, k, l = (int(v) for v in hkl)
    if all(v % 2 for v in (h, k, l)):
        return True
    return all(v % 2 == 0 for v in (h, k, l)) and (h + k + l) % 4 == 0


def lowest_allowed_si(hkl: np.ndarray) -> np.ndarray:
    """Igor's lowestAllowedHKL for diamond-structure Si."""
    primitive = hkl // int(np.gcd.reduce(np.abs(hkl)))
    for order in range(1, 9):
        if _allowed_diamond(primitive * order):
            return primitive * order
    raise ValueError(f"no allowed harmonic of {hkl}")


def fake_table(
    detector: DetectorGeometry,
    sample_rotation: np.ndarray,
    pixels: np.ndarray,
    *,
    noise_px: float = 0.0,
    seed: int = 0,
) -> CalibrationTable:
    """MakeFakeCalibrationData for Si: reflections near the given pixels, all with energies.

    Igor adds Gaussian noise of ``noise_px`` to positions and uniform noise of
    ``noise_px / 2000`` keV to energies.
    """
    rng = np.random.default_rng(seed)
    recip = sample_rotation @ si_reference().T  # Igor's recip: columns a*, b*, c*
    rows_hkl, rows_px, rows_kev = [], [], []
    for px, py in pixels:
        outgoing = detector.pixel_to_lab(np.array([px, py]))
        outgoing /= np.linalg.norm(outgoing)
        hkl = _closest_hkl(np.linalg.solve(recip, outgoing - _INCIDENT))
        if not hkl.any():
            continue
        hkl = lowest_allowed_si(hkl)
        q = recip @ hkl
        length = np.linalg.norm(q)
        qhat = q / length
        dot = -np.dot(_INCIDENT, qhat)
        kev = length * HC_KEV_NM / (4.0 * np.pi * dot)
        x, y = detector.q_to_pixel(q)
        if not (kev <= 99 and 0 <= x < detector.nx and 0 <= y < detector.ny):
            continue
        if noise_px > 0:
            x, y = x + rng.normal(0, noise_px), y + rng.normal(0, noise_px)
            kev += rng.uniform(-noise_px / 2000, noise_px / 2000)
        rows_hkl.append(hkl)
        rows_px.append((x, y))
        rows_kev.append(kev)
    n = len(rows_hkl)
    return CalibrationTable(
        np.array(rows_hkl), np.array(rows_px), np.array(rows_kev), ("",) * n, np.arange(n)
    )


def igor_error(table, detector, rotation_vector, translation, sample_axis, azimuth_deg=None) -> float:
    """CalibrationErrorRPinternal (+ the azimuth term of CalibrationErrorRPrho), literally."""
    d = detector_with(detector, rotation_vector, translation)
    rho_sample = Rotation.from_rotvec(sample_axis).as_matrix()
    recip0 = si_reference().T
    ki = np.array([0.0, 0.0, 1.0])
    n_rows = len(table)
    n_energy = int(np.sum(table.energy_kev > 0))
    weight_e = min((n_rows / max(n_energy, 1)) ** 2, 40)
    err2 = weight_sum = 0.0
    for i in range(n_rows):
        pixel_x, pixel_y = table.pixels[i]
        pixel_x = pixel_x if 0 <= pixel_x <= d.nx - 1 else np.nan
        pixel_y = pixel_y if 0 <= pixel_y <= d.ny - 1 else np.nan
        kev = table.energy_kev[i]
        kev = kev if np.isfinite(kev) and kev > 0 else np.nan
        have_qhat = np.isfinite(pixel_x + pixel_y)
        have_e = kev > 0
        if not have_qhat and not have_e:
            continue
        qcalc = rho_sample @ recip0 @ table.hkl[i]
        if have_qhat:
            kf = d.pixel_to_lab(np.array([pixel_x, pixel_y]))
            kf = kf / np.linalg.norm(kf)
            qmeas = kf - ki
        else:
            qmeas = qcalc.copy()
        qmeas = qmeas / np.linalg.norm(qmeas)
        radial_hat = qmeas.copy()
        q_len = -4 * np.pi * np.dot(ki, qmeas) * kev / HC_KEV_NM if have_e else np.linalg.norm(qcalc)
        if not np.isfinite(q_len) or q_len <= 0:
            continue
        qmeas = qmeas * q_len
        dq = qcalc - qmeas
        dq_rad = np.dot(radial_hat, dq)
        dq = dq - dq_rad * radial_hat
        dq_perp = np.linalg.norm(dq)
        dq2 = dq_perp**2 if have_qhat else 0.0
        dq2 += weight_e * dq_rad**2 if have_e else 0.0
        err2 += dq2 / q_len**2
        weight_sum += weight_e if have_e else 1.0
    rms = np.sqrt(err2 / weight_sum)
    if azimuth_deg is None:
        return float(rms)
    err_orient = 10 * np.radians(detector_azimuth_deg(d) - azimuth_deg)
    return float(np.sqrt(rms**2 + err_orient**2))
