# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Check of a calibration crystal's orientation against its surface normal."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
from pathlib import Path

import numpy as np

from lauelab.indexing import Crystal, InputError, load_crystal

_INCIDENT = np.array([0.0, 0.0, 1.0])
#: Surface normal of the 34-ID-E calibration standard: 45 degrees between the
#: reversed incident beam and the +y axis.
DEFAULT_SURFACE_NORMAL = (0.0, np.sqrt(0.5), -np.sqrt(0.5))


@dataclass(frozen=True)
class SurfaceCheck:
    """Direction of the surface reflection of an oriented crystal.

    Attributes
    ----------
    hkl
        The member of the requested family whose scattering vector is closest to
        the expected surface normal, shape ``(3,)``.
    direction
        Its unit scattering vector in the 34-ID-E laboratory frame, shape
        ``(3,)``.
    expected_normal
        The expected surface normal, a unit vector with shape ``(3,)``.
    exit_direction
        Direction of the beam that the expected surface would reflect
        specularly, shape ``(3,)``.
    angle_to_expected_deg
        Angle between ``direction`` and ``expected_normal``.
    angle_to_incident_deg
        Angle between ``direction`` and the reversed incident beam (-z).
    angle_to_exit_deg
        Angle between ``direction`` and ``exit_direction``.
    between
        Whether ``direction`` is closer to both the reversed incident beam and
        the exit direction than they are to each other, so that it lies between
        the two beams.

    Notes
    -----
    For a cubic crystal some member of a family such as {001} is always within
    about 55 degrees of any direction, so ``between`` can be true for a wrong
    orientation. ``angle_to_expected_deg`` shows the error: for the transpose
    of Igor's perfect-Si rotation it is 29.9 degrees, against 1.1 degrees for
    the rotation itself.
    """

    hkl: np.ndarray
    direction: np.ndarray
    expected_normal: np.ndarray
    exit_direction: np.ndarray
    angle_to_expected_deg: float
    angle_to_incident_deg: float
    angle_to_exit_deg: float
    between: bool

    def __post_init__(self) -> None:
        for name in ("hkl", "direction", "expected_normal", "exit_direction"):
            value = np.array(getattr(self, name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b))))


def _family(crystal: Crystal, hkl: np.ndarray) -> np.ndarray:
    if crystal.crystal_system == "cubic":
        members = {
            tuple(int(sign * value) for sign, value in zip(signs, order))
            for order in permutations(hkl)
            for signs in product((1, -1), repeat=3)
        }
    else:
        members = {tuple(int(v) for v in hkl), tuple(int(-v) for v in hkl)}
    members.discard((0, 0, 0))
    return np.array(sorted(members), dtype=np.int64)


def surface_check(
    crystal: Crystal | str | Path,
    reciprocal,
    *,
    family=(0, 0, 1),
    expected_normal=DEFAULT_SURFACE_NORMAL,
) -> SurfaceCheck:
    """Find where the surface reflection of an oriented crystal points.

    The calibration standard at 34-ID-E is assumed to be a Si wafer with a
    (001) surface whose normal is 45 degrees between the reversed incident beam
    and the +y axis. Its surface reflection, such as 004, then lies between the
    incident beam and the beam diffracted toward the detector. A sample
    rotation applied in the wrong direction (the transpose of the rotation)
    moves it elsewhere.

    Parameters
    ----------
    crystal
        The calibration crystal, or the path of its crystal XML file.
    reciprocal
        Reciprocal basis rows in 1/nm, including 2π, of the oriented crystal,
        such as ``Pattern.reciprocal`` or
        :attr:`DetectorCalibration.refined_reciprocal`.
    family
        Miller indices of the surface reflection. For a cubic crystal every
        signed permutation is a member of the family; otherwise ``family`` and
        its negative are.
    expected_normal
        Expected direction of the surface normal in the 34-ID-E laboratory
        frame. It is normalized.

    Returns
    -------
    SurfaceCheck
        The closest family member and its angles.
    """
    if isinstance(crystal, (str, Path)):
        crystal = load_crystal(crystal)
    if not isinstance(crystal, Crystal):
        raise InputError("crystal must be a Crystal or a crystal file path")
    reciprocal = np.asarray(reciprocal, dtype=np.float64)
    if reciprocal.shape != (3, 3) or not np.isfinite(reciprocal).all():
        raise InputError("reciprocal must be a finite (3, 3) array")
    hkl = np.asarray(family)
    if hkl.shape != (3,) or hkl.dtype.kind not in "iu" or not hkl.any():
        raise InputError("family must be three integers, not all zero")
    normal = np.asarray(expected_normal, dtype=np.float64)
    if normal.shape != (3,) or not np.isfinite(normal).all() or not normal.any():
        raise InputError("expected_normal must be a nonzero vector of three values")
    normal = normal / np.linalg.norm(normal)

    members = _family(crystal, hkl)
    q = members @ reciprocal
    q /= np.linalg.norm(q, axis=1)[:, None]
    best = int(np.argmax(q @ normal))
    direction = q[best]
    exit_direction = _INCIDENT - 2.0 * np.dot(_INCIDENT, normal) * normal
    reverse_incident = -_INCIDENT
    span = _angle_deg(reverse_incident, exit_direction)
    to_incident = _angle_deg(direction, reverse_incident)
    to_exit = _angle_deg(direction, exit_direction)
    return SurfaceCheck(
        members[best], direction, normal, exit_direction,
        _angle_deg(direction, normal), to_incident, to_exit,
        bool(to_incident < span and to_exit < span),
    )
