# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Backend-independent crystal orientation calculations."""

from __future__ import annotations

from itertools import combinations

import numpy as np


def lattice_params_to_reciprocal(
    a, b, c, alpha_deg, beta_deg, gamma_deg, *, space_group=None
):
    """Return the native reciprocal basis as rows in inverse input-length units.

    ``a``, ``b``, and ``c`` must use one common length unit; the returned basis
    uses its inverse (normally ``1/nm`` in the public analysis API). Angles are
    in degrees, and reciprocal vectors include the ``2*pi`` factor.

    The direct basis follows native ``setDirectRecip``: ``c`` is parallel to
    positive z, ``b`` lies in the yz plane, and ``a`` completes a right-handed
    basis. When ``space_group`` is provided, native crystal-system constraints
    are applied before constructing the basis. For a trigonal space group,
    angles within the native tolerance of ``(90, 90, 120)`` select hexagonal
    axes; all other angles select rhombohedral axes.
    """
    lengths = np.asarray([a, b, c], dtype=float)
    angles = np.radians(np.asarray([alpha_deg, beta_deg, gamma_deg], dtype=float))
    if not np.isfinite(lengths).all() or np.any(lengths <= 0):
        raise ValueError("cell lengths must be finite and positive")
    if not np.isfinite(angles).all() or np.any(angles <= 0) or np.any(angles >= np.pi):
        raise ValueError("cell angles must be finite and between 0 and 180 degrees")
    if space_group is not None:
        if not isinstance(space_group, (int, np.integer)) or not 1 <= space_group <= 230:
            raise ValueError("space_group must be between 1 and 230")
        a, b, c = lengths
        alpha, beta, gamma = angles
        using_hex_axes = (
            abs(np.pi / 2.0 - alpha)
            + abs(np.pi / 2.0 - beta)
            + abs(2.0 * np.pi / 3.0 - gamma)
        ) < 1e-9
        if space_group >= 195:
            b = c = a
            alpha = beta = gamma = np.pi / 2.0
        elif space_group >= 168:
            b = a
            alpha = beta = np.pi / 2.0
            gamma = 2.0 * np.pi / 3.0
        elif space_group >= 143:
            b = a
            if using_hex_axes:
                alpha = beta = np.pi / 2.0
                gamma = 2.0 * np.pi / 3.0
            else:
                c = a
                beta = gamma = alpha
        elif space_group >= 75:
            b = a
            alpha = beta = gamma = np.pi / 2.0
        elif space_group >= 16:
            alpha = beta = gamma = np.pi / 2.0
        elif space_group >= 3:
            alpha = gamma = np.pi / 2.0
    else:
        a, b, c = lengths
        alpha, beta, gamma = angles

    sin_alpha = np.sin(alpha)
    cos_alpha, cos_beta, cos_gamma = np.cos([alpha, beta, gamma])
    cos_alpha = 0.0 if abs(cos_alpha) < 1e-14 else cos_alpha
    cos_beta = 0.0 if abs(cos_beta) < 1e-14 else cos_beta
    cos_gamma = 0.0 if abs(cos_gamma) < 1e-14 else cos_gamma
    phi_squared = (
        1.0 - cos_alpha**2 - cos_beta**2 - cos_gamma**2
        + 2.0 * cos_alpha * cos_beta * cos_gamma
    )
    if phi_squared <= 0 or sin_alpha == 0:
        raise ValueError("cell parameters do not define a valid lattice")
    phi = np.sqrt(phi_squared)
    direct = np.array([
        [a * phi / sin_alpha, a * (cos_gamma - cos_alpha * cos_beta) / sin_alpha, a * cos_beta],
        [0.0, b * sin_alpha, b * cos_alpha],
        [0.0, 0.0, c],
    ])
    return 2.0 * np.pi * np.linalg.inv(direct).T


def reciprocal_to_orientation(reciprocal, reference_reciprocal):
    """Return the orientation matrix for measured and reference lattices.

    ``reciprocal`` may be one ``(3, 3)`` matrix or a stack with shape
    ``(..., 3, 3)``; ``reference_reciprocal`` is one matrix. The result has
    the shape of ``reciprocal``.
    """
    measured = np.asarray(reciprocal, dtype=float)
    reference = np.asarray(reference_reciprocal, dtype=float)
    if measured.shape[-2:] != (3, 3) or reference.shape != (3, 3):
        raise ValueError("reciprocal lattices must have shape (..., 3, 3) and (3, 3)")
    return np.swapaxes(measured, -1, -2) @ np.linalg.inv(reference.T)


def orientation_to_rodrigues(rotation):
    """Convert rotation matrices to ``axis * tan(angle / 2)``.

    ``rotation`` may be one ``(3, 3)`` matrix or a stack with shape
    ``(..., 3, 3)``; the result has shape ``(3,)`` or ``(..., 3)``.

    The Rodrigues magnitude is singular at 180 degrees. Near that singularity,
    the axis is recovered from the rotation eigenvectors and the effective angle
    is clamped to ``pi - 1e-7`` radians, with the first nonzero axis component
    chosen positive. Unlike the Laue Portal's zero-vector fallback, this retains
    a deterministic axis and decodes to approximately 180 degrees. A matrix
    with a non-finite entry maps to a vector of ``NaN``.
    """
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape[-2:] != (3, 3):
        raise ValueError("rotation must have shape (..., 3, 3)")
    stack = rotation.reshape((-1, 3, 3))
    result = np.full((len(stack), 3), np.nan)
    finite = np.isfinite(stack).all(axis=(1, 2))
    trace = np.trace(stack, axis1=1, axis2=2)
    angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    axis = np.stack([
        stack[:, 2, 1] - stack[:, 1, 2],
        stack[:, 0, 2] - stack[:, 2, 0],
        stack[:, 1, 0] - stack[:, 0, 1],
    ], axis=1)
    norm = np.linalg.norm(axis, axis=1)
    identity = finite & (angle < 1e-12)
    generic = finite & ~identity & (norm >= 1e-8)
    result[identity] = 0.0
    result[generic] = (
        axis[generic] / norm[generic, None] * np.tan(angle[generic] / 2.0)[:, None]
    )
    for index in np.flatnonzero(finite & ~identity & (norm < 1e-8)):
        values, vectors = np.linalg.eigh(stack[index] + np.eye(3))
        near_axis = vectors[:, np.argmax(values)]
        first = np.flatnonzero(np.abs(near_axis) > 1e-12)
        if first.size and near_axis[first[0]] < 0:
            near_axis = -near_axis
        result[index] = near_axis * np.tan((np.pi - 1e-7) / 2.0)
    return result.reshape(rotation.shape[:-2] + (3,))


def crystal_direction(rotation, lab_direction):
    """Express a lab-frame direction in crystal coordinates.

    ``rotation`` may be one ``(3, 3)`` matrix or a stack with shape
    ``(..., 3, 3)``; the result has the matching shape ``(..., 3)``. Every
    matrix in a stack is solved at once, so a singular matrix anywhere in
    the stack raises ``numpy.linalg.LinAlgError``.
    """
    rotation = np.asarray(rotation, dtype=float)
    direction = np.asarray(lab_direction, dtype=float)
    if rotation.shape[-2:] != (3, 3) or direction.shape != (3,):
        raise ValueError("rotation and lab_direction must have shapes (..., 3, 3) and (3,)")
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm == 0:
        raise ValueError("lab_direction must be a finite nonzero vector")
    result = np.linalg.solve(rotation, direction / norm)
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def _rotation_matrix(axis, angle_deg):
    axis = np.array(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    angle = np.radians(angle_deg)
    c, s, c1 = np.cos(angle), np.sin(angle), 1.0 - np.cos(angle)
    return np.array([
        [c + x * x * c1, x * y * c1 - z * s, x * z * c1 + y * s],
        [x * y * c1 + z * s, c + y * y * c1, y * z * c1 - x * s],
        [x * z * c1 - y * s, y * z * c1 + x * s, c + z * z * c1],
    ])


def _cubic_symmetry_ops():
    operations = [np.eye(3)]
    for axis in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
        operations.extend(_rotation_matrix(axis, angle) for angle in (90, 180, 270))
    for axis in ([1, 1, 1], [-1, 1, 1], [1, -1, 1], [-1, -1, 1]):
        operations.extend(_rotation_matrix(axis, angle) for angle in (120, 240))
    for axis in ([1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1], [0, 1, 1], [0, 1, -1]):
        operations.append(_rotation_matrix(axis, 180))
    return np.asarray(operations)


def _hexagonal_symmetry_ops():
    operations = [_rotation_matrix([0, 0, 1], angle) for angle in range(0, 360, 60)]
    for angle in range(0, 180, 30):
        operations.append(_rotation_matrix([np.cos(np.radians(angle)), np.sin(np.radians(angle)), 0], 180))
    return np.asarray(operations)


CUBIC_SYMMETRY = _cubic_symmetry_ops()
CUBIC_SYMMETRY.setflags(write=False)
HEXAGONAL_SYMMETRY = _hexagonal_symmetry_ops()
HEXAGONAL_SYMMETRY.setflags(write=False)


def symmetry_operations(symmetry: str | int) -> np.ndarray:
    """Return proper rotations for a supported symmetry name or space group."""
    if isinstance(symmetry, (int, np.integer)):
        if 195 <= symmetry <= 230:
            return CUBIC_SYMMETRY.copy()
        if 168 <= symmetry <= 194:
            return HEXAGONAL_SYMMETRY.copy()
    elif symmetry == "cubic":
        return CUBIC_SYMMETRY.copy()
    elif symmetry == "hexagonal":
        return HEXAGONAL_SYMMETRY.copy()
    raise ValueError("symmetry must be 'cubic', 'hexagonal', or a supported space group")


def _select_by_trace(rotations, right_factors):
    """Return ``rotations[n] @ right_factors[o]`` for the ``o`` maximizing the trace.

    ``trace(R @ M)`` equals ``sum(R * M.T)``, so the argmax is found without
    forming every candidate product; only the chosen products are formed.
    """
    traces = np.einsum("nij,oji->no", rotations, right_factors)
    best = np.argmax(traces, axis=1)
    return np.einsum("nij,njk->nik", rotations, right_factors[best])


def symmetry_reduce_orientation(rotation, *, operations=None):
    """Return the symmetry-equivalent orientation nearest to identity.

    ``rotation`` may be one ``(3, 3)`` matrix or a stack with shape
    ``(..., 3, 3)``; the result has the same shape.
    """
    rotation = np.asarray(rotation, dtype=float)
    if rotation.shape[-2:] != (3, 3):
        raise ValueError("rotation must have shape (..., 3, 3)")
    operations = np.eye(3)[None, ...] if operations is None else np.asarray(operations, dtype=float)
    stack = rotation.reshape((-1, 3, 3))
    reduced = _select_by_trace(stack, np.swapaxes(operations, 1, 2))
    return reduced.reshape(rotation.shape)


def misorientation_matrix(rotation_a, rotation_b, *, operations=None):
    """Return the minimum-angle misorientation from ``rotation_b`` to ``rotation_a``.

    ``rotation_a`` may be one ``(3, 3)`` matrix or a stack with shape
    ``(..., 3, 3)``; ``rotation_b`` is one matrix. The result has the shape
    of ``rotation_a``.
    """
    a = np.asarray(rotation_a, dtype=float)
    if a.shape[-2:] != (3, 3):
        raise ValueError("rotation_a must have shape (..., 3, 3)")
    b_inv = np.linalg.inv(np.asarray(rotation_b, dtype=float))
    operations = np.eye(3)[None, ...] if operations is None else np.asarray(operations, dtype=float)
    stack = a.reshape((-1, 3, 3))
    relative = _select_by_trace(stack, np.swapaxes(operations, 1, 2) @ b_inv)
    return relative.reshape(a.shape)


def misorientation_angle(rotation_a, rotation_b, *, operations=None):
    """Return the minimum misorientation angle in degrees."""
    relative = misorientation_matrix(rotation_a, rotation_b, operations=operations)
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def misorientation_from_reference(rotations, reference_index, *, operations=None):
    """Return Rodrigues vectors and angles relative to one orientation."""
    rotations = np.asarray(rotations, dtype=float)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError("rotations must have shape (n, 3, 3)")
    if not 0 <= reference_index < len(rotations):
        raise IndexError("reference_index is out of range")
    reference = rotations[reference_index]
    reduced = misorientation_matrix(rotations, reference, operations=operations)
    vectors = orientation_to_rodrigues(reduced)
    cosines = np.clip((np.trace(reduced, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    return vectors, np.degrees(np.arccos(cosines))


def pairwise_misorientation(rotations, *, indices=None, operations=None):
    """Return pairs and corresponding misorientation angles."""
    rotations = np.asarray(rotations, dtype=float)
    selected = range(len(rotations)) if indices is None else tuple(indices)
    pairs = tuple(combinations(selected, 2))
    angles = np.asarray([
        misorientation_angle(rotations[i], rotations[j], operations=operations)
        for i, j in pairs
    ])
    return pairs, angles
