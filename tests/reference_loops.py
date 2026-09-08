# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Per-row reference implementations for the vectorization equivalence tests.

These are verbatim copies of the loop forms that the vectorized library code
replaced. They exist only so that ``tests/test_vectorization.py`` can show
that the vectorized code returns the same values. Do not import them from
the package.
"""

import colorsys

import numpy as np

from lauelab.analysis.orientation import crystal_direction, orientation_to_rodrigues


def best_pattern_mask(pattern_frame_indices, pattern_indices):
    count = len(pattern_indices)
    selected = np.zeros(count, dtype=bool)
    for frame_index in np.unique(pattern_frame_indices):
        rows = np.flatnonzero(pattern_frame_indices == frame_index)
        if len(rows):
            selected[rows[np.argmin(pattern_indices[rows])]] = True
    return selected


def crystal_directions(rotations, normal):
    directions = np.full((len(rotations), 3), np.nan)
    for index, rotation in enumerate(rotations):
        if not np.isfinite(rotation).all():
            continue
        try:
            directions[index] = crystal_direction(rotation, normal)
        except (ValueError, np.linalg.LinAlgError):
            pass
    return directions


def cubic_ipf_colors(directions):
    values = np.asarray(directions, dtype=float)
    flat = values.reshape((-1, 3))
    result = np.full((len(flat), 3), 0.5)
    poles = np.column_stack([
        [0.0, 0.0, 1.0],
        [0.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
        [1.0 / np.sqrt(3.0)] * 3,
    ])
    for index, direction in enumerate(flat):
        norm = np.linalg.norm(direction)
        if not np.isfinite(norm):
            continue
        if norm < 1e-12:
            result[index] = 0.0
            continue
        folded = np.sort(np.abs(direction)) / norm
        coefficients = np.maximum(np.linalg.solve(poles, folded), 0.0)
        maximum = np.max(coefficients)
        result[index] = coefficients / maximum if maximum > 1e-12 else 0.0
    return result.reshape(values.shape)


def symmetry_reduce_orientation(rotation, operations):
    candidates = rotation @ np.swapaxes(operations, 1, 2)
    return candidates[np.argmax(np.trace(candidates, axis1=1, axis2=2))]


def misorientation_matrix(rotation_a, rotation_b, operations):
    b_inv = np.linalg.inv(rotation_b)
    candidates = rotation_a @ (np.swapaxes(operations, 1, 2) @ b_inv)
    return candidates[np.argmax(np.trace(candidates, axis1=1, axis2=2))]


def rodrigues_vectors(rotations):
    return np.asarray([orientation_to_rodrigues(rotation) for rotation in rotations])


def rodrigues_colors(vectors, max_angle_deg):
    flat = np.asarray(vectors, dtype=float).reshape((-1, 3))
    lengths = np.linalg.norm(flat, axis=1)
    result = np.zeros_like(flat)
    for index, (vector, length) in enumerate(zip(flat, lengths, strict=True)):
        if not np.isfinite(length) or length < 1e-12:
            continue
        angle = 2.0 * np.degrees(np.arctan(length))
        x, y, z = np.clip(vector / length * angle / max_angle_deg, -1.0, 1.0)
        result[index] = np.clip([
            max(x, 0) + max(-y, 0) / 2 + max(-z, 0) / 2,
            max(y, 0) + max(-x, 0) / 2 + max(-z, 0) / 2,
            max(z, 0) + max(-x, 0) / 2 + max(-y, 0) / 2,
        ], 0.0, 1.0)
    return result


def hsv_position_colors(dx, dy, radius):
    dx, dy = np.broadcast_arrays(np.asarray(dx, dtype=float), np.asarray(dy, dtype=float))
    result = np.ones(dx.shape + (3,))
    for index in np.ndindex(dx.shape):
        distance = np.hypot(dx[index], dy[index])
        if distance >= 1e-12:
            hue = np.arctan2(dy[index], dx[index]) % (2 * np.pi)
            result[index] = colorsys.hsv_to_rgb(hue / (2 * np.pi), min(1.0, distance / radius), 1.0)
    return result


def rgb_strings(values):
    array = np.asarray(values, dtype=float)
    if array.shape == (3,):
        array = array.reshape(1, 3)
    colors = []
    for value in array:
        if np.isfinite(value).all():
            red, green, blue = np.clip(np.rint(value * 255), 0, 255).astype(int)
            colors.append((red, green, blue))
        else:
            colors.append(None)
    return colors


def reciprocal_to_orientations(reciprocals, reference):
    rotations = np.full((len(reciprocals), 3, 3), np.nan)
    for index, reciprocal in enumerate(reciprocals):
        if np.isfinite(reciprocal).all():
            try:
                rotations[index] = reciprocal.T @ np.linalg.inv(reference.T)
            except np.linalg.LinAlgError:
                pass
    return rotations
