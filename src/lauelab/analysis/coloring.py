# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Backend-independent color calculations for crystallographic data."""

from __future__ import annotations

import colorsys

import numpy as np


def _round_significant(value, digits):
    if value == 0 or not np.isfinite(value):
        return value
    return round(value, digits - int(np.floor(np.log10(abs(value)))) - 1)


def cubic_ipf_colors(directions):
    """Map crystal directions to cubic IPF RGB values in ``[0, 1]``."""
    values = np.asarray(directions, dtype=float)
    if values.shape[-1:] != (3,):
        raise ValueError("directions must have shape (..., 3)")
    flat = values.reshape((-1, 3))
    result = np.full((len(flat), 3), 0.5)
    poles = np.column_stack([
        [0.0, 0.0, 1.0],
        [0.0, 1.0 / np.sqrt(2.0), 1.0 / np.sqrt(2.0)],
        [1.0 / np.sqrt(3.0)] * 3,
    ])
    norms = np.linalg.norm(flat, axis=1)
    finite = np.isfinite(norms)
    result[finite & (norms < 1e-12)] = 0.0
    usable = finite & (norms >= 1e-12)
    if np.any(usable):
        # Fold every direction into the standard triangle, then express it in
        # the pole basis: one solve for all rows with the poles as the matrix.
        folded = np.sort(np.abs(flat[usable]), axis=1) / norms[usable, None]
        coefficients = np.maximum(np.linalg.solve(poles, folded.T).T, 0.0)
        maximum = coefficients.max(axis=1)
        scaled = np.zeros_like(coefficients)
        positive = maximum > 1e-12
        scaled[positive] = coefficients[positive] / maximum[positive, None]
        result[usable] = scaled
    return result.reshape(values.shape)


def rodrigues_colors(vectors, *, max_angle_deg=None):
    """Map Rodrigues vectors to RGB values in ``[0, 1]``."""
    values = np.asarray(vectors, dtype=float)
    if values.shape[-1:] != (3,):
        raise ValueError("vectors must have shape (..., 3)")
    flat = values.reshape((-1, 3))
    lengths = np.linalg.norm(flat, axis=1)
    if max_angle_deg is None:
        component_angles = 2.0 * np.degrees(np.arctan(np.abs(flat.reshape(-1))))
        finite = component_angles[np.isfinite(component_angles)]
        maximum = _round_significant(float(np.percentile(finite, 95)), 2) if len(finite) else 0.0
        max_angle_deg = maximum if maximum > 0 else 45.0
    if not np.isfinite(max_angle_deg) or max_angle_deg <= 0:
        raise ValueError("max_angle_deg must be positive and finite")

    result = np.zeros_like(flat)
    usable = np.isfinite(lengths) & (lengths >= 1e-12)
    if np.any(usable):
        angles = 2.0 * np.degrees(np.arctan(lengths[usable]))
        scaled = np.clip(
            flat[usable] / lengths[usable, None] * (angles / max_angle_deg)[:, None],
            -1.0,
            1.0,
        )
        positive = np.maximum(scaled, 0.0)
        negative = np.maximum(-scaled, 0.0)
        result[usable] = np.clip(np.stack([
            positive[:, 0] + negative[:, 1] / 2 + negative[:, 2] / 2,
            positive[:, 1] + negative[:, 0] / 2 + negative[:, 2] / 2,
            positive[:, 2] + negative[:, 0] / 2 + negative[:, 1] / 2,
        ], axis=1), 0.0, 1.0)
    return result.reshape(values.shape)


def hsv_position_colors(dx, dy, *, radius=1.0):
    """Map Cartesian offsets to an HSV wheel with white at its center."""
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be positive and finite")
    dx, dy = np.broadcast_arrays(np.asarray(dx, dtype=float), np.asarray(dy, dtype=float))
    result = np.ones(dx.shape + (3,))
    for index in np.ndindex(dx.shape):
        distance = np.hypot(dx[index], dy[index])
        if distance >= 1e-12:
            hue = np.arctan2(dy[index], dx[index]) % (2 * np.pi)
            result[index] = colorsys.hsv_to_rgb(hue / (2 * np.pi), min(1.0, distance / radius), 1.0)
    return result


def closest_pole_colors(points, pattern_indices, count, *, center=(0.0, 0.0), radius=1.0):
    """Return each pattern's HSV color from its closest projected pole."""
    points = np.asarray(points, dtype=float)
    pattern_indices = np.asarray(pattern_indices, dtype=int)
    result = np.ones((count, 3))
    if not len(points):
        return result
    offsets = points - np.asarray(center, dtype=float)
    distances = np.sum(offsets**2, axis=1)
    minima = np.full(count, np.inf)
    np.minimum.at(minima, pattern_indices, distances)
    candidates = np.flatnonzero(distances == minima[pattern_indices])
    patterns, first = np.unique(pattern_indices[candidates], return_index=True)
    closest = candidates[first]
    result[patterns] = hsv_position_colors(offsets[closest, 0], offsets[closest, 1], radius=radius)
    return result


def cubic_ipf_key(resolution=256):
    """Return a cubic IPF reference triangle as a uint8 RGBA image."""
    if resolution < 2:
        raise ValueError("resolution must be at least 2")
    maximum = np.sqrt(2.0) - 1.0
    image = np.zeros((resolution, resolution, 4), dtype=np.uint8)
    for row in range(resolution):
        for column in range(resolution):
            x = column / (resolution - 1) * maximum
            y = (resolution - 1 - row) / (resolution - 1) * maximum
            denominator = 1.0 + x * x + y * y
            direction = np.array([2 * x, 2 * y, 1 - x * x - y * y]) / denominator
            if direction[1] > direction[0] + 1e-6 or direction[0] > direction[2] + 1e-6:
                continue
            image[row, column, :3] = np.clip(cubic_ipf_colors(direction) * 255, 0, 255)
            image[row, column, 3] = 255
    return image


def hsv_key(resolution=256):
    """Return a circular HSV reference key as a uint8 RGBA image."""
    if resolution < 2:
        raise ValueError("resolution must be at least 2")
    coordinate = np.linspace(-1.0, 1.0, resolution)
    x, y = np.meshgrid(coordinate, coordinate[::-1])
    inside = x * x + y * y <= 1.0
    image = np.zeros((resolution, resolution, 4), dtype=np.uint8)
    image[..., :3][inside] = np.clip(hsv_position_colors(x[inside], y[inside]) * 255, 0, 255)
    image[..., 3][inside] = 255
    return image
