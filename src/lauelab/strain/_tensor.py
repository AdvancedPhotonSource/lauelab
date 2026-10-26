# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Native-basis lattice and Igor stretch-strain calculations."""

from __future__ import annotations

import numpy as np

from lauelab.analysis.orientation import lattice_params_to_reciprocal
from lauelab.indexing.crystal import Cell, Crystal
from lauelab.indexing.errors import InputError

PARAMETERS = ("a", "b", "c", "alpha", "beta", "gamma")
COORDINATES = ("crystal", "beamline", "xhf", "sample")
_r = 1 / np.sqrt(2)
_FRAMES = {
    "beamline": np.eye(3),
    "xhf": np.array([[1, 0, 0], [0, _r, _r], [0, -_r, _r]]),
    "sample": np.array([[1, 0, 0], [0, -_r, -_r], [0, _r, -_r]]),
}


def _real(value, name):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or not np.isfinite(array).all():
        raise InputError(f"{name} must contain finite real numbers")
    return np.asarray(array, dtype=np.float64)


def _matrix(value, name="reciprocal"):
    matrix = _real(value, name)
    if matrix.shape[-2:] != (3, 3):
        raise InputError(f"{name} must have shape (..., 3, 3)")
    sign, logdet = np.linalg.slogdet(matrix)
    if np.any(sign <= 0) or not np.isfinite(logdet).all():
        raise InputError(f"{name} must be nonsingular and right-handed")
    return matrix


def _cell_parameters(reciprocal):
    direct = 2 * np.pi * np.linalg.inv(reciprocal)
    lengths = np.linalg.norm(direct, axis=0)
    angles = [
        np.degrees(
            np.arccos(
                np.clip(
                    np.dot(direct[:, i], direct[:, j]) / (lengths[i] * lengths[j]),
                    -1,
                    1,
                )
            )
        )
        for i, j in ((1, 2), (0, 2), (0, 1))
    ]
    return np.r_[lengths, angles]


def _canonical(parameters):
    try:
        return lattice_params_to_reciprocal(*parameters)
    except (ValueError, np.linalg.LinAlgError) as error:
        raise InputError(f"cell does not define a valid lattice: {error}") from error


def _reference(reference):
    if not isinstance(reference, (Cell, Crystal)):
        raise TypeError("reference must be a Cell or Crystal")
    cell = reference.cell if isinstance(reference, Crystal) else reference
    cell = cell.in_angstrom
    values = _real(
        [cell.a / 10, cell.b / 10, cell.c / 10, cell.alpha, cell.beta, cell.gamma],
        "reference cell",
    )
    group = reference.space_group if isinstance(reference, Crystal) else None
    try:
        reciprocal = lattice_params_to_reciprocal(*values, space_group=group)
    except ValueError as error:
        raise InputError(f"invalid effective reference cell: {error}") from error
    return _cell_parameters(reciprocal), reciprocal


def _mask(fit):
    if isinstance(fit, str):
        if len(fit) != 6 or set(fit) - {"0", "1"}:
            raise InputError(
                "fit must be six binary digits in a,b,c,alpha,beta,gamma order"
            )
        bits = np.array([c == "1" for c in fit])
    else:
        try:
            names = tuple(fit)
        except TypeError as error:
            raise InputError(
                "fit must be a binary mask or sequence of parameter names"
            ) from error
        if any(not isinstance(n, str) or n not in PARAMETERS for n in names):
            raise InputError(f"fit names must be drawn from {PARAMETERS}")
        if len(set(names)) != len(names):
            raise InputError("fit parameter names must be unique")
        bits = np.array([n in names for n in PARAMETERS])
    if not bits.any() or bits[:3].all():
        raise InputError(
            "fit requires a free cell parameter and at least one held length"
        )
    return bits, "".join("1" if bit else "0" for bit in bits)


def _coordinates(value):
    if value not in COORDINATES:
        raise InputError(f"coordinates must be one of {COORDINATES}")
    return value


def _orientation(reciprocal):
    canonical = _canonical(_cell_parameters(reciprocal))
    rotation = reciprocal.T @ np.linalg.inv(canonical.T)
    left, _, right = np.linalg.svd(rotation)
    return left @ right


def _normalize(reciprocal, reference):
    exponent = (np.linalg.slogdet(reference)[1] - np.linalg.slogdet(reciprocal)[1]) / 3
    return reciprocal * np.exp(exponent)[..., None, None]


def transform_strain(tensor, rotation):
    """Transform tensor components with ``R @ tensor @ R.T``.

    Parameters
    ----------
    tensor : numpy.ndarray
        Symmetric dimensionless tensor(s), shape ``(..., 3, 3)``, with tensor
        shear, without engineering-shear doubling.
    rotation : numpy.ndarray
        Proper orthonormal coordinate transform(s), shape ``(..., 3, 3)``.
        Leading dimensions broadcast against those of ``tensor``.

    Returns
    -------
    numpy.ndarray
        Independent float64 array of transformed components.

    Raises
    ------
    lauelab.indexing.InputError
        Invalid tensor/rotation, nonfinite entries or incompatible shapes.
    """
    tensor = _real(tensor, "tensor")
    rotation = _real(rotation, "rotation")
    if tensor.shape[-2:] != (3, 3) or rotation.shape[-2:] != (3, 3):
        raise InputError("tensor and rotation must have shape (..., 3, 3)")
    if not np.allclose(tensor, tensor.swapaxes(-1, -2), rtol=0, atol=1e-12):
        raise InputError("tensor must be symmetric")
    if not np.allclose(
        rotation @ rotation.swapaxes(-1, -2), np.eye(3), rtol=0, atol=1e-10
    ) or np.any(np.linalg.det(rotation) <= 0):
        raise InputError("rotation must be proper and orthonormal")
    try:
        return rotation @ tensor @ rotation.swapaxes(-1, -2)
    except ValueError as error:
        raise InputError(
            "tensor and rotation leading dimensions do not broadcast"
        ) from error


def _strain(reciprocal, reference, coordinates):
    normalized = _normalize(reciprocal, reference)
    deformation = np.linalg.inv(normalized) @ reference
    _, stretches, right = np.linalg.svd(deformation)
    epsilon = (right.T * stretches) @ right - np.eye(3)
    epsilon -= np.trace(epsilon) * np.eye(3) / 3
    if coordinates != "crystal":
        rotation = _FRAMES[coordinates] @ _orientation(normalized)
        epsilon = rotation @ epsilon @ rotation.T
    return (epsilon + epsilon.T) / 2


def strain_from_reciprocal(reciprocal, *, reference, coordinates="crystal"):
    """Calculate Igor deviatoric stretch strain from a refined lattice.

    Parameters
    ----------
    reciprocal : numpy.ndarray
        Reciprocal rows in 1/nm, including 2π, shape ``(..., 3, 3)``.
    reference : lauelab.indexing.Cell or lauelab.indexing.Crystal
        Explicit reference in the native c-parallel-z basis. Cell metrics
        are used as supplied; Crystal uses native space-group constraints.
    coordinates : str
        ``"crystal"`` (default), ``"beamline"``, ``"xhf"`` or ``"sample"``.
        Beamline conversion uses the fitted canonical-cell orientation;
        sample/XHF use the fixed Igor presets.

    Returns
    -------
    numpy.ndarray
        Dimensionless float64 tensor(s) with the input shape. Normalize to
        reference cell volume, then calculate ``dev(sqrt(F.T @ F) - I)``.
        The square root is a symmetric matrix square root.

    Notes
    -----
    The beamline transform uses Igor's orientation convention, which can
    differ from the polar rotation of F at finite shear. This function
    performs no measurement fit or uncertainty estimate. Invalid lattices
    or frame names raise InputError; an unsupported reference raises TypeError.
    """
    reciprocal = _matrix(reciprocal)
    _, ref = _reference(reference)
    _coordinates(coordinates)
    flat = reciprocal.reshape((-1, 3, 3))
    return np.array([_strain(b, ref, coordinates) for b in flat]).reshape(
        reciprocal.shape
    )
