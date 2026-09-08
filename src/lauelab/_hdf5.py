# Copyright 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Shared HDF5 conventions for lauelab-owned file formats."""

from __future__ import annotations

from datetime import datetime, timezone
from importlib.metadata import version as distribution_version
from numbers import Integral
from typing import Iterable

import h5py
import numpy as np

UTF8 = h5py.string_dtype(encoding="utf-8")


def write_root_attributes(
    target: h5py.File,
    *,
    format_name: str,
    version: int,
    created: str | None = None,
    source: str | None = None,
) -> None:
    """Write the common root attributes for a lauelab HDF5 file."""
    if isinstance(version, (bool, np.bool_)) or not isinstance(version, Integral):
        raise TypeError("version must be an integer")
    if created is None:
        created = datetime.now(timezone.utc).isoformat()

    target.attrs["format"] = format_name
    target.attrs["version"] = int(version)
    target.attrs["lauelab_version"] = distribution_version("lauelab")
    target.attrs["created"] = created
    if source is not None:
        target.attrs["source"] = str(source)


def set_units(dataset: h5py.Dataset, units: str | None) -> h5py.Dataset:
    """Attach physical units when a dataset represents a dimensioned value."""
    if units is not None:
        dataset.attrs["units"] = units
    return dataset


def check_format_version(
    source: h5py.File,
    *,
    format_name: str,
    supported_versions: Iterable[int],
) -> int:
    """Validate a lauelab HDF5 format marker and return its integer version."""
    actual_format = source.attrs.get("format")
    if isinstance(actual_format, bytes):
        actual_format = actual_format.decode("utf-8")
    if actual_format != format_name:
        raise ValueError(
            f"not a {format_name!r} file (format is {actual_format!r})"
        )

    version = source.attrs.get("version")
    if isinstance(version, np.generic):
        version = version.item()
    if isinstance(version, (bool, np.bool_)) or not isinstance(version, Integral):
        raise ValueError(f"{format_name!r} version must be an integer")

    version = int(version)
    supported = frozenset(int(value) for value in supported_versions)
    if version not in supported:
        expected = ", ".join(str(value) for value in sorted(supported))
        raise ValueError(
            f"unsupported {format_name!r} version {version}; supported versions: {expected}"
        )
    return version
