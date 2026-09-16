# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Laue diffraction indexing, analysis, visualization, and reconstruction."""

from importlib import import_module
from types import ModuleType

from ._publish import partial_path, publish_file

__all__ = [
    "analysis", "indexing", "is_results_file", "partial_path", "publish_file",
    "reconstruct", "visualization",
]


def is_results_file(path) -> bool:
    """Return whether *path* carries the lauelab indexing-results format marker.

    This reads only the root ``format`` attribute. It tells a results file apart
    from other HDF5 files such as detector frames; it does not check that the
    file is complete or consistent. Use
    :func:`lauelab.indexing.validate_results_file` for that.
    """
    import h5py

    from ._results_layout import FORMAT

    try:
        with h5py.File(path, "r") as source:
            value = source.attrs.get("format")
    except (OSError, TypeError, ValueError):
        return False
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value == FORMAT


def __getattr__(name: str) -> ModuleType:
    if name not in {"analysis", "indexing", "reconstruct", "visualization"}:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{name}", __name__)
    globals()[name] = module
    return module


def __dir__() -> list[str]:
    return sorted(__all__)
