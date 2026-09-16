# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Publish a completed output file by same-directory rename."""

from __future__ import annotations

import errno
import os
from pathlib import Path

_LINK_UNSUPPORTED = {errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EXDEV, errno.EMLINK}


def partial_path(final: str | Path) -> Path:
    """Return the in-progress name for *final*: the same name with ``.partial`` appended.

    Parameters
    ----------
    final
        Destination the finished file will have.

    Returns
    -------
    pathlib.Path
        ``final`` with ``.partial`` added after its full name, in the same
        directory, so that :func:`publish_file` can rename it into place.
    """
    final = Path(final)
    return final.with_name(final.name + ".partial")


def publish_file(partial: str | Path, final: str | Path, *, overwrite: bool = False) -> Path:
    """Move a closed, validated file into its final name in the same directory.

    Parameters
    ----------
    partial
        The finished file, normally from :func:`partial_path`. It must be
        closed; publish after every writer has released it.
    final
        Destination path in the same directory as ``partial``.
    overwrite
        Replace an existing ``final``. The default refuses to clobber: an
        existing destination raises ``FileExistsError`` and ``partial`` is left
        in place. The refusal is atomic with respect to another publisher on
        filesystems that support hard links.

    Returns
    -------
    pathlib.Path
        ``final``.

    Raises
    ------
    FileNotFoundError
        If ``partial`` does not exist.
    ValueError
        If the two paths are not in the same directory.
    FileExistsError
        If ``final`` exists and ``overwrite`` is `False`.
    OSError
        For other filesystem failures.

    Notes
    -----
    The rename is one filesystem operation, so a reader never observes a
    half-written ``final``. This function does not validate content; call the
    format's validator on ``partial`` first.
    """
    partial = Path(partial)
    final = Path(final)
    if partial.parent.resolve() != final.parent.resolve():
        raise ValueError(f"{partial} and {final} must be in the same directory")
    if not partial.is_file():
        raise FileNotFoundError(partial)
    if overwrite:
        os.replace(partial, final)
        return final
    try:
        os.link(partial, final)
    except FileExistsError:
        raise FileExistsError(final) from None
    except OSError as error:
        if error.errno not in _LINK_UNSUPPORTED:
            raise
        # Hard links are unavailable here; fall back to a checked rename.
        if final.exists():
            raise FileExistsError(final) from None
        os.replace(partial, final)
        return final
    os.unlink(partial)
    return final
