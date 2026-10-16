# Copyright 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Validate and concatenate closed indexing outputs without reconstructing frames."""

from __future__ import annotations

import os
import tempfile
from dataclasses import replace
from pathlib import Path
from xml.parsers import expat

import h5py
import numpy as np

from lauelab._hdf5 import UTF8, write_root_attributes
from lauelab._publish import partial_path, publish_file
from lauelab._results_layout import DATASETS, FORMAT

from .errors import InvalidResultsFile
from .results import ResultsFileSummary, validate_results_file

_COPY_ROWS = 65536
_COPY_BYTES = 1024 * 1024
_GROUPS = ("frames", "peaks", "patterns", "assignments")
_OFFSETS = {
    "/frames/peak_offsets": "peaks",
    "/frames/pattern_offsets": "patterns",
    "/patterns/assignment_offsets": "assignments",
}


def _paths(paths, destination, overwrite):
    if isinstance(paths, (str, bytes, os.PathLike)):
        raise TypeError("paths must be an iterable of file paths")
    paths = tuple(Path(path) for path in paths)
    if not paths:
        raise ValueError("paths must contain at least one file")
    destination = Path(destination)
    resolved = [path.resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("input paths repeat")
    for path, real in zip(paths, resolved):
        if real == destination.resolve() or (
            destination.exists()
            and path.exists()
            and os.path.samefile(path, destination)
        ):
            raise ValueError("destination must not be an input file")
    if not overwrite and destination.exists():
        raise FileExistsError(destination)
    return paths, destination


def _temporary(destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=destination.parent, prefix=partial_path(destination).name + "-"
    )
    os.close(descriptor)
    return Path(name)


def _equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return (
        np.array_equal(a, b, equal_nan=True)
        if a.dtype.kind in "fc" and b.dtype.kind in "fc"
        else np.array_equal(a, b)
    )


def _same_attributes(a, b):
    if not _attributes_equal(a.attrs, b.attrs):
        raise InvalidResultsFile(
            f"{b.file.filename}: incompatible attributes at {b.name}"
        )


def _attributes_equal(a, b):
    return set(a) == set(b) and all(_equal(a[k], b[k]) for k in a)


def _same_context(a, b):
    """Compare all static settings and their units, including absent crystal context."""
    _same_attributes(a, b)
    if set(a) != set(b):
        raise InvalidResultsFile(
            f"{b.file.filename}: incompatible metadata at {b.name}"
        )
    for name in a:
        first, other = a[name], b[name]
        if isinstance(first, h5py.Group) and isinstance(other, h5py.Group):
            _same_context(first, other)
        elif isinstance(first, h5py.Dataset) and isinstance(other, h5py.Dataset):
            _same_attributes(first, other)
            if (
                first.shape != other.shape
                or first.dtype != other.dtype
                or not _equal(first[()], other[()])
            ):
                raise InvalidResultsFile(
                    f"{b.file.filename}: incompatible metadata at {other.name}"
                )
        else:
            raise InvalidResultsFile(
                f"{b.file.filename}: incompatible metadata at {other.name}"
            )


def _columns(source):
    if set(source) - {*_GROUPS, "run", "geometry", "crystal"}:
        raise InvalidResultsFile(f"{source.filename}: unsupported result groups")
    names = {f"/{group}/{name}" for group in _GROUPS for name in source[group]}
    unknown = names - DATASETS.keys()
    if unknown:
        raise InvalidResultsFile(
            f"{source.filename}: unsupported result columns: {sorted(unknown)}"
        )
    return names


def merge_results_files(
    paths, destination, *, frame_ids=None, overwrite=False
) -> ResultsFileSummary:
    """Concatenate compatible HDF5 indexing files and validate before publication.

    Parameters
    ----------
    paths
        Nonempty iterable of closed results files. Files must remain unchanged
        during the call. File order, followed by row order within each file,
        determines the output frame order. Empty files are allowed.
    destination
        New results file. It must differ from every input, even with overwrite.
    frame_ids
        Optional expected identities in the exact concatenated order. This
        checks selection and ordering; it does not reorder frames.
    overwrite
        Replace an existing destination only after a successful merge.

    Returns
    -------
    ResultsFileSummary
        Validated counts, identities, and provenance of the destination.

    Raises
    ------
    InvalidResultsFile
        An input or output is invalid; identities repeat across inputs or mix
        strings and integers; expected identities disagree; or run, geometry,
        crystal, column attributes, or supported columns are incompatible.
    ValueError, TypeError
        Input paths are empty, repeated, or alias the destination, or ``paths``
        is a single path instead of an iterable.
    FileExistsError
        The destination exists and ``overwrite`` is false.
    OSError
        A file cannot be read, written, or published.

    Notes
    -----
    Scientific arrays are copied in blocks of at most 65,536 rows. Only the
    three ragged-array offsets are rebased; assignment peak indices remain
    frame-local. No detector images are read and no indexing is repeated.
    Frame identities and validation bookkeeping require memory proportional
    to the frame and pattern counts. The output owns its data, with no links
    to the input files. Optional scan-source columns absent from older files
    receive their documented empty-string and -1 values.

    Run, geometry, and crystal metadata must match exactly, including paths
    and units. Root attributes are copied from the first file, except that
    creation time and lauelab version describe this merge and ``source`` is
    removed. ``merged_sources`` records the input paths in order. A unique
    temporary file is removed on failure; an existing destination survives.
    """
    paths, destination = _paths(paths, destination, overwrite)
    summaries = [validate_results_file(path) for path in paths]
    identities = tuple(
        identity for summary in summaries for identity in summary.frame_ids
    )
    if len(set(identities)) != len(identities):
        raise InvalidResultsFile("frame identities repeat across input files")
    if len({type(identity) for identity in identities}) > 1:
        raise InvalidResultsFile("frame identities must not mix strings and integers")
    if frame_ids is not None and identities != tuple(frame_ids):
        raise InvalidResultsFile(
            "concatenated frame identities disagree with frame_ids"
        )
    counts = {
        group: sum(getattr(s, "n_" + group) for s in summaries) for group in _GROUPS
    }
    columns = {}
    with h5py.File(paths[0], "r") as first:
        for path in paths:
            with h5py.File(path, "r") as source:
                for name in ("run", "geometry", "crystal"):
                    if (name in first) != (name in source):
                        raise InvalidResultsFile(
                            f"{path}: incompatible {name} metadata"
                        )
                    if name in first:
                        _same_context(first[name], source[name])
                for name in _columns(source):
                    attrs = dict(source[name].attrs)
                    if name in columns and not _attributes_equal(columns[name], attrs):
                        raise InvalidResultsFile(
                            f"{path}: incompatible attributes at {name}"
                        )
                    columns.setdefault(name, attrs)
        partial = _temporary(destination)
        try:
            with h5py.File(partial, "w") as output:
                for name, value in first.attrs.items():
                    if name not in {"source", "merged_sources"}:
                        output.attrs[name] = value
                write_root_attributes(
                    output, format_name=FORMAT, version=summaries[0].version
                )
                output.attrs["merged_sources"] = np.asarray(
                    [str(p) for p in paths], dtype=UTF8
                )
                for name in ("run", "geometry", "crystal"):
                    if name in first:
                        first.copy(name, output, expand_external=True, expand_soft=True)
                for group in _GROUPS:
                    output.create_group(group)
                for name in sorted(columns):
                    spec = DATASETS[name]
                    dtype = UTF8 if spec.dtype.kind == "O" else spec.dtype
                    if (
                        name == "/frames/frame_ids"
                        and identities
                        and isinstance(identities[0], str)
                    ):
                        dtype = UTF8
                    total = counts[name.split("/")[1]] + (name in _OFFSETS)
                    target = output.create_dataset(
                        name,
                        shape=(total,) + spec.shape,
                        maxshape=(None,) + spec.shape,
                        chunks=(spec.chunk_rows,) + spec.shape,
                        dtype=dtype,
                        fillvalue=-1
                        if name == "/frames/source_depth_indices"
                        else None,
                    )
                    target.attrs.update(columns[name])
                positions = dict.fromkeys(_GROUPS, 0)
                for path, summary in zip(paths, summaries):
                    with h5py.File(path, "r") as source:
                        for name in _columns(source):
                            incoming, target = source[name], output[name]
                            skip = int(
                                name in _OFFSETS and positions[name.split("/")[1]] > 0
                            )
                            base = positions[_OFFSETS[name]] if name in _OFFSETS else 0
                            start = positions[name.split("/")[1]]
                            for begin in range(skip, len(incoming), _COPY_ROWS):
                                end = min(begin + _COPY_ROWS, len(incoming))
                                values = incoming[begin:end]
                                if name in _OFFSETS:
                                    values = values + base
                                target[start + begin : start + end] = values
                        for group in _GROUPS:
                            positions[group] += getattr(summary, "n_" + group)
            summary = validate_results_file(partial, frame_ids=identities)
            publish_file(partial, destination, overwrite=overwrite)
            return replace(summary, path=destination)
        finally:
            partial.unlink(missing_ok=True)


def _xml_body(source):
    """Find validated body byte offsets with an XML parser and no element tree."""
    parser = expat.ParserCreate()
    depth = count = 0
    first = last = None

    def fail(message):
        raise ValueError(f"{source.name}: {message}")

    def start(name, attrs):
        nonlocal depth, first
        if depth == 0:
            if name != "AllSteps" or attrs:
                fail("expected an AllSteps root without attributes")
            # Expat calls this only after the opening tag has been read.
            first = parser.CurrentByteIndex + parser.GetInputContext().index(b">") + 1
        elif depth == 1 and name != "step":
            fail(f"unexpected AllSteps child {name!r}")
        depth += 1

    def end(name):
        nonlocal depth, count, last
        depth -= 1
        if depth == 0:
            last = parser.CurrentByteIndex
        elif depth == 1:
            count += 1

    def declaration(version, encoding, standalone):
        if version != "1.0" or (
            encoding
            and encoding.upper().replace("-", "") not in {"UTF8", "USASCII", "ASCII"}
        ):
            fail("merging requires XML 1.0 with UTF-8 or ASCII encoding")

    parser.StartElementHandler, parser.EndElementHandler = start, end
    parser.XmlDeclHandler = declaration
    parser.StartDoctypeDeclHandler = lambda *args: fail(
        "DOCTYPE declarations are not supported"
    )
    parser.CharacterDataHandler = lambda text: (
        fail("text outside a step") if depth == 1 and text.strip() else None
    )
    try:
        while block := source.read(_COPY_BYTES):
            parser.Parse(block, False)
        parser.Parse(b"", True)
    except expat.ExpatError as error:
        fail(str(error))
    finally:
        # These callbacks capture parser; break the cycle to free each file's
        # parsing buffers immediately, including when cyclic GC is disabled.
        parser.StartElementHandler = None
        parser.EndElementHandler = None
    return first, last, count


def merge_xml_files(paths, destination, *, n_steps=None, overwrite=False) -> int:
    """Concatenate LaueGo AllSteps documents, preserving step content as bytes.

    Parameters
    ----------
    paths
        Nonempty iterable of closed UTF-8 or ASCII XML 1.0 files. Each must
        have an ``AllSteps`` root without attributes and only ``step`` element
        children. Files must remain unchanged during the call. Empty documents
        are allowed; file order and step order are preserved.
    destination
        Output XML document, distinct from all inputs.
    n_steps
        Optional expected total step count, a nonnegative integer.
    overwrite
        Replace an existing destination only after successful merging.

    Returns
    -------
    int
        Number of steps written.

    Raises
    ------
    ValueError, TypeError
        Invalid paths or step count; malformed or unsupported XML; or the
        total differs from ``n_steps``. DTDs are refused.
    FileExistsError
        The destination exists and ``overwrite`` is false.
    OSError
        A file cannot be read, written, or published.

    Notes
    -----
    Validation checks document structure and count, not scientific consistency
    or duplicate frame identities; use the HDF5 merge for those checks. Each
    source is parsed once, then its body is copied in 1 MiB blocks. No element
    tree or result objects are built. Numeric text, step attributes, and source
    paths are preserved. Memory use is independent of the number of steps.
    Publication uses a unique temporary file and preserves the destination on
    any failure, including an unexpected step count.
    """
    if n_steps is not None and (
        isinstance(n_steps, (bool, np.bool_))
        or not isinstance(n_steps, (int, np.integer))
        or n_steps < 0
    ):
        raise ValueError("n_steps must be a nonnegative integer or None")
    paths, destination = _paths(paths, destination, overwrite)
    partial = _temporary(destination)
    try:
        count = 0
        with open(partial, "wb") as output:
            output.write(b'<?xml version="1.0" encoding="utf-8"?>\n<AllSteps>')
            for path in paths:
                with open(path, "rb") as source:
                    first, last, steps = _xml_body(source)
                    source.seek(first)
                    remaining = last - first
                    while remaining:
                        block = source.read(min(_COPY_BYTES, remaining))
                        if not block:
                            raise OSError(f"{path}: input changed during merge")
                        output.write(block)
                        remaining -= len(block)
                    count += steps
            output.write(b"\n</AllSteps>")
        if n_steps is not None and count != n_steps:
            raise ValueError(f"merged XML has {count} steps, expected {n_steps}")
        publish_file(partial, destination, overwrite=overwrite)
        return count
    finally:
        partial.unlink(missing_ok=True)
