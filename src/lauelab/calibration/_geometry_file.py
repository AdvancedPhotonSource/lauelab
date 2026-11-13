# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Writing calibrated detectors into a LaueGo geometry file."""

from __future__ import annotations

import os
import re
import tempfile
import xml.etree.ElementTree as ET
import datetime
from pathlib import Path
from typing import Iterable

import numpy as np

from lauelab.indexing import DetectorGeometry, InputError, load_geometry

# Igor stores a detector note and its measurement time in 100-byte fields.
_MAX_FIELD = 99
_EPOCH_1904 = datetime.datetime(1904, 1, 1)
_DETECTOR_BLOCK = re.compile(r"<Detector\b[^>]*>.*?</Detector>", re.DOTALL)


def _field(name: str, text: str, limit: int | None = _MAX_FIELD) -> str:
    if not isinstance(text, str):
        raise InputError(f"{name} must be a string")
    if not text.isascii() or not text.isprintable() or any(c in text for c in "<>&"):
        raise InputError(f"{name} must be printable ASCII without <, > or &: {text!r}")
    if limit is not None and len(text) > limit:
        raise InputError(f"{name} must be at most {limit} characters; it has {len(text)}")
    return text


def _offset_hours(moment: datetime.datetime) -> str:
    offset = moment.utcoffset()
    hours = 0.0 if offset is None else offset.total_seconds() / 3600.0
    return f"{hours:g}"


def _date(moment: datetime.datetime) -> str:
    return f"{moment:%a, %b} {moment.day}, {moment.year}"


def _replace_element(block: str, tag: str, value: str, *, after: str | None = None) -> str:
    """Replace the text of ``<tag ...>`` in ``block``, or add the element.

    A missing element is added on the line after the element ``after`` or,
    without one, before the closing tag of the block.
    """
    pattern = re.compile(rf"(<{tag}\b[^>]*>)[^<]*(</{tag}>)")
    if pattern.search(block):
        return pattern.sub(lambda m: m.group(1) + value + m.group(2), block, count=1)
    anchor = re.search(rf"</{after}>[^\n]*\n", block) if after else None
    if anchor is None:
        anchor = re.search(r"\n([ \t]*)</\w+>\s*$", block)
        indent = (anchor.group(1) if anchor else "") + "\t"
        position = anchor.start() + 1 if anchor else len(block)
    else:
        line_start = block.rfind("\n", 0, anchor.start()) + 1
        indent = re.match(r"[ \t]*", block[line_start:]).group(0)
        position = anchor.end()
    return block[:position] + f"{indent}<{tag}>{value}</{tag}>\n" + block[position:]


def _check_unit(block: str, tag: str, unit: str, detector_id: str) -> None:
    match = re.search(rf"<{tag}\b([^>]*)>", block)
    if match is None:
        raise InputError(f"detector {detector_id!r} has no <{tag}> element")
    found = re.search(r'unit\s*=\s*"([^"]*)"', match.group(1))
    if found is None or found.group(1) != unit:
        raise InputError(f"<{tag}> of detector {detector_id!r} must have unit=\"{unit}\"")


def _numbers(values: np.ndarray) -> str:
    return " ".join(f"{value + 0.0:.10g}" for value in values)


def write_geometry(
    source: str | Path,
    destination: str | Path,
    detectors: Iterable[DetectorGeometry],
    *,
    note: str,
    file_note: str | None = None,
    written: datetime.datetime | None = None,
) -> Path:
    """Write a copy of a geometry file with calibrated detectors.

    Only these parts of ``source`` change:

    - ``<R>``, ``<P>``, ``<timeMeasured>`` and ``<note>`` of each detector
      whose ``<ID>`` equals the ``detector_id`` of one of ``detectors``;
    - ``<dateWritten>``, ``<timeWritten>`` and ``<EPOCH>``;
    - ``<fileNote>``, if ``file_note`` is given.

    Every other byte, including the other detectors, the wire and the sample,
    is copied. The C programs and Igor find elements by searching the text for
    their tags, so no new element names are added.

    The file is written to a temporary file in the destination folder, read
    back with :func:`~lauelab.indexing.load_geometry` and compared with
    ``detectors``, and then linked to ``destination``.

    Parameters
    ----------
    source
        The geometry XML file that the calibration started from.
    destination
        The new file. It must not exist.
    detectors
        Calibrated detectors, such as the ``refined`` detector of a
        :class:`DetectorCalibration`.
    note
        Text of each calibrated detector's ``<note>``: printable ASCII without
        ``<``, ``>`` or ``&``, at most 99 characters.
    file_note
        Text of the file's ``<fileNote>``, with the same characters allowed.
    written
        Time recorded as the measurement and writing time. The default is the
        current local time.

    Returns
    -------
    pathlib.Path
        ``destination``.

    Raises
    ------
    InputError
        If ``source`` is not a geometry XML file with these detectors and units
        ``radian`` and ``mm``, contains an XML tag inside a comment, or a note
        is not allowed.
    FileExistsError
        If ``destination`` exists.
    """
    source, destination = Path(source), Path(destination)
    detectors = list(detectors)
    if not detectors or not all(isinstance(d, DetectorGeometry) for d in detectors):
        raise InputError("detectors must contain at least one DetectorGeometry")
    note = _field("note", note)
    if file_note is not None:
        file_note = _field("file_note", file_note, limit=None)
    moment = written if written is not None else datetime.datetime.now()
    if moment.tzinfo is None:
        moment = moment.astimezone()
    measured = _field(
        "timeMeasured", f"{_date(moment)}, {moment:%H:%M:%S} ({_offset_hours(moment)})"
    )
    text = source.read_text(encoding="utf-8")
    try:
        ET.fromstring(text.encode("utf-8"))
    except ET.ParseError as error:
        raise InputError(f"{source} is not a geometry XML file: {error}") from error
    # The C programs find tags with a text search that also looks inside
    # comments, while Igor removes comments first. A comment containing a tag
    # can therefore give the two readers different values. Comments such as
    # "must be <= MAX_Ndetectors" contain no tag and are kept.
    for comment in re.findall(r"<!--(.*?)-->", text, re.DOTALL):
        if re.search(r"</?[A-Za-z_]", comment):
            raise InputError(
                f"{source} contains a comment with an XML tag, which the C programs and "
                "Igor read differently; remove it before calibrating"
            )
    if destination.exists():
        raise FileExistsError(destination)

    blocks = list(_DETECTOR_BLOCK.finditer(text))
    ids = [re.search(r"<ID\b[^>]*>([^<]*)</ID>", b.group(0)) for b in blocks]
    ids = [m.group(1).strip() if m else "" for m in ids]
    replacements = {}
    for detector in detectors:
        if ids.count(detector.detector_id) != 1:
            raise InputError(
                f"{source} must contain exactly one detector with ID {detector.detector_id!r}"
            )
        index = ids.index(detector.detector_id)
        block = blocks[index].group(0)
        _check_unit(block, "R", "radian", detector.detector_id)
        _check_unit(block, "P", "mm", detector.detector_id)
        block = _replace_element(block, "R", _numbers(detector.rotation_vector))
        block = _replace_element(block, "P", _numbers(detector.translation / 1000.0))
        block = _replace_element(block, "timeMeasured", measured, after="P")
        block = _replace_element(block, "note", note, after="timeMeasured")
        replacements[index] = block
    pieces, cursor = [], 0
    for index, match in enumerate(blocks):
        if index in replacements:
            pieces += [text[cursor:match.start()], replacements[index]]
            cursor = match.end()
    text = "".join(pieces) + text[cursor:]

    root = re.search(r"<geoN\b[^>]*>\n?", text)
    if root is None:
        raise InputError(f"{source} has no <geoN> element")
    header = text[root.end():]
    epoch = int((moment.replace(tzinfo=None) - _EPOCH_1904).total_seconds())
    for tag, value in (
        ("dateWritten", _date(moment)),
        ("timeWritten", f"{moment:%H:%M:%S}.{moment.microsecond // 100000} ({_offset_hours(moment)})"),
        ("EPOCH", str(epoch)),
    ) + ((("fileNote", file_note),) if file_note is not None else ()):
        header = _replace_element_in_header(header, tag, value)
    text = text[:root.end()] + header

    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        _verify(Path(temporary), source, detectors)
        os.link(temporary, destination)
    finally:
        os.unlink(temporary)
    return destination


def _replace_element_in_header(header: str, tag: str, value: str) -> str:
    """Replace a top-level element, or add it before the first child group."""
    pattern = re.compile(rf"(<{tag}\b[^>]*>)[^<]*(</{tag}>)")
    first_group = re.search(r"<(Sample|Detectors|Wire)\b", header)
    limit = first_group.start() if first_group else len(header)
    match = pattern.search(header, 0, limit)
    if match:
        return header[:match.start()] + match.group(1) + value + match.group(2) + header[match.end():]
    epoch = re.search(r"</EPOCH>[^\n]*\n", header[:limit])
    line_start = epoch.end() if epoch else header.rfind("\n", 0, limit) + 1
    indent = "\t"
    return header[:line_start] + f"{indent}<{tag}>{value}</{tag}>\n" + header[line_start:]


def _verify(path: Path, source: Path, detectors: list[DetectorGeometry]) -> None:
    written, original = load_geometry(path), load_geometry(source)
    if written.detector_count != original.detector_count:
        raise RuntimeError("the written geometry has a different number of detectors")
    changed = {d.detector_id: d for d in detectors}
    for slot in range(8):
        try:
            before = original.detector(slot)
        except ValueError:
            continue
        after = written.detector(slot)
        expected = changed.get(before.detector_id, before)
        if (
            after.detector_id != before.detector_id
            or not np.allclose(after.rotation_vector, expected.rotation_vector, rtol=0, atol=1e-9)
            or not np.allclose(after.translation, expected.translation, rtol=0, atol=1e-6)
        ):
            raise RuntimeError(f"the written geometry does not read back detector {before.detector_id!r}")
