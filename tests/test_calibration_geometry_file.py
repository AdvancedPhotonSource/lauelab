# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Writing calibrated detectors into a copy of a LaueGo geometry file."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from calibration_cases import GEOMETRY_2023
from lauelab.calibration import write_geometry
from lauelab.calibration._model import detector_with
from lauelab.indexing import InputError, load_geometry

WHEN = datetime(2026, 11, 13, 14, 5, 9, 300000, tzinfo=timezone(timedelta(hours=-6)))


def _calibrated(slot=0):
    detector = load_geometry(GEOMETRY_2023).detector(slot)
    return detector_with(
        detector,
        np.round(detector.rotation_vector + [1e-4, -2e-4, 3e-4], 8),
        detector.translation + [123.0, -45.0, 678.0],
    )


def _tags(text):
    return sorted(set(re.findall(r"<(\w+)", text)))


def test_written_file_reads_back_with_the_new_detector(tmp_path):
    refined = _calibrated()
    path = write_geometry(GEOMETRY_2023, tmp_path / "geoN_new.xml", [refined],
                          note="lauelab 0.1.0 Si 23pk 4E rms 0.0123px", file_note="frame Si_1.h5", written=WHEN)
    geometry = load_geometry(path)
    detector = geometry.detector(0)
    np.testing.assert_allclose(detector.rotation_vector, refined.rotation_vector, atol=1e-12)
    np.testing.assert_allclose(detector.translation, refined.translation, atol=1e-6)
    original = load_geometry(GEOMETRY_2023)
    for slot in (1, 2):
        np.testing.assert_array_equal(geometry.detector(slot).translation, original.detector(slot).translation)
    assert geometry.wire.diameter_um == original.wire.diameter_um


def test_only_the_calibrated_detector_and_header_change(tmp_path):
    source = GEOMETRY_2023.read_text()
    text = write_geometry(GEOMETRY_2023, tmp_path / "new.xml", [_calibrated()], note="calibrated",
                          written=WHEN).read_text()
    # No element name is added, so tag-searching readers see the same structure.
    assert _tags(text) == _tags(source)
    before = re.findall(r"<Detector\b.*?</Detector>", source, re.DOTALL)
    after = re.findall(r"<Detector\b.*?</Detector>", text, re.DOTALL)
    assert after[1:] == before[1:]
    for part in ("<Sample>", "<Wire>"):
        assert source[source.index(part):source.index(part.replace("<", "</"))] == \
            text[text.index(part):text.index(part.replace("<", "</"))]
    assert "<note>calibrated</note>" in after[0]
    assert "<timeMeasured>Fri, Nov 13, 2026, 14:05:09 (-6)</timeMeasured>" in after[0]
    assert "<dateWritten>Fri, Nov 13, 2026</dateWritten>" in text
    assert "<timeWritten>14:05:09.3 (-6)</timeWritten>" in text
    epoch = int((datetime(2026, 11, 13, 14, 5, 9) - datetime(1904, 1, 1)).total_seconds())
    assert f'unit="sec">{epoch}</EPOCH>' in text
    changed = [line for line in text.splitlines() if line not in source.splitlines()]
    assert len(changed) == 7  # R, P, timeMeasured, note, dateWritten, timeWritten, EPOCH


def test_file_note_is_added_in_the_header(tmp_path):
    text = write_geometry(GEOMETRY_2023, tmp_path / "new.xml", [_calibrated()], note="n",
                          file_note="calibrated from Si_1.h5", written=WHEN).read_text()
    header = text[:text.index("<Sample>")]
    assert "<fileNote>calibrated from Si_1.h5</fileNote>" in header


def test_side_detector_is_matched_by_id(tmp_path):
    refined = _calibrated(2)
    path = write_geometry(GEOMETRY_2023, tmp_path / "new.xml", [refined], note="d2", written=WHEN)
    geometry = load_geometry(path)
    np.testing.assert_allclose(geometry.detector(2).translation, refined.translation, atol=1e-6)
    np.testing.assert_array_equal(
        geometry.detector(0).translation, load_geometry(GEOMETRY_2023).detector(0).translation
    )


def test_existing_destination_is_refused(tmp_path):
    destination = tmp_path / "new.xml"
    destination.write_text("keep")
    with pytest.raises(FileExistsError):
        write_geometry(GEOMETRY_2023, destination, [_calibrated()], note="n", written=WHEN)
    assert destination.read_text() == "keep"
    assert list(tmp_path.iterdir()) == [destination]


@pytest.mark.parametrize("note", ["x" * 100, "a<b", "R&D", "line\nbreak", "µm"])
def test_notes_that_readers_cannot_store_are_refused(tmp_path, note):
    with pytest.raises(InputError):
        write_geometry(GEOMETRY_2023, tmp_path / "new.xml", [_calibrated()], note=note, written=WHEN)
    assert not any(tmp_path.iterdir())


def test_unknown_detector_is_refused(tmp_path):
    other = _calibrated()
    other = type(other)(other.nx, other.ny, other.size_x, other.size_y, "unknown", other.translation,
                        other.rotation_vector, other.rotation)
    with pytest.raises(InputError, match="exactly one detector"):
        write_geometry(GEOMETRY_2023, tmp_path / "new.xml", [other], note="n", written=WHEN)


def test_comment_with_markup_is_refused(tmp_path):
    """The C programs read tags inside comments; Igor removes comments first."""
    text = GEOMETRY_2023.read_text()
    r_line = re.search(r'<R unit="radian">[^<]+</R>', text).group(0)
    source = tmp_path / "commented.xml"
    source.write_text(text.replace(r_line, "<!-- old " + r_line + " -->\n\t\t\t" + r_line, 1))
    with pytest.raises(InputError, match="comment"):
        write_geometry(source, tmp_path / "new.xml", [_calibrated()], note="n", written=WHEN)
    assert not (tmp_path / "new.xml").exists()
