# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Tests for the reconstruction scan and point-file layout tables."""

from pathlib import Path
import re

import h5py
import numpy as np
import pytest

from lauelab.reconstruct import _scan_layout as layout
from lauelab.reconstruct._writer import PIXEL_DTYPES

DOCS = Path(__file__).resolve().parents[1] / "docs/development"
PAGE = DOCS / "reconstruction-scan-format.md"
PROTOCOL = DOCS / "reconstruction-scan-protocol.md"
FILL = {"": None, "empty": "", "-1": -1, "0": 0}


def _cells(line):
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _section_rows(heading, page=PAGE):
    """Rows of the first table under ``heading`` whose first cell is code."""
    text = page.read_text()
    section = re.split(r"^#{2,3} ", text.split(f"\n{heading}\n", 1)[1], flags=re.M)[0]
    return [
        _cells(line) for line in section.splitlines()
        if line.startswith("| `") or re.match(r"^\| \d", line)
    ]


def _is_variable_length(dtype):
    info = h5py.check_string_dtype(dtype)
    return info is not None and info.length is None


def _documented_dtype(spec):
    if isinstance(spec.dtype, str):
        return spec.dtype
    if _is_variable_length(spec.dtype):
        return "string"
    if h5py.check_enum_dtype(spec.dtype) is not None:
        return "enum " + spec.dtype.str.lstrip("|")
    return spec.dtype.str.lstrip("|")


def _documented_shape(shape):
    if len(shape) == 1:
        return f"({shape[0]},)"
    return "(" + ", ".join(str(value) for value in shape) + ")"


@pytest.mark.parametrize(("heading", "table"), [
    ("### Parameters and geometry", layout.SCAN_DATASETS),
    ("### Points", layout.POINT_COLUMNS),
])
def test_documented_catalog_tables_match_the_layout(heading, table):
    rows = _section_rows(heading)
    assert [row[0].strip("`") for row in rows] == list(table)
    for path, dtype, shape, units, fill, _ in rows:
        spec = table[path.strip("`")]
        assert dtype.strip("`") == _documented_dtype(spec), path
        assert shape.strip("`") == _documented_shape(spec.shape), path
        assert (units or None) == spec.units, path
        if fill == "NaN":
            assert np.isnan(spec.fill), path
        else:
            assert spec.fill == FILL[fill] and type(spec.fill) is type(FILL[fill]), path


def test_documented_root_attributes_match_the_layout():
    rows = _section_rows("### Root attributes")
    assert tuple(row[0].strip("`") for row in rows) == layout.SCAN_ATTRIBUTES
    assert layout.PROGRAM in rows[0][2]


def test_documented_point_file_table_matches_the_layout():
    rows = _section_rows("### Point file")
    assert [row[0].strip("`") for row in rows] == list(layout.POINT_DATASETS)
    for path, dtype, shape, units, when, _ in rows:
        spec = layout.POINT_DATASETS[path.strip("`")]
        assert dtype.strip("`") == _documented_dtype(spec), path
        assert shape.strip("`") == _documented_shape(spec.shape), path
        assert (units or None) == spec.units, path
        assert bool(when) == spec.optional, path


def test_documented_status_codes_match_the_layout():
    rows = [(int(code), name.strip("`")) for code, name, _ in
            _section_rows("## Status and completeness", PROTOCOL)]
    expected = [
        (member.value, member.name.lower())
        for codes in (layout.PointStatus, layout.RunStatus) for member in codes
    ]
    assert rows == expected


def test_format_identity():
    assert layout.SCAN_FORMAT == "lauelab-reconstruction-scan"
    assert layout.SCAN_VERSION == 2
    assert layout.RETIRED_SCAN_VERSIONS == {1}
    assert layout.POINT_FORMAT == "lauelab-reconstruction-point"
    assert layout.POINT_VERSION == 2
    text = PAGE.read_text()
    assert f"`{layout.SCAN_FORMAT}`" in text and f"`{layout.POINT_FORMAT}`" in text


def test_unwritten_status_reads_as_pending_and_settled_runs_are_terminal():
    assert layout.PointStatus(0) is layout.PointStatus.PENDING
    assert layout.RunStatus(0) is layout.RunStatus.RUNNING
    assert layout.PointStatus.PENDING not in layout.TERMINAL_POINT_STATUSES
    assert layout.PointStatus.WRITING not in layout.TERMINAL_POINT_STATUSES
    assert layout.SETTLED_RUN_STATUSES == {
        layout.RunStatus.FINISHED, layout.RunStatus.CANCELLED,
    }
    assert layout.POINT_COLUMNS["/points/statuses"].dtype == layout.POINT_STATUS_TYPE


def test_enumerated_types_name_every_code():
    assert h5py.check_enum_dtype(layout.POINT_STATUS_TYPE) == {
        status.name.lower(): status.value for status in layout.PointStatus}
    assert h5py.check_enum_dtype(layout.RUN_STATUS_TYPE) == {
        status.name.lower(): status.value for status in layout.RunStatus}
    pixel_types = h5py.check_enum_dtype(layout.PIXEL_TYPE)
    assert {code: PIXEL_DTYPES[code] for code in pixel_types.values() if code >= 0} == PIXEL_DTYPES
    assert all(np.dtype(name) == PIXEL_DTYPES[code] for name, code in pixel_types.items() if code >= 0)
    assert pixel_types["none"] == layout.POINT_COLUMNS["/points/dtypes"].fill


def test_parameters_are_named_as_the_reconstructor_arguments():
    from lauelab.reconstruct.scan import SETTING_NAMES
    names = [path.rsplit("/", 1)[1] for path in layout.SCAN_DATASETS if path.startswith("/parameters/")]
    assert names == ["detector", *SETTING_NAMES]


@pytest.mark.parametrize(("stem", "reason"), [
    ("Twin2_wire_1", None),
    ("Si wire 7 (µm)", None),
    ("", "empty"),
    (".hidden", "starts with '.'"),
    ("a\\b", "path separator"),
    ("a/b", "path separator"),
    ("a\tb", "control character"),
    ("a\x7fb", "control character"),
])
def test_point_file_names_come_from_safe_stems(stem, reason):
    found = layout.unsafe_stem(stem)
    assert found is None if reason is None else reason in found
    assert layout.point_path("Twin2_wire_1") == "points/Twin2_wire_1.h5"


def test_dtype_rules_resolve_for_every_pixel_type():
    data = layout.POINT_DATASETS["/entry1/data/data"]
    total = layout.POINT_DATASETS["/entry1/reconstruction/depth_intensity"]
    for stored in PIXEL_DTYPES.values():
        assert layout.resolve_dtype(data, stored=stored) == stored
        expected = np.dtype("<i8") if stored.kind in "iu" else np.dtype("<f8")
        assert layout.resolve_dtype(total, stored=stored) == expected
    raw = layout.POINT_DATASETS["/entry1/reconstruction/sum_raw"]
    assert layout.resolve_dtype(raw, input=np.uint16) == np.dtype("<i8")
    assert layout.resolve_dtype(raw, input=np.uint32) == np.dtype("<i8")
    assert layout.resolve_dtype(raw, input=np.uint64) == np.dtype("<f8")
    assert layout.resolve_dtype(raw, input=np.float32) == np.dtype("<f8")
    with pytest.raises(ValueError, match="unknown dtype rule"):
        layout.resolve_dtype(layout.DatasetSpec("other"))


def test_point_files_use_the_executable_paths_and_units():
    assert layout.POINT_DATASETS["/entry1/depth"].units == "micron"
    assert layout.POINT_DATASETS["/entry1/data/data"].attrs == {"signal": 1}
    for path in ("/entry1/microDiffraction/cosmic_filter", *layout.NORMALIZATION_DATASETS):
        assert layout.POINT_DATASETS[path].optional and layout.POINT_DATASETS[path].shape == (1,)
        assert path in layout.RESERVED_INPUT_PATHS


def test_layout_is_immutable():
    with pytest.raises(TypeError):
        layout.POINT_DATASETS["new"] = layout.POINT_DATASETS["/entry1/depth"]
    with pytest.raises(TypeError):
        layout.POINT_DATASETS["/entry1/data/data"].attrs["signal"] = 2
