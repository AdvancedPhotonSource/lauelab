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

PAGE = Path(__file__).resolve().parents[1] / "docs/development/reconstruction-scan-format.md"
MISSING = {"": None, "empty": "", "NaN": "nan", "-1": -1}


def _cells(line):
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _section_rows(heading):
    """Rows of the first table under ``heading`` whose first cell is code."""
    text = PAGE.read_text()
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
    return spec.dtype.str.lstrip("|")


def _documented_shape(shape):
    if len(shape) == 1:
        return f"({shape[0]},)"
    return "(" + ", ".join(str(value) for value in shape) + ")"


@pytest.mark.parametrize(("heading", "table"), [
    ("### Settings", layout.SETTINGS_DATASETS),
    ("### Scan run", layout.SCAN_RUN_DATASETS),
    ("### Catalog", layout.CATALOG_DATASETS),
    ("### Point file", layout.POINT_DATASETS),
])
def test_documented_tables_match_the_layout(heading, table):
    rows = _section_rows(heading)
    assert [row[0].strip("`") for row in rows] == list(table)
    for path, dtype, shape, units, missing, _ in rows:
        spec = table[path.strip("`")]
        assert dtype.strip("`") == _documented_dtype(spec), path
        assert shape.strip("`") == _documented_shape(spec.shape), path
        assert (units or None) == spec.units, path
        assert MISSING[missing] == spec.attrs.get("missing"), path


def test_documented_status_codes_match_the_layout():
    rows = [(int(code), name.strip("`")) for code, name, _ in
            _section_rows("## Status and completeness")]
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
    assert np.dtype("u1") == layout.CATALOG_DATASETS["/catalog/status"].dtype


def test_both_files_share_one_settings_table():
    tables = (layout.SCAN_RUN_DATASETS, layout.CATALOG_DATASETS, layout.POINT_DATASETS)
    for table in tables:
        assert not set(table) & set(layout.SETTINGS_DATASETS)
    assert not set(layout.SCAN_RUN_DATASETS) & set(layout.POINT_DATASETS)


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
    total = layout.POINT_DATASETS["/entry1/reconstruction/stored_depth_intensity/data"]
    for stored in PIXEL_DTYPES.values():
        assert layout.resolve_dtype(data, stored=stored) == stored
        expected = np.dtype("<i8") if stored.kind in "iu" else np.dtype("<f8")
        assert layout.resolve_dtype(total, stored=stored) == expected
    raw = layout.POINT_DATASETS["/entry1/reconstruction/sum_raw/data"]
    assert layout.resolve_dtype(raw, input=np.uint16) == np.dtype("<i8")
    assert layout.resolve_dtype(raw, input=np.uint32) == np.dtype("<i8")
    assert layout.resolve_dtype(raw, input=np.uint64) == np.dtype("<f8")
    assert layout.resolve_dtype(raw, input=np.float32) == np.dtype("<f8")
    with pytest.raises(ValueError, match="unknown dtype rule"):
        layout.resolve_dtype(layout.DatasetSpec("other"))


def test_value_kinds_separate_stored_and_computed_totals():
    assert layout.POINT_DATASETS["/entry1/reconstruction/stored_depth_intensity/data"].attrs["values"] == "stored"
    assert layout.POINT_DATASETS["/entry1/reconstruction/computed_depth_intensity/data"].attrs["values"] == "computed"
    assert layout.POINT_DATASETS["/entry1/reconstruction/sum_raw/data"].attrs["values"] == "raw"
    assert layout.POINT_DATASETS["/entry1/depth"].units == "um"


def test_layout_is_immutable():
    with pytest.raises(TypeError):
        layout.POINT_DATASETS["new"] = layout.POINT_DATASETS["/entry1/depth"]
    with pytest.raises(TypeError):
        layout.POINT_DATASETS["/entry1/data/data"].attrs["values"] = "computed"
