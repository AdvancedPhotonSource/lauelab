# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Tests for the reconstruction-scan HDF5 layout table."""

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
    ("### Run", layout.RUN_DATASETS),
    ("### Catalog", layout.CATALOG_DATASETS),
    ("### Point", layout.POINT_DATASETS),
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
    assert layout.FORMAT == "lauelab-reconstruction-scan"
    assert layout.VERSION == 1
    assert layout.SUPPORTED_VERSIONS == {1}
    assert layout.POINT_GROUP.format(index=0) == "/points/000000"
    assert f"`{layout.FORMAT}`" in PAGE.read_text()


def test_unwritten_status_reads_as_pending_and_published_states_are_terminal():
    assert layout.PointStatus(0) is layout.PointStatus.PENDING
    assert layout.RunStatus(0) is layout.RunStatus.RUNNING
    assert layout.PointStatus.PENDING not in layout.TERMINAL_POINT_STATUSES
    assert layout.PointStatus.WRITING not in layout.TERMINAL_POINT_STATUSES
    assert layout.PUBLISHABLE_RUN_STATUSES == {
        layout.RunStatus.FINISHED, layout.RunStatus.CANCELLED,
    }
    assert np.dtype("u1") == layout.CATALOG_DATASETS["/catalog/status"].dtype


def test_mutable_datasets_are_fixed_width():
    tables = (layout.RUN_DATASETS, layout.CATALOG_DATASETS, layout.POINT_DATASETS)
    for table in tables:
        for path, spec in table.items():
            if spec.mutable and not isinstance(spec.dtype, str):
                assert not _is_variable_length(spec.dtype), path
    assert layout.CATALOG_DATASETS["/catalog/errors"].dtype == np.dtype("S1024")


def test_dtype_rules_resolve_for_every_pixel_type():
    data = layout.POINT_DATASETS["data"]
    total = layout.POINT_DATASETS["reductions/depth_intensity"]
    for stored in PIXEL_DTYPES.values():
        assert layout.resolve_dtype(data, stored=stored) == stored
        expected = np.dtype("<i8") if stored.kind in "iu" else np.dtype("<f8")
        assert layout.resolve_dtype(total, stored=stored) == expected
    raw = layout.POINT_DATASETS["reference/sum_raw"]
    assert layout.resolve_dtype(raw, input=np.uint16) == np.dtype("<i8")
    assert layout.resolve_dtype(raw, input=np.uint32) == np.dtype("<i8")
    assert layout.resolve_dtype(raw, input=np.uint64) == np.dtype("<f8")
    assert layout.resolve_dtype(raw, input=np.float32) == np.dtype("<f8")
    with pytest.raises(ValueError, match="unknown dtype rule"):
        layout.resolve_dtype(layout.DatasetSpec("other"))


def test_value_kinds_separate_stored_and_computed_totals():
    assert layout.POINT_DATASETS["reductions/depth_intensity"].attrs["values"] == "stored"
    assert layout.POINT_DATASETS["computed/depth_intensity"].attrs["values"] == "computed"
    assert layout.POINT_DATASETS["reference/sum_raw"].attrs["values"] == "raw"
    assert layout.POINT_DATASETS["depth_um"].units == "um"


def test_layout_is_immutable():
    with pytest.raises(TypeError):
        layout.POINT_DATASETS["new"] = layout.POINT_DATASETS["depth_um"]
    with pytest.raises(TypeError):
        layout.POINT_DATASETS["data"].attrs["values"] = "computed"
