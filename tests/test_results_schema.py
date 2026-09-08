# Copyright 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Structural regression of the indexing-results file layout.

The listing under ``tests/data/results/`` records every object, dtype,
shape, chunking, and attribute name written for the synthetic frames. It is
the file-level counterpart of ``tests/test_results_layout.py``: a change to
what the writer emits shows up here as a diff, and the committed listing is
the reviewed decision. Regenerate it only with ``--regenerate-results-schema``.
"""

from pathlib import Path

import h5py
import pytest

from conftest import requires_liblaue

from lauelab.indexing import Indexer

ROOT = Path(__file__).resolve().parents[1]
GEOMETRY = ROOT / "tests/data/geo/geoN_2022-03-29_14-15-05.xml"
CRYSTAL = ROOT / "tests/config/Ni.xml"
FRAMES = ROOT / "tests/data/synthetic/frames"
LISTING = ROOT / "tests/data/results/layout_synthetic.txt"


def describe_layout(path) -> str:
    """Return a deterministic structural listing of an HDF5 file.

    Attribute values are omitted because timestamps, versions, and absolute
    paths vary between runs. Dataset sizes are included because the synthetic
    frames are deterministic.
    """
    lines = []

    def describe_attributes(prefix, node):
        for name in sorted(node.attrs):
            lines.append(f"{prefix}  @{name}")

    def visit(name, node):
        if isinstance(node, h5py.Group):
            lines.append(f"group /{name}")
            describe_attributes("", node)
            return
        dtype = "string" if h5py.check_string_dtype(node.dtype) is not None else node.dtype.str
        maxshape = tuple("unlimited" if value is None else value for value in node.maxshape)
        lines.append(
            f"dataset /{name} dtype={dtype} shape={node.shape} "
            f"maxshape={maxshape} chunks={node.chunks}"
        )
        describe_attributes("", node)

    with h5py.File(path, "r") as source:
        lines.append("group /")
        describe_attributes("", source)
        source.visititems(visit)
    return "\n".join(lines) + "\n"


def write_synthetic_results(path) -> None:
    indexer = Indexer(GEOMETRY, CRYSTAL)
    results = indexer.index_many(sorted(FRAMES.glob("*.h5")))
    indexer.write_results(results, path)


@requires_liblaue
def test_results_file_layout_matches_committed_listing(tmp_path, request):
    output = tmp_path / "results.h5"
    write_synthetic_results(output)
    actual = describe_layout(output)
    if request.config.getoption("--regenerate-results-schema"):
        LISTING.write_text(actual)
    assert actual == LISTING.read_text()
