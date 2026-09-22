# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Common point operations on existing per-depth reconstruction output."""

import h5py
import numpy as np
import pytest

from lauelab.indexing import InputError
from lauelab.reconstruct import PerDepthReader
from lauelab.reconstruct.inspection import ArrayPoint, depth_trace, reference_image


@pytest.fixture
def frames(tmp_path):
    values = np.arange(5 * 3 * 4, dtype=np.int32).reshape(5, 3, 4) - 20
    paths = []
    for index, frame in enumerate(values):
        path = tmp_path / f"depth_{index}.h5"
        with h5py.File(path, "w") as handle:
            handle.create_dataset("entry1/data/data", data=frame)
            handle.create_dataset("entry1/depth", data=[index - 2.0])
            handle.create_dataset("entry1/detector/ID", data=[b"detector"])
            handle.create_dataset("entry1/detector/Nx", data=[16])
            handle.create_dataset("entry1/detector/Ny", data=[12])
            for key, value in {"startx": 2, "starty": 3, "binx": 2, "biny": 2}.items():
                handle.create_dataset(f"entry1/detector/{key}", data=[value])
            handle.create_dataset("entry1/microDiffraction/norm_rescale", data=[32768.])
            handle.create_dataset("entry1/microDiffraction/norm_threshold", data=[5.])
        paths.append(path)
    return paths, values


def test_per_depth_reader_matches_array_contract_and_closes(frames):
    paths, values = frames
    array = ArrayPoint(values, np.arange(5) - 2.)
    blocks = list(array.iter_blocks(96))
    assert all(block.flags.owndata and block.nbytes <= 96 for _, block in blocks)
    np.testing.assert_array_equal(np.concatenate([block for _, block in blocks]), values)
    with PerDepthReader(reversed(paths), point_id="p") as point:
        assert point.paths == tuple(paths)
        assert point.point_id == "p"
        assert point.shape == values.shape and point.dtype == values.dtype
        assert point.detector_size == (16, 12)
        assert point.start == (2, 3) and point.group == (2, 2)
        assert point.detector_id == "detector"
        assert point.norm_rescale == 32768. and point.norm_threshold == 5.
        for index in range(5):
            frame = point.frame(index)
            assert frame.flags.owndata
            np.testing.assert_array_equal(frame, values[index])
        np.testing.assert_array_equal(point.region((1, 3, 2, 4), slice(1, 4)), values[1:4, 1:3, 2:4])
        np.testing.assert_array_equal(depth_trace(point).values, depth_trace(array).values)
        for bounds in [(0, 3, 0, 4), (1, 3, 2, 4)]:
            np.testing.assert_array_equal(depth_trace(point, bounds, max_bytes=96).values,
                                          depth_trace(array, bounds).values)
        np.testing.assert_array_equal(reference_image(point).image, values.sum(axis=0))
        blocks = list(point.iter_blocks(96))
        assert [first for first, _ in blocks] == [0, 2, 4]
        assert all(block.nbytes <= 96 for _, block in blocks)
        np.testing.assert_array_equal(np.concatenate([block for _, block in blocks]), values)
        for name in ("first_raw", "sum_raw"):
            with pytest.raises(InputError, match="not available"):
                reference_image(point, name)
    point.close()
    with pytest.raises(InputError, match="closed"):
        point.frame(0)


def test_open_reads_metadata_only_and_region_reads_only_its_selection(frames, monkeypatch):
    paths, values = frames
    original = h5py.Dataset.__getitem__

    def no_pixels(dataset, key):
        assert dataset.name != "/entry1/data/data", "opening the point read pixels"
        return original(dataset, key)

    monkeypatch.setattr(h5py.Dataset, "__getitem__", no_pixels)
    point = PerDepthReader(paths)
    read_direct = h5py.Dataset.read_direct
    calls = []

    def selected(dataset, destination, source_sel=None, dest_sel=None):
        calls.append((dataset.file.filename, source_sel, destination.nbytes))
        return read_direct(dataset, destination, source_sel, dest_sel)

    monkeypatch.setattr(h5py.Dataset, "read_direct", selected)
    np.testing.assert_array_equal(point.region((1, 2, 2, 4), slice(2, 4)), values[2:4, 1:2, 2:4])
    assert [path for path, _, _ in calls] == [str(paths[2]), str(paths[3])]
    assert all(selection == (slice(1, 2), slice(2, 4)) for _, selection, _ in calls)
    assert all(size == 16 for _, _, size in calls)
    point.close()


@pytest.mark.parametrize(("field", "value", "message"), [
    ("entry1/depth", -2., "unique depths"),
    ("entry1/depth", np.nan, "depth must be finite"),
    ("entry1/detector/binx", 4, "metadata disagrees"),
    ("entry1/microDiffraction/norm_rescale", 1., "metadata disagrees"),
])
def test_per_depth_reader_rejects_inconsistent_frames(frames, field, value, message):
    paths, _ = frames
    with h5py.File(paths[1], "r+") as handle:
        handle[field][...] = value
    with pytest.raises(InputError, match=message):
        PerDepthReader(paths)


def test_per_depth_reader_rejects_missing_depth_and_invalid_selection(frames):
    paths, _ = frames
    with PerDepthReader(paths) as point:
        with pytest.raises(IndexError):
            point.frame(-1)
        with pytest.raises(InputError, match="outside"):
            point.region((0, 4, 0, 4))
        with pytest.raises(InputError, match="step of 1"):
            point.region((0, 3, 0, 4), slice(None, None, 2))
        with pytest.raises(InputError, match="one frame"):
            next(point.iter_blocks(1))
    with h5py.File(paths[0], "r+") as handle:
        del handle["entry1/depth"]
    with pytest.raises(InputError, match="invalid per-depth frame"):
        PerDepthReader(paths)
    with pytest.raises(InputError, match="at least one"):
        PerDepthReader([])
