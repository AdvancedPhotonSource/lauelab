# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Full-size synthetic wire-scan point for the scan performance scripts.

The point has the size of a measured one, 2048 by 2048 pixels and several
hundred frames, so that timings include real I/O volume. Its content is the
128 by 128 regression scan regenerated with more wire steps and tiled 16 by
16. It is not a physical simulation, and its noise compresses differently from
measured frames, so compression ratios from it are indicative only.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import h5py
import numpy as np

_GENERATOR = Path(__file__).resolve().parents[1] / "data/reconstruction/generate_reference.py"
DEPTH_RANGE_UM = (-100.0, 100.0)
RESOLUTION_UM = 0.5
N_SCAN_IMAGES = 400
TILES = 16


def _generator():
    spec = importlib.util.spec_from_file_location("generate_reference", _GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def geometry_file() -> Path:
    return _generator().GEOMETRY_FILE


def write_full_size_point(path: Path) -> Path:
    """Write the point to ``path`` unless it already exists."""
    path = Path(path)
    if path.is_file():
        return path
    generator = _generator()
    generator.N_SCAN_IMAGES = N_SCAN_IMAGES
    generator.WIRE_Z_STEP_UM = 162.0 / N_SCAN_IMAGES
    small = path.with_suffix(".small.h5")
    generator.write_input_file(small)
    with h5py.File(small, "r") as source, h5py.File(path, "w") as target:
        for name, value in source.attrs.items():
            target.attrs[name] = value
        for name in source:
            if name != "entry1":
                source.copy(source[name], target, name=name)
        entry = target.create_group("entry1")
        for name in source["entry1"]:
            if name != "data":
                source.copy(source["entry1"][name], entry, name=name)
        frames = np.asarray(source["entry1/data/data"])
        data = entry.create_group("data").create_dataset(
            "data", shape=(len(frames), frames.shape[1] * TILES, frames.shape[2] * TILES),
            dtype=frames.dtype,
        )
        data.attrs["signal"] = np.int32(1)
        for index, frame in enumerate(frames):
            data[index] = np.tile(frame, (TILES, TILES))
        for name in ("binx", "biny"):
            entry["detector"][name][...] = 1
    small.unlink()
    return path
