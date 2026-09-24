#!/usr/bin/env python3
# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Measure chunk shapes and compression for point-file pixel data.

For each candidate the script writes one full-size point stripe by stripe, as
the point writer does, and then times the three read patterns of the format:
one full frame, a small ROI through every depth, and a larger ROI through
every depth. File pages are evicted before each read, so reads are cold.

The stripes are real reconstruction output of the synthetic full-size point
from ``scan_perf_input.py``. The script has no pass or fail threshold. Record
its table with the host, the filesystem, and the chosen storage settings when
``DATA_CHUNKS`` or ``COMPRESSION`` in ``_scan_writer.py`` change.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import h5py
import numpy as np

import scan_perf_input as perf_input
from lauelab.reconstruct import Reconstructor
from lauelab.reconstruct._reader import read_scan_info
from lauelab.reconstruct._scan_writer import store_stripe
from lauelab.reconstruct._writer import PIXEL_DTYPES

CANDIDATES = [
    ((1, 128, 128), None),
    ((1, 256, 256), None),
    ((1, 256, 2048), None),
    ((4, 128, 128), None),
    ((8, 64, 64), None),
    ((16, 128, 128), None),
    ((32, 64, 64), None),
    ((1, 128, 128), "gzip"),
    ((1, 256, 2048), "gzip"),
    ((16, 128, 128), "gzip"),
]
ROIS = {"roi16": (1000, 1016, 1000, 1016), "roi64": (1000, 1064, 1000, 1064)}


class _Collect:
    """Stripe sink that keeps converted stripes in memory."""

    output_files: list = []

    def __init__(self, dtype, threads) -> None:
        self.dtype = np.dtype(dtype)
        self.threads = threads
        self.stripes = []
        self.convert_seconds = 0.0

    def begin(self, *, depth_um, shape, **_) -> None:
        self.shape = (len(depth_um), *shape)

    def raw(self, row0, stripe) -> None:
        pass

    def write(self, row0, values) -> None:
        started = time.perf_counter()
        self.stripes.append((row0, store_stripe(values, self.dtype, n_threads=self.threads)[0]))
        self.convert_seconds += time.perf_counter() - started

    def finish(self, **_) -> None:
        pass

    def close(self) -> None:
        pass


def _evict(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def _timed_read(path: Path, selection) -> float:
    _evict(path)
    started = time.perf_counter()
    with h5py.File(path, "r") as source:
        source["data"][selection]
    return time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("workdir", type=Path, help="directory on the filesystem to measure")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--pixel-type", type=int, default=None,
                        help="output pixel type code (default: the input dtype)")
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)

    point = perf_input.write_full_size_point(args.workdir / "point.h5")
    reconstructor = Reconstructor(
        perf_input.geometry_file(), 0, depth_range=perf_input.DEPTH_RANGE_UM,
        resolution=perf_input.RESOLUTION_UM, num_threads=args.threads,
        output_pixel_type=args.pixel_type,
    )
    with h5py.File(point, "r") as source:
        info = read_scan_info(source)
        sink = _Collect(PIXEL_DTYPES[reconstructor._output_type(info.dtype)], reconstructor.num_threads)
        result = reconstructor._run_file(source, info, sink, False)
    assert result.success, result.error
    n_stripes = len(sink.stripes)
    print(f"stored dtype {sink.dtype}, shape {sink.shape}, {n_stripes} stripes, "
          f"convert {sink.convert_seconds / n_stripes:.2f} s per stripe")

    started = time.perf_counter()
    for _, stripe in sink.stripes:
        stripe.sum(axis=(1, 2), dtype=np.int64 if sink.dtype.kind in "iu" else np.float64)
        stripe.sum(axis=0, dtype=np.int64 if sink.dtype.kind in "iu" else np.float64)
    print(f"reductions {(time.perf_counter() - started) / n_stripes:.2f} s per stripe")

    print(f"{'chunks':>16s} {'filter':>6s} {'write s/stripe':>14s} {'GiB':>6s} "
          f"{'frame s':>8s} {'roi16 s':>8s} {'roi64 s':>8s}")
    for chunks, compression in CANDIDATES:
        path = args.workdir / "candidate.h5"
        started = time.perf_counter()
        with h5py.File(path, "w") as target:
            data = target.create_dataset(
                "data", shape=sink.shape, dtype=sink.dtype,
                chunks=tuple(min(a, b) for a, b in zip(chunks, sink.shape)),
                compression=compression, compression_opts=1 if compression else None,
                shuffle=bool(compression),
            )
            for row0, stripe in sink.stripes:
                data[:, row0:row0 + stripe.shape[1], :] = stripe
        _evict(path)
        write = (time.perf_counter() - started) / n_stripes
        size = path.stat().st_size / 2**30
        frame = _timed_read(path, (sink.shape[0] // 2,))
        rois = [
            _timed_read(path, (slice(None), slice(y0, y1), slice(x0, x1)))
            for y0, y1, x0, x1 in ROIS.values()
        ]
        print(f"{str(chunks):>16s} {compression or 'none':>6s} {write:14.2f} {size:6.2f} "
              f"{frame:8.3f} {rois[0]:8.3f} {rois[1]:8.3f}")
        path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
