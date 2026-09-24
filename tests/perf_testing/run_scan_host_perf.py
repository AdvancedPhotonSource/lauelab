#!/usr/bin/env python3
# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Compare worker and thread splits for reconstructing several full-size points.

For each worker count, ``reconstruct_scan`` runs the points in that many
worker processes, which split ``--threads`` between them, and writes a scan
directory. ``reconstruct_points`` runs the same split into per-depth files as a
comparison using independently written per-depth output.

The input is the synthetic full-size point from ``scan_perf_input.py``, linked
once per point under distinct names. The script has no pass or fail threshold.
Host load and the page cache affect every number; record the host, the
filesystem, and whether the threads are physical cores with the table.
"""

from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

import scan_perf_input as perf_input
from lauelab.reconstruct import reconstruct_points, reconstruct_scan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("workdir", type=Path, help="directory on the filesystem to measure")
    parser.add_argument("--points", type=int, default=4)
    parser.add_argument("--threads", type=int, default=20, help="threads available in total")
    parser.add_argument("--workers", type=int, nargs="*", default=[1, 2, 4])
    args = parser.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)

    point = perf_input.write_full_size_point(args.workdir / "point.h5")
    paths = []
    for index in range(args.points):
        link = args.workdir / f"point_{index}.h5"
        if not link.exists():
            link.symlink_to(point.name)
        paths.append(link)
    options = dict(geometry=perf_input.geometry_file(), detector=0,
                   depth_range=perf_input.DEPTH_RANGE_UM, resolution=perf_input.RESOLUTION_UM)

    print(f"{'strategy':<34s} {'wall s':>8s} {'s/point':>8s}")
    for workers in args.workers:
        threads = max(1, args.threads // workers)
        output = args.workdir / "host_scan"
        started = time.perf_counter()
        result = reconstruct_scan(paths, output, workers=workers, threads_per_worker=threads,
                                  **options)
        wall = time.perf_counter() - started
        assert result.complete
        label = f"scan directory, {workers} workers x{threads}"
        print(f"{label:<34s} {wall:8.1f} {wall / args.points:8.1f}")
        shutil.rmtree(output)

        output = args.workdir / "host_per_depth"
        started = time.perf_counter()
        results = reconstruct_points(paths, output, workers=workers,
                                     threads_per_worker=threads, **options)
        wall = time.perf_counter() - started
        assert all(item.success for item in results)
        label = f"per-depth files, {workers} workers x{threads}"
        print(f"{label:<34s} {wall:8.1f} {wall / args.points:8.1f}")
        shutil.rmtree(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
