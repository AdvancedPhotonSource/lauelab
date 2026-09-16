#!/usr/bin/env python3
# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Wall time and peak memory of Indexer.iter_index versus serial index_many.

Runs the four synthetic frames repeated ``--repeat`` times, serially and then
through ``iter_index`` for each worker count, and prints wall time, frames per
second, and the peak resident set size of this process and of its children.
There is no pass/fail threshold; record the numbers with the host description.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import resource
from time import perf_counter

from lauelab.indexing import FrameInput, Indexer, IndexParams, PeakParams

ROOT = Path(__file__).resolve().parents[2]
FRAMES = sorted((ROOT / "tests/data/synthetic/frames").glob("*.h5"))
PEAKS = PeakParams(boxsize=18, max_rfactor=0.5, min_size=3, min_separation=20,
                   threshold=None, threshold_ratio=4.0, max_peaks=200)
INDEXING = IndexParams(kev_max_calc=17.2, kev_max_test=35.0, angle_tolerance_deg=0.1)


def _peak_rss_mib(who) -> float:
    return resource.getrusage(who).ru_maxrss / 1024.0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=8)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--max-in-flight", type=int, default=None)
    parser.add_argument("--skip-serial", action="store_true")
    args = parser.parse_args(argv)

    indexer = Indexer(ROOT / "tests/data/geo/geoN_2022-03-29_14-15-05.xml",
                      ROOT / "tests/config/Ni.xml", peak_params=PEAKS, index_params=INDEXING)
    inputs = [FrameInput(path, input_id=(n, path.stem)) for n in range(args.repeat) for path in FRAMES]
    print(f"frames: {len(inputs)} ({len(FRAMES)} synthetic frames x {args.repeat})")

    if not args.skip_serial:
        started = perf_counter()
        results = indexer.index_many(item.frame for item in inputs)
        elapsed = perf_counter() - started
        print(f"serial          wall {elapsed:8.2f} s  {len(results) / elapsed:6.2f} frames/s  "
              f"parent peak RSS {_peak_rss_mib(resource.RUSAGE_SELF):7.1f} MiB")
        del results

    for workers in args.workers:
        started = perf_counter()
        count = 0
        failures = 0
        peak = 0
        with indexer.iter_index(inputs, workers=workers, max_in_flight=args.max_in_flight) as outcomes:
            for outcome in outcomes:
                count += 1
                failures += not outcome.ok
            peak = outcomes.peak_in_flight
        elapsed = perf_counter() - started
        print(f"workers={workers:<3d}    wall {elapsed:8.2f} s  {count / elapsed:6.2f} frames/s  "
              f"parent peak RSS {_peak_rss_mib(resource.RUSAGE_SELF):7.1f} MiB  "
              f"children peak RSS (largest) {_peak_rss_mib(resource.RUSAGE_CHILDREN):7.1f} MiB  "
              f"peak in flight {peak}  failures {failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
