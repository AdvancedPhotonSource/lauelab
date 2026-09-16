#!/usr/bin/env python3
# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Wall time and peak Python memory of streaming versus one-call XML output.

Indexes the four synthetic frames once, then writes them ``--repeat`` times as
one AllSteps document with ``XmlResultsWriter`` (one step in memory at a time)
and with ``write_combined_xml`` (every Step object built first). Reports wall
time, output size, and the tracemalloc peak of each pass. No pass/fail
threshold; record the numbers with the host description.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
from time import perf_counter
import tracemalloc

from lauelab.indexing import Indexer, PeakParams, XmlResultsWriter
from lauelab.indexing.xml_utils import write_combined_xml

ROOT = Path(__file__).resolve().parents[2]
FRAMES = sorted((ROOT / "tests/data/synthetic/frames").glob("*.h5"))
PEAKS = PeakParams(boxsize=18, max_rfactor=0.5, min_size=3, min_separation=20,
                   threshold=None, threshold_ratio=4.0, max_peaks=200)


def _measure(label, action, path):
    tracemalloc.start()
    started = perf_counter()
    action()
    elapsed = perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    size = path.stat().st_size / 2**20
    print(f"{label:<28s} wall {elapsed:7.2f} s   output {size:8.1f} MiB   peak Python memory {peak / 2**20:8.1f} MiB")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeat", type=int, default=250)
    parser.add_argument("--skip-combined", action="store_true")
    args = parser.parse_args(argv)

    indexer = Indexer(ROOT / "tests/data/geo/geoN_2022-03-29_14-15-05.xml",
                      ROOT / "tests/config/Ni.xml", peak_params=PEAKS)
    results = indexer.index_many(FRAMES)
    count = len(results) * args.repeat
    print(f"steps: {count} ({len(results)} results x {args.repeat})")
    with tempfile.TemporaryDirectory() as directory:
        streamed = Path(directory) / "streamed.xml"
        combined = Path(directory) / "combined.xml"

        def stream():
            with XmlResultsWriter(streamed) as writer:
                for _ in range(args.repeat):
                    for result in results:
                        writer.append(result)

        _measure("XmlResultsWriter", stream, streamed)
        if not args.skip_combined:
            def one_call():
                steps = [result.to_step() for _ in range(args.repeat) for result in results]
                write_combined_xml(steps, str(combined))

            _measure("write_combined_xml", one_call, combined)
            print(f"identical output: {streamed.read_bytes() == combined.read_bytes()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
