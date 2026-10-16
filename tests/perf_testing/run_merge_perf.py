#!/usr/bin/env python3
# Copyright 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""CPU-only merge timings from repeated synthetic results, with fresh-process RSS.

Fixture generation and imports are outside the timed region. Inputs are newly
written, so filesystem caches are warm. This measures local file merging, not
MPI communication or shared-filesystem performance. No pass/fail threshold.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
from time import perf_counter
from xml.etree import ElementTree as ET

import h5py
import numpy as np

from lauelab.indexing import (
    Indexer,
    PeakParams,
    XmlResultsWriter,
    merge_results_files,
    merge_xml_files,
    validate_results_file,
)

ROOT = Path(__file__).resolve().parents[2]
OFFSETS = {
    "frames/peak_offsets",
    "frames/pattern_offsets",
    "patterns/assignment_offsets",
}


def _repeat_hdf(base, destination, repeat, first_id):
    """Build larger valid fixtures in bulk; no frame-by-frame writer overhead."""
    with h5py.File(base, "r") as source, h5py.File(destination, "w") as target:
        target.attrs.update(source.attrs)
        for name in ("geometry", "crystal", "run"):
            source.copy(name, target)
        for group in ("frames", "peaks", "patterns", "assignments"):
            target.create_group(group)
            for name, column in source[group].items():
                key = group + "/" + name
                original = column[()]
                if key == "frames/frame_ids":
                    values = np.arange(
                        first_id,
                        first_id + len(original) * repeat,
                        dtype=original.dtype,
                    )
                elif key in OFFSETS:
                    values = np.append(
                        (
                            original[:-1] + np.arange(repeat)[:, None] * original[-1]
                        ).ravel(),
                        repeat * original[-1],
                    )
                else:
                    values = np.tile(original, (repeat,) + (1,) * (original.ndim - 1))
                target.create_dataset(
                    key, data=values, dtype=column.dtype
                ).attrs.update(column.attrs)


def _peak_rss_mib():
    # getrusage's lifetime peak can include the parent's RSS before exec.
    # VmHWM describes this executable's address space instead.
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError("Linux VmHWM is unavailable")


def _measure(plan, method):
    destination = Path(plan["directory"]) / (method + ".out")
    before = resource.getrusage(resource.RUSAGE_SELF)
    import_peak = _peak_rss_mib()
    started = perf_counter()
    if method == "hdf5":
        summary = merge_results_files(plan["hdf5"], destination, overwrite=True)
        count = summary.n_frames
    elif method == "validate-inputs":
        count = sum(validate_results_file(path).n_frames for path in plan["hdf5"])
    elif method == "xml":
        count = merge_xml_files(
            plan["xml"], destination, n_steps=plan["frames"], overwrite=True
        )
    else:
        root = ET.Element("AllSteps")
        for path in plan["xml"]:
            root.extend(ET.parse(path).getroot())
        count = len(root)
        ET.ElementTree(root).write(destination, encoding="utf-8", xml_declaration=True)
    elapsed = perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_SELF)
    assert count == plan["frames"]
    return {
        "method": method,
        "seconds": elapsed,
        "cpu_seconds": after.ru_utime
        + after.ru_stime
        - before.ru_utime
        - before.ru_stime,
        "peak_rss_mib": _peak_rss_mib(),
        "import_peak_rss_mib": import_peak,
        "output_bytes": destination.stat().st_size if destination.exists() else 0,
        "frames": count,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, nargs="+", default=[1024])
    parser.add_argument("--shards", type=int, nargs="+", default=[4])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--workdir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compare-tree", action="store_true")
    parser.add_argument(
        "--measure", choices=["hdf5", "xml", "validate-inputs", "xml-tree"]
    )
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    if args.measure:
        print(json.dumps(_measure(json.loads(args.plan.read_text()), args.measure)))
        return
    if args.runs < 1 or any(n < 1 for n in args.shards):
        parser.error("runs and shards must be positive")
    if any(n < 1 or n % (4 * shards) for n in args.frames for shards in args.shards):
        parser.error(
            "each frame count must be a positive multiple of 4 * each shard count"
        )
    report = {
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "h5py": h5py.__version__,
        "hdf5": h5py.version.hdf5_version,
        "cpu_count": os.cpu_count(),
        "warm_filesystem_cache": True,
        "workdir": str(
            args.workdir.resolve() if args.workdir else Path(tempfile.gettempdir())
        ),
        "cases": [],
    }
    with tempfile.TemporaryDirectory(dir=args.workdir) as temporary:
        directory = Path(temporary)
        indexer = Indexer(
            ROOT / "tests/data/geo/geoN_2022-03-29_14-15-05.xml",
            ROOT / "tests/config/Ni.xml",
            peak_params=PeakParams(
                boxsize=18,
                max_rfactor=0.5,
                min_size=3,
                min_separation=20,
                threshold=None,
                threshold_ratio=4,
                max_peaks=200,
            ),
        )
        results = indexer.index_many(
            sorted((ROOT / "tests/data/synthetic/frames").glob("*.h5"))
        )
        base = directory / "base.h5"
        indexer.write_results(results, base)
        xml = directory / "base.xml"
        with XmlResultsWriter(xml) as writer:
            for result in results:
                writer.append(result)
        body = xml.read_bytes().split(b"<AllSteps>", 1)[1].rsplit(b"</AllSteps>", 1)[0]
        for frames in args.frames:
            for shards in args.shards:
                case_dir = directory / f"{frames}-{shards}"
                case_dir.mkdir()
                plan = {
                    "directory": str(case_dir),
                    "frames": frames,
                    "shards": shards,
                    "hdf5": [],
                    "xml": [],
                }
                repeat = frames // (4 * shards)
                for number in range(shards):
                    hdf = case_dir / f"{number}.h5"
                    _repeat_hdf(base, hdf, repeat, number * frames // shards)
                    xml = case_dir / f"{number}.xml"
                    with open(xml, "wb") as output:
                        output.write(b'<?xml version="1.0"?>\n<AllSteps>')
                        for _ in range(repeat):
                            output.write(body)
                        output.write(b"</AllSteps>")
                    plan["hdf5"].append(str(hdf))
                    plan["xml"].append(str(xml))
                plan_path = case_dir / "plan.json"
                plan_path.write_text(json.dumps(plan))
                case = {
                    "frames": frames,
                    "shards": shards,
                    "samples": [],
                    "hdf5_input_bytes": sum(
                        Path(p).stat().st_size for p in plan["hdf5"]
                    ),
                    "xml_input_bytes": sum(Path(p).stat().st_size for p in plan["xml"]),
                }
                for method in ["validate-inputs", "hdf5", "xml"] + (
                    ["xml-tree"] if args.compare_tree else []
                ):
                    for _ in range(args.runs):
                        process = subprocess.run(
                            [
                                sys.executable,
                                __file__,
                                "--measure",
                                method,
                                "--plan",
                                str(plan_path),
                            ],
                            capture_output=True,
                            text=True,
                            check=True,
                        )
                        sample = json.loads(process.stdout)
                        case["samples"].append(sample)
                    chosen = [
                        sample
                        for sample in case["samples"]
                        if sample["method"] == method
                    ]
                    print(
                        json.dumps(
                            {
                                "frames": frames,
                                "shards": shards,
                                "method": method,
                                "median_seconds": statistics.median(
                                    s["seconds"] for s in chosen
                                ),
                                "max_peak_rss_mib": max(
                                    s["peak_rss_mib"] for s in chosen
                                ),
                            }
                        ),
                        flush=True,
                    )
                report["cases"].append(case)
                if args.output:
                    args.output.write_text(json.dumps(report, indent=2))
    if not args.output:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
