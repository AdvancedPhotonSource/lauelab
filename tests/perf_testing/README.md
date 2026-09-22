# Performance Benchmark Runner (Standalone)

This directory contains a minimal, standalone performance test runner for the reconstruction code. It is completely separate from pytest and will not be collected by default test runs.

`run_simulation_perf.py` measures the reflection simulator separately. It reports cold import and first-call time, warm-call median, peak resident memory, scientific inputs, spot count, and the private candidate limit.

Run the observational simulation benchmark from the repository root:

```console
$ python tests/perf_testing/run_simulation_perf.py --case ni --warm-runs 5
```

Use `--output report.json` to retain the machine-readable report. The benchmark has no pass or fail threshold because host load and cold filesystem state affect its measurements.

`run_reconstruct_perf.py` is a hand-run performance measurement, not an
automated test gate, for the production reconstruction executable and the
in-process driver. It runs Twin2 point 1 at 1, 8, 16, and 32 threads and reports
wall time, mean compute and I/O time per stripe, and the native-to-executable
kernel ratio. The results are observational because host load and filesystem
state affect the measurements. Point `LAUELAB_TWIN2_FIXTURE` at a directory
holding `Twin2_wire_1.h5` and `geoN_2023-04-06_03-07-11_cor6.xml`, then run:

```console
$ python tests/perf_testing/run_reconstruct_perf.py
```

`--fixture` overrides the environment variable. The script exits successfully
with a skip message when neither is set or the files are missing. Outputs use
temporary storage by default; pass `--output-dir` to retain them.

## Reconstruction scan file

`run_scan_storage_perf.py` and `run_scan_host_perf.py` measure the single-file
output of `reconstruct_scan`. Both build their input with `scan_perf_input.py`:
a synthetic point with detector-scale dimensions (401 stored frames of 2048 by 2048
`uint16`, 3.1 GiB) reconstructed to 401 depths. Both report measurements without pass/fail thresholds. Pass a work directory on the filesystem you want to measure; each
needs about 20 GiB there.

```console
$ python tests/perf_testing/run_scan_storage_perf.py /path/to/workdir --threads 20
$ python tests/perf_testing/run_scan_host_perf.py /path/to/workdir --points 4 --threads 20
```

The storage script times stripe writes and cold reads of one frame, a 16 by 16
ROI through depth, and a 64 by 64 ROI through depth for each chunk shape, with
and without gzip. The host script compares sequential points that use every
thread against point workers that split the threads.

### Recorded measurements

Recorded on 2026-09-21 on a 2-socket, 20-core Xeon E5-2680 v2 host with
125 GiB RAM, on local XFS, with h5py 3.14 and HDF5 1.14. These results informed
`DATA_CHUNKS = (4, 128, 128)`, uncompressed output by default, and sequential
point processing. The synthetic frames compress unusually well. The measured
Si-wire results below provide a detector-data comparison; GPFS testing
remains outstanding.

Storage, `uint16` output, seconds except size:

| Chunks | Filter | Write per stripe | GiB | Frame | ROI 16 | ROI 64 |
| --- | --- | --- | --- | --- | --- | --- |
| (1, 128, 128) | none | 1.45 | 3.14 | 0.030 | 0.131 | 0.445 |
| (1, 256, 256) | none | 1.16 | 3.13 | 0.027 | 0.255 | 1.084 |
| (1, 256, 2048) | none | 1.09 | 3.13 | 0.024 | 0.790 | 1.723 |
| (4, 128, 128) | none | 1.18 | 3.16 | 0.077 | 0.068 | 0.279 |
| (8, 64, 64) | none | 1.28 | 3.19 | 0.153 | 0.024 | 0.100 |
| (16, 128, 128) | none | 1.17 | 3.25 | 0.270 | 0.040 | 0.167 |
| (32, 64, 64) | none | 1.18 | 3.25 | 0.549 | 0.014 | 0.055 |
| (1, 128, 128) | gzip 1 | 1.78 | 0.16 | 0.023 | 0.146 | 0.449 |
| (1, 256, 2048) | gzip 1 | 1.79 | 0.16 | 0.017 | 0.621 | 1.216 |
| (16, 128, 128) | gzip 1 | 1.87 | 0.15 | 0.204 | 0.048 | 0.184 |

Conversion to the stored dtype costs 1.2 s and the reductions 0.3 s for each
stripe of 256 rows.

One point at 20 threads, by stage: reading 1.5 s, native compute 12.7 s, and
14.7 s wall with no output. Compute time is 23.0 s at 10 threads and 41.1 s at
5, so one point uses the node efficiently. Converting a stripe to its stored
dtype and reducing it in NumPy took 1.8 s per stripe, longer than the 1.6 s of
compute available to overlap with it, so the first scan writer took 22.2 s per point.
The native pass `laue_recon_store_stripe` does the same work in 0.12 s at
20 threads, and `reconstruct_scan` now takes 17.6 s per point end to end,
with HDF5 writes of about 0.5 s per stripe overlapping computation.
`compression="gzip"` raises that to 24.1 s and shrinks this file from
3.23 GiB to 0.22 GiB. Peak resident memory was 3.1 GiB in both cases.

Four points, wall seconds per point:

| Strategy | Per point |
| --- | --- |
| `reconstruct_scan`, sequential, 20 threads, NumPy conversion | 23.1 |
| `reconstruct_scan`, sequential, 20 threads, native conversion | 17.0 |
| `reconstruct_points`, 1 worker of 20 threads | 19.2 |
| `reconstruct_points`, 2 workers of 10 threads | 15.3 |
| `reconstruct_points`, 4 workers of 5 threads | 12.9 |
| `reconstructN_cpu`, 1 process of 20 threads | 31.8 |
| `reconstructN_cpu`, 2 processes of 10 threads | 17.7 |
| `reconstructN_cpu`, 4 processes of 5 threads | 14.6 |
| 4 concurrent `reconstruct_scan` writers into separate files, 5 threads each | 13.3 |

Point workers improve throughput by overlapping computation and serial
output work across processes. Sharing one output file would require
serializing writes and transferring each stripe, about 0.8 GiB, to the
writer process. Moving the
conversion and reductions into the native library recovered most of that
gain inside one process, so no point-worker pool was built for the scan file.
Merging four separately written point groups into one file costs a further
2.9 s per point, so the concurrent-writer route would not beat the sequential
native path by enough to justify it.

### Measured-data acceptance

Recorded on 2026-09-21 on the same host, from a one-off script that is not
tracked and calls only the public API. The point is `Si-wire_8.h5` from the
portal test workspace (`.../Run1/data/scan_1/`, 881 stored frames of 2048 by
2048 `uint16`, 7.39 GB) with `geoN_2023-04-06_03-07-11_cor6.xml`, detector 0,
and the settings the portal's test database records for that scan: leading
edge, depths -50 µm to 150 µm at 1 µm (201 depths), 100% of pixels. The file
was staged on local XFS first (163 s at 63 MiB/s from the NFS export), so
these timings cover reconstruction and output on local XFS. Performance on
the NFS export and GPFS remains unmeasured.

| Step | Result |
| --- | --- |
| `reconstruct_scan`, 1 point, 20 threads, `uint16` output | 28.7 s wall, 1.79 GB written, peak RSS 4.1 GiB, file validates complete |
| `reconstructN_cpu`, same settings, 1 process of 20 threads | 50.4 s wall, 201 per-depth files |
| Stored frames against the executable's files | 201 of 201 bit-identical, depths equal |
| `computed/depth_intensity` against the executable's summary | agrees to the summary's 6 significant digits; both peak at index 193 |
| Cold read of one frame; 16 by 16 and 64 by 64 ROI through 201 depths | 0.09 s; 0.04 s; 0.15 s |
| Depth inspection | full-frame trace peaks at 142 µm; three auto-placed 5 by 5 ROIs peak at 132, 142, and 142 µm; all normalize; each has 7 to 18 nonpositive samples omitted on a log axis |
| Indexing three `ScanFrame` depths, results reopened | each detector view's image is the exact stored plane with its depth; 0 to 11 peaks found, no orientation with default parameters and an unreviewed Si crystal file |
| `export_per_depth` against the executable's files | 201 files identical in objects, attributes, dtypes, and values; 6.8 s |

The stored-value peak (index 192) differs from the computed-value peak
(index 193) on this point: `uint16` output stores 0 for every negative
pixel, as the format page documents. GPFS behaviour and multi-point runs on
measured data remain unmeasured.


### Stripe-budget correction

Measured on the same local XFS host and Si-wire point on 2026-09-21, at
20 threads and 201 output depths. Each before/after case ran in a fresh
process, with the input warmed before timing. Timing includes reconstruction,
writing and publication. Each configuration was measured once; repeated runs
would be needed to estimate a speed difference. Peak RSS includes the
interpreter, native scratch and HDF5 buffers.

| Stripe budget | Wall before / after (s) | Peak RSS before / after (MiB) |
| --- | --- | --- |
| 8192 MiB (default) | 27.0 / 26.6 | 4179 / 3104 |
| 1024 MiB | 28.9 / 24.2 | 1380 / 1077 |

The correction accounts for converted stripes and reduction buffers when
choosing stripe rows and releases completed output arrays promptly. The
stripe budget does not limit total process RSS; frame-sized references,
normalization maps, native per-thread scratch and HDF5 buffers are additional.
No slowdown was observed in these runs. Reproduce the workload with
`reconstruct_scan([point_path], output_path, geometry=geometry_path, detector=0,
depth_range=(-50, 150), num_threads=20, memory_limit_mb=budget)` using the
measured-data inputs above; use a fresh process for each RSS measurement.

Key features:
- Requires two inputs: an HDF5 file and a staging directory (to test different filesystems).
- Pre-copies the input HDF5 into N replicas (one per parallel runner) once before all tests.
- Clears only the output folder between tests (input replicas remain).
- Runs CPU and/or GPU benchmarks over a grid:
  - CPU: Cartesian product of parallel reconstructions vs. number of threads per process.
  - GPU: Number of concurrent GPU reconstructions.
- Saves runtimes and test parameters to a JSONL file for later analysis.
- Uses the installed `lauelab` python API directly; no build steps required.

## Script

- `run_perf.py`: the CLI entry point.

## Usage

Basic example (CPU):
```
python tests/perf_testing/run_perf.py \
  --h5 /path/to/input.h5 \
  --staging-dir /path/to/staging \
  --geometry tests/config/geoN_2023-04-06_03-07-11.xml \
  --depth-range 0:300 \
  --mode cpu \
  --cpu-parallel 1,2,4 \
  --cpu-threads 1,2 \
  --runs 2
```

GPU example:
```
python tests/perf_testing/run_perf.py \
  --h5 /path/to/input.h5 \
  --staging-dir /path/to/staging \
  --geometry tests/config/geoN_2023-04-06_03-07-11.xml \
  --depth-range 0:300 \
  --mode gpu \
  --gpu-parallel 1,2,4 \
  --runs 3
```

Both CPU and GPU:
```
python tests/perf_testing/run_perf.py \
  --h5 /path/to/input.h5 \
  --staging-dir /path/to/staging \
  --geometry tests/config/geoN_2023-04-06_03-07-11.xml \
  --depth-range 0:300 \
  --mode both \
  --cpu-parallel 1,2,4 \
  --cpu-threads 1,2 \
  --gpu-parallel 1,2 \
  --runs 1
```

Dry-run (print planned configuration and exit):
```
python tests/perf_testing/run_perf.py --h5 ... --staging-dir ... --geometry ... --depth-range 0:300 --dry-run
```

## Arguments (minimal set)

Required:
- `--h5`: Path to the input HDF5 file to benchmark.
- `--staging-dir`: Staging directory for input replicas and outputs (allows testing IO on different filesystems).
- `--geometry`: Path to geometry XML file.
- `--depth-range`: Depth range as `start:end` (floats), e.g., `0:300`.

Optional (sane defaults):
- `--mode`: `cpu|gpu|both` (default: `both`)
- `--cpu-parallel`: comma-separated ints (default: `1,2`)
- `--cpu-threads`: comma-separated ints (default: `1,2`)
- `--gpu-parallel`: comma-separated ints (default: `1,2`)
- `--runs`: repetitions per grid point (default: `1`)
- `--resolution`: depth resolution microns (default: `1.0`)
- `--verbose`: verbosity 0–3 (default: `1`)
- `--percent-brightest`: percent brightest pixels (default: `100.0`)
- `--results`: output results JSONL path (default: `<staging>/results.jsonl`)
- `--tag`: freeform label for metadata
- `--dry-run`: print configuration only

## Outputs

- Input replicas: `<staging>/input/<basename>_0.h5`, `<basename>_1.h5`, …
- Per-run outputs: `<staging>/output/run_<timestamp>/proc_<i>/out*` (created for each process).
- Results file: `<staging>/results.jsonl` (or custom via `--results`)

Results JSONL includes:
- One `params` record with the test parameters and environment info.
- One `run` record per grid point repetition with:
  - mode, parallel, threads (CPU only), run_index
  - total_wall_s
  - per_proc timings with success/return_code
  - run_id and input_replicas

Example JSONL entries:
```
{"type":"params", "... test configuration and environment metadata ..."}
{"type":"run","mode":"cpu","parallel":4,"threads":2,"run_index":0,"total_wall_s":12.345,"per_proc":[{"i":0,"elapsed_s":...},...],"run_id":"20250812T150102Z","input_replicas":4,"success":true}
```

## Notes

- The script uses the installed `lauelab.reconstruct` Python API (`reconstruct` and `reconstruct_gpu`). Ensure the environment can resolve these.
- GPU tests will simply launch multiple concurrent GPU reconstructions; if you need per-GPU pinning on multi-GPU systems, this script can be extended to set `CUDA_VISIBLE_DEVICES` per process.
- Long runs are expected; use `--dry-run` first to verify the plan.
