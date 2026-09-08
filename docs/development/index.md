# Development

This page records the current local workflow for Python, native, test, and documentation changes.

## Build the project

Install the native dependencies listed in [Installation](../installation.md), then install an editable copy with the test and documentation extras:

```console
$ python -m pip install -e '.[test,docs]'
```

The editable install builds the native files once with CMake and places them in `site-packages`. Python changes under `src/` take effect immediately. After a change to C source or `CMakeLists.txt`, run the install command again.

`CMakeLists.txt` at the repository root is the only native build definition. `liblaue.so` embeds the GPL-licensed GSL using static `libgsl.a` and `libgslcblas.a` from the build environment, with their symbols hidden by `--exclude-libs`; importing the wheel therefore does not load the environment's `libgsl.so` or OpenBLAS. It has these options, set with `--config-settings=cmake.define.<OPTION>=<value>` on the pip command line:

| Option | Default | Effect |
| --- | --- | --- |
| `LAUE_BUILD_LAUEGO` | `ON` | Build `peaksearch`, `pix2qs`, and `euler` |
| `LAUE_BUILD_RECONSTRUCTION` | `ON` | Build `reconstructN_cpu` |
| `LAUE_NATIVE_OPTIMIZATION` | `OFF` | Add `-march=native` |

Reduced builds are for maintainers. They are not supported user configurations.

```{warning}
Files built with `LAUE_NATIVE_OPTIMIZATION=ON` are tuned to the build machine and can fail or give different results on other CPUs. Do not copy them to another machine or into a shared environment. Run the complete test suite after enabling it. Parity has been verified only on a host without FMA instructions.
```

To build without pip, for example to inspect compiler flags:

```console
$ cmake -S . -B build-cmake -G Ninja
$ cmake --build build-cmake
```

CUDA is not part of the package build. See `src/lauelab/reconstruct/source/recon_gpu/README.md`.

## Run tests

Run the complete suite from the repository root:

```console
$ python -m pytest
```

Run a focused module while developing a change:

```console
$ python -m pytest tests/test_indexer.py
```

Tests run against the installed package, never against `src/` directly. Native tests skip when the installed `lauelab` package does not contain `liblaue.so`. A passing run with skipped native tests does not validate the native code. Review the skip summary (`-rs`). Expected skips cover an unavailable GPU reconstruction executable and, outside CI, a missing Valgrind executable or C compiler for the native memory harness.

CI runs with `--require-native`. That option fails the session when `liblaue.so` is missing from the installed package or when any test skips for another reason.

All fixtures are in Git and are synthetic. `tests/data/synthetic/` holds simulated Laue frames and the LaueGo program outputs for them (see its README); the indexing tests compare the in-process indexer against those outputs.

The `integration` marker identifies tests that run compiled command-line programs. Exclude them with `-m "not integration"`.

### CPU reconstruction reference

`tests/data/reconstruction/` holds synthetic numerical references for `reconstructN_cpu`. `tests/test_reconstruct.py` checks the executable, and `tests/test_reconstructor.py` cross-checks the in-process driver against the same files. Do not regenerate a reference to make a build change pass.

The full-size Twin2 fixture is not in the repository. Set `LAUELAB_TWIN2_FIXTURE` to a directory holding `Twin2_wire_1.h5` through `Twin2_wire_3.h5`, `geoN_2023-04-06_03-07-11_cor6.xml`, and the `reference_rec8_point1/` output directory. The real-data tests and the performance script skip with `Twin2 wire-scan fixture not available` when the variable is unset or a file is missing. Run the real-data checks explicitly with:

```console
$ LAUELAB_TWIN2_FIXTURE=/path/to/twin2_wire python -m pytest -m "integration and slow" tests/test_reconstruct_realdata.py
```

Compare executable and in-process performance on Twin2 point 1 with:

```console
$ python tests/perf_testing/run_reconstruct_perf.py
```

The script runs both implementations at 1, 8, 16, and 32 threads and reports wall, compute, and I/O times. It has no pass or fail threshold because host load and filesystem state affect the measurements.

## Build the documentation

Install PyPA `build` (`python -m pip install build`) to produce the sdist and wheel with `python -m build`. Build the documentation with warnings treated as errors:

```console
$ python -m sphinx -E -W --keep-going -b html docs docs/_build/html
```

Open `docs/_build/html/index.html` or serve the directory locally:

```console
$ python -m http.server 8000 --directory docs/_build/html --bind 127.0.0.1
```

## Document public APIs

Use NumPy-style docstrings for public objects. Include a public object explicitly on a curated page under `docs/reference/`. Do not generate references for a whole module because that can expose private helpers and compatibility objects accidentally.

State shapes, dtypes, units, coordinate order, defaults, ownership, exceptions, and observable behavior where relevant. Use cross-references to connect guides and reference entries instead of copying long descriptions.

The documentation writing style is defined in `contributing/writing-style.md`. Apply its scientific notation, coordinate, example, and editing rules to narrative changes.

## Validate examples

Classify examples by what they need:

- Execute pure-Python examples in tests when practical.
- Validate native indexing examples with maintained integration fixtures.
- Keep illustrative snippets short and label synthetic data clearly.
- Do not require native execution during a basic Sphinx HTML build.

A code block that demonstrates exact output must have a test that checks that output. Avoid exact values when the example exists only to show control flow.

## Continuous integration

`.github/workflows/ci.yml` runs on every pull request and push to `main`. The job installs Valgrind and the package from a clean checkout on Ubuntu with Python 3.12, runs the test suite with `--require-native`, builds the documentation with warnings as errors, builds the sdist and wheel, and checks their contents with `.github/scripts/inspect_artifacts.py`. It then replaces the source installation with the wheel in the same Conda environment and runs `.github/scripts/smoke_test.py` from outside the checkout. Finally, it removes `liblaue.so` from the installed package and confirms that the test gate fails.

The workflow does not publish or keep build artifacts.

## Publish the site

`.github/workflows/docs.yml` builds the documentation for every pull request and deploys it to GitHub Pages after a successful push to `main`. The repository settings must select "GitHub Actions" as the Pages source. Do not commit `docs/_build/`.

## HDF5 file conventions

Files whose layout `lauelab` defines follow one set of conventions for root attributes, versioning, units, and ragged data. See [HDF5 file conventions](hdf5-conventions.md).

```{toctree}
:hidden:
:maxdepth: 1

hdf5-conventions
```
