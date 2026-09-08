# Installation

During alpha development, install `lauelab` from source in the Conda environment supplied by this repository. `python -m pip install .` compiles the native indexing library and command-line programs with CMake, then installs the Python package. The project does not publish prebuilt wheels or Conda packages.

Pip can create a temporary wheel while it installs the source tree. This is part of the build process and does not avoid native compilation. Wheels created with `python -m build` are development artifacts. They require compatible native libraries and are not supported outside the environment used to build them.

## Supported platforms

- Linux x86-64 with GCC.
- Python 3.11, 3.12, or 3.13. The maintained environment uses Python 3.12.
- NumPy 2.0 or later.

macOS, Windows, Linux ARM, and Clang are not supported.

## Native dependencies

The build needs these tools and libraries:

| Dependency | Purpose |
| --- | --- |
| GCC with C99 support | Compiles all native code |
| CMake 3.24 or later and Ninja | Build system (pip installs them from PyPI when they are absent) |
| GSL 2.8 with headers | Indexing and reconstruction |
| HDF5 1.14 with headers and `h5cc` | Peak search and reconstruction |
| zlib with headers | HDF5 compression |
| OpenMP (part of GCC) | CPU reconstruction |
| `patchelf` | Normalizes library search paths in the installed files (pip installs it from PyPI) |

Other minor versions of GSL and HDF5 can work, but the tests run against the versions above.

Running the complete native memory regression test also requires Valgrind and a C compiler available as `cc`. Both are included in `environment.yml`; on Debian or Ubuntu install `valgrind` alongside `build-essential`.

CUDA is not required. See [GPU reconstruction](#gpu-reconstruction).

## Install with conda (recommended)

Use this path at APS and for alpha testing. `environment.yml` provides the compiler and native libraries from conda-forge. Build, install, and run `lauelab` in this environment so the native programs can load the same GSL and HDF5 libraries.

1. Clone the repository:

   ```console
   $ git clone https://github.com/AdvancedPhotonSource/lauelab.git
   $ cd lauelab
   ```

2. Create and activate the environment:

   ```console
   $ conda env create -f environment.yml
   $ conda activate laue-analysis
   ```

3. Install the package:

   ```console
   $ python -m pip install .
   ```

## Using lauelab in Jupyter

The maintained Conda environment includes JupyterLab, IPython kernel support, and the `nbformat` package that Plotly uses for notebook display. Register the environment as a named kernel:

```console
$ conda activate laue-analysis
$ python -m ipykernel install --user --name laue-analysis --display-name "Python (laue-analysis)"
$ jupyter lab
```

Select `Python (laue-analysis)` in JupyterLab. Plotly figures display inline when you call `figure.show()`. The custom Matplotlib example in the [visualization guide](guides/visualization.md) requires `matplotlib`, which the maintained environment also includes.

## Install with a virtual environment

Use this path on a Linux system that already provides the native dependencies, for example through the distribution package manager or HPC environment modules.

1. Install the native dependencies. On Debian or Ubuntu:

   ```console
   $ sudo apt-get install build-essential libgsl-dev libhdf5-dev zlib1g-dev valgrind
   ```

2. Create and activate a virtual environment:

   ```console
   $ python -m venv .venv
   $ source .venv/bin/activate
   ```

3. Install the package:

   ```console
   $ python -m pip install .
   ```

Do not run `pip` with `sudo`.

The installed programs find GSL and HDF5 through the dynamic loader's search path. If environment modules or another non-system prefix provides the libraries, add its library directory to `LD_LIBRARY_PATH` at run time. A virtual environment does not inherit access to libraries from a separate Conda environment. Do not move a wheel built in one environment into another environment unless the target provides compatible native libraries.

## Verify the installation

Run these commands after installation.

1. Check the Python package:

   ```console
   $ python -c "from lauelab.indexing import Indexer, index_frame; print('import ok')"
   import ok
   ```

2. Check that the native indexing library loads:

   ```console
   $ python -c "from lauelab.indexing._liblaue import get_library; get_library(); print('native indexing ok')"
   native indexing ok
   ```

3. Check that the CPU reconstruction program runs:

   ```console
   $ python -c "from lauelab.reconstruct import find_executable; print(find_executable())"
   ```

The second command loads a private module only as an installation check. Do not use `_liblaue` as an application API.

## What the build produces

A default installation builds all of these:

| Component | Location in the installed package | Required |
| --- | --- | --- |
| `liblaue.so` | `lauelab/indexing/bin/` | Yes. The in-process indexing API needs it. |
| `peaksearch`, `pix2qs`, `euler` | `lauelab/indexing/bin/` | Yes. Used by the LaueGo compatibility functions. |
| `reconstructN_cpu` | `lauelab/reconstruct/bin/` | Yes. Used by `reconstruct()`. |

The in-process indexing API is the main path for new code. The LaueGo compatibility functions and their command-line programs remain supported for existing workflows. The build stops with an error when a required dependency is missing or a component fails to compile. It does not install a partial package.

Native code is compiled for the generic x86-64 baseline with SSE2. The same installed files run on any x86-64 Linux machine that has compatible GSL and HDF5 libraries.

(gpu-reconstruction)=

## GPU reconstruction

CPU reconstruction is the supported path. The package does not build or distribute the CUDA reconstruction program, and it does not detect CUDA during installation. `reconstruct_gpu()` can run an externally installed `reconstructN_gpu` executable from `PATH`. It raises `FileNotFoundError` when the executable is unavailable.

The source distribution and repository include the CUDA source and a stand-alone Makefile in `src/lauelab/reconstruct/source/recon_gpu/`. This build is not part of the maintained package installation.

## Troubleshooting

**The build reports that GSL or HDF5 was not found.** Install the development package that provides headers and `gsl-config` or `h5cc`, then run `pip install .` again. In a conda environment, confirm that the environment is active.

**The build reports that `patchelf` is required.** This appears only with `pip install --no-build-isolation`. Install `patchelf` into the environment (`pip install patchelf` or `conda install -c conda-forge patchelf`).

**`liblaue.so` fails to load with "cannot open shared object file".** The dynamic loader cannot find `libgsl` or `libhdf5`. Activate the environment that provides them, or add their directory to `LD_LIBRARY_PATH`.

**`reconstructN_gpu` is not found.** GPU reconstruction is not built by the package. See [GPU reconstruction](#gpu-reconstruction).

## Report a problem

Open an issue in the [lauelab repository](https://github.com/AdvancedPhotonSource/lauelab/issues). Include:

- The `lauelab` version or Git commit
- Operating system, Python version, and whether you used conda or a virtual environment
- The installation command
- The complete compiler, CMake, or linker error

Remove sample names, user names, local paths, and other sensitive experiment metadata before posting logs.
