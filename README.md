# lauelab

Python and native tools for Laue diffraction analysis at APS beamline 34-ID-E. The package supports indexing, wire-scan reconstruction, simulation, orientation analysis, and interactive visualization.

Built on [LaueGo](https://github.com/34IDE/LaueGo), developed by Jon Z. Tischler and collaborators. See [acknowledgments and citations](docs/acknowledgments.md).

> [!NOTE]
> `lauelab` is alpha software under active development. Public behavior can change before a stable release.

Read the [documentation](https://advancedphotonsource.github.io/lauelab/) for installation requirements, indexing workflows, concepts, and API details.

## Install

The package builds from source on Linux x86-64. It needs GCC, GSL, HDF5, and Python 3.11, 3.12, or 3.13. The conda environment in `environment.yml` provides all of them.

```bash
git clone https://github.com/AdvancedPhotonSource/lauelab.git
cd lauelab
conda env create -f environment.yml
conda activate laue-analysis
python -m pip install .
```

See the [installation guide](https://advancedphotonsource.github.io/lauelab/installation.html) for the virtual-environment path and troubleshooting.

## Index a frame

```python
from lauelab.indexing import index_frame

result = index_frame(
    "frame.h5",
    geometry="geometry.xml",
    crystal="crystal.xml",
)

print(result.n_peaks, result.n_indexed, result.n_patterns)
```

Use `Indexer` to reuse one geometry and crystal configuration across frames. The subprocess-based `lauego()` function remains available for existing LaueGo workflows.

## Test

```bash
python -m pytest
```
