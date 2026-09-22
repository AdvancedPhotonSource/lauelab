# lauelab

lauelab provides Python bindings to native Laue diffraction indexing and
wire-scan reconstruction, plus simulation, orientation analysis, and
visualization. It runs the analysis
behind the 34-ID-E Laue Portal, and its direct users are beamline scientists
working in Jupyter. Most of what they do with it we did not anticipate — so
the package must be predictable, honest about units and conventions, and
pleasant to explore interactively.

## What we can never compromise on

### 1. Conventions are settled physics, not style

Reciprocal lattices are rows a\*, b\*, c\* in 1/nm with the 2π factor.
The orientation reference basis is the native C convention (c ∥ z, including
its space-group angle forcing). Rodrigues vectors are axis·tan(θ/2).
These were decided deliberately against real alternatives. Never "fix" a
convention to match your intuition or another package — if you believe one is
wrong, stop and make the case to a human. A silently changed convention
produces plausible, wrong science.

### 2. The native code and LaueGo agree

The in-process indexer (`liblaue.so`) and the `lauego` subprocess pipeline are
cross-checks on each other, and the golden baselines pin both. The subprocess
pipeline is called **lauego**, never "legacy" — it and `reconstruct` are
supported public API. Reconstruction has the same pair: the in-process
`Reconstructor` (kernel in `liblaue.so`) and the `reconstructN_cpu` executable
behind `reconstruct()` cross-check each other, and the goldens under
`tests/data/reconstruction/` pin both to bit-identical images. Behavioral
divergence between the two paths is a bug or a documented decision; there is
no third state.

### 3. Verification is evidence, not vibes

A fix is done when its failure case runs and passes, not when the code looks
right. Never loosen a test tolerance to make a test pass — a tolerance change
needs a written justification. Never hand-edit golden baselines (see below).
Report failures verbatim; a gate you satisfied without meeting the requirement
behind it will be found, and it costs a full review round.

### 4. Simple beats impressive

This package is an API for scientists. Prefer deleting code
to abstracting it. Prefer one obvious path to two configurable ones. If a
change grows the public surface, say so explicitly and justify it.

## A small glossary

- **frame** — one detector image (2-D uint16 array or 34-ID-E HDF5 file).
- **point** — one measurement location. Not just a raster position: a point is
  a small scan in its own right (e.g. a wire sweep) that produces a stack of
  frames, which reconstruction consumes.
- **scan** — a sequence of points, most often rastered across the sample
  surface, though the stepped variable can also be energy or another factor.
- **indexing pipeline** — peak search → pixel-to-q → orientation indexing
  (native: one `laue_index` call; lauego: `peaksearch`/`pix2qs`/`euler` CLIs).
- **reconstruction** — resolving a wire-scan point into one frame per sample
  depth (native: `Reconstructor` over `laue_recon_stripe`; subprocess:
  `reconstructN_cpu` via `reconstruct()`). Depth is in µm along the incident
  beam relative to the Si origin of the geometry file.
- **pattern** — one candidate crystal orientation found for a frame.
- **lauego / LaueGo** — LaueGo is the original Igor Pro analysis suite for
  34-ID-E (https://github.com/34ide/lauego); lowercase `lauego` is this
  package's subprocess pipeline wrapping the LaueGo-derived
  `peaksearch`/`pix2qs`/`euler` CLIs.
- **baselines / goldens** — recorded lauego CLI outputs for the synthetic
  frames under `tests/data/synthetic/`, used to cross-check the in-process
  indexer; and recorded `reconstructN_cpu` outputs for the synthetic wire scan
  under `tests/data/reconstruction/` (one default plus seven named variants),
  used to cross-check the in-process reconstructor.
- **the portal** — the Laue Portal web app
  (https://github.com/advancedphotonsource/laue-portal), the main downstream
  consumer of this package.

## Common Mistakes

1. **Hand-editing baselines.** The files under `tests/data/synthetic/` are
   program output. Regenerate them only by running
   `tests/data/synthetic/generate.py` (deterministic; it also rewrites
   `provenance.json`, which is the cross-check). The same holds for
   `tests/data/reconstruction/`: regenerate a variant only with
   `generate_reference.py --variant <name>`, as a reviewed scientific decision
   recorded in the commit. An edited baseline is worse than a failing test: it
   records a claim the code never made.
2. **Testing against a stale native build.** After touching any `.c`/`.h` or
   CMake file, reinstall (`pip install -e . --no-build-isolation`) before
   trusting a single test result. The suite imports the *installed* package.
3. **Editing vendored code.** `src/lauelab/analysis/_vendor/` is a read-only
   snapshot. Wrap it, don't patch it.
4. **Committing working notes.** Review findings, fix plans, and agent
   scratch files never enter git. Keep them outside the worktree or expect
   them to be scrubbed from history later, which is expensive.
5. **Touching beamline data.** Paths under `/net/` are read-only user data.
   Local copies for hand-run tests live wherever `LAUELAB_TWIN2_FIXTURE`
   points; the git-ignored `sandbox/` is the usual place. Never write under
   `/net/`, never commit local data, and never name `sandbox/` in tracked
   files.

## Build and test

- Conda env: `envs/laue-analysis` (from `environment.yml`). Use its
  interpreter explicitly: `envs/laue-analysis/bin/python`.
- Rebuild after native changes: `pip install -e . --no-build-isolation`.
- Gates for any nontrivial change, all must pass:
  - `python -m pytest -q` — full suite; the only acceptable skips are the
    GPU reconstruction executable and the local Twin2 wire-scan fixture
    (set `LAUELAB_TWIN2_FIXTURE`, run those tests by hand with
    `-m "integration and slow"`, and record the result in the PR).
  - `python -m sphinx -E -W --keep-going -b html docs <builddir>` — docs
    build warning-free, and doc snippets are expected to actually run.
  - `git diff --check`
- Memory safety is release-gated: `tests/test_liblaue_safety.py` runs the
  native harnesses (`tests/native/peaksearch_memory.c` and
  `tests/native/recon_memory.c`) under Valgrind with all error kinds fatal and
  a checked-in libgomp suppression file. New native code paths need coverage
  there.

## Where code lives

- `src/lauelab/_native.py` — the one CFFI loader and cdef for `liblaue.so`,
  shared by `indexing` and `reconstruct`.
- `src/lauelab/indexing` — `Indexer`, `Geometry`, lauego wrappers; native C
  under `indexing/src/` (peaksearch, pixels2qs, euler, liblaue). The
  reconstruction kernel is `indexing/src/liblaue/liblaue_recon.c`.
- `src/lauelab/analysis` — orientation math, simulation (vendored JZT
  backend), coloring.
- `src/lauelab/visualization` — Plotly figures, tables, LaueGo XML loading.
- `src/lauelab/reconstruct` — `Reconstructor` (stripe pipeline over the native
  kernel), `reconstruct_points` process pool, `reconstruct_scan` with its
  single-file writer, `ScanReader`, and validator (layout in
  `_scan_layout.py`), HDF5 reader/writer conventions,
  and the supported CPU/GPU subprocess wrappers; the executables' C tree is
  `reconstruct/source/`.
- `tests/` — suite plus golden data; `tests/native/` has the C memory
  harness.
- `docs/` — Sphinx/MyST; quickstart and guides are runnable against
  `tests/data/`.

## Docs

Docs are a first-class deliverable of this project: the audience is a
scientist deciding whether to trust a number. Match the narrative voice in
`contributing/writing-style.md`. State units, conventions, shapes, and
ownership. Never describe lauego as legacy. If a change alters behavior a
user can observe, the docs change in the same commit.

## A note from the maintainer

Reviews of this codebase repeatedly found the same two failure modes: fixes
that satisfied the stated check but not the defect behind it, and complexity
added where a deletion was available. Push in the opposite direction of both.
When a rule here conflicts with the task in front of you, say so loudly and
get a human decision instead of quietly picking a side.
