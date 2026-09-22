# Simulate detector reflections

Use {func}`~lauelab.analysis.simulate_reflections` when you need a complete predicted pattern for one crystal orientation and one detector. The function returns one strongest representative for each signed harmonic direction that reaches the detector.

Use {meth}`~lauelab.analysis.SimulationResult.missing_from` when you need only directions that an indexed pattern does not represent. Detector-view preparation applies this operation for you. See [Add simulation to a detector view](detector-simulation.md) for that workflow.

## Supply the scientific inputs

Pass a `Crystal` and a `DetectorGeometry` to configure the simulation. The reciprocal matrix normally comes from an indexed {class}`~lauelab.indexing.Pattern`:

```python
from lauelab.analysis import simulate_reflections
from lauelab.indexing import load_crystal, load_geometry

crystal = load_crystal("crystal.xml")
geometry = load_geometry("geometry.xml")

detector_slot = geometry.find_detector("PE1621 723-3335")
if detector_slot < 0:
    raise ValueError("The detector is not present in the geometry file")
detector = geometry.detector(detector_slot)

pattern = frame_result.patterns[0]
simulation = simulate_reflections(
    crystal,
    pattern.reciprocal,
    detector,
    energy_range_kev=(6.0, 30.0),
    depth=0.0,
)
```

The example assumes that `frame_result` contains at least one indexed pattern. `pattern.reciprocal` is the fitted reciprocal matrix. Use this matrix directly because it contains the indexed orientation and any fitted lattice change.

```{warning}
Detector indices are physical geometry slots. Active detector slots can be sparse. Use `find_detector()` when you know the detector ID.
```

The inputs use these conventions:

| Input | Shape or type | Units and convention |
|---|---|---|
| `crystal` | {class}`~lauelab.indexing.Crystal` | Space group, unit cell, atom identity, fractional positions, and occupancy |
| `reciprocal` | `(3, 3)` | Basis vectors in rows, in `1/nm` |
| `detector` | {class}`~lauelab.indexing.DetectorGeometry` | One physical detector slot |
| `energy_range_kev` | two values | Inclusive lower and upper bounds in keV |
| `depth` | scalar | Sample depth in µm |

The reciprocal matrix must be finite and nonsingular. See the [results guide](results.md) for its basis and multiplication convention.

## Read the result

{class}`~lauelab.analysis.SimulationResult` contains aligned NumPy arrays:

| Field | Shape | Dtype | Meaning |
|---|---|---|---|
| `hkl` | `(n, 3)` | `numpy.int64` | Retained Miller-index representative |
| `q` | `(n, 3)` | `numpy.float64` | Reciprocal vector in `1/nm` |
| `detector_xy` | `(n, 2)` | `numpy.float64` | Zero-based, unbinned full-detector `(x, y)` pixels |
| `energy_kev` | `(n,)` | `numpy.float64` | Photon energy in keV |
| `relative_intensity` | `(n,)` | `numpy.float64` | Uncalibrated relative intensity |

Rows at the same array index describe one reflection. Construction copies the arrays and marks them read-only. A valid result with no accepted reflections uses shapes `(0, 3)`, `(0, 3)`, `(0, 2)`, `(0,)`, and `(0,)`.

Use `relative_intensity` to compare reflections within one simulation. These values are uncalibrated and exclude detector response, exposure, incident spectrum, and other experimental corrections.

## Derive missing directions

Pass the indexed HKLs with shape `(m, 3)` to `missing_from()`:

```python
missing = simulation.missing_from(pattern.hkl)

for hkl, xy, energy in zip(
    missing.hkl,
    missing.detector_xy,
    missing.energy_kev,
    strict=True,
):
    print(tuple(hkl), tuple(xy), energy)
```

The comparison uses signed primitive directions. An indexed `(2, 2, 2)` suppresses a simulated `(1, 1, 1)` because both are positive harmonics. It does not suppress `(-1, -1, -1)`. Duplicate indexed rows do not change the result.

The returned object owns new read-only arrays and retains the simulation order.

## Interpret numerical behavior

Energy bounds are inclusive. A reflection at either requested bound remains eligible.

The result contains one row per positive harmonic direction. The function keeps the row with the greatest relative intensity. Equal intensities prefer lower energy, then lexicographically smaller HKL. Final rows use descending relative intensity with the same tie-breakers.

Simulation keeps only finite intersections inside the selected full detector. ROI origin, detector-pixel grouping, image shape, and display-axis inversion do not affect `SimulationResult`.

## Handle errors and limits

Simulation reports input and calculation errors with these exceptions:

| Exception | Meaning |
|---|---|
| `TypeError` | An input object or numeric value has an unsupported type. |
| `ValueError` | An input has an invalid shape, range, or finite-value constraint. |
| `RuntimeError` | The private simulator cannot load, execute, return valid numbers, or finish before its candidate limit. |

Simulation requires a `Crystal` with atoms. A successful calculation with no accepted reflections returns an empty result. Simulator failures and candidate-limit exhaustion raise `RuntimeError`; the function provides neither partial patterns nor a fallback calculation.

## Validation and model limits

The public crystal model accepts International Tables space-group numbers 1 through 230. Tests exercise the code paths for all seven crystal systems. The reviewed numerical fixtures cover Ni, CdTe, and synthetic Si. This coverage does not establish experimental validation for every space group or material.

The private simulator uses atom identity, fractional position, and occupancy. The public {class}`~lauelab.indexing.Atom` model does not represent thermal displacement, valence, Wyckoff metadata, or other extended structure fields.

The JZT-based simulator loads when simulation is first requested. Its inputs and outputs are exposed through the package models documented here.

See the [simulation API reference](../reference/simulation.md) for complete signatures.
