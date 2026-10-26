# Refine deviatoric lattice strain

Use `lauelab.strain` to refine orientation and lattice shape from assigned Laue reflections. The reference is an explicit {class}`~lauelab.indexing.Cell` or {class}`~lauelab.indexing.Crystal`. White-beam positions determine shape, with the overall lattice scale fixed by reference-volume normalization. They do not measure hydrostatic expansion.

The module follows the modern LaueGo `DeviatoricStrainRefine` direction objective and stretch-strain definition. Controlled numerical cases are tested. **Empirical agreement with exported Igor results remains pending**; see the [validation procedure](../development/strain-validation.md).

## Run a controlled example

This example creates directions from a known deformation. It uses no detector image or experimental data. Arrays are NumPy-compatible; calculations use float64.

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from scipy.spatial.transform import Rotation

from lauelab.analysis import lattice_params_to_reciprocal
from lauelab.indexing import Cell
from lauelab.strain import (
    StrainRefiner, StrainReflections, refine_deviatoric,
    strain_from_reciprocal, save_strain_results, load_strain_results,
    export_strain_csv,
)

reference = Cell(0.35, 0.35, 0.35, unit="nm")
b0 = lattice_params_to_reciprocal(0.35, 0.35, 0.35, 90, 90, 90)
rotation = Rotation.from_rotvec([0.4, -0.1, 0.2]).as_matrix()
stretch = np.array([
    [1.0010, 0.0004, 0.0001],
    [0.0004, 0.9996, -0.0002],
    [0.0001, -0.0002, 1.0002],
])
deformation = rotation @ stretch
measured_reciprocal = b0 @ np.linalg.inv(deformation)
hkl = np.array([
    [1, 1, 1], [1, 2, 3], [2, 1, -3], [2, 3, 1],
    [4, 1, 2], [1, -2, 1], [-2, 3, 1], [2, 1, 1],
    [-1, 1, 1], [1, 3, -2],
])
qhat = hkl @ measured_reciprocal
qhat /= np.linalg.norm(qhat, axis=1)[:, None]
initial_reciprocal = b0 @ rotation.T

# Functional measurement refinement returns a complete fit result.
result = refine_deviatoric(
    qhat, hkl, reference=reference,
    initial_reciprocal=initial_reciprocal,
)
assert result.success, result.message
assert result.rank == result.n_parameters == 8
expected = stretch / np.linalg.det(stretch)**(1/3) - np.eye(3)
expected -= np.trace(expected) * np.eye(3) / 3
np.testing.assert_allclose(result.tensor_in("crystal"), expected, atol=1e-10)

# The configured API uses the same numerical fit.
measurements = StrainReflections(qhat, hkl, initial_reciprocal)
refiner = StrainRefiner(reference, coordinates="beamline")
configured = refiner.refine(measurements)
np.testing.assert_allclose(configured.tensor, result.tensor, atol=1e-12)

# Pure lattice conversion performs no measurement fit and returns an array.
converted = strain_from_reciprocal(measured_reciprocal, reference=reference)
np.testing.assert_allclose(converted, expected, atol=1e-12)

with TemporaryDirectory() as directory:
    directory = Path(directory)
    save_strain_results(result, directory / "strain.h5")
    reopened = load_strain_results(directory / "strain.h5")[0]
    np.testing.assert_array_equal(reopened.tensor, result.tensor)
    export_strain_csv([result], directory / "catalogue.csv", directory / "reflections.csv")
```

`result.tensor` is a dimensionless `(3, 3)` tensor in the configured frame. Its off-diagonals are tensor shear, without engineering-shear doubling. `result.rotation` is the fitted proper rotation in Lauelab's native crystal-basis convention. `result.reciprocal` contains reciprocal vectors as rows in `1/nm`, including `2*pi`.

## Use an indexed frame or saved catalogue

For an existing `indexer` and `indexed_frame`, bind shared context once:

```python
refiner = StrainRefiner(
    reference=indexer.crystal,
    geometry=indexer.geometry,
    detector_index=indexer.detector_index,
)
result = refiner.refine(indexed_frame, pattern=0)
```

`FrameResult` does not retain its crystal or geometry objects. For this high-level path, measured centroids are converted to directions using the effective geometry, physical detector slot, frame ROI/grouping and depth. Stored indexing `qhat` is not silently combined with a new calibration. An explicit depth, including zero, overrides stored depth.

{func}`~lauelab.strain.prepare_reflections` exposes this preparation stage separately. It returns {class}`~lauelab.strain.StrainReflections` with aligned directions, HKLs, original measured peak indices and provenance. Prepared measurements retain their fixed calibration when passed to a refiner; prepare the original source again to change measurement context.

The same `refine` method accepts a `ResultSet`, `VisualizationDataset`, Igor indexing XML, or indexing-results HDF5. For a source with multiple frames, provide `frame_id` and `pattern`. XML frame IDs default to step ordinals; HDF5 retains stored frame IDs. A configured explicit reference is never replaced by the source indexing crystal.

Saved-input preparation reuses the existing readers and assignment joins. XML errors are checked before display-oriented truncation. Missing, out-of-range, repeated or mismatched associations make the selected pattern invalid; unrelated invalid patterns can remain in the catalogue. No HKLs are reassigned. Coincident harmonics cannot be counted as independent directions. Old HDF5 centroids retain their stored float32 precision even when loaded into float64 arrays; provenance identifies this source precision.

## Select a model and reflections

`fit="110111"` is the default, in `a,b,c,alpha,beta,gamma` order. A one means refine. Equivalent named input is `fit=("a", "b", "alpha", "beta", "gamma")`. Three rotations are always fitted. A mask needs at least one free cell parameter and at least one held length.

Excluded working-cell values come from the reference. The initial lattice provides the starting orientation and free values; the effective initial model is recorded. This needs no extra public option. The held length is a scale anchor: final reference-volume normalization changes all three lengths together. Held angles and ratios between held lengths remain fixed; a held length does not imply zero reported strain along that axis.

Use `refiner.replace(fit=("a", "b"))` for a new validated configuration. The original refiner remains unchanged. Fit parameters, reference, coordinates and numerical policy are read-only configuration. Per-call arguments select data, `pattern`, `frame_id`, original `peak_indices`, or depth. A reflection subset follows source assignment order and retains original peak identities.

## Inspect support and failure

`success` means that optimization converged and the joint orientation/shape Jacobian passed numerical support checks. It does not establish unbiased centroids, correct calibration, correct phase or calibrated experimental confidence.

| Status | Meaning |
|---|---|
| `converged` | Optimizer convergence and full-rank, bounded-condition numerical support |
| `insufficient_support` | Too few observations or deficient joint Jacobian rank |
| `ill_conditioned` | Joint Jacobian condition exceeds the configured numerical limit |
| `nonconverged` | Optimizer stopped without convergence |
| `invalid_input` | Expected item validation error recorded by a batch |

Failed results contain a NaN tensor. A finite candidate reciprocal lattice can remain for diagnosis. Inspect `message`, `rank`, `condition_number`, `n_parameters`, `nfev` and `rms_error_deg`. `result.reflections` is an immutable notebook table with original peak IDs, measured/predicted directions and pixels, angular/pixel residuals and calculated energies. Calculated energies are predictions, not measured energy inputs. Direction-only fits have unavailable pixel columns.

The default `condition_limit=1e8` applies to dimensionless rotation/length-log/angle-radian parameters. It is a numerical guard, not an experimental strain tolerance. No automatic downweighting, rematching or outlier rejection occurs. Users select reflections explicitly and inspect residuals. Calibrated uncertainty is deferred.

## Frames, batches and saved results

Frames are `"crystal"`, `"beamline"` (fit default), `"xhf"` and `"sample"`. `result.tensor_in(name)` changes the reported components without refitting. The XHF/sample frames are the fixed Igor presets; `transform_strain(tensor, rotation)` accepts a proper orthonormal coordinate transform. See the [algorithm](../concepts/algorithms/strain.md) for the precise fitted-orientation convention and finite-shear distinction.

`refiner.refine_many(source)` processes all patterns serially in source order. Use `patterns="best"` for the lowest rank per frame or a sequence of ranks, and `frame_ids` for explicit frame filtering. The batch does not inherit visualization support filters. A saved catalogue is parsed once, with association indexes reused. Each result is independent; previous fits do not become implicit starting guesses.

A malformed single-pattern input raises an exception. Batches record expected per-pattern input failures and continue, including insufficient or nonconverged fits. Invalid shared configuration, unreadable/corrupt source structure and unexpected solver/internal errors stop execution. Batch results retain source identities and order.

`save_strain_results` stores an ordered collection, including failed items, in standalone HDF5. It saves actual fitting inputs, reference and candidate lattices, predictions, coordinate/strain conventions, settings and calibration provenance. `load_strain_results` validates the format and reconstructs read-only results without reopening source images or geometry paths. New scientific arrays use float64.

`export_strain_csv` writes separate catalogue and reflection tables. They are exchange views; HDF5 is the complete evidence artifact. Saving is explicit, refuses existing destinations unless `overwrite=True`, and publishes validated temporary outputs. CSV files are published independently, not as a multi-file transaction. Source indexing files are never destinations for these writers.

See the [API reference](../reference/strain.md) for signatures and [storage format](../development/strain-format.md) for validation details. Measured-energy full strain, Q-profile fitting, calibrated error bars, peak refitting and portal interfaces are outside this release.
