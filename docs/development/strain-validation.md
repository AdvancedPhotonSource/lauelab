# Strain validation and Igor reference exports

Two evidence gates are separate:

1. **Controlled numerical correctness.** Known direct-lattice deformations, rotations and scale changes test the tensor math and recovery from assigned directions. Integration cases test centroids, sparse detector slots, grouping/depth, nonzero patterns, strict XML/HDF5 adapters, failure statuses and saved results. Repository tests and warning-free documentation remain required.
2. **Empirical Igor parity.** Run audited cases through the chosen Igor `DeviatoricStrainRefine` branch and compare explicit inputs, references, objectives, frames and tensors. This gate remains **pending**. There is no runnable Igor installation or established exported strain baseline in the inspected development environment.

Controlled tests do not establish empirical parity. They also do not calibrate experimental strain uncertainty. No test baseline is presented as an Igor export unless Igor actually produced it.

## Controlled cases

`tests/strain_cases.py` constructs native-basis direct lattices independently by reverse Cholesky factorization of the metric. Tests cover cubic, hexagonal, triclinic and rhombohedral cells, finite symmetric stretch/shear, pure rotation, hydrostatic scale invariance, reduced masks and alternate held-length anchors. Expected crystal tensors come from the known imposed symmetric stretch, with explicit volume normalization.

No tolerance was loosened to accommodate a failing calculation. The documented precision levels distinguish tensor algebra/recovery from existing float32 indexing-file coordinates. A saved float64 array cannot recover bits absent from a source file.

## Export a current Igor analysis

The companion `tests/data/strain/export_igor.ipf` exports the measured/refined diagnostic waves after running the position-only Igor routine. Load it with the existing LaueGo procedures. Select the original calibrated geometry in Igor, then supply explicit fitted/indexed peak waves, a constraint mask and an output prefix in Igor's native file-path syntax.

Use pattern 0 for initial comparisons: the checked position-only source initializes from `recip_lattice0` even when another pattern is requested. The portable implementation's nonzero-pattern behavior is separately tested. Use the `DeviatoricStrainRefine` branch, without `USE_ENERGY_STRAIN_REFINE`; the newer total-strain path has a different objective.

Export first:

- An unambiguous, well-supported cubic case.
- A hexagonal/non-orthogonal case.
- A controlled finite-shear case and its rotated counterpart.
- One reduced-mask case.
- A rhombohedral case with the source/reference basis conversion stated explicitly before comparison.

Retain the full original centroid list, HKL/peak association evidence, starting reciprocal matrix, source reference metric, geometry/ROI/group/depth, exact constraint mask and source revisions. The exporter includes wave notes and the refinement's measured/predicted direction table; retain the geometry file alongside them. Igor's nearest-direction rematching must be audited for duplicate/coincident support before it can be a trusted reference.

## Audit before comparing

1. Confirm units, handedness, HKL basis, reciprocal column/row conversion, source crystal setting and geometry origin.
2. Confirm that the effective canonical reference metric and reference-volume normalization agree. The Python API's explicit reference must match the source nominal cell for a parity case.
3. Match the unit-direction chord objective and uniform weighting. Do not compare against the older infinitesimal strain or magnitude-aware objective by accident.
4. Convert Igor's reciprocal columns to Lauelab rows. Reconcile the special rhombohedral embedding explicitly; a transpose alone is insufficient.
5. Compare fitted lattices and direction predictions before tensor components. Then compare the specified tensor frame, tensor shear and Igor scalar normalization.
6. Choose tolerances from the retained measurement precision and controlled sensitivity, record the justification, and generate any accepted reference artifact reproducibly. Never hand-edit exported or generated golden values.

The export procedure has been inspected against the source interface but has **not been run in Igor here**. Treat its first execution as part of reference preparation. Its text outputs are source evidence to audit, not automatically accepted regression goldens. The implementation's result provenance records empirical parity as pending until reviewed exports justify a new claim.

Numeric wave exports use 17 significant digits and explicit zero-based row/column/layer indices. Igor's routine rounds reciprocal and cell-parameter strings in wave notes to eight significant digits. These strings are not full-precision lattice references; comparison tolerances must account for their precision separately from the exported tensor and direction waves.
