# Igor strain reference preparation

No empirical Igor strain goldens are available yet. `export_igor.ipf` is a
source-inspected helper that has **not been executed in Igor**. The controlled
Python cases in `tests/strain_cases.py` are independent synthetic tests.

Load the exporter with the LaueGo position-only `DeviatoricStrainRefine`
procedures. Disable `USE_ENERGY_STRAIN_REFINE`, activate the calibrated geometry
and use an already indexed, well-supported pattern 0. Supply the actual running
LaueGo revision and a prefix in a fresh directory, using Igor path syntax:

```igorpro
LauelabExportIgorStrain(0, "110111", FullPeakListOrange, FullPeakIndexed, "Disk:exports:case1_", "<running-LaueGo-commit>")
```

This invokes the normal Igor refinement and updates its diagnostic waves.
It exports all peak lists referenced by the indexing wave, the original indexed
wave, measured/predicted direction diagnostics, four strain tensors and their
wave notes. Numeric CSVs use explicit zero-based row, column and layer indices
with 17 significant digits; reconstruct the arrays by those indices rather than
assuming Python memory order matches Igor. `PeaksForStrain` columns follow the
audited source: HKL 0–2, measured directions 3–5, predicted directions 6–8,
intensity 9, predicted energy 10, predicted pixels 11–12, angular error 13 and
detector slot 14.

Keep the original image/indexing source, effective reference, active geometry
XML and ROI/group/depth with the export. Igor's nearest-direction association
must be audited against original measured peaks, including duplicates. Notes
contain the initial and refined reciprocal lattices, but the Igor routine rounds
its lattice notes to eight significant digits. Do not treat those strings as
full-precision lattice goldens; retain the full-precision tensor and direction
waves and justify comparison tolerances accordingly.

The helper refuses existing destinations. An interrupted or failed execution can
leave partial evidence: only an export containing the final metadata and all
expected waves is complete. Audit the first execution and any local procedure
changes before accepting an export as scientific reference data. The validation
guide in `docs/development/strain-validation.md` specifies basis, objective,
scale and frame comparisons, including the special rhombohedral basis.
