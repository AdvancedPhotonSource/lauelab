# Synthetic monochromatic energy fixtures

`rings.h5` contains a deterministic analytic ring phantom. The matching dark image, exclusion mask and calibration are in `dark.h5`, `mask.h5` and `geometry.xml`. The detector is rectangular, translated and rotated, and occupies physical slot 2. The selected ROI contains a masked NaN that is excluded before arithmetic. Rings are centered at 6.5 and 13 `1/nm`; the expected profile is the mean over pixels assigned to each Q bin, not the continuous ring function evaluated at the bin center.

`reference.npz` contains scalar reference values for explicit and automatic Q grids. `reference.py` calculates detector rays, Q, sorting, grouping and correction sums independently of the production NumPy code, with no `lauelab` imports. Source revision, source hashes, calculation conventions, numeric tolerances and their justification are recorded in `provenance.json`. Separate flat-detector tests compare the geometry with analytic expected values. The images are synthetic and the references are calculated in Python; these fixtures contain no measured data or Igor-runtime outputs.

Regenerate only these fixtures from the repository root:

```console
python -m tests.data.energy.generate
```

To verify source hashes against a local LaueGo checkout:

```console
python -m tests.data.energy.generate --lauego-root /path/to/LaueGo
```

The generator requires no Igor license or runtime and writes only the energy fixtures. Tests compare bin membership and counts exactly and floating-point values within the recorded tolerances.
