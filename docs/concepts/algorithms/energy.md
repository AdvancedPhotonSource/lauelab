# Monochromatic intensity versus Q

{func}`~lauelab.energy.q_distribution` calculates a scattering-vector magnitude Q for each pixel from the photon energy and detector geometry. It reports the mean corrected detector intensity over pixels assigned to each uniform Q center. Only measured, selected pixels contribute to the profile; unmeasured scattering directions are not included.

## Geometry and units

Let $E$ be photon energy in keV, $\mathbf{k}_i$ the unit incident direction, and $\mathbf{k}_f$ the unit ray from the sample to a detector pixel in the established laboratory convention. Then

$$
Q=\frac{2\pi E}{hc}|\mathbf{k}_f-\mathbf{k}_i|
 =\frac{4\pi E}{hc}\sin\theta,
$$

where $2\theta$ is the scattering angle and $hc=1.239841857\ \mathrm{keV\,nm}$, as in `EnergyScansN.ipf`. Q is in `1/nm` and includes the factor of two pi. Reflection simulation uses $hc=1.2398419739\ \mathrm{keV\,nm}$; this difference matters when comparing the two calculations at high precision.

The detector transform calculates laboratory positions in µm. The function subtracts sample depth from each position's beam-axis coordinate before normalizing the outgoing ray. It then calculates Q from the photon energy and the difference between outgoing and incident directions. {meth}`~lauelab.indexing.Geometry.pixels_to_q` returns unit scattering-vector directions; it does not calculate energy-dependent magnitudes.

For full-detector origin $s$, grouping $g$, and zero-based stored-frame coordinate $i$, the full-detector coordinate is $s+ig+(g-1)/2$. Use this fractional group center and the supplied depth consistently in the Q range, histogram and spacing estimate.

## Igor waves and NumPy arrays

The Igor routines use `image[p][q]`, where `p` is the detector x coordinate and `q` is y. Igor calls these its row and column dimensions. Its memory order is column-major, so its first dimension varies fastest. NumPy inputs use `image[y, x]`; C-order traversal also varies x fastest. The independent scalar reference loops over y outside x to reproduce the same physical traversal. The first NumPy subscript therefore corresponds to Igor's `q`. See the [WaveMetrics multidimensional wave conventions](https://docs.wavemetrics.com/igorpro/igor-basics/multidimensional-waves).

An Igor wave has dimension scales and units in addition to its values. `SetScale/I` specifies the first and last **sample coordinates**. `QDistribution.q_nm_inv` contains these coordinates explicitly, including both endpoint centers. See the [SetScale reference](https://docs.wavemetrics.com/igorpro/commands/setscale).

Tests use nonsquare images with unequal intensities, asymmetric geometry, grouping and depth so that axis swaps, flattening errors and omitted fractional offsets cannot cancel each other.

## Grid and membership

The function calculates automatic endpoints from the minimum and maximum geometric Q of included pixels, regardless of intensity. To estimate spacing, it uses one diagonal grouped-pixel step at the approximate ROI midpoint. The midpoint is rounded to a stored pixel, then converted to its fractional full-detector group center. Both probe positions use the supplied depth. The center count is

$$
N=\operatorname{round}_{\rm Igor}\!\left(\frac{Q_{\max}-Q_{\min}}{dQ_{\rm est}}\right)+1.
$$

After choosing $N$, the actual spacing is $(Q_{\max}-Q_{\min})/(N-1)$. A pixel maps to index $\operatorname{round}_{\rm Igor}((Q-Q_{\min})/dQ)$. [Igor rounding](https://docs.wavemetrics.com/igorpro/commands/round) sends half ties away from zero; NumPy's round-to-even is not equivalent.

Explicit `q_first`, `q_last` and `q_points` use the same assignment rule. Endpoint bins extend half a step beyond the endpoint centers, with exact outer half ties falling outside the valid index range. The function omits out-of-range pixels from the profile and reports their count in the diagnostics. Empty bins are included in the result. No interpolation, endpoint clamping or automatic coarsening is applied.

The function raises `InputError` for zero spans, undefined or nonpositive spacing, fewer than two centers, spacing that cannot be represented in float64, or a count above the allocation limit.

## Correction formula

Subtract a matching supplied dark to obtain corrected pixel intensity $J$. Let exposure be $t_e$ in seconds, I0 count be $C_0$, scaler duration be $t_s$ in seconds and I0 gain be $G$. The default multiplicative correction is

$$
F=\frac{10^5}{t_e}\frac{t_s}{C_0}\frac{10^9}{G}.
$$

When I0 gain is omitted, the function uses $G=10^9$ and records this assumption in the provenance. The other quantities are required, and all supplied correction values must be positive and finite. For exposure-only normalization, $F=1/t_e$; for raw mode, $F=1$. A bin's sum is $F\sum J$, and its mean is that sum divided by assigned pixel occupancy.

Masked pixels are excluded before arithmetic and are omitted from both sums and occupancy. Zero and negative corrected intensities are valid. If a bin contains a nonfinite corrected intensity, its sum and mean are NaN, and its nonfinite-pixel count is positive. Other bins are unaffected. Finite-input overflow raises `InputError`. Float64 summation can depend slightly on accumulation order; integer occupancy and bin assignment are exact for the specified grid.

## Igor interoperability

The calculation is based on the single-image routine in [34IDE/LaueGo](https://github.com/34IDE/LaueGo/blob/69a3b0229dfca2e06a135621391051dd46e5928a/microN/EnergyScansN.ipf). When comparing profiles, account for the mask convention and occupancy: Igor's energy routine includes pixels whose mask value is 1 and counts excluded pixels with zero intensity. `q_distribution` uses nonzero mask values to exclude pixels from both sums and occupancy. See {ref}`Select pixels and Q centers <energy-pixels>` for mask conversion.

`q_distribution` uses fractional group centers and the supplied depth in the range, spacing estimate and histogram. It requires valid correction metadata and does not fit peaks, estimate a pedestal, or calculate statistical uncertainties.

## Validation

`tests/data/energy/reference.py` implements scalar detector geometry, Q assignment and correction sums independently of the production NumPy code, with no `lauelab` imports. Source revision, source hashes, fixture parameters and numeric tolerances are recorded in `tests/data/energy/provenance.json`. Tests compare bin membership and counts exactly and compare floating-point values within the documented tolerances.

Separate flat-detector cases have analytic expected Q values. Synthetic rings at known Q exercise the image, correction and persistence workflow. The validation evidence consists of these analytic and source-derived Python checks; it does not include measured monochromatic data or execution in Igor.
