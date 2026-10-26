# Deviatoric strain refinement

The module fits assigned measured directions using the modern position-only `DeviatoricStrainRefine` objective in LaueGo `microN/IndexingN.ipf`. It does not use the different relative-Q objective in `TotalStrainRefine`, or the older `sym(F)-I` strain from `micro/Refinement.ipf`.

## Reference and lattice basis

Reciprocal lattice vectors occupy rows in $B$, in `1/nm`, including $2\pi$. A Miller-index row maps to laboratory reciprocal space as $q=hkl\,B$. Direct lattice vectors are columns of $A=2\pi B^{-1}$, in nm.

An explicit `Cell` supplies its exact metric after unit and validity checks. `Crystal` uses native space-group metric coercion and records the effective cell. The native reference basis has c parallel to z. A deformed lattice is constructed without imposing reference space-group metric constraints. No HKL reindexing, primitive/conventional cell conversion or reference symmetry selection is inferred from measurements.

The initial reciprocal matrix supplies a canonical-cell orientation and initial free shape values. Frozen working-cell values come from the reference. Scale the initial lengths to one held reference length, project excluded parameters to reference values and record that effective start. A starting point that cannot define a valid cell is an input error.

## Objective and numerical parameters

Normalize each measured direction and each predicted $hkl\,B$. Igor's uniform mismatch is

$$r_{\rm chord}=\sqrt{\frac{1}{N}\sum_i2(1-\hat q_i\cdot\hat g_i)}.$$

The solver minimizes the equivalent vector residuals $\hat g_i-\hat q_i$. This avoids cancellation in `1-dot` near exact agreement. The objective is chord distance, approximately angular RMS for small residuals. Reported angular residuals use `atan2(norm(cross), dot)` and degrees.

The optimizer varies a local rotation in radians about the supplied starting orientation. Free lengths use dimensionless logarithmic increments; free angles use radian increments. Excluded values remain reference values. This parameter scaling does not change the available Igor-like models. Initial increments are exactly zero so floating-point deviations in nominal reference values do not collapse the initial trust-region size.

SciPy bounded trust-region least squares uses central numerical derivatives and uniform weights. Positive lengths, valid cell metrics and proper rotations are required. Nonphysical trial metrics receive a large residual outside the model; final candidate lattices are validated. The finite log-length bounds protect against numerical overflow; they are not calibrated experimental strain bounds.

The usual mask has five shape and three rotational parameters. Each direction supplies two independent observations, so the nominal gate is $2N>n_{\rm parameters}$. Rank and condition of the joint Jacobian also matter. Reflections confined to a plane can remain underconstrained despite satisfying the count gate. Coincident harmonic directions and repeated measured peak identities cannot create independent support.

## Normalize before calculating strain

Position-only data leave a common lattice scale unobservable. Normalize the refined cell to the reference volume, then calculate

$$F=A_{\rm measured}A_{\rm reference}^{-1},\qquad
U=\sqrt{F^TF},\qquad
\epsilon=U-I,\qquad
\epsilon_{\rm dev}=\epsilon-\tfrac13\operatorname{tr}(\epsilon)I.$$

The implementation obtains the positive right stretch from an SVD. This is deviatoric Biot strain. The square root is a symmetric matrix square root, not an elementwise operation. It differs from Green–Lagrange, logarithmic and infinitesimal strain at finite deformation.

Reference-volume normalization is a convention for the missing scale, not a measurement of physical incompressibility. Removing trace alone is not a substitute: at finite strain, scaling $U$ also scales $\operatorname{dev}(U-I)$. Both the fitter and standalone lattice conversion use the same normalization order.

## Frame and scalar conventions

The crystal-basis tensor is transformed to beamline components using the fitted canonical-cell orientation $\rho$, as Igor does:

$$\epsilon_{\rm BL}=\rho\epsilon_{\rm crystal}\rho^T.$$

For finite shear, $\rho$ need not equal the polar rotation of the complete deformation gradient. Substituting that rotation would change the scientific convention. The proper fitted rotation is stored independently; the existing direct lattice-quotient orientation helper is not treated as a pure rotation for a strained cell.

With $r=1/\sqrt2$, the fixed beamline-to-preset transforms, written as rows, are:

```text
XHF    = [[1, 0, 0], [0,  r,  r], [0, -r,  r]]
sample = [[1, 0, 0], [0, -r, -r], [0,  r, -r]]
```

Tensor shear is never doubled. Igor's scalar `vonMisesStrain` is $\sqrt{3\epsilon_{\rm dev}:\epsilon_{\rm dev}/2}$, which is 1.5 times the common equivalent-strain normalization $\sqrt{2\epsilon_{\rm dev}:\epsilon_{\rm dev}/3}$. `sum_abs_epsilon_crystal` is a crystal-frame component sum and is not rotation invariant.

Igor has a special symmetric rhombohedral embedding. Lauelab retains its native basis; empirical comparisons must explicitly convert the exported basis. Controlled rhombohedral calculations are tested, but this does not establish empirical all-system Igor parity.

## Limits and evidence

Centroid bias, peak overlap, phase/assignment errors, detector calibration errors and wrong depth can produce plausible fitted strain. Numerical convergence and good conditioning do not remove those error sources. Current peak results contain no centroid covariance, so this module does not report calibrated strain uncertainty.

White-beam positions cannot recover uniform expansion; independently measured energies or spacings are needed for lattice scale. [Zhang et al. (2017)](https://journals.iucr.org/s/issues/2017/04/00/fv5067/index.html) discusses this observability limit and the effects of image noise, calibration and reflection count. [Petit et al. (2015)](https://journals.iucr.org/s/issues/2015/04/00/co5059/index.html) describes the separate Laue-DIC approach to distorted spot images.

The source reference is [LaueGo, commit 69a3b02, IndexingN.ipf](https://github.com/34IDE/LaueGo/blob/69a3b0229dfca2e06a135621391051dd46e5928a/microN/IndexingN.ipf#L6699). Igor range, column-initializer and clipped-index semantics were reviewed explicitly. The port preserves scientific definitions while using explicit supplied associations and proper pattern selection. [Validation](../../development/strain-validation.md) separates controlled numerical recovery from pending empirical Igor export comparisons.
