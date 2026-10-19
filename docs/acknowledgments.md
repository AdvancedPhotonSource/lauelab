# Acknowledgments and citations

`lauelab` is a Python adaptation of [LaueGo](https://github.com/34IDE/LaueGo),
the Igor Pro analysis suite developed by Jon Z. Tischler and collaborators
for APS beamline 34-ID-E. It retains and adapts the original native indexing
and wire-scan reconstruction code.

Reflection simulation also uses Tischler's `pydiffract` code, incorporated
through the Laue Portal. See the [simulator provenance](https://github.com/AdvancedPhotonSource/lauelab/blob/main/src/lauelab/analysis/_vendor/jzt/README.md)
for source details.

## Citations

When publishing results obtained with `lauelab`, identify the package version
or commit and cite Tischler (2014) for the LaueGo analysis software. For
wire-scan reconstruction, also cite Larson et al. (2002); Yang et al. (2004)
provides additional background on differential-aperture X-ray microscopy.

- J. Z. Tischler (2014).
  [Reconstructing 2D and 3D X-ray Orientation Maps From White-Beam Laue](https://doi.org/10.1142/9781908979636_0010).
  In *Strain and Dislocation Gradients from Diffraction*, edited by
  R. Barabash and G. Ice, chapter 10, pp. 358–375. Imperial College Press.
- B. C. Larson, W. Yang, G. E. Ice, J. D. Budai and J. Z. Tischler (2002).
  [Three-dimensional X-ray structural microscopy with submicrometre resolution](https://doi.org/10.1038/415887a).
  *Nature* **415**, 887–890.
- W. Yang, B. C. Larson, J. Z. Tischler, G. E. Ice, J. D. Budai and W. Liu (2004).
  [Differential-aperture X-ray structural microscopy: a submicron-resolution three-dimensional probe of local microstructure and strain](https://doi.org/10.1016/j.micron.2004.02.004).
  *Micron* **35**(6), 431–439.
