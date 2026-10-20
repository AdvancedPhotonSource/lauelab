# Depth reconstruction

Depth reconstruction assigns the intensity that disappears between consecutive wire-scan frames to depths along the incident beam. This page describes the transformations and binning implemented by the native kernel. It uses the coordinate conventions of the 34-ID-E geometry file without assigning physical axis names.

## Pixel positions

The kernel converts a binned image coordinate to a laboratory position in four steps.

1. Map the image column and row to full-detector pixel coordinates using the image geometry's `start` and `group`. The result is the center of the pixel group:

   ```{math}
   i = c\,g_x + s_x + \frac{g_x - 1}{2}, \qquad
   j = r\,g_y + s_y + \frac{g_y - 1}{2}
   ```

   Here `c` and `r` are the zero-based image column and row, `(s_x, s_y)` is `start`, and `(g_x, g_y)` is `group`, all in unbinned pixels.

2. Convert to detector-local physical coordinates using the detector's pixel size and full dimensions:

   ```{math}
   x_d = \left(i - \tfrac{N_x - 1}{2}\right) p_x + P_x, \qquad
   y_d = \left(j - \tfrac{N_y - 1}{2}\right) p_y + P_y, \qquad
   z_d = P_z
   ```

   `N_x` and `N_y` are the full-detector dimensions in unbinned pixels, `p_x` and `p_y` are the pixel sizes in µm, and `P` is the detector translation from the geometry file.

3. Apply the detector rotation matrix from the geometry file.

4. Apply the wire-alignment rotation `\rho` described below.

The kernel evaluates two positions for every pixel: the column edges `c - 0.5` and `c + 0.5`. They bound the pixel along the column direction and give each pixel a finite depth interval for one wire position.

## Wire positions

Raw wire positions are `(x, y, z)` values in the acquisition coordinate system. The kernel transforms each one in this order:

1. Apply the selected positioner correction. `"pm500"` adds a periodic table correction to `y` and `z`. `"alio"` and `"none"` apply no correction.
2. Subtract the wire origin from the geometry file.
3. Apply the wire rotation matrix from the geometry file.
4. Apply `\rho`.

`\rho` is the rotation that brings the wire axis onto the x axis. It is derived from the wire axis vector in the geometry file. When the wire axis is already parallel to x, `\rho` is the identity. After this rotation the wire is a cylinder along x, and the depth construction uses only y and z components.

The incident-beam direction used by the depth calculation is:

```{math}
\hat{k}_i = \rho\,(0, 0, 1)
```

## Depth of one pixel and wire position

Let `(y_p, z_p)` be the rotated pixel position and `(y_w, z_w)` the rotated wire-center position. With wire radius `a`, the kernel calculates:

```{math}
\tan\varphi_0 = \frac{z_w - z_p}{y_w - y_p}, \qquad
\tan\Delta\varphi = \frac{a}{\sqrt{(y_w - y_p)^2 + (z_w - z_p)^2 - a^2}}
```

`\varphi_0` is the direction from the pixel to the wire center, and `\Delta\varphi` is the half-angle subtended by the wire. The tangent ray from the pixel past the wire edge has direction `\varphi_0 - \Delta\varphi` for the leading edge and `\varphi_0 + \Delta\varphi` for the trailing edge:

```{math}
\tan\varphi = \frac{\tan\varphi_0 \mp \tan\Delta\varphi}{1 \pm \tan\varphi_0 \tan\Delta\varphi}
```

The kernel intersects this ray, written as `z = z_p + (y - y_p)\tan\varphi`, with the incident-beam line through the origin along `\hat{k}_i`. The depth is the signed projection of the intersection point `\mathbf{s}` onto the beam direction:

```{math}
D = \hat{k}_i \cdot \mathbf{s}
```

`D` is in µm and increases along `\hat{k}_i`. Depth is measured from the laboratory-frame origin. Subtracting the wire origin expresses the raw positioner readings in that frame, so `D = 0` corresponds to the Si origin recorded in the geometry file.

## Depth grid

The output depths form a regular grid. The requested `depth_range` endpoints are rounded to the nearest multiple of `resolution`:

```{math}
D_0 = \operatorname{round}(D_\text{start} / \delta)\,\delta, \qquad
n = \operatorname{round}\!\left(\frac{D_1 - D_0}{\delta}\right) + 1
```

where `\delta` is `resolution` in µm and `D_1` is the rounded end. Output depth `k` is `D_0 + k\delta`, and its bin covers `[D_0 + (k - \tfrac12)\delta,\; D_0 + (k + \tfrac12)\delta)`. `ReconstructionResult.depth_um` lists `D_0 + k\delta` for `k = 0, \dots, n - 1`. Equal endpoints give one depth.

## Frame differencing

For one pixel, the kernel reads its value from each scan frame into a trace `I_0, \dots, I_{N-1}`, applies the per-frame scale and the per-pixel normalization plane, and optionally runs the cosmic-ray filter. It then forms the differences:

```{math}
d_k = I_k - I_{k+1}, \qquad k = 0, \dots, N - 3
```

The final difference `I_{N-2} - I_{N-1}` is never formed. This reproduces the executable's loop bound. A zero difference deposits nothing.

Scan frame `k` is paired with wire positions `w_k` and `w_{k+1}`. For the leading edge, the kernel deposits `d_k` with positive sign; for the trailing edge, it deposits `-d_k`. With `wire_edge="both"`, it performs both deposits for the same difference, each with its own edge depths.

## Trapezoid deposit

For frame pair `k`, the pixel's two edge positions and two wire positions give four depths:

| Symbol | Pixel edge | Wire position |
| --- | --- | --- |
| `D_\text{ps}` | front (`c + 0.5`) | `w_k` |
| `D_\text{fs}` | back (`c - 0.5`) | `w_k` |
| `D_\text{fe}` | front | `w_{k+1}` |
| `D_\text{pe}` | back | `w_{k+1}` |

`D_\text{fs}` and `D_\text{fe}` are swapped when reversed. The weight is a trapezoid: zero outside `[D_\text{ps}, D_\text{pe}]`, rising linearly to one at `D_\text{fs}`, one between `D_\text{fs}` and `D_\text{fe}`, and falling linearly to zero at `D_\text{pe}`. Its total area is `(D_\text{pe} - D_\text{ps}) / 2`.

Each output bin that overlaps the trapezoid receives:

```{math}
\Delta\text{out}_k = d \cdot \frac{\text{area of the trapezoid inside the bin}}{\text{total trapezoid area}}
```

The deposit is skipped when the trapezoid lies entirely outside the depth grid or its area is negative or not a number. Bins are clamped to the grid, so intensity from a trapezoid that partly extends outside the range is lost rather than folded into the edge bins.

## Intensity mask

The intensity map is the first scan frame unless you pass `intensity_map` with array input. The kernel reconstructs pixels whose intensity-map value is at least this cutoff:

```{math}
\text{cutoff} = \max\!\left(1,\; \lfloor v_{\lfloor n\,(100 - p)/100 \rfloor} \rfloor\right)
```

Here `v` is the ascending sorted intensity map with `n` pixels, and `p` is `percent_brightest`. Because the cutoff is at least 1, a pixel whose intensity-map value is below 1 is never reconstructed, even with `percent_brightest=100`.

## Normalization

Two normalizations can scale the trace before differencing.

**Per-frame scale.** File input reads the HDF5 vector named by `normalization` and divides it by a fixed beamline divisor: 102 for `mA` and 88100 for `cnt3`. Other tags have no divisor. Array input takes the equivalent dimensionless `scale` array directly.

**Exponent normalization.** With `norm_exponent` set to `e`, every frame is multiplied element-wise by a plane derived from the intensity map `M`:

```{math}
w(x, y) = \begin{cases} T^{-e} & M(x, y) < T \\ M(x, y)^{-e} & \text{otherwise} \end{cases}
```

When `norm_threshold` is `None`, `T` is the mean plus five standard deviations of the lowest half of the sorted intensity-map values, using the sample variance. The exponent and threshold are rounded to single precision to match the executable. When files use an integer output type, written images are multiplied by a type-dependent rescale factor. Retained result images are not rescaled.

**Cosmic-ray filter.** With `cosmic_filter=True`, the kernel replaces isolated spikes in each pixel's trace with a running median before differencing. A value counts as a spike when it exceeds the local median by a fixed count or a fixed factor. Traces shorter than seven frames are not filtered.

## Conventions requiring review

The kernel reproduces the executable's arithmetic and is verified against it by regression references. The following interpretations are not established by the implementation alone:

- The physical sign convention for depth relative to the sample surface.
- Physical names for the laboratory axes and the wire scan direction.
- Scientific guidance for `percent_brightest`, `norm_exponent`, and the choice of wire edge.

Until a 34-ID-E source confirms them, treat `depth_um` as the projection onto `\hat{k}_i` defined above and validate edge and normalization choices against the executable on representative data.
