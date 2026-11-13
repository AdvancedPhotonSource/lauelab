# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Fit of one peak at a position chosen by the user."""

from __future__ import annotations

from numbers import Real

import numpy as np
from scipy.optimize import least_squares

from lauelab.indexing import InputError
from lauelab.indexing.indexer import PEAK_SEARCH_DTYPE

_HWHM_PER_SIGMA = np.sqrt(2.0 * np.log(2.0))


def fit_peak(image, x: float, y: float, *, half_width: int = 10) -> np.ndarray | None:
    """Fit a two-dimensional Gaussian to the peak nearest a chosen position.

    This replaces LaueGo's "Add Gaussian to pkList". The fit region is the
    square of ``2 * half_width + 1`` pixels centred on the brightest pixel
    within ``half_width`` of ``(x, y)``. The model is a constant background
    plus a Gaussian with independent widths along x and y.

    Parameters
    ----------
    image
        Two-dimensional frame indexed ``image[y, x]``.
    x, y
        Zero-based frame pixel coordinates of the chosen position.
    half_width
        Half-width in pixels of the search and fit regions, at least 3.

    Returns
    -------
    numpy.ndarray or None
        One peak with shape ``(1,)`` and the fields of ``PeakSearch.peaks``,
        so it can be appended to them with :func:`numpy.concatenate`, or `None`
        if no peak can be fitted: the fit fails, the amplitude is not positive,
        or the centre is not inside the fit region. ``intens`` is the Gaussian
        amplitude above ``background``, ``integral`` its integral,
        ``hwhm_x`` and ``hwhm_y`` its half widths at half maximum in pixels,
        ``tilt`` is zero, and ``chisq`` is the mean square fit residual.
    """
    image = np.asarray(image)
    if image.ndim != 2 or image.dtype.kind not in "iuf":
        raise InputError("image must be a two-dimensional numeric array")
    if not all(isinstance(v, Real) and np.isfinite(v) for v in (x, y)):
        raise InputError("x and y must be finite numbers")
    if isinstance(half_width, bool) or not isinstance(half_width, (int, np.integer)) or half_width < 3:
        raise InputError("half_width must be an integer of at least 3")
    ny, nx = image.shape
    if not (0 <= x <= nx - 1 and 0 <= y <= ny - 1):
        raise InputError(f"({x}, {y}) is outside the {nx} x {ny} image")

    def window(cx, cy):
        x0, x1 = max(int(round(cx)) - half_width, 0), min(int(round(cx)) + half_width + 1, nx)
        y0, y1 = max(int(round(cy)) - half_width, 0), min(int(round(cy)) + half_width + 1, ny)
        return x0, x1, y0, y1

    x0, x1, y0, y1 = window(x, y)
    search = image[y0:y1, x0:x1].astype(np.float64)
    peak_y, peak_x = np.unravel_index(int(np.argmax(search)), search.shape)
    x0, x1, y0, y1 = window(x0 + peak_x, y0 + peak_y)
    data = image[y0:y1, x0:x1].astype(np.float64)
    if data.size < 16:
        return None
    grid_y, grid_x = np.mgrid[y0:y1, x0:x1].astype(np.float64)
    background = float(np.median(data))
    amplitude = float(data.max() - background)
    if amplitude <= 0:
        return None
    start_y, start_x = np.unravel_index(int(np.argmax(data)), data.shape)

    def model(p):
        b, a, cx, cy, sx, sy = p
        return b + a * np.exp(-0.5 * (((grid_x - cx) / sx) ** 2 + ((grid_y - cy) / sy) ** 2))

    p0 = [background, amplitude, x0 + start_x, y0 + start_y, 1.5, 1.5]
    lower = [-np.inf, 0.0, x0, y0, 0.3, 0.3]
    upper = [np.inf, np.inf, x1 - 1, y1 - 1, float(half_width), float(half_width)]
    fit = least_squares(lambda p: (model(p) - data).ravel(), p0, bounds=(lower, upper))
    if not fit.success:
        return None
    b, a, cx, cy, sx, sy = fit.x
    on_edge = cx <= x0 + 0.5 or cx >= x1 - 1.5 or cy <= y0 + 0.5 or cy >= y1 - 1.5
    if a <= 0 or on_edge:
        return None
    peak = np.zeros(1, dtype=PEAK_SEARCH_DTYPE)
    peak["fit_x"], peak["fit_y"] = cx, cy
    peak["intens"] = a
    peak["integral"] = 2.0 * np.pi * a * sx * sy
    peak["hwhm_x"], peak["hwhm_y"] = _HWHM_PER_SIGMA * sx, _HWHM_PER_SIGMA * sy
    peak["tilt"] = 0.0
    peak["chisq"] = float(np.mean(fit.fun**2))
    peak["background"] = b
    return peak
