# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Known-value fixtures for the reconstruction storage and inspection contract.

Every expected value in this module was worked by hand from the rules in
``docs/development/reconstruction-scan-format.md``. Nothing here is program
output, and nothing here is derived from the recorded goldens under
``tests/data/reconstruction/``. ``tests/test_reconstruction_contract.py``
checks the values against HDF5 itself and against brute-force oracles; the
storage, reduction, and ROI implementations are then tested against the same
values.

Arrays use ``(depth, y, x)``. ROI bounds are half-open ``(y0, y1, x0, x1)``.
Click coordinates are ``(x, y)`` in stored-image pixels, where an integer is a
pixel centre.
"""

from __future__ import annotations

import numpy as np

DEPTH_UM = np.array([-1.5, -0.5, 0.5, 1.5])

# Unscaled float64 kernel output. Plane 0 mixes signs and fractions, plane 1
# is a zero image, plane 2 crosses every integer saturation boundary, and
# plane 3 has no positive pixel.
COMPUTED = np.array([
    [[1.9, -1.9, 0.5, -0.5],
     [2.0, 3.0, -4.0, 0.0],
     [10.25, -10.75, 100.0, -100.0]],
    [[0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0],
     [0.0, 0.0, 0.0, 0.0]],
    [[300.0, -300.0, 255.9, 256.0],
     [40000.0, -40000.0, 65535.9, 65536.0],
     [70000.0, -70000.0, 32767.9, -32768.9]],
    [[-1.0, -2.0, -3.0, 0.0],
     [-0.9, -0.1, 0.0, 0.0],
     [-5.5, -6.5, -7.5, -8.5]],
])

# Stored pixels are HDF5's conversion of ``COMPUTED * rescale``: truncation
# toward zero, then saturation at the limits of the stored dtype.
STORED = {
    # Pixel type 1, the both-edge default. No saturation; signs survive.
    "int32": {
        "pixel_type": 1,
        "rescale": 1.0,
        "data": np.array([
            [[1, -1, 0, 0], [2, 3, -4, 0], [10, -10, 100, -100]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[300, -300, 255, 256], [40000, -40000, 65535, 65536],
             [70000, -70000, 32767, -32768]],
            [[-1, -2, -3, 0], [0, 0, 0, 0], [-5, -6, -7, -8]],
        ], dtype=np.int32),
        "depth_intensity": np.array([1, 0, 131581, -32], dtype=np.int64),
        "sum_reconstructed": np.array([
            [300, -303, 252, 256],
            [40002, -39997, 65531, 65536],
            [70005, -70016, 32860, -32876],
        ], dtype=np.int64),
    },
    # Pixel type 3. Every negative pixel stores 0, so stored totals exceed the
    # computed totals and the all-negative plane stores a zero image.
    "uint16": {
        "pixel_type": 3,
        "rescale": 1.0,
        "data": np.array([
            [[1, 0, 0, 0], [2, 3, 0, 0], [10, 0, 100, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[300, 0, 255, 256], [40000, 0, 65535, 65535], [65535, 0, 32767, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        ], dtype=np.uint16),
        "depth_intensity": np.array([116, 0, 270183, 0], dtype=np.int64),
        "sum_reconstructed": np.array([
            [301, 0, 255, 256],
            [40002, 3, 65535, 65535],
            [65545, 0, 32867, 0],
        ], dtype=np.int64),
    },
    # Pixel type 2 with exponent normalization, whose integer rescale is 128.
    # The rescale is applied once, before conversion.
    "int16_rescaled": {
        "pixel_type": 2,
        "rescale": 128.0,
        "data": np.array([
            [[243, -243, 64, -64], [256, 384, -512, 0], [1312, -1376, 12800, -12800]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[32767, -32768, 32755, 32767], [32767, -32768, 32767, 32767],
             [32767, -32768, 32767, -32768]],
            [[-128, -256, -384, 0], [-115, -12, 0, 0], [-704, -832, -960, -1088]],
        ], dtype=np.int16),
        "depth_intensity": np.array([64, 0, 131052, -4479], dtype=np.int64),
        "sum_reconstructed": np.array([
            [32882, -33267, 32435, 32703],
            [32908, -32396, 32255, 32767],
            [33375, -34976, 44607, -46656],
        ], dtype=np.int64),
    },
}

# The computed per-depth totals, which are ``ReconstructionResult.
# depth_intensity``. They are not the stored totals of any integer variant.
COMPUTED_DEPTH_INTENSITY = np.array([0.5, 0.0, 131582.8, -35.0])

# Float-to-integer conversion at each boundary: (computed value, stored value).
# NaN is absent because HDF5 converts it inconsistently between integer dtypes;
# the scan file stores 0 for it, which its own tests cover.
CONVERSION = {
    "int8": [(-0.9, 0), (0.9, 0), (-1.5, -1), (2.5, 2), (127.9, 127), (128.0, 127),
             (-128.9, -128), (-129.0, -128), (np.inf, 127), (-np.inf, -128)],
    "uint8": [(-0.9, 0), (-1.0, 0), (0.9, 0), (2.5, 2), (255.9, 255), (256.0, 255),
              (np.inf, 255), (-np.inf, 0)],
    "int16": [(-0.9, 0), (0.9, 0), (-1.5, -1), (32767.9, 32767), (32768.0, 32767),
              (-32768.9, -32768), (-32769.0, -32768), (np.inf, 32767),
              (-np.inf, -32768)],
    "uint16": [(-0.9, 0), (-1.0, 0), (0.9, 0), (65535.9, 65535), (65536.0, 65535),
               (1e300, 65535), (np.inf, 65535), (-np.inf, 0)],
    "int32": [(-0.9, 0), (0.9, 0), (-2.5, -2), (2147483647.0, 2147483647),
              (2147483647.5, 2147483647), (2147483648.0, 2147483647),
              (-2147483648.9, -2147483648), (-2147483649.0, -2147483648),
              (1e300, 2147483647), (-1e300, -2147483648)],
}

# ROI placement on a (rows, columns) = (3, 4) image:
# (size, click (x, y), bounds (y0, y1, x0, x1), centre (x, y)).
IMAGE_SHAPE = (3, 4)
PLACEMENTS = [
    (1, (0.0, 0.0), (0, 1, 0, 1), (0.0, 0.0)),
    (1, (0.49, -0.49), (0, 1, 0, 1), (0.0, 0.0)),
    (1, (3.49, 2.49), (2, 3, 3, 4), (3.0, 2.0)),
    # An odd size ties at a pixel edge; the lower pixel wins.
    (1, (3.5, 2.5), (2, 3, 3, 4), (3.0, 2.0)),
    (1, (1.5, 0.5), (0, 1, 1, 2), (1.0, 0.0)),
    # An even size has a half-integer centre and ties at a pixel centre.
    (2, (2.5, 1.5), (1, 3, 2, 4), (2.5, 1.5)),
    (2, (1.0, 1.0), (0, 2, 0, 2), (0.5, 0.5)),
    (2, (1.01, 1.01), (1, 3, 1, 3), (1.5, 1.5)),
    (2, (2.9, 0.2), (0, 2, 2, 4), (2.5, 0.5)),
    # The largest square that fits has one legal row position.
    (3, (1.0, 1.0), (0, 3, 0, 3), (1.0, 1.0)),
    (3, (2.4, 1.4), (0, 3, 1, 4), (2.0, 1.0)),
]

# Placements outside the image are rejected, never clipped or shifted:
# (size, click (x, y)).
REJECTED_PLACEMENTS = [
    (1, (-0.5, 0.0)),      # tie between pixel -1 and pixel 0 resolves to -1
    (1, (-0.51, 0.0)),
    (1, (3.51, 0.0)),
    (1, (0.0, 2.51)),
    (2, (0.0, 1.0)),       # x0 would be -1
    (2, (3.4, 1.0)),       # x1 would be 5
    (3, (0.0, 1.0)),
    (3, (1.0, 0.4)),       # y0 would be -1
    (3, (1.0, 1.6)),       # y1 would be 4
    (4, (1.5, 1.0)),       # wider than the image has rows
    (1, (np.nan, 0.0)),
    (1, (0.0, np.inf)),
]

# A size must be a positive integer; ``True`` is not an integer here.
INVALID_SIZES = [0, -1, 2.5, np.nan, True, "2", None]

# ROI sum traces through depth on ``STORED["int32"]["data"]``:
# name -> (bounds (y0, y1, x0, x1), trace).
ROI_TRACES = {
    "single_pixel": ((0, 1, 0, 1), np.array([1, 0, 300, -1], dtype=np.int64)),
    "even_square": ((1, 3, 2, 4), np.array([-4, 0, 131070, -15], dtype=np.int64)),
    "odd_square": ((0, 3, 0, 3), np.array([101, 0, 98557, -24], dtype=np.int64)),
    "nonpositive": ((0, 1, 1, 2), np.array([-1, 0, -300, -2], dtype=np.int64)),
    "full_frame": ((0, 3, 0, 4), np.array([1, 0, 131581, -32], dtype=np.int64)),
}

# Normalization divides a trace by its own maximum when that maximum is
# positive, and is otherwise unavailable (``None``). Signs are preserved.
NORMALIZED = [
    (np.array([1, 0, 300, -1]), np.array([1 / 300, 0.0, 1.0, -1 / 300])),
    (np.array([2.0, -4.0, 1.0]), np.array([1.0, -2.0, 0.5])),
    (np.array([-1, 0, -300, -2]), None),
    (np.array([-3.0, -1.0, -2.0]), None),
    (np.array([0, 0, 0]), None),
]

# A logarithmic axis omits nonpositive samples and reports how many:
# (trace, indices kept, number omitted).
LOG_SAMPLES = [
    (np.array([1, 0, 300, -1]), np.array([0, 2]), 2),
    (np.array([-4, 0, 131070, -15]), np.array([2]), 3),
    (np.array([5, 6, 7]), np.array([0, 1, 2]), 0),
    (np.array([-1, 0, -300, -2]), np.array([], dtype=np.int64), 4),
]
