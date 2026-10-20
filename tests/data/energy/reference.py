# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Independent scalar IPF-derived reference; NOT Igor-runtime output.

No production lauelab imports. Igor's first subscript p is physical detector x,
and varies fastest on flattening; our image[y,x] loops preserve that traversal.
SetScale/I gives inclusive endpoint CENTERS. Approved mask/zero/coordinate fixes
are applied explicitly, not claimed to reproduce those original source bugs.
"""

import itertools
import math

import numpy as np

HC = 1.239841857
UPSTREAM = {
    "repository": "https://github.com/34IDE/LaueGo",
    "commit": "69a3b0229dfca2e06a135621391051dd46e5928a",
    "sha256": {
        "microN/EnergyScansN.ipf": "f37a33ff4c4c4f1067e9d26c99d8528b55b2e4c1b7c8056b4aeda476077c204d",
        "microN/microGeometryN.ipf": "fbb1d18f8e8748a321681de8e8b6c6f1511d4d4b62ec5cdf9a8319b183933c78",
    },
}


def round_away(value):
    magnitude = abs(value)
    integer = math.floor(magnitude)
    return math.copysign(integer + (magnitude-integer >= 0.5), value)


def rotation_matrix(vector):
    """Rodrigues rotation-vector matrix, using scalar arithmetic."""
    angle = math.sqrt(sum(float(v)**2 for v in vector))
    if angle == 0:
        return [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    x, y, z = [v/angle for v in vector]
    c, s, d = math.cos(angle), math.sin(angle), 1-math.cos(angle)
    return [[c+x*x*d, x*y*d-z*s, x*z*d+y*s],
            [y*x*d+z*s, c+y*y*d, y*z*d-x*s],
            [z*x*d-y*s, z*y*d+x*s, c+z*z*d]]


def q_at(x, y, calibration, energy, depth=0):
    """Scalar detector coordinates -> half scattering angle -> Q magnitude."""
    nx, ny = calibration["pixels"]
    sx, sy = calibration["size_um"]
    p = calibration["translation_um"]
    local = [(x-(nx-1)/2)*sx/nx+p[0], (y-(ny-1)/2)*sy/ny+p[1], p[2]]
    matrix = rotation_matrix(calibration["rotation_vector_rad"])
    lab = [sum(matrix[i][j]*local[j] for j in range(3)) for i in range(3)]
    lab[2] -= depth
    scattering_angle = math.atan2(math.hypot(lab[0], lab[1]), lab[2])
    return 4*math.pi*energy/HC*math.sin(scattering_angle/2)


def profile(image, calibration, energy, *, start=(0, 0), group=(1, 1), depth=0,
            roi=None, mask=None, dark=None, factor=1, grid=None):
    """Sort by Q and sum scalar groups independently of production bincount."""
    ny, nx = image.shape
    y0, y1, x0, x1 = roi or (0, ny, 0, nx)
    samples = []
    for y in range(y0, y1):
        for x in range(x0, x1):
            if mask is not None and mask[y, x] != 0:
                continue
            full_x = start[0]+x*group[0]+(group[0]-1)/2
            full_y = start[1]+y*group[1]+(group[1]-1)/2
            q = q_at(full_x, full_y, calibration, energy, depth)
            value = float(image[y, x]) - (float(dark[y, x]) if dark is not None else 0)
            samples.append((q, value))
    samples.sort(key=lambda item: item[0])
    if grid is None:
        low, high = samples[0][0], samples[-1][0]
        ix, iy = round_away((x0+x1-1)/2), round_away((y0+y1-1)/2)
        x = start[0]+ix*group[0]+(group[0]-1)/2
        y = start[1]+iy*group[1]+(group[1]-1)/2
        estimated = abs(q_at(x+group[0], y+group[1], calibration, energy, depth)
                        - q_at(x, y, calibration, energy, depth))
        n = int(round_away((high-low)/estimated))+1
    else:
        low, high, n = grid
    step = (high-low)/(n-1)
    groups = itertools.groupby(samples, key=lambda item: int(round_away((item[0]-low)/step)))
    sums, counts, invalid = [0.]*n, [0]*n, [0]*n
    for index, members in groups:
        if not 0 <= index < n:
            continue
        values = [value for _, value in members]
        counts[index] = len(values)
        invalid[index] = sum(not math.isfinite(value) for value in values)
        sums[index] = math.nan if invalid[index] else factor*math.fsum(values)
    means = [sums[i]/counts[i] if counts[i] else math.nan for i in range(n)]
    return {"q_nm_inv": np.linspace(low, high, n), "intensity_mean": np.array(means),
            "intensity_sum": np.array(sums), "pixel_count": np.array(counts, dtype=np.int64),
            "nonfinite_pixel_count": np.array(invalid, dtype=np.int64)}
