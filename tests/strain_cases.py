# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Controlled strain cases constructed independently from the fitted model."""

from dataclasses import replace
from itertools import product

import numpy as np
from scipy.spatial.transform import Rotation

from lauelab.indexing import Cell, Crystal, FrameResult, Geometry, Pattern
from lauelab.indexing.indexer import PEAK_DTYPE

HKL = np.array(
    [
        [1, 1, 1],
        [1, 2, 3],
        [2, 1, -3],
        [2, 3, 1],
        [4, 1, 2],
        [1, -2, 1],
        [-2, 3, 1],
        [2, 1, 1],
        [-1, 1, 1],
        [1, 3, -2],
    ]
)


def direct(cell):
    """Recover c-parallel-z direct columns by reverse Cholesky of the metric."""
    a, b, c = cell.a, cell.b, cell.c
    alpha, beta, gamma = np.radians([cell.alpha, cell.beta, cell.gamma])
    metric = np.array(
        [
            [a * a, a * b * np.cos(gamma), a * c * np.cos(beta)],
            [a * b * np.cos(gamma), b * b, b * c * np.cos(alpha)],
            [a * c * np.cos(beta), b * c * np.cos(alpha), c * c],
        ]
    )
    reverse = np.eye(3)[::-1]
    return reverse @ np.linalg.cholesky(reverse @ metric @ reverse).T @ reverse


def case(cell=None, *, scale=1, finite=False):
    cell = cell or Cell(0.35, 0.35, 0.35)
    a0 = direct(cell)
    size = 0.04 if finite else 0.001
    stretch = np.array(
        [
            [1 + size, size * 0.4, size * 0.1],
            [size * 0.4, 1 - size * 0.4, -size * 0.2],
            [size * 0.1, -size * 0.2, 1 + size * 0.2],
        ]
    )
    rotation = Rotation.from_rotvec([0.4, -0.1, 0.2]).as_matrix()
    b = 2 * np.pi * np.linalg.inv(scale * rotation @ stretch @ a0)
    qhat = HKL @ b
    qhat /= np.linalg.norm(qhat, axis=1)[:, None]
    initial = (
        2
        * np.pi
        * np.linalg.inv(Rotation.from_rotvec([0.401, -0.1001, 0.2001]).as_matrix() @ a0)
    )
    expected = stretch / np.linalg.det(stretch) ** (1 / 3) - np.eye(3)
    expected -= np.trace(expected) * np.eye(3) / 3
    return cell, b, initial, qhat, expected


def frame_case(tmp_path, *, group=(1, 1), start=(0, 0), depth=0):
    path = tmp_path / "geometry.xml"
    path.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<geoN><Detectors Ndetectors="1"><Detector N="2">
<Npixels>2048 2048</Npixels><size unit="mm">400 400</size>
<R unit="radian">0 0 0</R><P unit="mm">0 0 -100</P>
<ID>STRAIN-SYNTHETIC</ID></Detector></Detectors></geoN>""")
    geometry = Geometry(path)
    cell, b, initial, _, expected = case()
    h = np.array(list(product(range(-3, 4), repeat=3)))
    h = h[np.any(h, axis=1)]
    primitive = h // np.gcd.reduce(abs(h), axis=1)[:, None]
    _, indices = np.unique(primitive, axis=0, return_index=True)
    h = h[np.sort(indices)]
    xy = geometry.detector(2).q_to_pixel(h @ b, depth=depth, on_detector=True)
    keep = (
        np.isfinite(xy).all(axis=1)
        & (xy > np.array(start) + 4).all(axis=1)
        & (xy < 2000).all(axis=1)
    )
    h, xy = h[keep], xy[keep]
    take = np.linspace(0, len(h) - 1, 15, dtype=int)
    h, xy = h[take], xy[take]
    xy = (xy - np.array(start) - (np.array(group) - 1) / 2) / np.array(group)
    peaks = np.zeros(len(xy), dtype=PEAK_DTYPE)
    peaks["fit_x"], peaks["fit_y"] = xy.T
    peaks["qhat"] = [1, 0, 0]  # Stored directions deliberately stale.
    pat = Pattern(
        np.zeros(3),
        np.eye(3),
        initial,
        10,
        0.1,
        h,
        np.arange(len(h)),
        np.zeros(len(h)),
        np.ones(len(h)) * 10,
        np.ones(len(h)),
    )
    frame = FrameResult(
        peaks,
        (
            replace(
                pat,
                hkl=h[:2],
                pk_index=np.arange(2),
                err_deg=np.zeros(2),
                energy_kev=np.ones(2) * 10,
                pred_intens=np.ones(2),
            ),
            pat,
        ),
        1,
        1,
        1,
        len(h),
        0,
        0,
        metadata={"detector_id": "STRAIN-SYNTHETIC"},
        image_shape=(
            int((2048 - start[1]) / group[1]),
            int((2048 - start[0]) / group[0]),
        ),
        start=start,
        group=group,
        depth=depth,
    )
    crystal = Crystal("controlled", 1, cell, ())
    return geometry, crystal, frame, b, expected


def write_xml(path, frame):
    def values(a):
        return " ".join(format(float(x), ".17g") for x in a)

    patterns = []
    for i, pat in enumerate(frame.patterns):
        b = pat.reciprocal
        patterns.append(f'''<pattern num="{i}" Nindexed="{len(pat.hkl)}"><recip_lattice>
<astar>{values(b[0])}</astar><bstar>{values(b[1])}</bstar><cstar>{values(b[2])}</cstar></recip_lattice>
<hkl_s><h>{values(pat.hkl[:, 0])}</h><k>{values(pat.hkl[:, 1])}</k><l>{values(pat.hkl[:, 2])}</l>
<PkIndex>{values(pat.pk_index)}</PkIndex></hkl_s></pattern>''')
    path.write_text(f'''<AllSteps><step><depth>{frame.depth}</depth><detector><detectorID>STRAIN-SYNTHETIC</detectorID>
<Nx>{frame.image_shape[1]}</Nx><Ny>{frame.image_shape[0]}</Ny>
<ROI startx="{frame.start[0]}" starty="{frame.start[1]}" groupx="{frame.group[0]}" groupy="{frame.group[1]}"/>
<peaksXY Npeaks="{len(frame.peaks)}"><fitX>{values(frame.peaks["fit_x"])}</fitX>
<fitY>{values(frame.peaks["fit_y"])}</fitY></peaksXY></detector>
<indexing>{"".join(patterns)}</indexing></step></AllSteps>''')
    return path
