# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Generate only the energy fixtures; no Igor or production lauelab dependency.

Run with python -m tests.data.energy.generate [--lauego-root PATH]. Optional
source verification checks the pinned IPF hashes, without executing Igor.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

import h5py
import numpy as np

from .reference import HC, UPSTREAM, profile, q_at

CALIBRATION = {"pixels": [96, 80], "size_um": [19200., 16000.],
               "translation_um": [1500., -700., 40000.],
               "rotation_vector_rad": [0.04, -0.025, 0.015]}
ENERGY = 12.0
ROI = (5, 73, 7, 91)
GRID = (0., 20., 201)
FIELDS = ("q_nm_inv", "intensity_mean", "intensity_sum", "pixel_count", "nonfinite_pixel_count")


def geometry_xml(calibration=CALIBRATION, slot=2, detector_id="SYNTHETIC-Q"):
    nx, ny = calibration["pixels"]
    sx, sy = calibration["size_um"]
    rotation = " ".join(map(str, calibration["rotation_vector_rad"]))
    position = " ".join(str(value/1000) for value in calibration["translation_um"])
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<geoN xmlns="http:example.com/geoN"><Detectors Ndetectors="1">
<Detector N="{slot}"><Npixels>{nx} {ny}</Npixels>
<size unit="mm">{sx/1000} {sy/1000}</size>
<R unit="radian">{rotation}</R><P unit="mm">{position}</P>
<ID>{detector_id}</ID><note>Synthetic analytic ring phantom, not measured data</note>
</Detector></Detectors></geoN>
'''


def write_image(path, values, *, monitors=True, detector_id="SYNTHETIC-Q", exposure=2.):
    with h5py.File(path, "w") as target:
        target.attrs["file_time"] = "2026-01-01T00:00:00Z"
        data = target.create_dataset("entry1/data/data", data=values)
        data.attrs["units"] = "synthetic intensity"
        for name, value in {
            "entry1/title": "Synthetic monochromatic rings",
            "entry1/detector/ID": detector_id,
            "entry1/detector/exposure": exposure,
            "entry1/detector/startx": 0, "entry1/detector/starty": 0,
            "entry1/detector/binx": 1, "entry1/detector/biny": 1,
            "entry1/microDiffraction/MonoMode": "monochromatic",
            "entry1/sample/incident_energy": ENERGY,
        }.items():
            target.create_dataset(name, data=[value] if not isinstance(value, str) else value)
        if monitors:
            target.create_dataset("entry1/monitor/I0", data=[40000.])
            target.create_dataset("entry1/monitor/ScalerCountTime", data=[4.])


def generate(directory=None, *, lauego_root=None):
    directory = Path(directory) if directory is not None else Path(__file__).parent
    directory.mkdir(parents=True, exist_ok=True)
    if lauego_root is not None:
        for name, expected in UPSTREAM["sha256"].items():
            actual = hashlib.sha256((Path(lauego_root)/name).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"source hash mismatch: {name}")
    (directory/"geometry.xml").write_text(geometry_xml())
    ny, nx = CALIBRATION["pixels"][::-1]
    image, dark = np.empty((ny, nx)), np.empty((ny, nx))
    mask = np.zeros((ny, nx), dtype=np.uint8)
    mask[:, :4], mask[9:12, 19:23], mask[41:49, 58:71] = 255, 255, 1
    for y in range(ny):
        for x in range(nx):
            q = q_at(x, y, CALIBRATION, ENERGY)
            signal = 2+0.015*x+0.03*y + 110*math.exp(-0.5*((q-6.5)/0.3)**2) + 70*math.exp(-0.5*((q-13.)/0.45)**2)
            dark[y, x] = 3+(x % 5)*0.2+(y % 7)*0.1
            image[y, x] = signal/5 + dark[y, x]
    image[10, 20] = np.nan  # excluded before arithmetic; not a missing ring bin
    write_image(directory/"rings.h5", image)
    write_image(directory/"dark.h5", dark, monitors=False)
    write_image(directory/"mask.h5", mask, monitors=False)
    explicit = profile(image, CALIBRATION, ENERGY, roi=ROI, mask=mask, dark=dark, factor=5., grid=GRID)
    automatic = profile(image, CALIBRATION, ENERGY, roi=ROI, mask=mask, dark=dark, factor=5.)
    np.savez(directory/"reference.npz", **explicit, **{f"automatic_{key}": value for key, value in automatic.items()})
    provenance = {
        "data_kind": "synthetic analytic known-Q rings; NOT measured data or Igor-runtime output",
        "reference_kind": "independent scalar IPF-derived Python, plus analytic tests",
        "upstream": UPSTREAM, "hc_kev_nm": HC, "energy_kev": ENERGY,
        "geometry": CALIBRATION, "roi_y0_y1_x0_x1": ROI, "grid_first_last_count": GRID,
        "normalization_factor": 5., "ring_centers_nm_inv": [6.5, 13.],
        "random_seed": None, "noise": "none", "input_shape_y_x": [ny, nx],
        "comparison": {"rtol": 1e-12, "atol": 1e-12,
                       "reason": "float64 vector ray vs scalar atan2 and independent summation order; counts/membership exact"},
        "approved_departures": ["excluded pixels omitted from sums AND occupancy", "zero input remains valid",
                                "fractional group centers and depth used consistently", "strict correction metadata"],
        "array_conventions": "Igor image[p][q]=image[x][y]; NumPy image[y,x]; Igor first dimension / NumPy last dimension varies fastest",
        "generator_versions": {"numpy": np.__version__, "h5py": h5py.__version__},
    }
    (directory/"provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True)+"\n")
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lauego-root", type=Path)
    generate(lauego_root=parser.parse_args().lauego_root)
