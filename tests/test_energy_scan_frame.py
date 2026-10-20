# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Verify explicit plane selection with the actual reconstructed point writer."""

import h5py
import numpy as np
import pytest

from conftest import requires_liblaue
from lauelab.energy import load_q_distribution, q_distribution, save_q_distribution
from lauelab.indexing import InputError, ScanFrame
from lauelab.reconstruct import PointReader, reconstruct_point
from tests.data.reconstruction.generate_reference import DEPTH_RANGE_UM, GEOMETRY_FILE, write_input_file

pytestmark = requires_liblaue


@pytest.fixture(scope="module")
def point(tmp_path_factory):
    directory = tmp_path_factory.mktemp("energy-point")
    raw, output = directory/"raw.h5", directory/"point.h5"
    write_input_file(raw)
    with h5py.File(raw, "a") as source:
        for path, value in {"entry1/microDiffraction/MonoMode": "monochromatic",
                            "entry1/sample/incident_energy": [12.],
                            "entry1/detector/exposure": [2.],
                            "entry1/monitor/I0": [100., 200.],
                            "entry1/monitor/ScalerCountTime": [4.]}.items():
            if path in source:
                del source[path]
            source.create_dataset(path, data=value)
    outcome = reconstruct_point(raw, output, geometry=GEOMETRY_FILE, detector=0, point_id="energy-point",
                                depth_range=DEPTH_RANGE_UM, num_threads=1, output_pixel_type=5)
    assert outcome.status == "complete", outcome.error
    return output


def test_scan_frame_matches_one_owned_plane_and_keeps_processing(point, monkeypatch, tmp_path):
    with PointReader(point) as reader:
        index = min(2, reader.shape[0]-1)
        image, depth = reader.frame(index), reader.depth_um[index]
        start, group, detector_id = reader.start, reader.group, reader.detector_id
    reference = ScanFrame(point, "energy-point", index)
    calls = []
    real = PointReader.frame
    def frame(self, i):
        calls.append(i)
        return real(self, i)
    monkeypatch.setattr(PointReader, "frame", frame)
    actual = q_distribution(reference, geometry=GEOMETRY_FILE, normalization="none")
    assert calls == [index]
    expected = q_distribution(image, geometry=GEOMETRY_FILE, energy_kev=12, monochromatic=True,
                              detector_id=detector_id, normalization="none", start=start, group=group, depth=depth)
    for field in ("q_nm_inv", "intensity_mean", "intensity_sum", "pixel_count", "nonfinite_pixel_count"):
        np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
    p = actual.provenance
    assert p["source"]["depth_index"] == index and p["source"]["stored_depth_um"] == depth
    assert "prior_processing" in p["file_metadata"]
    reopened = load_q_distribution(save_q_distribution(actual, tmp_path/"profile.h5"))
    assert reopened.provenance == p
    np.testing.assert_array_equal(reopened.intensity_mean, actual.intensity_mean)
    assert calls == [index]  # Saving/loading does not reopen the source point.
    zero = q_distribution(reference, geometry=GEOMETRY_FILE, normalization="none", depth=0)
    assert zero.provenance["selection"]["depth_um"] == 0
    assert zero.provenance["source"]["stored_depth_um"] == depth


def test_scan_monitor_vectors_require_explicit_scalar_override(point):
    reference = ScanFrame(point, "energy-point", 0)
    with pytest.raises(InputError, match="i0_counts.*scalar"):
        q_distribution(reference, geometry=GEOMETRY_FILE)
    result = q_distribution(reference, geometry=GEOMETRY_FILE, metadata={"i0_counts": 100.})
    assert result.provenance["normalization"]["factor"] == 2000
    with pytest.raises(InputError, match="holds point"):
        q_distribution(ScanFrame(point, "wrong", 0), geometry=GEOMETRY_FILE)
