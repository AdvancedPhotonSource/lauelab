# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
from dataclasses import replace
import inspect
from pathlib import Path
import re
from xml.etree import ElementTree

import h5py
import numpy as np
import pytest

from conftest import requires_liblaue

from lauelab.analysis import simulate_reflections
from lauelab.indexing import (
    FrameMetadata, Indexer, IndexParams, PeakParams, index_frame, load_crystal,
    load_geometry,
)
from lauelab.visualization import (
    DataScope, ResultSet, load_visualization_xml, prepare_detector_view,
    prepare_pole_figure,
)


ROOT = Path(__file__).resolve().parents[1]
pytestmark = requires_liblaue
GEOMETRY = ROOT / "tests/data/geo/geoN_2022-03-29_14-15-05.xml"
CRYSTAL = ROOT / "tests/config/Ni.xml"
FRAMES = ROOT / "tests/data/synthetic/frames"
BASELINE = ROOT / "tests/data/synthetic/baseline"


def _table_after(path: Path, marker: str, delimiter=None) -> np.ndarray:
    lines = path.read_text().splitlines()
    start = next(index for index, line in enumerate(lines) if marker in line) + 1
    return np.loadtxt(lines[start:], delimiter=delimiter, ndmin=2)

def test_indexer_uses_unified_constructor_names():
    signature = inspect.signature(Indexer)
    assert tuple(signature.parameters)[:2] == ("geometry", "crystal")


def test_geometry_and_crystal_can_be_preloaded():
    geometry = load_geometry(GEOMETRY)
    crystal = load_crystal(CRYSTAL)
    indexer = Indexer(geometry, crystal)

    assert indexer.geometry is geometry
    assert indexer.geometry_path == GEOMETRY
    assert indexer.crystal is crystal
    assert not hasattr(indexer, "geo_file")
    assert not hasattr(indexer, "crystal_file")
    assert not hasattr(indexer, "crystal_model")


def test_public_crystal_is_editable_by_replacement():
    crystal = load_crystal(CRYSTAL)
    modified = replace(crystal, space_group=229)

    assert crystal.space_group == 225
    assert crystal.crystal_system == "cubic"
    assert modified.space_group == 229
    assert modified is not crystal
    Indexer(GEOMETRY, modified)


def test_indexer_matches_lauego_peak_and_q_reference():
    stem = "synthetic_ni_two_grains"
    with h5py.File(FRAMES / f"{stem}.h5") as source:
        image = source["entry1/data/data"][...]

    indexer = Indexer(
        GEOMETRY,
        peak_params=PeakParams(
            boxsize=18,
            max_rfactor=0.5,
            min_size=3,
            min_separation=20,
            threshold=None,
            threshold_ratio=4.0,
            peak_shape="Lorentzian",
            max_peaks=200,
        ),
    )
    result = indexer.index(image)
    expected_peaks = _table_after(BASELINE / "peaks" / f"peaks_{stem}.txt", "$peakList")
    expected_q = _table_after(BASELINE / "p2q" / f"p2q_{stem}.txt", "$N_Ghat+Intens", delimiter=",")

    assert result.indexed is False
    assert result.patterns == ()
    assert result.n_peaks == len(expected_peaks) == 48
    np.testing.assert_allclose(result.peaks["fit_x"], expected_peaks[:, 0], atol=5e-4, rtol=0)
    np.testing.assert_allclose(result.peaks["fit_y"], expected_peaks[:, 1], atol=5e-4, rtol=0)
    np.testing.assert_allclose(result.peaks["intens"], expected_peaks[:, 2], atol=5e-4, rtol=1e-8)
    np.testing.assert_allclose(result.peaks["integral"], expected_peaks[:, 3], atol=5e-6, rtol=1e-8)
    np.testing.assert_allclose(result.peaks["hwhm_x"], expected_peaks[:, 4], atol=5e-4, rtol=0)
    np.testing.assert_allclose(result.peaks["hwhm_y"], expected_peaks[:, 5], atol=5e-4, rtol=0)
    np.testing.assert_allclose(result.peaks["tilt"], expected_peaks[:, 6], atol=5e-4, rtol=0)
    np.testing.assert_allclose(result.peaks["chisq"], expected_peaks[:, 7], atol=5e-6, rtol=1e-5)
    # The reference q vectors were computed from peak positions rounded to 0.001 px.
    np.testing.assert_allclose(result.peaks["qhat"], expected_q[:, :3], atol=2e-7, rtol=0)


def test_max_peaks_is_an_exact_cap():
    indexer = Indexer(
        GEOMETRY,
        peak_params=PeakParams(
            boxsize=18,
            max_rfactor=0.5,
            min_size=3,
            min_separation=20,
            threshold=None,
            max_peaks=5,
        ),
    )

    result = indexer.index(FRAMES / "synthetic_ni_two_grains.h5")

    assert result.n_peaks == 5
    assert result.to_step().detector.peaksXY.NpeakMax == 5


def test_mask_values_are_treated_as_boolean():
    image = np.arange(1, 8 * 12 + 1, dtype=np.uint16).reshape(8, 12)
    mask = np.zeros_like(image, dtype=np.int64)
    mask[0, 0] = 256

    result = Indexer(GEOMETRY, peak_params=PeakParams(threshold=1000)).index(
        image, mask=mask
    )

    assert result.total_sum == np.sum(image) - image[0, 0]


def test_smoothing_preserves_raw_frame_statistics():
    image = np.arange(1, 8 * 12 + 1, dtype=np.uint16).reshape(8, 12)
    mask = np.zeros_like(image, dtype=bool)
    mask[0, 0] = True
    threshold = 50

    result = Indexer(
        GEOMETRY,
        peak_params=PeakParams(threshold=threshold, smooth=True),
    ).index(image, mask=mask)
    selected = image[~mask]

    assert result.total_sum == np.sum(selected)
    assert result.sum_above_threshold == np.sum(selected[selected > threshold])
    assert result.num_above_threshold == np.count_nonzero(selected > threshold)


def test_auto_threshold_with_smoothing_keeps_raw_sums():
    image = np.full((64, 64), 20, dtype=np.uint16)
    image[30:34, 30:34] = 500

    result = Indexer(
        GEOMETRY,
        peak_params=PeakParams(threshold=None, smooth=True),
    ).index(image)

    # threshold_used derives from the smoothed image; the sums must still
    # describe the raw input pixels.
    assert np.isfinite(result.threshold_used)
    assert result.total_sum == np.sum(image)
    raw_above = image[image > result.threshold_used]
    assert result.sum_above_threshold == np.sum(raw_above)
    assert result.num_above_threshold == raw_above.size


def test_blank_frame_with_auto_threshold_returns_empty_result():
    image = np.zeros((8, 12), dtype=np.uint16)
    result = Indexer(
        GEOMETRY,
        CRYSTAL,
        peak_params=PeakParams(threshold=None),
    ).index(image)

    assert result.n_peaks == 0
    assert result.n_patterns == 0
    assert result.indexed is False
    assert np.isnan(result.threshold_used)
    assert result.total_sum == 0
    assert result.sum_above_threshold == 0
    assert result.num_above_threshold == 0
    assert result.image is image


@pytest.mark.parametrize(
    ("threshold", "threshold_ratio", "expected"),
    [(None, None, 4.0), (None, 3.25, 3.25), (100.0, 3.25, np.nan)],
)
def test_xml_records_only_active_threshold_ratio(
    tmp_path, threshold, threshold_ratio, expected
):
    indexer = Indexer(
        GEOMETRY,
        peak_params=PeakParams(threshold=threshold, threshold_ratio=threshold_ratio),
    )

    result = indexer.index(np.zeros((8, 12), dtype=np.uint16))
    output = tmp_path / "result.xml"
    result.write_xml(output)
    peak_data = ElementTree.parse(output).find("./step/detector/peaksXY")

    if np.isnan(expected):
        assert np.isnan(result.threshold_ratio)
        assert peak_data.get("thresholdRatio") is None
    else:
        assert result.threshold_ratio == expected
        assert float(peak_data.get("thresholdRatio")) == expected


@pytest.mark.parametrize(
    "index_file",
    sorted((BASELINE / "index").glob("index_*.txt")),
    ids=lambda path: path.stem.removeprefix("index_"),
)
def test_indexer_matches_all_lauego_index_references(index_file):
    stem = index_file.stem.removeprefix("index_")
    with h5py.File(FRAMES / f"{stem}.h5") as source:
        image = source["entry1/data/data"][...]
    indexer = Indexer(
        GEOMETRY,
        CRYSTAL,
        peak_params=PeakParams(
            boxsize=18, max_rfactor=0.5, min_size=3, min_separation=20,
            threshold=None, threshold_ratio=4.0, max_peaks=200,
        ),
        index_params=IndexParams(
            kev_max_calc=17.2, kev_max_test=35.0, angle_tolerance_deg=0.1,
            cone_deg=72.0, hkl_prefer=(0, 0, 1),
        ),
    )

    result = indexer.index(image)
    text = index_file.read_text()
    expected_count = int(re.search(r"\$NpatternsFound\s+(\d+)", text).group(1))
    assert len(result.patterns) == expected_count

    lines = text.splitlines()
    for index, pattern in enumerate(result.patterns):
        euler_text = re.search(rf"\$EulerAngles{index}\s+\{{([^}}]+)", text).group(1)
        expected_euler = np.asarray([float(value) for value in euler_text.split(",")])
        rotation_text = re.search(rf"\$rotation_matrix{index}\s+(.+)", text).group(1)
        expected_rotation = np.asarray([
            float(value) for value in re.findall(r"[-+]?\d+\.\d+", rotation_text)
        ]).reshape((3, 3)).T
        marker = next(i for i, line in enumerate(lines) if line.startswith(f"$array{index}"))
        count = int(lines[marker].split()[2])
        rows = [re.findall(r"[-+]?\d+(?:\.\d+)?", line) for line in lines[marker + 1:marker + 1 + count]]
        expected_hkl = np.asarray([[int(value) for value in row[4:7]] for row in rows])
        expected_energy = np.asarray([float(row[8]) for row in rows])
        expected_indices = np.asarray([int(row[-1]) for row in rows])

        assert pattern.n_indexed == count
        np.testing.assert_array_equal(pattern.hkl, expected_hkl)
        np.testing.assert_array_equal(pattern.pk_index, expected_indices)
        np.testing.assert_allclose(pattern.energy_kev, expected_energy, atol=5e-4, rtol=0)
        np.testing.assert_allclose(pattern.euler_deg, expected_euler, atol=5e-4, rtol=0)
        np.testing.assert_allclose(pattern.rotation, expected_rotation, atol=2e-6, rtol=0)


def test_reported_hkl_and_energy_describe_the_same_reflection():
    indexer = Indexer(
        GEOMETRY,
        CRYSTAL,
        peak_params=PeakParams(
            boxsize=18, max_rfactor=0.5, min_size=3, min_separation=20,
            threshold=None, threshold_ratio=4.0, max_peaks=200,
        ),
        index_params=IndexParams(
            kev_max_calc=17.2, kev_max_test=35.0, angle_tolerance_deg=0.1,
            cone_deg=72.0, hkl_prefer=(0, 0, 1),
        ),
    )
    result = indexer.index(FRAMES / "synthetic_ni_two_grains.h5")

    reported_hkls = {
        tuple(hkl) for pattern in result.patterns for hkl in pattern.hkl
    }
    # In FCC Ni, (0, 2, 2) is the lowest allowed harmonic along (0, 1, 1).
    assert (0, 2, 2) in reported_hkls

    for pattern in result.patterns:
        reciprocal_per_angstrom = pattern.hkl @ pattern.reciprocal / 10.0
        sin_theta = -result.peaks["qhat"][pattern.pk_index, 2]
        expected_energy = (
            12.3984187
            * np.linalg.norm(reciprocal_per_angstrom, axis=1)
            / (4.0 * np.pi * sin_theta)
        )
        np.testing.assert_allclose(
            pattern.energy_kev, expected_energy, atol=0, rtol=1e-5
        )


def test_indexer_matches_peak_positions_for_all_synthetic_frames():
    indexer = Indexer(
        GEOMETRY,
        peak_params=PeakParams(
            boxsize=18,
            max_rfactor=0.5,
            min_size=3,
            min_separation=20,
            threshold=None,
            threshold_ratio=4.0,
            max_peaks=200,
        ),
    )

    for frame_file in sorted((FRAMES).glob("*.h5")):
        with h5py.File(frame_file) as source:
            result = indexer.index(source["entry1/data/data"][...])
        peaks_file = BASELINE / "peaks" / f"peaks_{frame_file.stem}.txt"
        lines = peaks_file.read_text().splitlines()
        start = next(index for index, line in enumerate(lines) if "$peakList" in line) + 1
        expected = np.loadtxt(lines[start:], ndmin=2) if lines[start:] else np.empty((0, 8))

        assert result.n_peaks == len(expected), frame_file.name
        if len(expected):
            actual_xy = np.column_stack((result.peaks["fit_x"], result.peaks["fit_y"]))
            np.testing.assert_allclose(actual_xy, expected[:, :2], atol=5e-4, rtol=0)


def _gaussian_frame(center, shape=(96, 128)):
    y, x = np.indices(shape)
    signal = 2000 * np.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2) / 8)
    return np.asarray(10 + signal, dtype=np.uint16)


@pytest.mark.parametrize("peak_shape", ["Lorentzian", "Gaussian"])
@pytest.mark.parametrize("center", [(6, 48), (121, 48), (64, 6), (64, 89)])
def test_edge_peak_is_recovered(peak_shape, center):
    indexer = Indexer(
        GEOMETRY,
        peak_params=PeakParams(
            boxsize=8,
            min_size=2,
            min_separation=5,
            threshold=100.0,
            peak_shape=peak_shape,
        ),
    )

    result = indexer.index(_gaussian_frame(center))

    assert result.n_peaks == 1
    np.testing.assert_allclose(
        [result.peaks["fit_x"][0], result.peaks["fit_y"][0]],
        center,
        atol=0.5,
        rtol=0,
    )
    assert result.peaks["hwhm_x"][0] == pytest.approx(
        result.peaks["hwhm_y"][0], rel=0.05
    )
    assert np.isfinite(result.peaks["integral"][0])


def test_edge_peak_search_is_deterministic():
    image = _gaussian_frame((6, 48))
    indexer = Indexer(
        GEOMETRY,
        peak_params=PeakParams(
            boxsize=8,
            max_rfactor=10.0,
            min_size=2,
            min_separation=5,
            threshold=100.0,
        ),
    )

    results = [indexer.index(image) for _ in range(5)]

    assert [result.n_peaks for result in results] == [results[0].n_peaks] * 5
    for result in results[1:]:
        np.testing.assert_array_equal(result.peaks, results[0].peaks)


def test_index_frame_exposes_frame_options():
    signature = inspect.signature(index_frame)
    assert tuple(signature.parameters) == (
        "frame",
        "geometry",
        "crystal",
        "peak_params",
        "index_params",
        "detector_index",
        "detector_id",
        "cosmic_filter",
        "start",
        "group",
        "depth",
        "mask",
        "metadata",
        "keep_image",
    )


def test_index_frame_and_image_retention_defaults():
    image = np.zeros((8, 12), dtype=np.uint16)
    result = index_frame(
        image,
        geometry=GEOMETRY,
    )
    assert result.image is image
    assert result.n_patterns == 0
    assert result.indexed_peak_indices.size == 0
    np.testing.assert_array_equal(result.unindexed_peak_indices, np.arange(result.n_peaks))

    indexer = Indexer(GEOMETRY)
    batch = indexer.index_many([image, image])
    assert all(item.image is None for item in batch)


def test_result_is_independent_of_indexer_configuration():
    indexer = Indexer(
        GEOMETRY,
        peak_params=PeakParams(boxsize=8, min_size=6),
    )
    result = indexer.index(np.zeros((8, 12), dtype=np.uint16))
    peak_data = result.to_step().detector.peaksXY
    first_step = result.to_step()
    first_step.detector.peaksXY.boxsize = 123
    indexer.peak_params = replace(indexer.peak_params, boxsize=99, min_size=99)

    assert result.peak_boxsize == peak_data.boxsize == 8
    assert result.peak_minwidth == peak_data.minwidth == 1.5
    assert result.peak_maxwidth == peak_data.maxwidth == 12.0
    assert result.peak_max_cent_to_fit == peak_data.maxCentToFit == 8.0
    assert result.to_step().detector.peaksXY.boxsize == 8


def test_indexer_requires_no_metadata_for_in_memory_frame():
    indexer = Indexer(GEOMETRY)
    result = indexer.index(np.zeros((8, 12), dtype=np.uint16))
    step = result.to_step()

    assert result.metadata == {}
    assert step.detector.Nx == 12
    assert step.detector.Ny == 8
    assert step.detector.detectorID == indexer.detector_id


def test_indexer_accepts_optional_manual_metadata():
    indexer = Indexer(GEOMETRY)
    image = np.zeros((8, 12), dtype=np.uint16)
    image[2:6, 3:7] = 100
    result = indexer.index(
        image,
        start=(100, 200),
        group=(2, 3),
        metadata=FrameMetadata(
            sample_name="Ni foil",
            scan_number=42,
            detector_id=indexer.detector_id,
            exposure_seconds=0.25,
        ),
    )
    step = result.to_step()

    assert step.sampleName == "Ni foil"
    assert step.scanNum == 42
    assert step.detector.detectorID == indexer.detector_id
    assert step.detector.Nx == 12
    assert step.detector.Ny == 8
    assert step.detector.roi.startx == 100
    assert step.detector.roi.endx == 123
    assert step.detector.roi.starty == 200
    assert step.detector.roi.endy == 223


def test_xml_omits_missing_optional_metadata(tmp_path):
    indexer = Indexer(GEOMETRY)
    result = indexer.index(np.zeros((8, 12), dtype=np.uint16))
    path = tmp_path / "minimal.xml"

    result.write_xml(path)
    root = ElementTree.parse(path).getroot()
    step = root.find("step")
    detector = step.find("detector")

    for name in ("scanNum", "Xsample", "Ysample", "Zsample", "depth", "energy"):
        assert step.find(name) is None
    assert detector.find("inputImage") is None
    assert detector.find("exposure") is None


def test_indexer_accepts_partial_hdf5_metadata(tmp_path):
    path = tmp_path / "minimal.h5"
    with h5py.File(path, "w") as output:
        output.create_dataset("entry1/data/data", data=np.zeros((8, 12), dtype=np.uint16))
        output.create_dataset("entry1/sample/name", data=np.asarray([b"partial"]))

    indexer = Indexer(GEOMETRY)
    result = indexer.index(path)
    step = result.to_step()

    assert result.metadata == {"sample_name": "partial"}
    assert step.sampleName == "partial"
    assert step.detector.Nx == 12
    assert step.detector.Ny == 8


def test_indexer_indexes_file_and_builds_step(tmp_path):
    stem = "synthetic_ni_two_grains"
    indexer = Indexer(
        GEOMETRY,
        CRYSTAL,
        peak_params=PeakParams(
            boxsize=18, max_rfactor=0.5, min_size=3, min_separation=20,
            threshold=None, threshold_ratio=4.0, max_peaks=200,
        ),
        index_params=IndexParams(
            kev_max_calc=17.2, kev_max_test=35.0, angle_tolerance_deg=0.1,
            cone_deg=72.0, hkl_prefer=(0, 0, 1),
        ),
    )
    result = indexer.index(FRAMES / f"{stem}.h5")
    step = result.to_step()
    output = tmp_path / "result.xml"
    result.write_xml(output)

    assert result.input_image.endswith(f"{stem}.h5")
    assert step.sampleName == "synthetic Ni"
    assert step.detector.detectorID == "PE1621 723-3335"
    assert step.detector.peaksXY.Npeaks == 48
    assert step.indexing.NpatternsFound == 2
    assert step.indexing.Nindexed == 47
    assert output.read_text().startswith('<?xml version="1.0" ?>')


def test_reciprocal_convention_is_consistent_across_live_xml_and_visualization(
    tmp_path,
):
    stem = "synthetic_ni_two_grains"
    indexer = Indexer(
        GEOMETRY,
        CRYSTAL,
        peak_params=PeakParams(
            boxsize=18, max_rfactor=0.5, min_size=3, min_separation=20,
            threshold=None, threshold_ratio=4.0, max_peaks=200,
        ),
        index_params=IndexParams(
            kev_max_calc=17.2, kev_max_test=35.0, angle_tolerance_deg=0.1,
            cone_deg=72.0, hkl_prefer=(0, 0, 1),
        ),
    )
    result = indexer.index(FRAMES / f"{stem}.h5")
    live = ResultSet.from_indexer(indexer, (result,))
    live_data = live.to_visualization()

    output = tmp_path / "result.xml"
    result.write_xml(output)
    xml_data = load_visualization_xml(output, geometry=indexer.geometry)

    np.testing.assert_allclose(
        live_data.pattern_reciprocals,
        xml_data.pattern_reciprocals,
        atol=0,
        rtol=0,
    )
    np.testing.assert_allclose(
        live_data.pattern_rotations,
        xml_data.pattern_rotations,
        atol=1e-12,
        rtol=0,
    )
    np.testing.assert_allclose(live_data.peaks["background"], xml_data.peaks["background"])
    np.testing.assert_allclose(
        live_data.assignment_error_deg, xml_data.assignment_error_deg
    )
    np.testing.assert_allclose(
        live_data.assignment_energy_kev, xml_data.assignment_energy_kev
    )
    np.testing.assert_allclose(
        live_data.assignment_predicted_intensity,
        xml_data.assignment_predicted_intensity,
    )
    for pattern in result.patterns:
        calculated = pattern.hkl @ pattern.reciprocal
        calculated /= np.linalg.norm(calculated, axis=1, keepdims=True)
        measured = result.peaks["qhat"][pattern.pk_index]
        cosine = np.clip(np.sum(calculated * measured, axis=1), -1, 1)
        angles = np.degrees(np.arccos(cosine))
        # The native indexer and this NumPy recomputation differ in how they
        # normalize and accumulate before the arccos, which near 0.1 degree
        # amplifies a ~1e-13 cosine difference to ~2e-8 degrees. Observed
        # deviations reach 2.05e-8 degrees on the synthetic frames, so the
        # earlier 2e-8 tolerance sat at the noise level and failed on other
        # hosts. 5e-8 degrees (about 1e-9 radians) still verifies the
        # convention: a wrong basis or a missing 2*pi would be off by degrees,
        # and the indexing tolerance itself is 0.1 degree.
        np.testing.assert_allclose(angles, pattern.err_deg, atol=5e-8, rtol=0)
        assert np.max(angles) < indexer.index_params.angle_tolerance_deg

    pattern = result.patterns[0]
    simulated = simulate_reflections(
        indexer.crystal,
        pattern.reciprocal,
        indexer.detector,
        energy_range_kev=(6.0, 35.0),
        depth=0.0 if result.depth is None else result.depth,
    )
    np.testing.assert_allclose(simulated.q, simulated.hkl @ pattern.reciprocal)
    indexed_divisors = np.gcd.reduce(np.abs(pattern.hkl), axis=1)
    simulated_divisors = np.gcd.reduce(np.abs(simulated.hkl), axis=1)
    indexed_directions = {
        tuple(row) for row in pattern.hkl // indexed_divisors[:, None]
    }
    simulated_directions = {
        tuple(row) for row in simulated.hkl // simulated_divisors[:, None]
    }
    assert indexed_directions <= simulated_directions

    live_view = prepare_detector_view(live, frame_id=0, patterns="all")
    xml_view = prepare_detector_view(xml_data, frame_id=0, patterns="all")
    for live_pattern, xml_pattern in zip(
        live_view.patterns, xml_view.patterns, strict=True
    ):
        np.testing.assert_allclose(
            live_pattern.predicted_xy, xml_pattern.predicted_xy, atol=1e-12, rtol=0
        )
        measured = live_view.measured_xy[live_pattern.measured_peak_indices]
        error = np.linalg.norm(live_pattern.predicted_xy - measured, axis=1)
        assert np.max(error) < 0.5

    scope = DataScope(patterns="all", min_indexed=0)
    live_poles = prepare_pole_figure(live, scope=scope)
    xml_poles = prepare_pole_figure(xml_data, scope=scope)
    np.testing.assert_allclose(
        live_poles.points, xml_poles.points, atol=1e-12, rtol=0
    )
    np.testing.assert_array_equal(
        live_poles.pattern_indices, xml_poles.pattern_indices
    )


def test_indexer_selects_detector_by_id():
    indexer = Indexer(
        GEOMETRY,
        detector_id="PE1621 723-3335",
    )
    assert indexer.detector_index == 0

    with pytest.raises(ValueError, match="not present"):
        Indexer(GEOMETRY, detector_id="missing")


def test_indexer_replace_preserves_detector_slot_unless_id_is_explicit(tmp_path):
    original_id = "PE1621 723-3335"
    other_id = "PE0822 883-4841"
    reordered_text = GEOMETRY.read_text().replace(original_id, "temporary detector")
    reordered_text = reordered_text.replace(other_id, original_id)
    reordered_text = reordered_text.replace("temporary detector", other_id)
    reordered = tmp_path / "reordered.xml"
    reordered.write_text(reordered_text)

    indexer = Indexer(GEOMETRY, detector_id=original_id)
    replaced = indexer.replace(geometry=reordered)

    assert replaced.detector_id == other_id
    assert replaced.detector.detector_id == other_id
    assert replaced.detector_index == 0

    by_id = indexer.replace(geometry=reordered, detector_id=original_id)
    assert by_id.detector_id == original_id
    assert by_id.detector_index == 1

    crystal = load_crystal(CRYSTAL)
    assert indexer.replace(crystal=crystal).crystal is crystal
    assert indexer.replace(cosmic_filter=True).detector_index == indexer.detector_index


def test_indexer_validates_detector_id_metadata_for_all_input_kinds(tmp_path):
    path = tmp_path / "wrong-detector.h5"
    with h5py.File(path, "w") as output:
        output.create_dataset("entry1/data/data", data=np.zeros((8, 12), dtype=np.uint16))
        output.create_dataset("entry1/detector/ID", data=np.asarray([b"wrong detector"]))

    indexer = Indexer(GEOMETRY)
    with pytest.raises(ValueError, match="does not match selected detector"):
        indexer.index(path)
    with pytest.raises(ValueError, match="does not match selected detector"):
        indexer.index(
            np.zeros((8, 12), dtype=np.uint16),
            metadata={"detector_id": "wrong detector"},
        )


def test_explicit_detector_id_metadata_overrides_hdf5_metadata(tmp_path):
    path = tmp_path / "wrong-detector.h5"
    with h5py.File(path, "w") as output:
        output.create_dataset("entry1/data/data", data=np.zeros((8, 12), dtype=np.uint16))
        output.create_dataset("entry1/detector/ID", data=np.asarray([b"wrong detector"]))

    indexer = Indexer(GEOMETRY)
    result = indexer.index(path, metadata={"detector_id": indexer.detector_id})

    assert result.metadata["detector_id"] == indexer.detector_id
