# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""The indexing stages run separately: peak_search, Geometry.pixels_to_q, index_orientations."""

from dataclasses import asdict, fields
import inspect
from pathlib import Path
import re

import h5py
import numpy as np
import pytest

from conftest import requires_liblaue

from lauelab.indexing import (
    Indexer, IndexParams, InputError, PeakParams, PeakSearch, index_orientations,
    load_crystal, load_geometry, peak_search,
)


ROOT = Path(__file__).resolve().parents[1]
pytestmark = requires_liblaue
GEOMETRY = ROOT / "tests/data/geo/geoN_2022-03-29_14-15-05.xml"
CRYSTAL = ROOT / "tests/config/Ni.xml"
FRAMES = ROOT / "tests/data/synthetic/frames"
BASELINE = ROOT / "tests/data/synthetic/baseline"
# Settings of the lauego baselines; see tests/data/synthetic/generate.py.
PEAK_SETTINGS = dict(
    boxsize=18, max_rfactor=0.5, min_size=3, min_separation=20,
    threshold=None, threshold_ratio=4.0, max_peaks=200,
)
INDEX_SETTINGS = dict(
    kev_max_calc=17.2, kev_max_test=35.0, angle_tolerance_deg=0.1,
    cone_deg=72.0, hkl_prefer=(0, 0, 1),
)
STEMS = sorted(path.stem for path in FRAMES.glob("*.h5"))


def _table_after(path: Path, marker: str, delimiter=None) -> np.ndarray:
    lines = path.read_text().splitlines()
    start = next(index for index, line in enumerate(lines) if marker in line) + 1
    return np.loadtxt(lines[start:], delimiter=delimiter, ndmin=2)


def _frame(stem):
    with h5py.File(FRAMES / f"{stem}.h5") as source:
        detector = source["entry1/detector"]
        start = (int(detector["startx"][0]), int(detector["starty"][0]))
        group = (int(detector["binx"][0]), int(detector["biny"][0]))
        return source["entry1/data/data"][...], start, group


@pytest.mark.parametrize(
    ("function", "params"), [(peak_search, PeakParams), (index_orientations, IndexParams)]
)
def test_stage_keywords_mirror_the_parameter_dataclasses(function, params):
    keywords = {
        name: parameter for name, parameter in inspect.signature(function).parameters.items()
        if parameter.kind is inspect.Parameter.KEYWORD_ONLY and name != "mask"
    }

    assert list(keywords) == [item.name for item in fields(params)]
    for item in fields(params):
        assert keywords[item.name].default == item.default, item.name
        assert keywords[item.name].annotation == item.type, item.name


@pytest.mark.parametrize("stem", STEMS)
def test_peak_search_matches_lauego_peaksearch(stem):
    image, _, _ = _frame(stem)
    expected = _table_after(BASELINE / "peaks" / f"peaks_{stem}.txt", "$peakList")

    found = peak_search(image, **PEAK_SETTINGS)

    assert isinstance(found, PeakSearch)
    assert found.n_peaks == len(expected)
    assert "qhat" not in found.peaks.dtype.names
    np.testing.assert_allclose(found.peaks["fit_x"], expected[:, 0], atol=5e-4, rtol=0)
    np.testing.assert_allclose(found.peaks["fit_y"], expected[:, 1], atol=5e-4, rtol=0)
    np.testing.assert_allclose(found.peaks["intens"], expected[:, 2], atol=5e-4, rtol=1e-8)
    np.testing.assert_allclose(found.peaks["integral"], expected[:, 3], atol=5e-6, rtol=1e-8)


@pytest.mark.parametrize("stem", STEMS)
def test_peak_search_peaks_convert_to_lauego_q_vectors(stem):
    image, start, group = _frame(stem)
    expected = _table_after(BASELINE / "p2q" / f"p2q_{stem}.txt", "$N_Ghat+Intens", delimiter=",")

    found = peak_search(image, **PEAK_SETTINGS)
    qhat = load_geometry(GEOMETRY).pixels_to_q(found.peaks, start=start, group=group)

    # The reference q vectors were computed from peak positions rounded to 0.001 px.
    np.testing.assert_allclose(qhat, expected[:, :3], atol=2e-7, rtol=0)


@pytest.mark.parametrize(
    "index_file",
    sorted((BASELINE / "index").glob("index_*.txt")),
    ids=lambda path: path.stem.removeprefix("index_"),
)
def test_index_orientations_matches_lauego_euler_on_its_input_file(index_file):
    """Use the q vectors from the input file of the euler CLI, so that only crystal indexing is compared."""
    stem = index_file.stem.removeprefix("index_")
    qhat = _table_after(BASELINE / "p2q" / f"p2q_{stem}.txt", "$N_Ghat+Intens", delimiter=",")[:, :3]

    patterns = index_orientations(qhat, CRYSTAL, **INDEX_SETTINGS)

    text = index_file.read_text()
    lines = text.splitlines()
    assert len(patterns) == int(re.search(r"\$NpatternsFound\s+(\d+)", text).group(1))
    for index, pattern in enumerate(patterns):
        euler_text = re.search(rf"\$EulerAngles{index}\s+\{{([^}}]+)", text).group(1)
        expected_euler = np.asarray([float(value) for value in euler_text.split(",")])
        marker = next(i for i, line in enumerate(lines) if line.startswith(f"$array{index}"))
        count = int(lines[marker].split()[2])
        rows = [re.findall(r"[-+]?\d+(?:\.\d+)?", line) for line in lines[marker + 1:marker + 1 + count]]

        assert pattern.n_indexed == count
        np.testing.assert_array_equal(pattern.hkl, [[int(value) for value in row[4:7]] for row in rows])
        np.testing.assert_array_equal(pattern.pk_index, [int(row[-1]) for row in rows])
        np.testing.assert_allclose(pattern.energy_kev, [float(row[8]) for row in rows], atol=5e-4, rtol=0)
        np.testing.assert_allclose(pattern.euler_deg, expected_euler, atol=5e-4, rtol=0)


def _assert_patterns_identical(actual, expected):
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
        for item in fields(right):
            np.testing.assert_array_equal(getattr(left, item.name), getattr(right, item.name))


@pytest.mark.parametrize("stem", STEMS)
def test_stages_reproduce_indexer_index_exactly(stem):
    indexer = Indexer(
        GEOMETRY, CRYSTAL,
        peak_params=PeakParams(**PEAK_SETTINGS), index_params=IndexParams(**INDEX_SETTINGS),
    )
    result = indexer.index(FRAMES / f"{stem}.h5")

    found = peak_search(result.image, **asdict(indexer.peak_params))
    qhat = indexer.geometry.pixels_to_q(
        found.peaks, detector_index=indexer.detector_index,
        start=result.start, group=result.group, depth=result.depth,
    )
    patterns = index_orientations(qhat, indexer.crystal, **asdict(indexer.index_params))

    assert found.params == indexer.peak_params
    for name in found.peaks.dtype.names:
        np.testing.assert_array_equal(found.peaks[name], result.peaks[name])
    np.testing.assert_array_equal(qhat, result.peaks["qhat"])
    _assert_patterns_identical(patterns, result.patterns)
    for name in (
        "threshold_used", "threshold_ratio", "total_sum", "sum_above_threshold",
        "num_above_threshold", "peak_minwidth", "peak_maxwidth", "peak_max_cent_to_fit",
        "peak_boxsize",
    ):
        np.testing.assert_array_equal(getattr(found, name), getattr(result, name))


def test_peak_search_params_configure_an_indexer_identically():
    image, _, _ = _frame("synthetic_ni_two_grains")
    found = peak_search(image, **{**PEAK_SETTINGS, "boxsize": 18.0, "min_size": 3})

    assert found.params.boxsize == 18 and isinstance(found.params.boxsize, int)
    assert found.params.min_size == 3.0 and isinstance(found.params.min_size, float)
    assert Indexer(GEOMETRY, peak_params=found.params).peak_params == found.params


def test_peak_search_masks_like_indexer():
    image, _, _ = _frame("synthetic_ni_two_grains")
    mask = np.zeros(image.shape, dtype=bool)
    mask[:, :300] = True

    found = peak_search(image, mask=mask, **PEAK_SETTINGS)
    result = Indexer(GEOMETRY, peak_params=PeakParams(**PEAK_SETTINGS)).index(image, mask=mask)

    np.testing.assert_array_equal(found.peaks["fit_x"], result.peaks["fit_x"])
    assert found.total_sum == result.total_sum


@pytest.mark.parametrize(
    "settings",
    [
        dict(boxsize=0), dict(boxsize=2.5), dict(min_size=0), dict(min_size=True),
        dict(min_size=float("nan")), dict(min_separation=0), dict(max_peaks=0),
        dict(max_rfactor=0), dict(threshold_ratio=0), dict(peak_shape="L"),
    ],
)
def test_peak_search_rejects_settings_as_the_indexer_does(settings):
    with pytest.raises(InputError) as expected:
        Indexer(GEOMETRY, peak_params=PeakParams(**settings))
    with pytest.raises(InputError) as actual:
        peak_search(np.zeros((8, 8), dtype=np.uint16), **settings)

    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize(
    "settings",
    [
        dict(kev_max_calc=0), dict(cone_deg=-1), dict(hkl_prefer=(0, 1)),
        dict(hkl_prefer=(0, 0.5, 1)), dict(max_data=1), dict(max_data=3.5),
    ],
)
def test_index_orientations_rejects_settings_as_the_indexer_does(settings):
    with pytest.raises(InputError) as expected:
        Indexer(GEOMETRY, index_params=IndexParams(**settings))
    with pytest.raises(InputError) as actual:
        index_orientations(np.eye(3), CRYSTAL, **settings)

    assert str(actual.value) == str(expected.value)


@pytest.mark.parametrize(
    ("image", "mask", "message"),
    [
        (np.zeros((2, 3, 4), dtype=np.uint16), None, "2D array"),
        (np.zeros((8, 8), dtype=np.int64), None, "2D array"),
        (np.full((8, 8), np.nan), None, "non-finite"),
        (np.zeros((8, 8), dtype=np.uint16), np.zeros((8, 7)), "mask shape"),
    ],
)
def test_peak_search_rejects_invalid_images_and_masks(image, mask, message):
    with pytest.raises(InputError, match=message):
        peak_search(image, mask=mask)


def test_index_orientations_rejects_invalid_vectors_and_crystals():
    with pytest.raises(InputError, match=r"shape \(n, 3\)"):
        index_orientations(np.zeros((4, 2)), CRYSTAL)
    with pytest.raises(InputError, match="non-finite"):
        index_orientations([[0.0, 0.6, -0.8], [np.inf, 0.0, -1.0]], CRYSTAL)
    with pytest.raises(InputError, match="peak 1 has an invalid q vector"):
        index_orientations([[0.0, 0.6, -0.8], [0.0, 0.0, 0.0]], CRYSTAL)
    with pytest.raises(InputError, match="crystal must be a Crystal"):
        index_orientations(np.eye(3), None)


def test_index_orientations_returns_no_patterns_for_fewer_than_two_vectors():
    assert index_orientations(np.empty((0, 3)), CRYSTAL) == ()
    assert index_orientations([[0.0, 0.6, -0.8]], load_crystal(CRYSTAL)) == ()


def test_index_orientations_uses_only_the_first_max_data_rows():
    qhat = _table_after(
        BASELINE / "p2q" / "p2q_synthetic_ni_two_grains.txt", "$N_Ghat+Intens", delimiter=","
    )[:, :3]
    settings = {**INDEX_SETTINGS, "max_data": 30}

    limited = index_orientations(qhat, CRYSTAL, **settings)

    assert limited
    assert all(pattern.pk_index.max() < 30 for pattern in limited)
    _assert_patterns_identical(limited, index_orientations(qhat[:30], CRYSTAL, **settings))


def test_pixels_to_q_rejects_structured_peaks_without_coordinates():
    geometry = load_geometry(GEOMETRY)
    with pytest.raises(ValueError, match="fit_x and fit_y"):
        geometry.pixels_to_q(np.zeros(2, dtype=[("fit_x", float)]))
    with pytest.raises(ValueError, match="fit_x and fit_y"):
        geometry.pixels_to_q(np.zeros((2, 2), dtype=[("fit_x", float), ("fit_y", float)]))
    assert geometry.pixels_to_q(np.zeros(0, dtype=[("fit_x", float), ("fit_y", float)])).shape == (0, 3)


def test_stages_guide_example():
    """Run the steps of the examples in docs/guides/stages.md."""
    geometry = load_geometry(GEOMETRY)
    crystal = load_crystal(CRYSTAL)
    with h5py.File(FRAMES / "synthetic_ni_two_grains.h5") as source:
        image = source["entry1/data/data"][...]

    settings = dict(boxsize=18, max_rfactor=0.5, min_separation=20, threshold=None, max_peaks=200)
    thresholds = [
        peak_search(image, threshold_ratio=ratio, **settings).threshold_used
        for ratio in (3.0, 4.0, 6.0)
    ]
    found = peak_search(image, threshold_ratio=4.0, **settings)
    qhat = geometry.pixels_to_q(found.peaks)
    patterns = index_orientations(qhat, crystal, kev_max_calc=17.2, angle_tolerance_deg=0.1)
    bright = found.peaks["intens"] > np.median(found.peaks["intens"])
    bright_patterns = index_orientations(qhat[bright], crystal, kev_max_calc=17.2, angle_tolerance_deg=0.1)
    indexer = Indexer(
        geometry, crystal, peak_params=found.params,
        index_params=IndexParams(kev_max_calc=17.2, angle_tolerance_deg=0.1),
    )
    again = peak_search(image, **asdict(indexer.peak_params))

    assert thresholds == sorted(thresholds) and len(set(thresholds)) == 3
    assert found.n_peaks == 48
    assert len(patterns) == 2
    assert bright.sum() == 24 and len(bright_patterns) == 1
    assert bright_patterns[0].pk_index.max() < 24
    _assert_patterns_identical(indexer.index(image).patterns, patterns)
    np.testing.assert_array_equal(again.peaks, found.peaks)
