# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Analytic, source-derived and adapter evidence; no Igor-runtime goldens."""

import hashlib
import json
import math
from pathlib import Path
import pickle
import shutil

import h5py
import numpy as np
import pytest

from conftest import requires_liblaue
from lauelab.energy import QDistribution, q_distribution
from lauelab.energy import _core
from lauelab.indexing import FrameMetadata, Geometry, InputError, ScanFrame, load_mask
from lauelab._frame import read_h5_frame
from tests.data.energy.generate import CALIBRATION, ENERGY, GRID, ROI, geometry_xml, generate
from tests.data.energy.reference import profile as scalar_profile

pytestmark = requires_liblaue
DATA = Path(__file__).parent/"data/energy"
FIELDS = ("q_nm_inv", "intensity_mean", "intensity_sum", "pixel_count", "nonfinite_pixel_count")


@pytest.fixture
def flat_geometry(tmp_path):
    calibration = {"pixels": [3, 3], "size_um": [300000., 300000.],
                   "translation_um": [0., 0., 100000.], "rotation_vector_rad": [0., 0., 0.]}
    path = tmp_path/"flat.xml"
    path.write_text(geometry_xml(calibration, slot=0))
    return path


def flat(image, geometry, **changes):
    options = dict(geometry=geometry, energy_kev=1.239841857, monochromatic=True,
                   normalization="none", q_first=0., q_last=6., q_points=7)
    return q_distribution(image, **(options | changes))


def rings(**changes):
    options = dict(geometry=DATA/"geometry.xml", mask=load_mask(DATA/"mask.h5"),
                   dark=DATA/"dark.h5", roi=ROI, q_first=GRID[0], q_last=GRID[1], q_points=GRID[2])
    return q_distribution(DATA/"rings.h5", **(options | changes))


def test_flat_detector_has_closed_form_membership_and_constant_intensity(flat_geometry):
    # lambda=1 nm, pitch=distance. Q is 0, 4*pi*sin(atan(1)/2),
    # or 4*pi*sin(atan(sqrt(2))/2), independently of production geometry.
    axial = 4*math.pi*math.sin(math.atan(1)/2)
    corner = 4*math.pi*math.sin(math.atan(math.sqrt(2))/2)
    assert 4.5 < axial < 5.5 < corner < 6.5
    result = flat(np.full((3, 3), 7, np.uint32), flat_geometry)
    np.testing.assert_array_equal(result.pixel_count, [1, 0, 0, 0, 0, 4, 4])
    np.testing.assert_array_equal(result.intensity_sum, [7, 0, 0, 0, 0, 28, 28])
    np.testing.assert_array_equal(result.intensity_mean[[0, 5, 6]], [7, 7, 7])
    assert np.isnan(result.intensity_mean[1:5]).all()


def test_rectangular_asymmetric_grouped_image_matches_literal_coordinates(tmp_path):
    # A transpose, wrong flattening order, swapped group, or missing half group
    # moves unequal intensities to different bins. Square constant data cannot
    # establish these conventions.
    calibration = {"pixels": [7, 5], "size_um": [700000., 500000.],
                   "translation_um": [10000., -20000., 100000.], "rotation_vector_rad": [0., 0., 0.]}
    path = tmp_path/"rectangle.xml"
    path.write_text(geometry_xml(calibration, slot=2))
    image = np.array([[3., 13., 23.], [33., 43., 53.]])
    expected_counts, expected_sums = np.zeros(49, dtype=np.int64), np.zeros(49)
    for row, fy in enumerate([0.5, 2.5]):
        for column, fx in enumerate([1.5, 3.5, 5.5]):
            transverse = math.hypot((fx-3)*100000+10000, (fy-2)*100000-20000)
            q = 4*math.pi*math.sin(math.atan2(transverse, 96500)/2)
            index = math.floor(q/0.25+0.5)
            expected_counts[index] += 1
            expected_sums[index] += image[row, column]
    result = q_distribution(image, geometry=path, start=(1, 0), group=(2, 2), depth=3500,
                            energy_kev=1.239841857, monochromatic=True, normalization="none",
                            q_first=0, q_last=12, q_points=49)
    np.testing.assert_array_equal(result.pixel_count, expected_counts)
    np.testing.assert_array_equal(result.intensity_sum, expected_sums)
    assert result.provenance["geometry"]["detector_index"] == 2


def test_igor_rounding_and_adjacent_representable_half_values():
    values = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5])
    np.testing.assert_array_equal(_core._round_igor(values), [-3, -2, -1, 1, 2, 3])
    np.testing.assert_array_equal(_core._round_igor([np.nextafter(.5, 0), np.nextafter(-.5, 0)]), [0, 0])


def test_explicit_grid_uses_centers_and_drops_outer_half_ties(flat_geometry, monkeypatch):
    monkeypatch.setattr(_core, "_q", lambda pixels, *args: np.array([1.5, 2.5, 3.5])[pixels[:, 0].astype(int)])
    result = flat(np.array([[10, 20, 30]]), flat_geometry, q_first=2, q_last=4, q_points=3)
    np.testing.assert_array_equal(result.pixel_count, [0, 1, 1])
    np.testing.assert_array_equal(result.intensity_sum, [0, 20, 30])
    assert result.provenance["diagnostics"]["outside_grid_pixel_count"] == 1
    monkeypatch.setattr(_core, "_q", lambda pixels, *args: np.full(len(pixels), 4.5))
    with pytest.raises(InputError, match="no selected pixels"):
        flat(np.ones((1, 3)), flat_geometry, q_first=2, q_last=4, q_points=3)


@pytest.mark.parametrize("dtype", [np.uint8, np.int8, np.uint16, np.int16, np.uint32,
                                  np.int32, np.uint64, np.int64, np.float16, np.float32, np.float64, np.longdouble])
def test_real_dtypes_are_accepted_without_indexer_whitelist(flat_geometry, dtype):
    result = flat(np.full((3, 3), 7, dtype=dtype), flat_geometry)
    np.testing.assert_array_equal(result.intensity_mean[[0, 5, 6]], [7., 7., 7.])


def test_wide_integer_precision_is_documented_not_a_dtype_error(flat_geometry):
    value = 2**53+1
    result = flat(np.full((3, 3), value, dtype=np.uint64), flat_geometry)
    assert result.intensity_mean[0] == float(value)
    assert result.provenance["source"]["dtype"] == np.dtype(np.uint64).str
    assert result.provenance["arithmetic_dtype"] == "float64"


@pytest.mark.parametrize("dtype", [complex, object, str, bool])
def test_non_real_numeric_image_contract(flat_geometry, dtype):
    with pytest.raises(InputError, match="2D real"):
        flat(np.ones((3, 3), dtype=dtype), flat_geometry)


def test_zero_negative_dark_and_per_bin_nonfinite_behavior(flat_geometry):
    counts = np.array([1, 0, 0, 0, 0, 4, 4])
    result = flat(np.zeros((3, 3)), flat_geometry)
    np.testing.assert_array_equal(result.pixel_count, counts)
    assert (result.intensity_mean[counts > 0] == 0).all()
    result = flat(np.zeros((3, 3), dtype=np.uint16), flat_geometry, dark=np.ones((3, 3), dtype=np.uint16))
    assert (result.intensity_mean[counts > 0] == -1).all()
    image = np.ones((3, 3))
    image[0, 1], image[2, 1] = np.nan, np.inf
    result = flat(image, flat_geometry)
    assert result.pixel_count[5] == 4 and result.nonfinite_pixel_count[5] == 2
    assert np.isnan(result.intensity_mean[5]) and np.isnan(result.intensity_sum[5])
    assert result.intensity_mean[6] == 1 and result.intensity_mean[0] == 1
    mask = np.zeros((3, 3)); mask[0, 1], mask[2, 1] = np.nan, 255
    result = flat(image, flat_geometry, mask=mask)
    assert result.pixel_count[5] == 2 and result.intensity_mean[5] == 1
    assert result.nonfinite_pixel_count.sum() == 0
    result = flat(np.full((3, 3), np.inf), flat_geometry)
    assert result.pixel_count.sum() == result.nonfinite_pixel_count.sum() == 9
    assert np.isnan(result.intensity_mean).all()


def test_nonfinite_dark_is_reported_and_masked_before_arithmetic(flat_geometry):
    dark = np.zeros((3, 3)); dark[1, 1] = np.nan
    result = flat(np.ones((3, 3)), flat_geometry, dark=dark)
    assert result.nonfinite_pixel_count[0] == 1
    mask = np.zeros((3, 3), bool); mask[1, 1] = True
    result = flat(np.ones((3, 3)), flat_geometry, dark=dark, mask=mask)
    assert result.pixel_count[0] == result.nonfinite_pixel_count[0] == 0


@pytest.mark.parametrize("mode, factor", [("none", 1), (None, 1), ("exposure", .5), ("i0", 20000)])
def test_normalization_has_hand_computed_factors(flat_geometry, mode, factor):
    metadata = FrameMetadata(exposure_seconds=2, i0_counts=10, scaler_count_time_seconds=4)
    result = flat(np.full((3, 3), 7), flat_geometry, metadata=metadata, normalization=mode)
    assert result.intensity_mean[0] == 7*factor
    assert result.provenance["normalization"]["factor"] == factor
    if mode == "i0":
        assert result.provenance["normalization"]["i0_gain_assumed"] is True
        gained = flat(np.full((3, 3), 7), flat_geometry, normalization=mode,
                      metadata={**metadata.as_dict(), "i0_gain": 5e8})
        assert gained.intensity_mean[0] == 14*factor


@pytest.mark.parametrize("field", ["exposure_seconds", "i0_counts", "scaler_count_time_seconds", "i0_gain"])
@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, [2, 3], "4", True])
def test_required_correction_fields_never_silently_fallback(flat_geometry, field, value):
    meta = dict(exposure_seconds=2, i0_counts=10, scaler_count_time_seconds=4, i0_gain=1e9)
    meta[field] = value
    with pytest.raises(InputError, match=field):
        flat(np.ones((3, 3)), flat_geometry, normalization="i0", metadata=meta)


def test_missing_monitors_require_override_or_explicit_raw_mode(flat_geometry):
    with pytest.raises(InputError, match="exposure_seconds"):
        flat(np.ones((3, 3)), flat_geometry, normalization="i0")
    with pytest.raises(InputError, match="exposure_seconds"):
        flat(np.ones((3, 3)), flat_geometry, normalization="exposure")


@pytest.mark.parametrize("energy", [0, -1, np.nan, np.inf, [12, 13], True])
def test_invalid_energy_is_not_a_valid_scattering_calibration(flat_geometry, energy):
    with pytest.raises(InputError, match="energy_kev"):
        flat(np.ones((3, 3)), flat_geometry, energy_kev=energy)


def test_monochromatic_qualification_and_white_override(flat_geometry):
    image = np.ones((3, 3))
    with pytest.raises(InputError, match="monochromatic illumination"):
        flat(image, flat_geometry, monochromatic=None)
    with pytest.raises(InputError, match="monochromatic illumination"):
        flat(image, flat_geometry, monochromatic=np.bool_(False), metadata={"mono_mode": "monochromatic"})
    with pytest.raises(InputError, match="white-beam metadata"):
        flat(image, flat_geometry, metadata={"mono_mode": "white slitted"})
    result = flat(image, flat_geometry, monochromatic=np.bool_(True),
                  metadata={"mono_mode": "white slitted"}, monochromatic_reason="Header is stale; monochromator selected")
    assert result.provenance["beam"]["reason"]


def test_degenerate_automatic_grid_and_allocation_limit(flat_geometry):
    automatic = dict(q_first=None, q_last=None, q_points=None)
    with pytest.raises(InputError, match="degenerate"):
        flat(np.ones((1, 1)), flat_geometry, **automatic)
    with pytest.raises(InputError, match="max_bins"):
        flat(np.ones((3, 3)), flat_geometry, q_points=1_000_001)
    with pytest.raises(InputError, match="max_bins"):
        rings(max_bins=2, **automatic)
    result = flat(np.ones((3, 3)), flat_geometry, max_bins=10)
    assert len(result.q_nm_inv) == 7


@pytest.mark.parametrize("kwargs", [dict(q_first=None), dict(q_points=1), dict(q_points=True),
                                    dict(q_first=-1), dict(q_last=0), dict(q_last=np.inf),
                                    dict(roi=(0, 4, 0, 3)), dict(roi=(0., 2, 0, 2)),
                                    dict(roi=(0, 0, 0, 1)), dict(mask=np.zeros((1, 1))),
                                    dict(dark=np.zeros((1, 1))), dict(start=(-1, 0)),
                                    dict(group=(0, 1)), dict(depth=np.inf),
                                    dict(start=(2**64, 0)), dict(group=(2**63, 1))])
def test_invalid_selection_and_grid_raise_clear_errors(flat_geometry, kwargs):
    with pytest.raises(InputError):
        flat(np.ones((3, 3)), flat_geometry, **kwargs)


def test_empty_selection_vs_valid_zero_image(flat_geometry):
    with pytest.raises(InputError, match="no unmasked"):
        flat(np.zeros((3, 3)), flat_geometry, mask=np.ones((3, 3)))
    with pytest.raises(InputError, match="no selected pixels"):
        flat(np.zeros((3, 3)), flat_geometry, q_first=100, q_last=110)


def test_stacks_are_not_implicitly_squeezed_or_sliced(flat_geometry, tmp_path):
    values = np.ones((1, 3, 3))
    with pytest.raises(InputError, match="2D"):
        flat(values, flat_geometry)
    path = tmp_path/"stack.h5"
    with h5py.File(path, "w") as source:
        source.create_dataset("entry1/data/data", data=values)
    with pytest.raises(InputError, match="2D"):
        flat(path, flat_geometry)


def test_finite_arithmetic_overflow_is_not_source_missing_data(flat_geometry):
    with pytest.raises(InputError, match="overflows"):
        flat(np.full((3, 3), 1e308), flat_geometry)
    with pytest.raises(InputError, match="overflows"):
        flat(np.full((3, 3), 1e308), flat_geometry, dark=np.full((3, 3), -1e308))
    if np.finfo(np.longdouble).max > np.finfo(float).max:
        with pytest.raises(InputError, match="overflows"):
            flat(np.full((3, 3), np.finfo(np.longdouble).max), flat_geometry)


@pytest.mark.parametrize("automatic", [False, True])
def test_retained_ring_fixture_matches_independent_scalar_reference(automatic):
    changes = dict(q_first=None, q_last=None, q_points=None) if automatic else {}
    result = rings(**changes)
    ref = np.load(DATA/"reference.npz")
    for field in FIELDS:
        expected = ref[("automatic_" if automatic else "")+field]
        if "count" in field:
            np.testing.assert_array_equal(getattr(result, field), expected)
        else:
            np.testing.assert_allclose(getattr(result, field), expected, rtol=1e-12, atol=1e-12)


def test_grouping_crop_depth_and_rotation_match_scalar_reference():
    image = np.arange(12*18, dtype=np.float64).reshape(12, 18)-50
    mask = np.zeros(image.shape); mask[3:7, 5:10] = 1
    dark = np.arange(image.size).reshape(image.shape)*.01
    roi = (1, 11, 2, 17)
    kwargs = dict(start=(4, 7), group=(2, 3), depth=-3500, roi=roi, mask=mask, dark=dark)
    result = q_distribution(image, geometry=DATA/"geometry.xml", energy_kev=ENERGY,
                            monochromatic=True, normalization="none", **kwargs)
    expected = scalar_profile(image, CALIBRATION, ENERGY, **kwargs)
    for name in FIELDS:
        if "count" in name:
            np.testing.assert_array_equal(getattr(result, name), expected[name])
        else:
            np.testing.assert_allclose(getattr(result, name), expected[name], rtol=1e-12, atol=1e-12)


def test_layout_endianness_blocks_ownership_and_provenance(monkeypatch):
    image, _, _ = read_h5_frame(DATA/"rings.h5")
    dark, _, _ = read_h5_frame(DATA/"dark.h5")
    mask = load_mask(DATA/"mask.h5")
    meta = dict(energy_kev=12, mono_mode="monochromatic", exposure_seconds=2,
                i0_counts=40000, scaler_count_time_seconds=4)
    def run(values):
        return q_distribution(values, geometry=Geometry(DATA/"geometry.xml"), metadata=meta, dark=dark,
                              mask=mask, roi=ROI, q_first=0, q_last=20, q_points=201)
    expected = run(image)
    monkeypatch.setattr(_core, "_BLOCK_PIXELS", 137)
    for values in (np.asfortranarray(image), image.astype(">f8"), image[:, ::-1][:, ::-1]):
        actual = run(values)
        for name in FIELDS:
            np.testing.assert_allclose(getattr(actual, name), getattr(expected, name), rtol=1e-12, atol=1e-12)
        assert actual.provenance["source"]["sha256"] == hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest()
    old = expected.intensity_mean.copy()
    image[:] = -100; dark[:] = 0; meta["energy_kev"] = 500
    expected.provenance["effective_metadata"]["energy_kev"] = 999
    np.testing.assert_array_equal(expected.intensity_mean, old)
    assert expected.provenance["effective_metadata"]["energy_kev"] == 12
    for name in FIELDS:
        assert not getattr(expected, name).flags.writeable


def test_reader_monitor_override_precedence_and_indexing_compatibility(tmp_path):
    source = tmp_path/"vectors.h5"; shutil.copyfile(DATA/"rings.h5", source)
    with h5py.File(source, "a") as handle:
        del handle["entry1/monitor/I0"]
        handle.create_dataset("entry1/monitor/I0", data=[40000., 80000.])
    _, old_metadata, _ = read_h5_frame(source)
    assert "i0_counts" not in old_metadata  # historical indexer adapter contract
    with pytest.raises(InputError, match="scalar override"):
        q_distribution(source, geometry=DATA/"geometry.xml")
    result = q_distribution(source, geometry=DATA/"geometry.xml", metadata={"i0_counts": 40000},
                            energy_kev=13, start=(90, 90), group=(9, 9), depth=0)
    assert result.provenance["normalization"]["factor"] == 5
    assert result.provenance["selection"]["start"] == [0, 0]
    assert result.provenance["selection"]["group"] == [1, 1]
    assert result.provenance["effective_metadata"]["energy_kev"] == 13
    assert result.provenance["file_metadata"]["energy_kev"] == 12
    assert result.provenance["file_metadata"]["i0_counts"] == [40000., 80000.]
    with h5py.File(source, "a") as handle:
        del handle["entry1/microDiffraction/MonoMode"]
        handle.create_dataset("entry1/microDiffraction/MonoMode", data="white slitted")
    with pytest.raises(InputError, match="white-beam"):
        q_distribution(source, geometry=DATA/"geometry.xml", metadata={"mono_mode": "monochromatic"}, normalization="none")


def test_explicit_declaration_resolves_mode_vector_but_preserves_white_history(tmp_path):
    source = tmp_path/"mode-vector.h5"; shutil.copyfile(DATA/"rings.h5", source)
    with h5py.File(source, "a") as handle:
        del handle["entry1/microDiffraction/MonoMode"]
        handle.create_dataset("entry1/microDiffraction/MonoMode",
                              data=["white slitted", "monochromatic"], dtype=h5py.string_dtype())
    with pytest.raises(InputError, match="white-beam"):
        q_distribution(source, geometry=DATA/"geometry.xml", metadata={"mono_mode": "monochromatic"}, normalization="none")
    result = q_distribution(source, geometry=DATA/"geometry.xml", monochromatic=True,
                            monochromatic_reason="Explicitly selected monochromatic image", normalization="none")
    assert result.provenance["file_metadata"]["mono_mode"] == ["white slitted", "monochromatic"]


@pytest.mark.parametrize("field,value", [("entry1/detector/ID", "wrong"), ("entry1/detector/exposure", 3.),
                                        ("entry1/detector/startx", 1), ("entry1/detector/binx", 2)])
def test_known_dark_mismatch_is_rejected_without_rescaling(tmp_path, field, value):
    path = tmp_path/"dark.h5"; shutil.copyfile(DATA/"dark.h5", path)
    with h5py.File(path, "a") as handle:
        del handle[field]; handle.create_dataset(field, data=value)
    with pytest.raises(InputError, match="dark"):
        rings(dark=path)


@pytest.mark.parametrize("role", ["primary", "dark"])
@pytest.mark.parametrize("fields,key", [(("startx", "starty"), "start"), (("binx", "biny"), "group")])
@pytest.mark.parametrize("missing", [0, 1])
def test_partial_hdf5_coordinates_are_rejected_for_energy_only(tmp_path, role, fields, key, missing):
    original = DATA/("rings.h5" if role == "primary" else "dark.h5")
    path = tmp_path/"partial.h5"; shutil.copyfile(original, path)
    with h5py.File(path, "a") as source:
        del source[f"entry1/detector/{fields[missing]}"]
        source[f"entry1/detector/{fields[1-missing]}"][()] = 9
    # The indexing adapter retains its established interpretation. Energy
    # cannot discard a known axis and then assume matching coordinates.
    assert key not in read_h5_frame(path)[2]
    frame, dark = (path, DATA/"dark.h5") if role == "primary" else (DATA/"rings.h5", path)
    with pytest.raises(InputError, match=f"{fields[0]} and {fields[1]}"):
        q_distribution(frame, geometry=DATA/"geometry.xml", dark=dark)


@pytest.mark.parametrize("role", ["primary", "dark"])
@pytest.mark.parametrize("fields", [("startx", "starty"), ("binx", "biny")])
def test_absent_hdf5_coordinate_pairs_still_allow_defaults(tmp_path, role, fields):
    path = tmp_path/"absent.h5"
    shutil.copyfile(DATA/("rings.h5" if role == "primary" else "dark.h5"), path)
    with h5py.File(path, "a") as source:
        for field in fields:
            del source[f"entry1/detector/{field}"]
    frame, dark = (path, DATA/"dark.h5") if role == "primary" else (DATA/"rings.h5", path)
    actual = q_distribution(frame, geometry=DATA/"geometry.xml", dark=dark)
    expected = q_distribution(DATA/"rings.h5", geometry=DATA/"geometry.xml", dark=DATA/"dark.h5")
    for name in FIELDS:
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


def test_geometry_snapshot_survives_source_changes_and_tracks_unapplied_maps(tmp_path):
    path = tmp_path/"geometry.xml"
    path.write_text(geometry_xml().replace("</Detector>", "<distortionMap>calibration.itx</distortionMap></Detector>"))
    geometry = Geometry(path)
    path.unlink()
    result = rings(geometry=geometry)
    assert "calibration.itx" in result.provenance["geometry"]["xml"]
    assert result.provenance["geometry"]["unapplied_distortion_maps"] == ["calibration.itx"]
    assert result.provenance["corrections"]["distortion"] is False


def test_detector_selection_conflicts_and_old_public_class_identity():
    with pytest.raises(InputError, match="conflict"):
        rings(detector_index=0)
    with pytest.raises(InputError, match="conflict"):
        rings(detector_id="wrong")
    assert FrameMetadata.__module__ == "lauelab.indexing.indexer"
    assert ScanFrame.__module__ == "lauelab.indexing._frame"
    reference = ScanFrame("point.h5", "point", 2)
    assert pickle.loads(pickle.dumps(reference)) == reference


def test_generator_reproduces_numeric_artifacts_without_lauelab_or_igor(tmp_path):
    generate(tmp_path)
    old, new = np.load(DATA/"reference.npz"), np.load(tmp_path/"reference.npz")
    for field in old.files:
        np.testing.assert_array_equal(old[field], new[field])
    for name in ("rings.h5", "dark.h5", "mask.h5"):
        with h5py.File(DATA/name) as old, h5py.File(tmp_path/name) as new:
            np.testing.assert_array_equal(old["entry1/data/data"][...], new["entry1/data/data"][...])
    provenance = json.loads((DATA/"provenance.json").read_text())
    assert "NOT measured" in provenance["data_kind"]
