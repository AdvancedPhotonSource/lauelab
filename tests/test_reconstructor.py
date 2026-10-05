# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Tests for the in-process reconstruction driver."""

import json
from pathlib import Path
import re

import h5py
import numpy as np
import pytest

from conftest import requires_liblaue
from lauelab.indexing import InputError
from lauelab.reconstruct import (
    ImageGeometry, ReconstructionResult, Reconstructor, find_executable, reconstruct,
    reconstruct_points,
)
from lauelab.reconstruct._reader import cutoff_mask, positioner_from_file_time, read_scan_info
from lauelab.reconstruct.reconstruct import _find_executable
from tests.data.reconstruction.generate_reference import (
    BINNING, DEPTH_RANGE_UM, FULL_PIXELS, GEOMETRY_FILE, VARIANTS,
    write_input_file,
)

pytestmark = requires_liblaue
REFERENCE_DIR = Path(__file__).parent / "data/reconstruction"


def _reconstructor(variant=None, **changes):
    options = VARIANTS.get(variant, {})
    values = dict(
        geometry=GEOMETRY_FILE,
        detector=0,
        depth_range=DEPTH_RANGE_UM,
        wire_edge=options.get("wire_edge", "leading"),
        normalization=options.get("normalization"),
        norm_exponent=options.get("norm_exponent"),
        cosmic_filter=options.get("cosmic_filter", False),
        output_pixel_type=options.get("output_pixel_type", 5),
        num_threads=1,
        rows_per_stripe=31,
    )
    values.update(changes)
    return Reconstructor(**values)


def _wire_positions(path, n_images):
    with h5py.File(path) as handle:
        return np.column_stack([
            np.asarray(handle[f"entry1/wire/wire{name}"])[2:n_images + 3]
            for name in "XYZ"
        ])


@pytest.mark.parametrize("variant", [None, *VARIANTS])
def test_reconstructor_matches_golden(tmp_path, variant):
    options = VARIANTS.get(variant, {})
    source = tmp_path / "synthetic.h5"
    write_input_file(source, write_mA=options.get("write_mA", False),
                     write_microdiffraction=options.get("write_microdiffraction", False))
    result = _reconstructor(variant).reconstruct(source, return_images=True)
    suffix = f"_{variant}" if variant else ""
    expected = np.load(REFERENCE_DIR / f"cpu_reference{suffix}.npz")
    metadata = json.loads((REFERENCE_DIR / f"cpu_reference{suffix}.json").read_text())

    assert result.success, result.error
    np.testing.assert_array_equal(result.depth_um, expected["depth_um"])
    actual = result.images
    if np.issubdtype(expected["images"].dtype, np.integer):
        info = np.iinfo(expected["images"].dtype)
        actual = np.clip(np.trunc(actual), info.min, info.max).astype(expected["images"].dtype)
    tolerance = metadata["comparison"]
    np.testing.assert_allclose(actual, expected["images"], rtol=tolerance["rtol"], atol=tolerance["atol"])


def test_default_stripe_is_capped_at_256_rows_and_memory_limit():
    reconstructor = _reconstructor(rows_per_stripe=None)
    assert reconstructor.memory_limit_mb == 8192
    assert reconstructor._stripe_rows(10, 11, 1000, 32, 2) == 256
    constrained = _reconstructor(rows_per_stripe=None, memory_limit_mb=1)
    assert constrained._stripe_rows(10, 11, 1000, 32, 2) == 151


def test_array_and_file_paths_are_identical(tmp_path):
    source = tmp_path / "synthetic.h5"
    write_input_file(source)
    reconstructor = _reconstructor()
    file_result = reconstructor.reconstruct(source, return_images=True)
    with h5py.File(source) as handle:
        images = np.asarray(handle["entry1/data/data"])[1:-1]
        intensity = np.asarray(handle["entry1/data/data"])[1]
        wire = np.column_stack([
            np.asarray(handle[f"entry1/wire/wire{name}"])[2:len(images) + 3]
            for name in "XYZ"
        ])
    image_geometry = ImageGeometry(FULL_PIXELS, FULL_PIXELS, group=(BINNING, BINNING),
                                   n_rows=images.shape[1], n_cols=images.shape[2])
    array_result = reconstructor.reconstruct_array(
        images, wire, intensity_map=intensity, positioner="alio",
        image_geometry=image_geometry,
    )
    assert array_result.success
    np.testing.assert_array_equal(array_result.images, file_result.images)


@pytest.mark.parametrize(("frame_skip", "wire_skip"), [(1, 1), (0, 0), (0, 1), (3, 2)])
def test_skips_select_the_executable_slices_and_wire_entries(tmp_path, frame_skip, wire_skip):
    source = tmp_path / "synthetic.h5"
    write_input_file(source, write_mA=True)
    reconstructor = _reconstructor(normalization="mA", frame_skip=frame_skip, wire_skip=wire_skip)
    file_result = reconstructor.reconstruct(source, return_images=True)
    with h5py.File(source) as handle:
        stored = np.asarray(handle["entry1/data/data"])
        images = stored[frame_skip:-1]
        # The executable reads header vectors from entry 1, then skips
        # wire_skip more wire entries; frame_skip does not shift normalization.
        wire = np.column_stack([
            np.asarray(handle[f"entry1/wire/wire{name}"])[1 + wire_skip:len(images) + 2 + wire_skip]
            for name in "XYZ"
        ])
        scale = np.asarray(handle["entry1/mA"])[1:len(images) + 1] / 102.0
    image_geometry = ImageGeometry(FULL_PIXELS, FULL_PIXELS, group=(BINNING, BINNING),
                                   n_rows=images.shape[1], n_cols=images.shape[2])
    array_result = reconstructor.reconstruct_array(
        images, wire, intensity_map=stored[frame_skip], positioner="alio",
        image_geometry=image_geometry, scale=scale,
    )
    assert file_result.success and array_result.success
    np.testing.assert_array_equal(array_result.images, file_result.images)


@pytest.mark.parametrize("name", ["frame_skip", "wire_skip"])
@pytest.mark.parametrize("value", [-1, 1.0, True])
def test_skips_must_be_non_negative_integers(name, value):
    with pytest.raises(InputError, match=f"{name} must be a non-negative integer"):
        _reconstructor(**{name: value})


def test_summary_records_skips_only_when_they_differ_from_the_executable(tmp_path):
    source = tmp_path / "synthetic.h5"
    write_input_file(source)
    output_base = tmp_path / "skipped_"
    assert _reconstructor(frame_skip=2, wire_skip=0).reconstruct(source, output_base).success
    tags, _ = _summary(f"{output_base}summary.txt")
    assert (tags["frame_skip"], tags["wire_skip"]) == ("2", "0")


def test_zero_wire_rotation_is_finite_on_native_and_executable_paths(tmp_path):
    source = tmp_path / "synthetic.h5"
    write_input_file(source)
    geometry = tmp_path / "zero_rotation.xml"
    text = GEOMETRY_FILE.read_text()
    text = re.sub(r"<R unit=\"radian\">[^<]+</R>(?=\s*<Axis>)",
                  '<R unit="radian">0 0 0</R>', text)
    text = re.sub(r"<Axis>[^<]+</Axis>", "<Axis>1 0 0</Axis>", text)
    geometry.write_text(text)

    native = _reconstructor(geometry=geometry).reconstruct(source, return_images=True)
    output_base = tmp_path / "executable_"
    executable = find_executable()
    try:
        external = reconstruct(
            source, output_base, geometry, DEPTH_RANGE_UM,
            output_pixel_type=5, num_threads=1, executable=executable,
        )
    finally:
        _find_executable.cache_clear()

    assert native.success, native.error
    assert external.success, external.error
    assert np.isfinite(native.images).all()
    assert np.any(native.images)
    with h5py.File(external.output_files[0]) as output:
        executable_image = np.asarray(output["entry1/data/data"])
    assert np.isfinite(executable_image).all()
    assert np.any(executable_image)


@pytest.mark.parametrize(
    ("dtype", "expected_output_dtype"),
    [(np.uint32, np.float64), (np.float32, np.float32)],
)
def test_file_input_accepts_numeric_dtypes(tmp_path, dtype, expected_output_dtype):
    source = tmp_path / "synthetic.h5"
    write_input_file(source)
    with h5py.File(source, "r+") as handle:
        data = np.asarray(handle["entry1/data/data"], dtype=dtype)
        del handle["entry1/data/data"]
        handle["entry1/data"].create_dataset("data", data=data)

    expected = _reconstructor().reconstruct_array(
        data[1:-1].astype(np.uint16),
        _wire_positions(source, len(data) - 2),
        intensity_map=data[1],
        positioner="alio",
        image_geometry=ImageGeometry(
            FULL_PIXELS, FULL_PIXELS, group=(BINNING, BINNING),
            n_rows=data.shape[1], n_cols=data.shape[2],
        ),
    )
    output_base = tmp_path / "numeric_"
    actual = _reconstructor(output_pixel_type=None).reconstruct(
        source, output_base, return_images=True
    )
    array_actual = _reconstructor().reconstruct_array(
        data[1:-1], _wire_positions(source, len(data) - 2),
        intensity_map=data[1], positioner="alio",
        image_geometry=ImageGeometry(
            FULL_PIXELS, FULL_PIXELS, group=(BINNING, BINNING),
            n_rows=data.shape[1], n_cols=data.shape[2],
        ),
    )

    assert actual.success, actual.error
    assert array_actual.success, array_actual.error
    np.testing.assert_array_equal(actual.images, expected.images)
    np.testing.assert_array_equal(array_actual.images, expected.images)
    with h5py.File(actual.output_files[0]) as output:
        assert output["entry1/data/data"].dtype == np.dtype(expected_output_dtype)


@pytest.mark.parametrize("threads", [4])
def test_driver_is_thread_invariant(tmp_path, threads):
    source = tmp_path / "synthetic.h5"
    write_input_file(source)
    expected = _reconstructor(num_threads=1).reconstruct(source, return_images=True)
    actual = _reconstructor(num_threads=threads).reconstruct(source, return_images=True)
    np.testing.assert_array_equal(actual.images, expected.images)
    np.testing.assert_array_equal(actual.depth_intensity, expected.depth_intensity)


def test_output_files_use_hdf5_conversion_and_drop_wire_group(tmp_path):
    source = tmp_path / "synthetic.h5"
    write_input_file(source)
    output_base = tmp_path / "out" / "recon_"
    result = _reconstructor("out_uint16").reconstruct(source, output_base)

    assert result.success, result.error
    assert result.images is None
    assert result.output_files[-1] == f"{output_base}summary.txt"
    with h5py.File(result.output_files[0]) as output:
        assert output["entry1/data/data"].dtype == np.dtype("uint16")
        assert "entry1/wire" not in output
        assert output["entry1/data/data"].attrs["signal"] == 1
        assert "Facility/facility_name" in output


def test_integer_output_rescales_files_but_not_result_images(tmp_path):
    source = tmp_path / "synthetic.h5"
    write_input_file(source, write_microdiffraction=True)
    output_base = tmp_path / "normalized_"
    result = _reconstructor(
        "norm_exponent", output_pixel_type=3
    ).reconstruct(source, output_base, return_images=True)

    assert result.success, result.error
    with h5py.File(result.output_files[0]) as output:
        written = np.asarray(output["entry1/data/data"])
        expected = np.clip(np.trunc(result.images[0] * 255), 0, 65535)
        np.testing.assert_array_equal(expected, written)
        np.testing.assert_array_equal(
            output["entry1/microDiffraction/norm_rescale"], [255.0]
        )


def _objects(handle):
    objects = {}
    handle.visititems(lambda name, obj: objects.setdefault(name, obj))
    return objects


def _summary(path):
    lines = Path(path).read_text().splitlines()
    tags = {}
    array_start = None
    for index, line in enumerate(lines):
        match = re.match(r"^\$(\w+)\s+(.*?)(\s*//.*)?$", line)
        if match:
            tags[match.group(1)] = match.group(2).rstrip()
            if match.group(1) == "array0":
                array_start = index + 1
    return tags, "\n".join(lines[array_start:])


def test_native_output_is_structurally_equivalent_to_subprocess(tmp_path):
    source = tmp_path / "synthetic.h5"
    write_input_file(source)
    native_base = tmp_path / "native_"
    subprocess_base = tmp_path / "subprocess_"
    native = _reconstructor().reconstruct(source, native_base)
    executable = find_executable()
    try:
        external = reconstruct(
            source, subprocess_base, GEOMETRY_FILE, DEPTH_RANGE_UM,
            output_pixel_type=5, num_threads=1, executable=executable,
        )
    finally:
        _find_executable.cache_clear()
    assert native.success and external.success
    for index in range(len(native.depth_um)):
        with h5py.File(f"{native_base}{index}.h5") as left, h5py.File(f"{subprocess_base}{index}.h5") as right:
            left_objects = _objects(left)
            right_objects = _objects(right)
            assert left_objects.keys() == right_objects.keys()
            for name in left_objects:
                a, b = left_objects[name], right_objects[name]
                assert type(a) is type(b)
                assert a.attrs.keys() == b.attrs.keys()
                for attribute in a.attrs:
                    np.testing.assert_array_equal(a.attrs[attribute], b.attrs[attribute])
                if isinstance(a, h5py.Dataset):
                    assert a.dtype == b.dtype
                    assert a.shape == b.shape
                    np.testing.assert_array_equal(a, b)

    native_tags, native_array = _summary(f"{native_base}summary.txt")
    external_tags, external_array = _summary(f"{subprocess_base}summary.txt")
    # Paths/program names differ by design, timings are nondeterministic, and
    # each driver selects its stripe size independently.
    excluded = {"ws_outfile", "program_name", "executionTime", "rows_at_one_time"}
    for tag, value in native_tags.items():
        if tag not in excluded:
            assert external_tags[tag] == value
    assert native_array == external_array


def test_runtime_write_failure_returns_partial_result(tmp_path, monkeypatch):
    from lauelab.reconstruct.reconstructor import write_stripe as real_write_stripe

    source = tmp_path / "synthetic.h5"
    write_input_file(source)
    calls = 0

    def fail_second_write(handles, row0, values):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("output became unavailable")
        real_write_stripe(handles, row0, values)

    monkeypatch.setattr("lauelab.reconstruct.reconstructor.write_stripe", fail_second_write)
    output_base = tmp_path / "failed_"
    result = _reconstructor(rows_per_stripe=32).reconstruct(source, output_base)
    assert not result.success
    assert "output became unavailable" in result.error
    assert result.last_completed_stripe == 0
    with h5py.File(f"{output_base}0.h5") as output:
        image = np.asarray(output["entry1/data/data"])
    assert np.any(image[:32])
    assert not np.any(image[32:])


def test_missing_wire_vector_normalization_raises_input_error(tmp_path):
    source = tmp_path / "synthetic.h5"
    write_input_file(source)
    with pytest.raises(InputError, match="normalization vector 'mA' is missing"):
        _reconstructor(normalization="mA").reconstruct(source)


def test_invalid_geometry_detector_raises_before_work(tmp_path):
    source = tmp_path / "synthetic.h5"
    write_input_file(source)
    with h5py.File(source, "r+") as handle:
        handle["entry1/detector/Nx"][0] = FULL_PIXELS // 2
    output_base = tmp_path / "missing" / "failed_"
    with pytest.raises(InputError, match="image dimensions do not match the detector"):
        _reconstructor().reconstruct(source, output_base)
    assert not output_base.parent.exists()


def test_reader_requires_five_stored_slices(tmp_path):
    source = tmp_path / "short.h5"
    with h5py.File(source, "w") as output:
        output.create_dataset("entry1/data/data", shape=(4, 2, 2), dtype=np.uint16)
    with h5py.File(source) as handle:
        with pytest.raises(InputError, match="needs at least 5 stored slices"):
            read_scan_info(handle)
        with pytest.raises(InputError, match=r"needs at least 6 stored slices \(2 skipped"):
            read_scan_info(handle, frame_skip=2)


def test_cutoff_mask_clamps_extreme_percentile_to_last_pixel():
    intensity = np.array([[1, 2], [3, 4]])
    np.testing.assert_array_equal(
        cutoff_mask(intensity, np.nextafter(0.0, 1.0)),
        [[0, 0], [0, 1]],
    )


def test_partial_sample_position_is_written_independently(tmp_path):
    source = tmp_path / "synthetic.h5"
    write_input_file(source)
    with h5py.File(source, "r+") as handle:
        handle["entry1/sample/sampleX"][0] = 1.5
        handle["entry1/sample/sampleY"][0] = np.nan
        handle["entry1/sample/sampleZ"][0] = 3.5
        info = read_scan_info(handle)
    np.testing.assert_equal(info.sample_position, (1.5, np.nan, 3.5))

    output_base = tmp_path / "partial_sample_"
    result = _reconstructor().reconstruct(source, output_base)
    assert result.success, result.error
    tags, _ = _summary(f"{output_base}summary.txt")
    assert tags["X1"] == "1.5"
    assert not {"Y1", "Z1", "H1", "F1"} & tags.keys()


@pytest.mark.parametrize(
    ("file_time", "expected"),
    [
        ("2008-01-01 12:00:00", "pm500"),
        ("2008-01-01T12:00:00", "none"),
        ("2023-04-13T03:39:25-06:00", "none"),
    ],
)
def test_positioner_from_file_time_matches_executable(file_time, expected):
    assert positioner_from_file_time(file_time) == expected


def test_single_depth_range_reconstructs_one_image(tmp_path):
    source = tmp_path / "synthetic.h5"
    write_input_file(source)
    result = _reconstructor(depth_range=(0.0, 0.0)).reconstruct(
        source, return_images=True
    )
    assert result.success, result.error
    np.testing.assert_array_equal(result.depth_um, [0.0])
    assert result.images.shape[0] == 1


def test_reconstruct_points_keeps_results_when_one_point_fails(tmp_path):
    source = tmp_path / "synthetic.h5"
    write_input_file(source)
    results = reconstruct_points(
        [source, tmp_path / "missing.h5"], tmp_path / "batch",
        geometry=GEOMETRY_FILE, detector=0, workers=1, threads_per_worker=1,
        depth_range=DEPTH_RANGE_UM, output_pixel_type=5, rows_per_stripe=32,
    )
    assert results[0].success, results[0].error
    assert Path(results[0].output_files[0]).name == "synthetic_0.h5"
    assert not results[1].success
    assert "does not exist" in results[1].error
    assert results[1].command == "liblaue"


def test_reconstruct_points_rejects_num_threads(tmp_path):
    with pytest.raises(ValueError, match="threads_per_worker"):
        reconstruct_points(
            [], tmp_path / "batch", geometry=GEOMETRY_FILE, detector=0,
            threads_per_worker=1, num_threads=1, depth_range=DEPTH_RANGE_UM,
        )
    assert not (tmp_path / "batch").exists()


def test_result_preserves_six_positional_fields():
    result = ReconstructionResult(True, ["a"], "log", None, "command", 0)
    assert result.success
    assert result.images is None
    assert result._asdict()["command"] == "command"
