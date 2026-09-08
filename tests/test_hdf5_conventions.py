# Copyright 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Tests for package-wide HDF5 conventions."""

from importlib.metadata import version as distribution_version

import h5py
import numpy as np
import pytest

from lauelab._hdf5 import check_format_version, set_units, write_root_attributes

FORMAT = "test-format"


def test_root_attributes_and_units(tmp_path):
    path = tmp_path / "result.h5"
    with h5py.File(path, "w") as target:
        write_root_attributes(
            target,
            format_name=FORMAT,
            version=1,
            created="2026-09-04T12:34:56+00:00",
            source="input.xml",
        )
        dataset = set_units(target.create_dataset("values", data=[1.0]), "um")

        assert target.attrs["format"] == FORMAT
        assert target.attrs["version"] == 1
        assert target.attrs["lauelab_version"] == distribution_version("lauelab")
        assert target.attrs["created"] == "2026-09-04T12:34:56+00:00"
        assert target.attrs["source"] == "input.xml"
        assert dataset.attrs["units"] == "um"


def test_root_attributes_generate_utc_timestamp(tmp_path):
    with h5py.File(tmp_path / "result.h5", "w") as target:
        write_root_attributes(target, format_name=FORMAT, version=1)
        assert target.attrs["created"].endswith("+00:00")
        assert "source" not in target.attrs


@pytest.mark.parametrize("version", [True, 1.0, "1"])
def test_root_attributes_require_integer_version(tmp_path, version):
    with h5py.File(tmp_path / "result.h5", "w") as target:
        with pytest.raises(TypeError, match="version must be an integer"):
            write_root_attributes(target, format_name=FORMAT, version=version)


def test_check_format_version_accepts_supported_version_and_bytes_format(tmp_path):
    with h5py.File(tmp_path / "result.h5", "w") as source:
        source.attrs["format"] = np.bytes_(FORMAT)
        source.attrs["version"] = np.int32(1)
        assert check_format_version(
            source, format_name=FORMAT, supported_versions={1}
        ) == 1


@pytest.mark.parametrize(
    ("attributes", "message"),
    [
        ({"format": "other", "version": 1}, "not a 'test-format' file"),
        ({"format": FORMAT, "version": 2}, "unsupported 'test-format' version 2"),
        ({"format": FORMAT, "version": "1"}, "version must be an integer"),
    ],
)
def test_check_format_version_rejects_wrong_conventions(tmp_path, attributes, message):
    with h5py.File(tmp_path / "result.h5", "w") as source:
        source.attrs.update(attributes)
        with pytest.raises(ValueError, match=message):
            check_format_version(source, format_name=FORMAT, supported_versions={1})
