# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
import csv
from dataclasses import fields

import h5py
import numpy as np
import pytest

from lauelab.indexing import InputError
from lauelab.strain import (
    export_strain_csv,
    load_strain_results,
    refine_deviatoric,
    save_strain_results,
)
from strain_cases import HKL, case


def results():
    ref, _, initial, q, _ = case()
    return [
        refine_deviatoric(q, HKL, reference=ref, initial_reciprocal=initial),
        refine_deviatoric(q[:3], HKL[:3], reference=ref, initial_reciprocal=initial),
    ]


def test_complete_float64_roundtrip_includes_unsuccessful_results(tmp_path):
    values = results()
    path = save_strain_results(values, tmp_path / "strain.h5")
    loaded = load_strain_results(path)
    assert [r.status for r in loaded] == ["converged", "insufficient_support"]
    for original, reopened in zip(values, loaded):
        for item in fields(original):
            value = getattr(original, item.name)
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(getattr(reopened, item.name), value)
                assert not getattr(reopened, item.name).flags.writeable
        np.testing.assert_array_equal(
            reopened.measurements.qhat, original.measurements.qhat
        )
        assert reopened.provenance == original.provenance
    with h5py.File(path) as source:
        assert source["results/0/tensor"].dtype == np.dtype("<f8")
        assert source["results/0/measurements/xy"].dtype == np.dtype("<f8")
        assert "units" not in source["results/0/tensor"].attrs
        assert "units" not in source["results/0/measurements/hkl"].attrs
        assert source["results/0/cell_parameters"].attrs[
            "component_units"
        ].tolist() == ["nm", "nm", "nm", "deg", "deg", "deg"]
        assert h5py.check_enum_dtype(source["results/0/status"].dtype) == {
            "converged": 0,
            "insufficient_support": 1,
            "ill_conditioned": 2,
            "nonconverged": 3,
            "invalid_input": 4,
        }


@pytest.mark.parametrize(
    "damage",
    [
        "version",
        "units",
        "tensor",
        "count",
        "hkl",
        "rotation",
        "predicted_qhat",
        "cell_parameters",
        "status",
        "component_units",
    ],
)
def test_corrupt_files_do_not_load_as_valid_strain(tmp_path, damage):
    path = save_strain_results(results(), tmp_path / "strain.h5")
    with h5py.File(path, "r+") as source:
        if damage == "version":
            source.attrs["version"] = 999
        elif damage == "units":
            source["results/0/tensor"].attrs["units"] = "percent"
        elif damage == "tensor":
            source["results/0/tensor"][0, 0] = 1
        elif damage == "count":
            source.attrs["count"] = 9
        elif damage == "hkl":
            source["results/0/measurements/hkl"][0] = 0
        elif damage == "status":
            source["results/0/status"][()] = 99
        elif damage == "component_units":
            source["results/0/cell_parameters"].attrs["component_units"] = ["nm"] * 6
        else:
            source[f"results/0/{damage}"][...] = np.nan
    with pytest.raises(InputError):
        load_strain_results(path)


def test_existing_output_preserved_and_overwrite_explicit(tmp_path):
    path = tmp_path / "strain.h5"
    path.write_bytes(b"existing user artifact")
    with pytest.raises(FileExistsError):
        save_strain_results(results(), path)
    assert path.read_bytes() == b"existing user artifact"
    save_strain_results(results(), path, overwrite=True)
    assert len(load_strain_results(path)) == 2
    assert not list(tmp_path.glob("*.partial"))


def test_csv_catalogue_and_reflection_granularities(tmp_path):
    values = results()
    summary, reflections = export_strain_csv(
        values, tmp_path / "catalogue.csv", tmp_path / "reflections.csv"
    )
    with summary.open() as handle:
        catalogue = list(csv.DictReader(handle))
    with reflections.open() as handle:
        spots = list(csv.DictReader(handle))
    assert len(catalogue) == 2
    assert len(spots) == len(HKL) + 3
    assert catalogue[0]["coordinates"] == "beamline"
    assert catalogue[0]["shear_convention"] == "tensor"
    assert catalogue[1]["status"] == "insufficient_support"
    assert "angular_residual_deg" in spots[0]
    with pytest.raises(FileExistsError):
        export_strain_csv(values, summary, reflections)


def test_empty_collection_is_a_valid_saved_analysis(tmp_path):
    path = save_strain_results([], tmp_path / "empty.h5")
    assert load_strain_results(path) == []
