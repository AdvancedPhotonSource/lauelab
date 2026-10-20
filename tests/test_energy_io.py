# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Standalone artifact, malformed-file, publication and plot checks."""

import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from conftest import requires_liblaue
from lauelab.energy import (
    QDistribution, q_distribution, save_q_distribution, load_q_distribution,
    validate_q_distribution_file, export_q_distribution_csv,
)
from lauelab.energy import _core, _io, _layout
from lauelab.indexing import InputError, load_mask
from lauelab.visualization import plot_q_distribution

pytestmark = requires_liblaue
DATA = Path(__file__).parent/"data/energy"


@pytest.fixture
def result():
    return q_distribution(DATA/"rings.h5", geometry=DATA/"geometry.xml", dark=DATA/"dark.h5",
                          mask=load_mask(DATA/"mask.h5"), roi=(5, 73, 7, 91),
                          q_first=0, q_last=20, q_points=201)


@pytest.fixture
def saved(tmp_path, result):
    return save_q_distribution(result, tmp_path/"profile.h5")


def test_hdf5_round_trip_is_standalone_and_uses_layout_table(result, saved):
    loaded = load_q_distribution(saved)
    validate_q_distribution_file(saved)
    assert loaded.provenance == result.provenance
    for name in _layout.PROFILE:
        np.testing.assert_array_equal(getattr(loaded, name), getattr(result, name))
        assert not getattr(loaded, name).flags.writeable
    with h5py.File(saved) as source:
        assert source.attrs["format"] == _layout.FORMAT and source.attrs["version"] == 1
        assert set(source) == {"profile", "parameters", "geometry", "provenance"}
        for name, (dtype, units) in _layout.PROFILE.items():
            assert source[f"profile/{name}"].dtype == np.dtype(dtype)
            assert source[f"profile/{name}"].attrs.get("units") == _io._units(units, "i0")
        for path, (dtype, shape, units) in _layout.SETTINGS.items():
            assert source[path].shape == shape
            assert source[path].attrs.get("units") == _io._units(units, "i0")
        assert "distortion" in source["provenance/json"].asstr()[()]
        assert "entry1" not in source


def test_round_trip_handles_nan_source_metadata_and_invalid_bins(tmp_path):
    with h5py.File(DATA/"rings.h5") as source:
        image = source["entry1/data/data"][...]
    image[:] = np.nan
    result = q_distribution(image, geometry=DATA/"geometry.xml", energy_kev=12,
                            monochromatic=True, normalization="none", metadata={"unused": np.inf})
    path = save_q_distribution(result, tmp_path/"nan.h5")
    loaded = load_q_distribution(path)
    assert np.isnan(loaded.intensity_mean).all()
    assert loaded.nonfinite_pixel_count.sum() == image.size
    assert loaded.provenance["effective_metadata"]["unused"] == {"nonfinite": "Infinity"}


def test_csv_preserves_columns_counts_values_and_identifies_corrections(tmp_path, result):
    path = export_q_distribution_csv(result, tmp_path/"profile.csv")
    text = path.read_text()
    assert "1/nm" in text and '"mode": "i0"' in text and "NaN" in text
    with path.open() as stream:
        rows = list(csv.DictReader(line for line in stream if not line.startswith("#")))
    assert list(rows[0]) == list(_layout.PROFILE)
    for name in _layout.PROFILE:
        actual = np.array([row[name] for row in rows], dtype=np.int64 if "count" in name else np.float64)
        np.testing.assert_array_equal(actual, getattr(result, name))


@pytest.mark.parametrize("writer,suffix", [(save_q_distribution, "h5"), (export_q_distribution_csv, "csv")])
def test_default_no_overwrite_and_explicit_replace(tmp_path, result, writer, suffix):
    path = tmp_path/f"profile.{suffix}"
    path.write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        writer(result, path)
    assert path.read_bytes() == b"existing"
    writer(result, path, overwrite=True)
    assert path.read_bytes() != b"existing"
    assert not list(tmp_path.glob("*.partial"))


def test_failed_validation_cleans_only_its_own_partial(tmp_path, result, monkeypatch):
    other = tmp_path/"profile.h5.partial"; other.write_bytes(b"another writer")
    def fail(*args, **kwargs):
        raise InputError("injected validation failure")
    monkeypatch.setattr(_io, "validate_q_distribution_file", fail)
    with pytest.raises(InputError, match="injected"):
        save_q_distribution(result, tmp_path/"profile.h5")
    assert other.read_bytes() == b"another writer"
    assert not (tmp_path/"profile.h5").exists()
    assert list(tmp_path.iterdir()) == [other]


def test_racing_publisher_is_not_overwritten(tmp_path, result, monkeypatch):
    real = _io.publish_file
    def race(partial, path, **kwargs):
        Path(path).write_bytes(b"other publisher")
        return real(partial, path, **kwargs)
    monkeypatch.setattr(_io, "publish_file", race)
    path = tmp_path/"profile.h5"
    with pytest.raises(FileExistsError):
        save_q_distribution(result, path)
    assert path.read_bytes() == b"other publisher"
    assert not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize("change", ["version", "format", "units", "dtype", "length", "missing", "mean", "counts", "q", "json", "setting", "negative_energy", "diagnostics"])
def test_malformed_artifacts_are_rejected(saved, change):
    with h5py.File(saved, "a") as source:
        if change in ("version", "format"):
            source.attrs[change] = 99 if change == "version" else "lauelab-indexing-results"
        elif change == "units":
            source["profile/q_nm_inv"].attrs["units"] = "1/angstrom"
        elif change in ("dtype", "length", "missing"):
            del source["profile/pixel_count"]
            if change != "missing":
                source.create_dataset("profile/pixel_count", data=np.zeros(201 if change == "dtype" else 2, dtype=np.float64 if change == "dtype" else np.int64))
        elif change == "mean":
            source["profile/intensity_mean"][100] = 999999.
        elif change == "counts":
            source["profile/nonfinite_pixel_count"][0] = 10
        elif change == "q":
            source["profile/q_nm_inv"][100] += .01
        elif change == "json":
            source["provenance/json"][()] = "not JSON"
        elif change == "setting":
            source["parameters/depth_um"][()] = 42.
        else:
            p = json.loads(source["provenance/json"].asstr()[()])
            if change == "negative_energy":
                p["effective_metadata"]["energy_kev"] = -12
                source["parameters/energy_kev"][()] = -12
            else:
                p["diagnostics"]["outside_grid_pixel_count"] += 1
            source["provenance/json"][()] = json.dumps(p)
    with pytest.raises(InputError):
        load_q_distribution(saved)


@pytest.fixture(params=[
    pytest.param((("selection", "group"), [0, 0]), id="zero-group"),
    pytest.param((("selection", "group"), [1.5, 1]), id="fractional-group"),
    pytest.param((("selection", "group"), [True, 1]), id="boolean-group"),
    pytest.param((("selection", "start"), [-1, 0]), id="negative-start"),
    pytest.param((("selection", "start"), [1, 0]), id="frame-outside-detector"),
    pytest.param((("selection", "roi"), [5, 5, 7, 91]), id="empty-roi"),
    pytest.param((("selection", "roi"), [5, 81, 7, 91]), id="roi-outside-frame"),
    pytest.param((("selection", "roi"), [5., 73, 7, 91]), id="fractional-roi-type"),
    pytest.param((("selection", "roi"), [4, 73, 7, 91]), id="roi-count-mismatch"),
    pytest.param((("source", "shape"), [0, 96]), id="empty-source"),
    pytest.param((("source", "shape"), [1, 80, 96]), id="stack-source"),
    pytest.param((("geometry", "calibration", "nx"), -96), id="negative-detector-dimension"),
    pytest.param((("geometry", "calibration", "ny"), 0), id="zero-detector-dimension"),
    pytest.param((("geometry", "calibration", "size_x"), -19200.), id="negative-detector-size"),
    pytest.param((("geometry", "calibration", "size_y"), 0.), id="zero-detector-size"),
    pytest.param((("geometry", "calibration", "size_y"), np.inf), id="nonfinite-detector-size"),
    pytest.param((("geometry", "calibration", "translation"), [0., 0., np.nan]), id="nonfinite-translation"),
    pytest.param((("geometry", "calibration", "rotation_vector"), [0., np.inf, 0.]), id="nonfinite-rotation-vector"),
    pytest.param((("geometry", "calibration", "rotation"), [[np.nan, 0., 0.], [0., 1., 0.], [0., 0., 1.]]), id="nonfinite-rotation"),
    pytest.param((("geometry", "detector_index"), -1), id="negative-detector-slot"),
    pytest.param((("geometry", "detector_index"), 3), id="unsupported-detector-slot"),
])
def invalid_settings(request, result):
    keys, value = request.param
    provenance = result.provenance
    target = provenance
    for key in keys[:-1]:
        target = target[key]
    target[keys[-1]] = value
    return provenance


def test_invalid_scientific_settings_cannot_be_published(tmp_path, result, invalid_settings):
    bad = QDistribution(*(getattr(result, name) for name in _layout.PROFILE), invalid_settings)
    path = tmp_path/"invalid.h5"
    path.write_bytes(b"existing result")
    with pytest.raises(InputError):
        save_q_distribution(bad, path, overwrite=True)
    assert path.read_bytes() == b"existing result"
    assert list(tmp_path.iterdir()) == [path]


def test_matching_invalid_settings_are_rejected_on_validation_and_load(saved, invalid_settings):
    # Keep numeric settings and JSON in agreement: duplicated-field checks
    # alone cannot establish that the saved selection/calibration is valid.
    settings = _io._settings(invalid_settings, _core._json_text(invalid_settings))
    with h5py.File(saved, "a") as source:
        for path, value in settings.items():
            source[path][...] = value
    with pytest.raises(InputError):
        validate_q_distribution_file(saved)
    with pytest.raises(InputError):
        load_q_distribution(saved)


def test_saving_requires_explicit_roi_bounds(tmp_path, result):
    provenance = result.provenance
    provenance["selection"]["roi"] = None
    bad = QDistribution(*(getattr(result, name) for name in _layout.PROFILE), provenance)
    path = tmp_path/"missing-roi.h5"
    with pytest.raises(InputError, match="roi"):
        save_q_distribution(bad, path)
    assert not path.exists()


def test_valid_grouped_crop_and_depth_survive_saved_settings_checks(tmp_path):
    result = q_distribution(np.arange(12*18).reshape(12, 18), geometry=DATA/"geometry.xml",
                            energy_kev=12, monochromatic=True, normalization="none",
                            start=(4, 7), group=(2, 3), depth=-3500, roi=(1, 11, 2, 17))
    path = save_q_distribution(result, tmp_path/"grouped.h5")
    loaded = load_q_distribution(path)
    assert loaded.provenance == result.provenance
    for name in _layout.PROFILE:
        np.testing.assert_array_equal(getattr(loaded, name), getattr(result, name))


def test_loading_checks_bin_guard_before_reading_large_dataset(tmp_path):
    path = tmp_path/"huge.h5"
    from lauelab._hdf5 import write_root_attributes
    with h5py.File(path, "w") as target:
        write_root_attributes(target, format_name=_layout.FORMAT, version=1)
        target.create_dataset("profile/q_nm_inv", shape=(10**10,), chunks=(1024,), dtype=np.float64)
    assert path.stat().st_size < 100_000
    with pytest.raises(InputError, match="max_bins"):
        load_q_distribution(path)


def test_explicitly_raised_loading_limit(saved):
    with pytest.raises(InputError, match="max_bins"):
        load_q_distribution(saved, max_bins=100)
    assert len(load_q_distribution(saved, max_bins=201).q_nm_inv) == 201


def test_result_constructor_rejects_invalid_scientific_arrays(result):
    values = {name: getattr(result, name).copy() for name in _layout.PROFILE}
    values["pixel_count"] = values["pixel_count"].astype(float)+.5
    with pytest.raises(InputError, match="integers"):
        QDistribution(**values, provenance=result.provenance)


@pytest.mark.parametrize("mode,units", [("none", "synthetic intensity"), ("exposure", "synthetic intensity/s"), ("i0", "unspecified")])
def test_units_and_extended_float_metadata_survive_round_trip(tmp_path, mode, units):
    result = q_distribution(DATA/"rings.h5", geometry=DATA/"geometry.xml", normalization=mode,
                            metadata={"exposure_seconds": np.longdouble(2)})
    path = save_q_distribution(result, tmp_path/"units.h5")
    assert result.intensity_units == units
    assert load_q_distribution(path).intensity_units == units
    with h5py.File(path) as source:
        assert source["profile/intensity_mean"].attrs["units"] == units
        assert source["parameters/normalization_factor"].attrs.get("units") == {"none": None, "exposure": "1/s", "i0": "unspecified"}[mode]


def test_documented_layout_paths_match_registry():
    import re
    text = (Path(__file__).parents[1]/"docs/development/energy-format.md").read_text()
    paths = set(re.findall(r"\| `((?:profile|parameters|geometry|provenance)/[^`]+)` \|", text))
    assert paths == {*(f"profile/{name}" for name in _layout.PROFILE), *list(_layout.SETTINGS)}


def test_energy_guide_complete_example_executes(tmp_path, monkeypatch):
    import re
    root = Path(__file__).parents[1]
    monkeypatch.chdir(root)
    text = (root/"docs/guides/energy.md").read_text()
    blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
    assert blocks
    namespace = {}
    for block in blocks:
        exec(compile(block, "docs/guides/energy.md", "exec"), namespace)
    assert namespace["profile"].pixel_count.sum() == 5596


def test_plot_matches_igor_presentation_and_accepts_loaded_results(result, saved):
    loaded = load_q_distribution(saved)
    figure = plot_q_distribution(loaded)
    trace = figure.data[0]
    np.testing.assert_array_equal(trace.x, result.q_nm_inv)
    np.testing.assert_array_equal(trace.y, result.intensity_mean)
    assert trace.mode == "lines+markers" and trace.line.dash == "dash"
    assert trace.connectgaps is False and trace.meta["role"] == "profile"
    assert figure.layout.xaxis.type == figure.layout.yaxis.type == "linear"
    assert figure.layout.yaxis.range[0] == 0
    assert figure.layout.xaxis.ticks == "inside" and figure.layout.xaxis.mirror
    assert figure.layout.xaxis.linecolor == figure.layout.yaxis.linecolor == "black"
    assert figure.layout.xaxis.mirror == "ticks"
    assert "nm" in figure.layout.xaxis.title.text and "I₀" in figure.layout.yaxis.title.text
    assert not trace.error_y.visible and not figure.layout.shapes
    changed = plot_q_distribution(loaded, trace_update={"profile": {"line": {"color": "red"}}},
                                   layout_update={"yaxis": {"range": [-10, 20]}})
    assert changed.data[0].line.color == "red" and changed.layout.yaxis.range[0] == -10


@pytest.mark.parametrize("value", [0.002, np.finfo(float).max])
def test_plot_autoscale_preserves_small_and_extreme_finite_intensity(value):
    result = QDistribution([0, 1], [value, value], [value, value], [1, 1], [0, 0],
                           {"normalization": {"mode": "none"}})
    upper = plot_q_distribution(result).layout.yaxis.range[1]
    assert np.isfinite(upper) and upper >= value
    if value < 1:
        assert upper < .01
