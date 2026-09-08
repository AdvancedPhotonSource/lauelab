# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
from dataclasses import fields
from pathlib import Path
import time

import h5py
import numpy as np
import pytest

from conftest import requires_liblaue

from lauelab import is_results_file
from lauelab.indexing import Atom, Cell, Crystal, FrameResult, Geometry, Pattern
from lauelab.indexing._liblaue import NativeCrystal
from lauelab.visualization import (
    DataScope,
    ResultSet,
    VisualizationDataset,
    convert_xml,
    load_results,
    load_visualization_xml,
    prepare_map,
)


GEOMETRY = Path(__file__).parent / "data/geo/geoN_2022-03-29_14-15-05.xml"


def _step(frame, patterns):
    pattern_xml = "".join(
        f"""<pattern num="{rank}" rms_error="0.00{rank + 5}" goodness="{150 - 30 * rank}" Nindexed="{count}">
        <recip_lattice><astar>-10 5 -6</astar><bstar>-15 -2 -1</bstar><cstar>-2 13 8</cstar></recip_lattice>
        <hkl_s><h>{' '.join(['1'] * count)}</h><k>{' '.join(['0'] * count)}</k>
        <l>{' '.join(['0'] * count)}</l><PkIndex>{' '.join(str(i) for i in range(count))}</PkIndex></hkl_s>
        </pattern>"""
        for rank, count in enumerate(patterns)
    )
    peak_count = max(patterns, default=2)
    values = " ".join(str(i) for i in range(peak_count))
    return f"""<step><Xsample>{100 + frame}</Xsample><Ysample>200</Ysample><Zsample>300</Zsample>
    <depth>nan</depth><energy>20</energy><scanNum>{1001 + frame}</scanNum>
    <detector><inputImage>image_{frame}.h5</inputImage><detectorID>TEST-DET</detectorID>
    <Nx>2048</Nx><Ny>2048</Ny><ROI startx="0" starty="0" groupx="1" groupy="1"/>
    <peaksXY Npeaks="{peak_count}"><Xpixel>{values}</Xpixel><Ypixel>{values}</Ypixel>
    <Intens>{values}</Intens><Integral>{values}</Integral><Qx>{values}</Qx><Qy>{values}</Qy><Qz>{values}</Qz></peaksXY></detector>
    <indexing Npatterns="{len(patterns)}">{pattern_xml}<xtl><structureDesc>TestMaterial</structureDesc>
    <SpaceGroup>225</SpaceGroup><latticeParameters unit="nm">0.4 0.4 0.4 90 90 90</latticeParameters>
    <atom symbol="Ni" label="Ni001">0 0 0</atom></xtl></indexing></step>"""


@pytest.fixture
def legacy_xml(tmp_path):
    path = tmp_path / "indexing.xml"
    path.write_text(f"<AllSteps>{_step(0, (4, 3))}{_step(1, (2,))}</AllSteps>")
    return path


def test_load_visualization_xml_normalizes_legacy_data(legacy_xml):
    dataset = load_visualization_xml(legacy_xml)

    assert dataset.n_frames == 2
    assert dataset.n_patterns == 3
    assert dataset.n_assignments == 9
    assert dataset.frame_n_peaks.tolist() == [4, 2]
    assert dataset.pattern_ids() == ((0, 0),)
    assert dataset.pattern_ids(DataScope(patterns="all", min_indexed=0)) == (
        (0, 0),
        (0, 1),
        (1, 0),
    )
    assert dataset.crystal.name == "TestMaterial"
    assert dataset.crystal.space_group == 225
    assert np.isfinite(dataset.pattern_rotations).all()
    np.testing.assert_allclose(dataset.peaks["fit_x"][:2], [0, 1])
    np.testing.assert_allclose(dataset.peaks["qhat"][1], [1, 1, 1])


@pytest.mark.parametrize("space_group", ["227:2", "not-a-number"])
def test_invalid_or_suffixed_space_group_degrades_to_no_crystal(tmp_path, space_group):
    path = tmp_path / "space-group.xml"
    step = _step(0, (1,)).replace(
        "<SpaceGroup>225</SpaceGroup>", f"<SpaceGroup>{space_group}</SpaceGroup>"
    )
    path.write_text(f"<AllSteps>{step}</AllSteps>")

    dataset = load_visualization_xml(path)

    assert dataset.crystal is None
    assert dataset.n_patterns == 1


@requires_liblaue
def test_atom_occupancy_round_trips_through_indexing_xml(tmp_path):
    crystal = Crystal(
        "mixed",
        1,
        Cell(0.5, 0.5, 0.5),
        (
            Atom("Ni", (0, 0, 0), occupancy=0.5),
            Atom("Ni", (0.5, 0.5, 0.5), occupancy=0.9),
        ),
    )
    from lauelab.analysis import simulate_reflections
    from lauelab.indexing import Indexer

    indexer = Indexer(GEOMETRY, crystal)
    result = indexer.index(np.zeros((2, 2), dtype=np.uint16))
    path = tmp_path / "occupancy.xml"
    result.write_xml(path)

    loaded = load_visualization_xml(path)

    assert [atom.occupancy for atom in loaded.crystal.atoms] == [0.5, 0.9]
    reciprocal = NativeCrystal.create(crystal).reciprocal()
    expected = simulate_reflections(
        crystal, reciprocal, indexer.detector, energy_range_kev=(6, 15)
    )
    actual = simulate_reflections(
        loaded.crystal, reciprocal, indexer.detector, energy_range_kev=(6, 15)
    )
    np.testing.assert_array_equal(actual.hkl, expected.hkl)
    np.testing.assert_allclose(actual.relative_intensity, expected.relative_intensity)


def test_xml_cell_units_are_normalized_for_orientation(tmp_path):
    reciprocal = 2.0 * np.pi / 0.4

    def document(length, unit):
        return f"""<AllSteps><step><detector/><indexing><pattern>
        <recip_lattice><astar>{reciprocal} 0 0</astar><bstar>0 {reciprocal} 0</bstar>
        <cstar>0 0 {reciprocal}</cstar></recip_lattice></pattern>
        <xtl><structureDesc>Test</structureDesc><SpaceGroup>225</SpaceGroup>
        <latticeParameters unit="{unit}">{length} {length} {length} 90 90 90</latticeParameters>
        </xtl></indexing></step></AllSteps>"""

    rotations = []
    for length, unit in ((0.4, "nm"), (4.0, "angstrom")):
        path = tmp_path / f"cell-{unit}.xml"
        path.write_text(document(length, unit))
        dataset = load_visualization_xml(path)
        assert dataset.crystal.cell.unit == unit
        rotations.append(dataset.pattern_rotations[0])

    np.testing.assert_allclose(rotations, np.tile(np.eye(3), (2, 1, 1)), atol=1e-15)


@requires_liblaue
@pytest.mark.parametrize(
    ("space_group", "cell"),
    [
        (168, Cell(0.4, 0.4, 0.6, 90, 90, 120)),
        (1, Cell(0.4, 0.5, 0.6, 70, 80, 75)),
    ],
    ids=("hexagonal", "triclinic"),
)
def test_live_and_xml_rotations_use_the_same_native_basis(tmp_path, space_group, cell):
    rotation = np.array([
        [0.93629336, -0.27509585, 0.21835066],
        [0.28962948, 0.95642509, -0.03695701],
        [-0.19866933, 0.09784340, 0.97517033],
    ])
    crystal = Crystal("test", space_group, cell)
    reference = NativeCrystal.create(crystal).reciprocal()
    pattern = Pattern(
        euler_deg=np.zeros(3),
        rotation=rotation,
        reciprocal=reference @ rotation.T,
        goodness=1.0,
        rms_error_deg=0.0,
        hkl=np.empty((0, 3), dtype=int),
        pk_index=np.empty(0, dtype=int),
        err_deg=np.empty(0),
        energy_kev=np.empty(0),
        pred_intens=np.empty(0),
    )
    result = FrameResult(
        peaks=np.empty(0, dtype=[]),
        patterns=(pattern,),
        threshold_used=100.0,
        total_sum=0.0,
        sum_above_threshold=0.0,
        num_above_threshold=0,
        peaksearch_seconds=0.0,
        indexing_seconds=0.0,
        metadata={"sample_position": (0.0, 0.0, 0.0)},
    )
    live = ResultSet((result,), crystal=crystal)
    step = _step(0, (0,))
    reciprocal_text = "".join(
        f"<{name}>{' '.join(map(str, vector))}</{name}>"
        for name, vector in zip(("astar", "bstar", "cstar"), pattern.reciprocal, strict=True)
    )
    step = step.replace(
        '<recip_lattice><astar>-10 5 -6</astar><bstar>-15 -2 -1</bstar><cstar>-2 13 8</cstar></recip_lattice>',
        f"<recip_lattice>{reciprocal_text}</recip_lattice>",
    ).replace(
        '<SpaceGroup>225</SpaceGroup><latticeParameters unit="nm">0.4 0.4 0.4 90 90 90</latticeParameters>',
        f'<SpaceGroup>{space_group}</SpaceGroup><latticeParameters unit="nm">'
        f"{cell.a} {cell.b} {cell.c} {cell.alpha} {cell.beta} {cell.gamma}</latticeParameters>",
    )
    path = tmp_path / f"{space_group}.xml"
    path.write_text(f"<AllSteps>{step}</AllSteps>")

    loaded = load_visualization_xml(path)

    np.testing.assert_allclose(
        loaded.pattern_rotations, live.to_visualization().pattern_rotations, atol=1e-8, rtol=0
    )
    if space_group == 168:
        for color, kwargs in (
            ("rodrigues", {}),
            ("misorientation", {"misorientation_reference": (0, 0)}),
        ):
            np.testing.assert_allclose(
                prepare_map(loaded, color=color, **kwargs).colors,
                prepare_map(live, color=color, **kwargs).colors,
                atol=1e-8,
                rtol=0,
            )


def test_declared_peak_count_preserves_rows_with_missing_fields(tmp_path):
    path = tmp_path / "partial.xml"
    path.write_text(
        """<AllSteps><step><detector><peaksXY Npeaks="3">
        <Xpixel>1 2</Xpixel><Ypixel>3 4 5</Ypixel>
        </peaksXY></detector></step></AllSteps>"""
    )
    dataset = load_visualization_xml(path)
    assert dataset.frame_n_peaks.tolist() == [3]
    assert len(dataset.peaks) == 3
    assert np.isnan(dataset.peaks["fit_x"][2])
    assert dataset.peaks["fit_y"][2] == 5


@pytest.mark.parametrize(
    ("fragment", "field"),
    [
        ("<Nx>nan</Nx>", "Nx"),
        ('<Nx>1</Nx><peaksXY Npeaks="nan"/>', "peaksXY.Npeaks"),
    ],
)
def test_malformed_integer_fields_report_file_step_and_field(tmp_path, fragment, field):
    path = tmp_path / "malformed.xml"
    path.write_text(f"<AllSteps><step><detector>{fragment}</detector></step></AllSteps>")

    with pytest.raises(ValueError, match=rf"{path}.*step 0.*{field}"):
        load_visualization_xml(path)


def test_load_visualization_xml_accepts_explicit_geometry(legacy_xml):
    geometry = Geometry(GEOMETRY)
    dataset = load_visualization_xml(legacy_xml, geometry=geometry)
    assert dataset.geometry is geometry


@requires_liblaue
def test_embedded_relative_geometry_resolves_from_xml_directory(tmp_path):
    geometry = tmp_path / "geometry.xml"
    geometry.write_text(GEOMETRY.read_text())
    path = tmp_path / "indexing.xml"
    path.write_text(
        """<AllSteps><step><detector><geoFile>geometry.xml</geoFile></detector></step></AllSteps>"""
    )

    dataset = load_visualization_xml(path)

    assert dataset.geometry.path == geometry


def test_missing_embedded_geometry_is_not_required(tmp_path):
    path = tmp_path / "minimal.xml"
    path.write_text(
        """<AllSteps><step><Xsample>0</Xsample><Ysample>1</Ysample><Zsample>2</Zsample>
        <detector><geoFile>/missing/geometry.xml</geoFile><Nx>2</Nx><Ny>3</Ny>
        <peaksXY Npeaks="1"><Xpixel>0</Xpixel><Ypixel>1</Ypixel></peaksXY></detector>
        </step></AllSteps>"""
    )
    dataset = load_visualization_xml(path)
    assert dataset.n_frames == 1
    assert dataset.geometry is None
    assert dataset.n_patterns == 0


def test_load_visualization_xml_validates_frame_ids(legacy_xml):
    with pytest.raises(ValueError, match="contain 2"):
        load_visualization_xml(legacy_xml, frame_ids=(1,))

    dataset = load_visualization_xml(
        legacy_xml, frame_ids=np.array([10, 20], dtype=np.int64)
    )
    assert dataset.frame_ids == (10, 20)
    assert all(type(value) is int for value in dataset.frame_ids)


def _assert_converted_equal(actual, expected):
    assert actual.frame_ids == expected.frame_ids
    assert actual.detector_ids == expected.detector_ids
    assert actual.input_images == expected.input_images
    for item in fields(VisualizationDataset):
        if item.name in {"frame_ids", "detector_ids", "input_images", "images", "crystal", "geometry"}:
            continue
        actual_value = getattr(actual, item.name)
        expected_value = getattr(expected, item.name)
        if item.name == "peaks":
            for field_name in actual_value.dtype.names:
                np.testing.assert_allclose(
                    actual_value[field_name], expected_value[field_name],
                    rtol=2e-6, atol=2e-6, equal_nan=True,
                )
        else:
            np.testing.assert_allclose(
                actual_value, expected_value,
                rtol=2e-6, atol=2e-6, equal_nan=True,
            )
    assert actual.crystal == expected.crystal


def test_convert_xml_round_trips_normalized_arrays_and_refuses_overwrite(legacy_xml):
    expected = load_visualization_xml(legacy_xml)
    output = convert_xml(legacy_xml)

    assert output == legacy_xml.with_suffix(".h5")
    assert is_results_file(output)
    _assert_converted_equal(load_results(output), expected)
    with pytest.raises(FileExistsError):
        convert_xml(legacy_xml)
    assert convert_xml(legacy_xml, overwrite=True) == output


def test_convert_xml_records_unresolved_geometry_and_allows_missing_crystal(tmp_path):
    path = tmp_path / "minimal.xml"
    path.write_text(
        """<AllSteps><step><detector><geoFile>missing/geometry.xml</geoFile>
        <Nx>2</Nx><Ny>3</Ny></detector></step></AllSteps>"""
    )

    output = convert_xml(path)
    loaded = load_results(output)

    assert loaded.crystal is None
    assert loaded.geometry is None
    with h5py.File(output) as source:
        assert source["geometry"].attrs["path"] == "missing/geometry.xml"
        assert "xml" not in source["geometry"]
        assert "crystal" not in source


def test_conversion_is_byte_stable_at_fixed_timestamp(legacy_xml, frozen_results_clock):
    first = convert_xml(legacy_xml, legacy_xml.with_name("first.h5"))
    time.sleep(1.1)
    second = convert_xml(legacy_xml, legacy_xml.with_name("second.h5"))
    assert first.read_bytes() == second.read_bytes()


def test_conversion_rejects_conflicting_run_parameters_before_overwriting(tmp_path):
    xml_path = tmp_path / "mixed.xml"
    xml_path.write_text('''<AllSteps>
        <step><indexing cone="72" /></step>
        <step><indexing cone="60" /></step>
        </AllSteps>''')
    output = tmp_path / "existing.h5"
    output.write_bytes(b"keep existing output")
    with pytest.raises(ValueError, match="conflicting run parameter 'cone_deg'"):
        convert_xml(xml_path, output, overwrite=True)
    assert output.read_bytes() == b"keep existing output"


def test_conversion_collects_parameters_from_later_steps_and_old_attribute_names(tmp_path):
    xml_path = tmp_path / "parameters.xml"
    xml_path.write_text('''<AllSteps>
        <step><detector><cosmicFilter>None</cosmicFilter>
        <peaksXY maxRfactor="None" max_number="None" peakProgram="None" />
        </detector><indexing cone="None" hklPrefer="None" keVmaxTest="nan" /></step>
        <step><detector><cosmicFilter>False</cosmicFilter>
        <peaksXY max_number="50" min_separation="10" maskFile="mask.h5" />
        </detector><indexing indexProgram="euler" cone="72" /></step>
        </AllSteps>''')
    with h5py.File(convert_xml(xml_path)) as source:
        run = source["run"].attrs
        assert run["program"] == "euler"
        assert run["cone_deg"] == 72
        assert not bool(run["cosmic_filter"])
        assert run["max_peaks"] == 50 and run["min_separation"] == 10
        assert run["mask_file"] == "mask.h5"
        assert "max_rfactor" not in run and "hkl_prefer" not in run
        assert "kev_max_test" not in run and "peak_program" not in run


def test_is_results_file_rejects_frame_hdf5_and_non_hdf5(tmp_path):
    frame = tmp_path / "frame.h5"
    with h5py.File(frame, "w") as target:
        target.create_dataset("entry1/data/data", data=np.zeros((2, 2)))

    assert not is_results_file(frame)
    assert not is_results_file(tmp_path / "missing.h5")
    assert not is_results_file(__file__)
