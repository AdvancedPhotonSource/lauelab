# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Depth inspection: ROI placement, traces, normalization, and the Plotly builders.

The numerical tests run the L1 contract fixtures through ``ArrayPoint`` and
need no native library. The file-based tests check the same functions on
reconstructed point files.
"""

import re

import numpy as np
import pytest

from lauelab.indexing import InputError
from lauelab.reconstruct.inspection import (
    ArrayPoint, DepthTrace, ReferenceImage, bounds_center, depth_trace, reference_image,
    roi_traces, square_bounds,
)
from lauelab.visualization import (
    DEPTH_AXIS_OPTIONS, INTENSITY_OPTIONS, REFERENCE_OPTIONS, RoiOverlay,
    plot_depth_trace, plot_reference_image, plot_roi_traces,
)
from tests.data.reconstruction_contract import fixtures


def _roles(figure):
    return [trace.meta["role"] for trace in figure.data]


@pytest.fixture
def point():
    return ArrayPoint(fixtures.STORED["int32"]["data"], fixtures.DEPTH_UM, "fixture")


# --- ROI placement -------------------------------------------------------------

@pytest.mark.parametrize(("size", "click", "bounds", "centre"), fixtures.PLACEMENTS)
def test_square_bounds_follow_the_contract_fixtures(size, click, bounds, centre):
    assert square_bounds(size, *click, fixtures.IMAGE_SHAPE) == bounds
    assert bounds_center(bounds) == centre
    y0, y1, x0, x1 = bounds
    assert (y1 - y0) * (x1 - x0) == size * size


@pytest.mark.parametrize(("size", "click"), fixtures.REJECTED_PLACEMENTS)
def test_out_of_image_squares_are_rejected_not_clipped(size, click):
    with pytest.raises(InputError, match="does not fit|finite"):
        square_bounds(size, *click, fixtures.IMAGE_SHAPE)


@pytest.mark.parametrize("size", fixtures.INVALID_SIZES, ids=repr)
def test_square_size_must_be_a_positive_integer(size):
    with pytest.raises(InputError, match="ROI size"):
        square_bounds(size, 1.0, 1.0, fixtures.IMAGE_SHAPE)


def test_square_bounds_accept_numpy_scalars():
    assert square_bounds(np.int64(2), np.float32(2.5), np.float64(1.5), (3, 4)) == (1, 3, 2, 4)


# --- Traces --------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(fixtures.ROI_TRACES))
def test_roi_trace_sums_the_stored_pixels_in_the_bounds(point, name):
    bounds, expected = fixtures.ROI_TRACES[name]
    trace = depth_trace(point, bounds)
    np.testing.assert_array_equal(trace.values, expected)
    assert trace.values.dtype == np.int64
    assert trace.bounds == bounds
    assert trace.n_pixels == (bounds[1] - bounds[0]) * (bounds[3] - bounds[2])
    np.testing.assert_array_equal(trace.depth_um, fixtures.DEPTH_UM)
    np.testing.assert_array_equal(trace.depth_index, [0, 1, 2, 3])
    assert not trace.values.flags.writeable


def test_full_frame_trace_is_the_embedded_reduction(point):
    trace = depth_trace(point)
    np.testing.assert_array_equal(trace.values, fixtures.STORED["int32"]["depth_intensity"])
    assert trace.bounds is None and trace.n_pixels == 0 and trace.label == "Full frame"
    assert trace.point_id == "fixture"


def test_roi_traces_keep_names_and_order(point):
    rois = {name: bounds for name, (bounds, _) in fixtures.ROI_TRACES.items()}
    traces = roi_traces(point, rois)
    assert list(traces) == list(rois)
    assert all(trace.label == name for name, trace in traces.items())
    assert roi_traces(point, {}) == {}


def test_traces_reject_bounds_outside_the_image(point):
    for bounds in [(0, 4, 0, 1), (0, 1, 0, 5), (1, 1, 0, 1), (0, 1, 0), (0.0, 1, 0, 1)]:
        with pytest.raises(InputError):
            depth_trace(point, bounds)


@pytest.mark.parametrize(("values", "expected"), fixtures.NORMALIZED)
def test_normalization_needs_a_positive_maximum(values, expected):
    trace = DepthTrace(values, np.arange(len(values), dtype=float), "p", None, "t")
    result = trace.normalized()
    if expected is None:
        assert not result.available and result.values is None
        assert "not positive" in result.reason and "t:" in result.reason
    else:
        assert result.available and result.reason == ""
        np.testing.assert_array_equal(result.values, expected)
        assert result.values.max() == 1.0


@pytest.mark.parametrize(("values", "kept", "omitted"), fixtures.LOG_SAMPLES)
def test_log_axis_omits_nonpositive_samples(values, kept, omitted):
    trace = DepthTrace(values, np.arange(len(values), dtype=float), "p", None, "t")
    samples = trace.log_samples()
    np.testing.assert_array_equal(samples.kept, kept)
    assert samples.n_omitted == omitted


def test_float_point_traces_accumulate_in_float64():
    point = ArrayPoint(fixtures.COMPUTED.astype(np.float32), fixtures.DEPTH_UM)
    trace = depth_trace(point, (0, 3, 0, 4))
    assert trace.values.dtype == np.float64
    np.testing.assert_allclose(trace.values, fixtures.COMPUTED.astype(np.float32).sum(axis=(1, 2)))


# --- Reference images and ArrayPoint -------------------------------------------

def test_array_point_offers_only_the_reconstructed_reference(point):
    image = reference_image(point)
    np.testing.assert_array_equal(image.image, fixtures.STORED["int32"]["sum_reconstructed"])
    assert (image.kind, image.values, image.point_id) == ("sum_reconstructed", "stored", "fixture")
    assert image.shape == (3, 4) and not image.image.flags.writeable
    for kind in ("first_raw", "sum_raw"):
        with pytest.raises(InputError, match="not available"):
            reference_image(point, kind)
    with pytest.raises(InputError, match="kind must be one of"):
        reference_image(point, "sum")


def test_array_point_reads_frames_and_regions_as_owned_arrays(point):
    frame = point.frame(2)
    frame[:] = 0
    np.testing.assert_array_equal(point.frame(2), fixtures.STORED["int32"]["data"][2])
    assert point.region((1, 3, 2, 4), slice(1, 3)).shape == (2, 2, 2)
    with pytest.raises(IndexError):
        point.frame(4)
    with pytest.raises(InputError):
        ArrayPoint(np.zeros((2, 2)), [0.0, 1.0])
    with pytest.raises(InputError):
        ArrayPoint(np.zeros((2, 2, 2)), [0.0, np.nan])


# --- Figures -------------------------------------------------------------------

def test_reference_image_figure_draws_squares_on_pixel_edges(point):
    image = reference_image(point)
    rois = [RoiOverlay("a", (0, 1, 0, 1), "red"), RoiOverlay("b", (1, 3, 2, 4), "blue")]
    figure = plot_reference_image(image, rois, limits=(0.0, 10.0))
    assert _roles(figure) == ["image", "roi", "roi"]
    assert figure.data[0].uid == "reference-sum_reconstructed"
    assert figure.data[0].meta["values"] == "stored"
    assert (figure.data[0].zmin, figure.data[0].zmax) == (0.0, 10.0)
    assert list(figure.data[1].x) == [-0.5, 0.5, 0.5, -0.5, -0.5]
    assert list(figure.data[1].y) == [-0.5, -0.5, 0.5, 0.5, -0.5]
    assert figure.data[2].uid == "roi-62" and figure.data[2].line.color == "blue"  # b.hex()
    assert "centre: (2.5, 1.5)" in figure.data[2].hovertemplate
    assert list(figure.layout.xaxis.range) == [-0.5, 3.5]
    assert list(figure.layout.yaxis.range) == [2.5, -0.5]
    assert figure.layout.xaxis.scaleanchor == "y"
    assert figure.layout.uirevision == "reference-4x3"

    with pytest.raises(ValueError, match="outside"):
        plot_reference_image(image, [RoiOverlay("c", (0, 4, 0, 1), "red")])
    with pytest.raises(TypeError):
        plot_reference_image(image.image)
    with pytest.raises(ValueError, match="limits"):
        plot_reference_image(image, limits=(1.0, 1.0))


def test_roi_uids_are_valid_css_class_names_for_any_roi_name(point):
    """Plotly selects removed traces by ``.cb<uid>``; a space in a uid breaks the update."""
    names = ["ROI 1", "ROI 11", "a.b#c", "µ spot", "ROI 1 "]
    figure = plot_reference_image(
        reference_image(point), [RoiOverlay(name, (0, 1, 0, 1), "red") for name in names]
    )
    traces = plot_roi_traces(roi_traces(point, {name: (0, 1, 0, 1) for name in names}))
    for uids in ([trace.uid for trace in figure.data[1:]], [trace.uid for trace in traces.data]):
        assert all(re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", uid) for uid in uids)
        assert len(set(uids)) == len(names)
    assert [trace.meta["name"] for trace in traces.data] == names


def test_depth_trace_figure_supports_both_axes_log_and_selection(point):
    trace = depth_trace(point)
    linear = plot_depth_trace(trace, selected_index=2)
    assert _roles(linear) == ["trace"]
    assert list(linear.data[0].x) == list(fixtures.DEPTH_UM)
    assert linear.layout.xaxis.title.text == "Depth (um)"
    assert linear.layout.shapes[0].x0 == 0.5
    assert linear.layout.uirevision == "depth-trace-depth"

    by_index = plot_depth_trace(trace, axis="index", log=True)
    assert list(by_index.data[0].x) == [0, 1, 2, 3]
    np.testing.assert_array_equal(by_index.data[0].y, [1, np.nan, 131581, np.nan])
    assert by_index.layout.yaxis.type == "log"
    assert by_index.layout.xaxis.title.text == "Depth index"
    assert "2 of 4 samples are not positive" in by_index.layout.annotations[0].text

    with pytest.raises(ValueError, match="axis"):
        plot_depth_trace(trace, axis="um")
    with pytest.raises(ValueError, match="selected_index"):
        plot_depth_trace(trace, selected_index=4)


def test_roi_traces_figure_keeps_identity_and_reports_unavailable_normalization(point):
    traces = roi_traces(point, {
        "spot": (0, 1, 0, 1), "dark": (0, 1, 1, 2), "square": (1, 3, 2, 4),
    })
    figure = plot_roi_traces(traces, colors={"dark": "black"})
    assert _roles(figure) == ["roi", "roi", "roi"]
    assert [trace.meta["name"] for trace in figure.data] == ["spot", "dark", "square"]
    assert figure.data[1].line.color == "black" and figure.data[1].uid == "roi-trace-" + b"dark".hex()
    assert figure.layout.yaxis.title.text == "Stored intensity"
    assert not figure.layout.annotations

    normalized = plot_roi_traces(traces, normalized=True, trace_update={"roi": {"mode": "lines"}})
    assert [trace.meta["name"] for trace in normalized.data] == ["spot", "square"]
    assert max(normalized.data[0].y) == 1.0
    assert normalized.data[0].mode == "lines"
    assert "dark: maximum 0 is not positive" in normalized.layout.annotations[0].text
    assert normalized.layout.yaxis.title.text == "Normalized intensity"

    logged = plot_roi_traces(traces, axis="index", log=True)
    assert list(logged.data[0].x) == [0, 1, 2, 3]
    assert "9 of 12 samples are not positive" in logged.layout.annotations[0].text

    empty = plot_roi_traces({})
    assert not empty.data and "No ROIs selected" in empty.layout.annotations[0].text
    with pytest.raises(ValueError, match="unknown trace roles"):
        plot_roi_traces(traces, trace_update={"image": {}})


def test_options_are_discoverable():
    assert [choice.value for choice in DEPTH_AXIS_OPTIONS] == ["depth", "index"]
    assert [choice.value for choice in INTENSITY_OPTIONS] == ["sum", "normalized"]
    assert [choice.value for choice in REFERENCE_OPTIONS] == [
        "sum_reconstructed", "first_raw", "sum_raw",
    ]
    assert isinstance(ReferenceImage(np.zeros((2, 2)), "sum_raw", "raw", "p", "x"), ReferenceImage)


# --- Scan file and the guide example ---------------------------------------------

@pytest.fixture(scope="module")
def scan_point(tmp_path_factory):
    """The synthetic scan reconstructed into a scan directory, as the guide uses it."""
    from conftest import LIBLAUE_AVAILABLE
    from lauelab.reconstruct import reconstruct_scan
    from tests.data.reconstruction.generate_reference import (
        DEPTH_RANGE_UM, GEOMETRY_FILE, write_input_file,
    )

    if not LIBLAUE_AVAILABLE:
        pytest.skip("liblaue.so is not present in the installed lauelab package")
    work = tmp_path_factory.mktemp("scan")
    source = work / "synthetic.h5"
    write_input_file(source)
    result = reconstruct_scan(
        [source], work / "run", geometry=GEOMETRY_FILE, detector=0,
        point_ids=["scan12_p1"], depth_range=DEPTH_RANGE_UM, wire_edge="both",
        threads_per_worker=1, rows_per_stripe=31,
    )
    assert result.complete
    return result.path


def test_guide_example_runs_as_documented(scan_point):
    """The flow shown in docs/guides/depth-inspection.md."""
    from lauelab.reconstruct import ScanReader

    with ScanReader(scan_point) as scan:
        point = scan.point("scan12_p1")
        background = reference_image(point, "sum_reconstructed")
        assert background.values == "stored"
        rows, columns = background.shape
        spot = square_bounds(5, 64.2, 63.8, (rows, columns))
        edge = square_bounds(4, 52.0, 70.0, (rows, columns))
        assert spot == (62, 67, 62, 67)
        assert bounds_center(spot) == (64.0, 64.0)
        assert bounds_center(edge) == (51.5, 69.5)
        whole = depth_trace(point)
        traces = roi_traces(point, {"spot": spot, "edge": edge})
        np.testing.assert_array_equal(whole.values, point.depth_intensity())
        np.testing.assert_array_equal(
            traces["spot"].values, point.region(spot).sum(axis=(1, 2), dtype=np.int64)
        )
        raw = reference_image(point, "first_raw")
        assert raw.values == "raw" and raw.image.dtype == np.uint16
        pixels = point.region(spot)
        assert pixels.shape == (51, 5, 5) and pixels.dtype == np.int32

    # The synthetic spots switch off at 0, -10, and +12 um, and the reference
    # README records that reconstruction places each within one or two bins.
    peak_depth_um = whole.depth_um[whole.values.argmax()]
    spot_peak_um = traces["spot"].depth_um[traces["spot"].values.argmax()]
    assert peak_depth_um == 0.0 and spot_peak_um == 0.0
    edge_peak_um = traces["edge"].depth_um[traces["edge"].values.argmax()]
    assert abs(edge_peak_um - (-10.0)) <= 2.0

    normalized = traces["spot"].normalized()
    assert normalized.available and normalized.values.max() == 1.0
    samples = traces["edge"].log_samples()
    assert len(samples.kept) + samples.n_omitted == 51 and samples.n_omitted > 0

    overlays = [RoiOverlay("spot", spot, "rgb(230,90,60)"), RoiOverlay("edge", edge, "rgb(60,140,230)")]
    image_figure = plot_reference_image(background, overlays)
    whole_figure = plot_depth_trace(whole, axis="depth", selected_index=int(whole.values.argmax()))
    roi_figure = plot_roi_traces(
        traces, colors={"spot": "rgb(230,90,60)", "edge": "rgb(60,140,230)"}, normalized=True,
    )
    assert _roles(image_figure) == ["image", "roi", "roi"]
    assert whole_figure.layout.shapes[0].x0 == 0.0
    assert [trace.meta["name"] for trace in roi_figure.data] == ["spot", "edge"]
    assert image_figure.layout.uirevision == "reference-128x128"


def test_array_point_matches_the_scan_file(scan_point):
    from lauelab.reconstruct import ScanReader

    with ScanReader(scan_point) as scan:
        point = scan.point("scan12_p1")
        stack = np.concatenate([block for _, block in point.iter_blocks(10**9)])
        memory = ArrayPoint(stack, point.depth_um, "scan12_p1")
        bounds = square_bounds(3, 68, 58, point.shape[1:])
        np.testing.assert_array_equal(depth_trace(memory, bounds).values, depth_trace(point, bounds).values)
        np.testing.assert_array_equal(depth_trace(memory).values, depth_trace(point).values)
        np.testing.assert_array_equal(
            reference_image(memory).image, reference_image(point).image
        )


def test_roi_reduction_bounds_live_pixel_reads(monkeypatch):
    import weakref

    images = np.arange(17 * 11 * 13, dtype=np.int32).reshape(17, 11, 13)
    point = ArrayPoint(images, np.arange(17))
    bounds = (2, 10, 1, 12)
    plane_bytes = 8 * 11 * 4
    budget = 3 * plane_bytes
    read = point.region
    previous = None
    selections = []

    def bounded_region(bounds, depths=None):
        nonlocal previous
        assert previous is None or previous() is None, "previous pixel block was retained"
        assert depths is not None
        block = read(bounds, depths)
        assert block.nbytes <= budget
        previous = weakref.ref(block)
        selections.append(depths)
        return block

    monkeypatch.setattr(point, "region", bounded_region)
    trace = depth_trace(point, bounds, max_bytes=budget)
    np.testing.assert_array_equal(trace.values, images[:, 2:10, 1:12].sum(axis=(1, 2)))
    assert len(selections) == 6
    assert selections[-1] == slice(15, 17)
    with pytest.raises(InputError, match="one ROI plane"):
        depth_trace(point, bounds, max_bytes=plane_bytes - 1)


@pytest.mark.parametrize("normalized", [False, True])
def test_log_figures_leave_gaps_at_invalid_samples(normalized):
    trace = DepthTrace(np.array([1, 0, -1, 4]), np.arange(4.), "p", None, "roi")
    figures = [plot_depth_trace(trace, log=True),
               plot_roi_traces({"roi": trace}, log=True, normalized=normalized)]
    for figure in figures:
        series = figure.data[0]
        np.testing.assert_array_equal(series.x, [0, 1, 2, 3])
        assert np.isnan(series.y[1:3]).all()
        assert series.y[0] > 0 and series.y[3] > 0
        assert series.connectgaps is False
        np.testing.assert_array_equal(series.customdata, [0, 1, 2, 3])
