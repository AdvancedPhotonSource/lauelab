# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from conftest import requires_liblaue


ROOT = Path(__file__).resolve().parents[1]
pytestmark = requires_liblaue


def _run_python(script, *arguments):
    environment = dict(os.environ)  # child imports the same installed lauelab as the test runner
    return subprocess.run(
        [sys.executable, "-c", script, *map(str, arguments)],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_invalid_native_calls_return_errors_without_terminating_python():
    script = r'''
from lauelab.indexing._liblaue import ffi, get_library

lib = get_library()
error = ffi.new("char[256]")
result = ffi.new("laue_frame_result *")
params = ffi.new("laue_peak_params *")
info = ffi.new("laue_detector_info *")

assert lib.laue_geometry_from_file(ffi.NULL, error, 256) == ffi.NULL
assert ffi.string(error) == b"geometry path is NULL"
assert lib.laue_crystal_create(b"bad", 0, 1, 1, 1, 90, 90, 90,
                               ffi.NULL, 0, error, 256) == ffi.NULL
assert ffi.string(error) == b"invalid crystal parameters"

assert lib.laue_geometry_detector_count(ffi.NULL) == 0
assert lib.laue_geometry_find_detector(ffi.NULL, b"detector") == -1
assert lib.laue_crystal_reciprocal(ffi.NULL, ffi.NULL) == 1
assert lib.laue_geometry_detector_info(ffi.NULL, 0, info, error, 256) == 1
assert ffi.string(error) == b"invalid detector index"
wire = ffi.new("laue_wire_info *")
assert lib.laue_geometry_wire_info(ffi.NULL, wire, error, 256) == 1
assert ffi.string(error) == b"invalid wire geometry output"
assert lib.laue_recon_create(ffi.NULL, 0, ffi.NULL, error, 256) == ffi.NULL
assert ffi.string(error) == b"invalid reconstruction parameters"
assert lib.laue_recon_set_wire_positions(ffi.NULL, ffi.NULL, 0, 0) == 1
assert lib.laue_recon_stripe(ffi.NULL, ffi.NULL, 0, 0, 0, 0, ffi.NULL,
                             ffi.NULL, ffi.NULL, ffi.NULL, 0, ffi.NULL) == 1
assert lib.laue_recon_n_depths(ffi.NULL) == 0
assert lib.laue_recon_depth_um(ffi.NULL, 0) != lib.laue_recon_depth_um(ffi.NULL, 0)
assert ffi.string(lib.laue_recon_last_error(ffi.NULL)) == b"reconstruction context is NULL"
lib.laue_recon_free(ffi.NULL)


assert lib.laue_find_peaks(ffi.NULL, 1, 1, params, result) == 1
assert result.status == 1
assert ffi.string(result.message) == b"invalid peak-search input"
assert lib.laue_find_peaks_typed(ffi.NULL, lib.LAUE_PIXEL_I32, 1, 1, params, result) == 1
assert ffi.string(result.message) == b"invalid peak-search input"
pixel = ffi.new("unsigned short[1]", [5])
assert lib.laue_find_peaks_typed(pixel, 99, 1, 1, params, result) == 1
assert ffi.string(result.message) == b"unsupported pixel type 99"
params.boxsize = params.min_size = params.min_separation = 1
params.max_peaks = -1
assert lib.laue_find_peaks(pixel, 1, 1, params, result) == 1
assert ffi.string(result.message) == b"invalid or unsupported peak-search parameters"
assert lib.laue_pixels_to_q(ffi.NULL, 0, result) == 1
assert result.status == 1
assert ffi.string(result.message) == b"geometry is NULL"
assert lib.laue_index(ffi.NULL, ffi.NULL, result) == 1
assert result.status == 1
assert ffi.string(result.message) == b"invalid indexing input"

assert lib.laue_find_peaks(ffi.NULL, 1, 1, params, ffi.NULL) == 1
assert lib.laue_find_peaks_typed(ffi.NULL, lib.LAUE_PIXEL_U16, 1, 1, params, ffi.NULL) == 1
assert lib.laue_pixels_to_q(ffi.NULL, 0, ffi.NULL) == 1
assert lib.laue_index(ffi.NULL, ffi.NULL, ffi.NULL) == 1
lib.laue_frame_result_free(result)
lib.laue_frame_result_free(result)
lib.laue_frame_result_free(ffi.NULL)
lib.laue_geometry_free(ffi.NULL)
lib.laue_crystal_free(ffi.NULL)
'''
    completed = _run_python(script)

    assert completed.returncode == 0, completed.stderr


def test_out_of_range_detector_slot_is_rejected_without_crashing(tmp_path):
    source = ROOT / "tests/data/geo/geoN_2022-03-29_14-15-05.xml"
    path = tmp_path / "slot-3.xml"
    path.write_text(source.read_text().replace('<Detector N="2">', '<Detector N="3">'))
    script = r'''
import sys
from lauelab.indexing._liblaue import Geometry

try:
    Geometry(sys.argv[1])
except ValueError as error:
    assert "unable to read detector geometry" in str(error)
else:
    raise AssertionError("invalid detector slot was accepted")
'''

    completed = _run_python(script, path)

    assert completed.returncode == 0, completed.stderr


def test_saturated_frame_does_not_overflow_native_stack():
    script = r'''
import numpy as np
from lauelab.indexing import Indexer, PeakParams

indexer = Indexer(
    "tests/data/geo/geoN_2022-03-29_14-15-05.xml",
    peak_params=PeakParams(threshold=100),
)
indexer.index(np.full((350, 350), 500, np.uint16))
'''

    completed = _run_python(script)

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize(
    ("source_name", "arguments"),
    [
        ("peaksearch_memory.c", ()),
        ("recon_memory.c", (ROOT / "tests/data/geo/geoN_2022-03-29_14-15-05.xml",)),
    ],
)
def test_native_calls_release_allocations(tmp_path, source_name, arguments):
    valgrind = shutil.which("valgrind")
    compiler = shutil.which("cc")
    if valgrind is None:
        pytest.skip("Valgrind is required for the native memory regression test")
    if compiler is None:
        pytest.skip("a C compiler is required for the native memory regression test")

    from importlib import resources

    library = Path(str(resources.files("lauelab.indexing.bin") / "liblaue.so"))
    source = ROOT / "tests/native" / source_name
    executable = tmp_path / source.stem.replace("_", "-")
    compiled = subprocess.run(
        [
            compiler,
            "-std=c99",
            "-O0",
            "-g",
            f"-I{ROOT / 'src/lauelab/indexing/src/liblaue'}",
            str(source),
            str(library),
            "-lm",
            "-o",
            str(executable),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert compiled.returncode == 0, compiled.stderr

    checked = subprocess.run(
        [
            valgrind,
            "--quiet",
            "--leak-check=full",
            "--show-leak-kinds=all",
            "--errors-for-leak-kinds=all",
            f"--suppressions={ROOT / 'tests/native/libgomp.supp'}",
            "--track-origins=yes",
            "--error-exitcode=99",
            str(executable),
            *map(str, arguments),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert checked.returncode == 0, checked.stdout + checked.stderr
