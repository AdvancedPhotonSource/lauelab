# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""HDF5 and summary output for in-process wire-scan reconstruction."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np


PIXEL_DTYPES = {
    0: np.dtype("<f4"),
    1: np.dtype("<i4"),
    2: np.dtype("<i2"),
    3: np.dtype("<u2"),
    5: np.dtype("<f8"),
    6: np.dtype("i1"),
    7: np.dtype("u1"),
}


def pixel_type(dtype: np.dtype) -> int:
    dtype = np.dtype(dtype)
    for number, candidate in PIXEL_DTYPES.items():
        if dtype.kind == candidate.kind and dtype.itemsize == candidate.itemsize:
            return number
    raise ValueError(f"unsupported output dtype {dtype}")


def normalization_rescale(output_type: int) -> float:
    return float({1: 1 << 15, 2: 1 << 7, 3: (1 << 8) - 1,
                  6: 1 << 3, 7: (1 << 4) - 1}.get(output_type, 1))


def _copy_metadata(source: h5py.File, target: h5py.File, *, exclude_entry=(), exclude_data=()) -> None:
    for key, value in source.attrs.items():
        target.attrs[key] = value
    for name in source:
        if name == "entry1":
            entry = target.create_group(name)
            for key, value in source[name].attrs.items():
                entry.attrs[key] = value
            for child in source[name]:
                if child == "wire" or child in exclude_entry:
                    continue
                if child == "data":
                    group = entry.create_group("data")
                    for key, value in source[name][child].attrs.items():
                        group.attrs[key] = value
                    for nested in source[name][child]:
                        if nested != "data" and nested not in exclude_data:
                            source.copy(source[name][child][nested], group, name=nested)
                else:
                    source.copy(source[name][child], entry, name=child)
        else:
            source.copy(source[name], target, name=name)


def create_outputs(source: h5py.File, output_base: str | Path, depth_um: np.ndarray,
                   shape: tuple[int, int], dtype: np.dtype, *, cosmic_filter: bool,
                   norm_exponent: float | None, norm_threshold: float | None,
                   norm_rescale: float) -> tuple[list[h5py.File], list[str]]:
    """Create all output files and hold them open for stripe writes."""
    output_base = str(output_base)
    Path(output_base).parent.mkdir(parents=True, exist_ok=True)
    handles = []
    paths = []
    try:
        for index, depth in enumerate(depth_um):
            path = f"{output_base}{index}.h5"
            target = h5py.File(path, "w")
            handles.append(target)
            paths.append(path)
            _copy_metadata(source, target)
            data = target["entry1/data"].create_dataset("data", shape=shape, dtype=dtype)
            data.attrs["signal"] = np.int32(1)
            if "entry1/depth" in target:
                depth_data = target["entry1/depth"]
            else:
                depth_data = target["entry1"].create_dataset(
                    "depth", data=np.asarray([0.0], dtype=np.float64)
                )
                depth_data.attrs["units"] = "micron"
            depth_data[...] = np.asarray([depth], dtype=np.float64)
            if cosmic_filter or norm_exponent is not None:
                group = target["entry1"].require_group("microDiffraction")
                if cosmic_filter:
                    group.create_dataset("cosmic_filter", data=np.asarray([1], dtype=np.int32))
                if norm_exponent is not None:
                    group.create_dataset("norm_exponent", data=np.asarray([norm_exponent]))
                    group.create_dataset("norm_threshold", data=np.asarray([norm_threshold]))
                    group.create_dataset("norm_rescale", data=np.asarray([norm_rescale]))
        return handles, paths
    except Exception:
        for handle in handles:
            handle.close()
        raise


def write_stripe(handles: list[h5py.File], row0: int, values: np.ndarray) -> None:
    """Write one float64 stripe, allowing HDF5 to perform numeric conversion."""
    row1 = row0 + values.shape[1]
    for index, handle in enumerate(handles):
        handle["entry1/data/data"][row0:row1, :] = values[index]


def write_summary(path: str | Path, *, input_path: str, output_base: str,
                  geometry_path: str, detector: int, depth_um: np.ndarray,
                  resolution: float, wire_edge: int, output_type: int,
                  percent_brightest: float, memory_limit_mb: int,
                  cosmic_filter: bool, normalization: str | None,
                  norm_exponent: float | None, norm_threshold: float | None,
                  norm_rescale: float, scan_number: int | None,
                  sample_position: tuple[float, float, float] | None,
                  energy_kev: float | None, image_geometry,
                  rows_per_stripe: int, elapsed: float,
                  depth_intensity: np.ndarray, verbose: int = 1) -> None:
    """Write the reconstruction parameter block and intensity-vs-depth array."""
    lines = [
        "$filetype\tgeometryFileN;depthSortedInfo",
        "",
        f"$ws_infile\t\t\t\t{input_path}",
        f"$ws_outfile\t\t\t\t{output_base}",
        f"$ws_geofile\t\t\t\t{geometry_path}",
        "$ws_fileExtension\t\th5",
        f"$ws_detectorNumber\t\t{detector}",
        f"$ws_depthStart\t\t\t{depth_um[0]:g}",
        f"$ws_depthEnd\t\t\t{depth_um[-1]:g}",
        f"$ws_depthResolution\t\t{resolution:g}",
        "$ws_firstInputIndex\t\t0",
        "$ws_lastInputIndex\t\t0",
    ]
    if normalization:
        lines.append(f"$ws_normalization\t\t{normalization}")
    lines.extend([
        f"$ws_wireEdge\t\t\t{wire_edge}",
        f"$ws_outputPixelType\t\t{output_type}",
        f"$ws_percentOfPixels\t\t{percent_brightest:g}",
        f"$ws_MiB_RAM\t\t\t\t{memory_limit_mb}",
        f"$cosmic_filter\t\t\t{int(cosmic_filter)}",
        f"$ws_verbose\t\t\t\t{verbose}",
        "$program_name\t\tliblaue",
        f"$norm_exponent\t\t\t{0 if norm_exponent is None else norm_exponent:g}",
        f"$norm_threshold\t\t\t{'nan' if norm_threshold is None else format(norm_threshold, 'g')}",
        f"$norm_rescale\t\t\t{norm_rescale:g}",
    ])
    if scan_number is not None:
        lines.append(f"$scanNum\t\t\t\t{scan_number}")
    startx, starty = image_geometry.start
    groupx, groupy = image_geometry.group
    rows, cols = image_geometry.shape
    lines.extend([
        f"$startx\t\t\t\t\t{startx}",
        f"$endx\t\t\t\t\t{startx + cols * groupx - 1}",
        f"$groupx\t\t\t\t\t{groupx}",
        f"$starty\t\t\t\t\t{starty}",
        f"$endy\t\t\t\t\t{starty + rows * groupy - 1}",
        f"$groupy\t\t\t\t\t{groupy}",
    ])
    if sample_position is not None:
        x, y, z = sample_position
        if np.isfinite(x):
            lines.append(f"$X1\t\t\t\t\t\t{x:g}")
        if np.isfinite(y + z):
            lines.extend([
                f"$Y1\t\t\t\t\t\t{y:g}",
                f"$Z1\t\t\t\t\t\t{z:g}",
                f"$H1\t\t\t\t\t\t{(z + y) / np.sqrt(2):g}",
                f"$F1\t\t\t\t\t\t{(z - y) / np.sqrt(2):g}",
            ])
    if energy_kev is not None and np.isfinite(energy_kev):
        lines.append(f"$keV\t\t\t\t\t{energy_kev:g}")
    elapsed_text = f"{elapsed:.1f}" if elapsed > 2 else f"{elapsed:.3f}"
    lines.extend([
        f"$rows_at_one_time\t\t{rows_per_stripe}",
        f"$executionTime\t\t\t{elapsed_text}",
        "",
        f"$array0peakIndex\t\t{int(np.argmax(depth_intensity))}",
        f"$array0peakDepth\t\t{depth_um[int(np.argmax(depth_intensity))]:g}",
        f"$array0\t3,{len(depth_um)},Index,depth(micron),Intensity",
    ])
    lines.extend(f"{index}\t{depth:g}\t{intensity:g}" for index, (depth, intensity) in enumerate(zip(depth_um, depth_intensity)))
    Path(path).write_text("\n".join(lines) + "\n")
