# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Export one point of a reconstruction-scan file to per-depth HDF5 files."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from lauelab.indexing.errors import InputError

from . import _scan_layout as layout
from ._reader import ImageGeometry
from ._scan_reader import ScanReader
from ._writer import pixel_type, write_summary

_EDGE = {"leading": 1, "trailing": 0, "both": -1}


def _scalar(source, name):
    value = source[name][()]
    return value.decode() if isinstance(value, bytes) else value


def export_per_depth(path, point_id: str, output_base) -> list[str]:
    """Write one point of a scan file as LaueGo-compatible per-depth files.

    The files are the ones :meth:`Reconstructor.reconstruct` writes for
    ``output_base``: ``<output_base><i>.h5`` for depth index ``i`` and
    ``<output_base>summary.txt``. Pixels are copied from the scan file in
    their stored dtype without any conversion or rescaling; the depth,
    detector metadata, and normalization provenance come with them. Nothing
    is reconstructed.

    Parameters
    ----------
    path : pathlib.Path or str
        Reconstruction-scan file.
    point_id : str
        A complete point of that file.
    output_base : pathlib.Path or str
        Output filename prefix, not a directory. Existing files are replaced.

    Returns
    -------
    list of str
        The per-depth paths in depth order followed by the summary path.

    Raises
    ------
    InputError
        If the point does not exist or is not complete.
    OSError
        If the scan file cannot be opened or an output cannot be written.

    Notes
    -----
    The summary reports ``$rows_at_one_time`` as the frame height and
    ``$executionTime`` as zero, because the export did not run a
    reconstruction; the values a run wrote are not kept in the scan file.
    """
    output_base = str(output_base)
    with ScanReader(path) as scan:
        try:
            point = scan.point(point_id)
        except KeyError as error:
            raise InputError(f"{path} has no point {point_id!r}") from error
        run = scan._file
        source = run[layout.POINT_GROUP.format(index=point.entry.index)][layout.POINT_SOURCE_GROUP]
        settings = {name: _scalar(run, f"settings/{name}") for name in (
            "detector", "resolution", "wire_edge", "percent_brightest", "normalization",
            "norm_exponent", "cosmic_filter", "memory_limit_mb",
        )}
        Path(output_base).parent.mkdir(parents=True, exist_ok=True)
        n_depths, rows, columns = point.shape
        paths = []
        for index in range(n_depths):
            target_path = f"{output_base}{index}.h5"
            with h5py.File(target_path, "w") as target:
                for key, value in source.attrs.items():
                    target.attrs[key] = value
                for name in source:
                    source.copy(source[name], target, name=name)
                data = target["entry1/data"].create_dataset("data", data=point.frame(index))
                data.attrs["signal"] = np.int32(1)
                if "entry1/depth" in target:
                    depth = target["entry1/depth"]
                else:
                    depth = target["entry1"].create_dataset("depth", data=np.asarray([0.0]))
                    depth.attrs["units"] = "micron"
                depth[...] = np.asarray([point.depth_um[index]])
                exponent = None if np.isnan(settings["norm_exponent"]) else settings["norm_exponent"]
                if settings["cosmic_filter"] or exponent is not None:
                    group = target["entry1"].require_group("microDiffraction")
                    if settings["cosmic_filter"]:
                        group.create_dataset("cosmic_filter", data=np.asarray([1], dtype=np.int32))
                    if exponent is not None:
                        group.create_dataset("norm_exponent", data=np.asarray([exponent]))
                        group.create_dataset("norm_threshold", data=np.asarray([point.norm_threshold]))
                        group.create_dataset("norm_rescale", data=np.asarray([point.norm_rescale]))
            paths.append(target_path)

        entry = point.entry
        summary = f"{output_base}summary.txt"
        write_summary(
            summary, input_path=entry.source_path, output_base=output_base,
            geometry_path=_scalar(run, "geometry/path"), detector=int(settings["detector"]),
            depth_um=point.depth_um, resolution=float(settings["resolution"]),
            wire_edge=_EDGE[settings["wire_edge"]], output_type=pixel_type(point.dtype),
            percent_brightest=float(settings["percent_brightest"]),
            memory_limit_mb=int(settings["memory_limit_mb"]),
            cosmic_filter=bool(settings["cosmic_filter"]),
            normalization=settings["normalization"] or None,
            norm_exponent=None if np.isnan(settings["norm_exponent"]) else float(settings["norm_exponent"]),
            norm_threshold=point.norm_threshold, norm_rescale=point.norm_rescale,
            scan_number=entry.scan_number, sample_position=entry.sample_position,
            energy_kev=entry.energy_kev,
            image_geometry=ImageGeometry(*point.detector_size, point.start, point.group, rows, columns),
            rows_per_stripe=rows, elapsed=0.0,
            depth_intensity=point.computed_depth_intensity(),
        )
        paths.append(summary)
    return paths
