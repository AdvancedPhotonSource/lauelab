# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Export a reconstructed point file to per-depth HDF5 files."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from ._reader import ImageGeometry
from ._scan_reader import PointReader
from ._writer import _copy_metadata, pixel_type, write_summary

_EDGE = {"leading": 1, "trailing": 0, "both": -1}


def export_per_depth(path, output_base) -> list[str]:
    """Write a point file as LaueGo-compatible per-depth files.

    Output follows the :meth:`Reconstructor.reconstruct` naming convention: ``<output_base><i>.h5`` for depth index ``i`` and
    ``<output_base>summary.txt``. Pixels are copied from the point file in
    their stored dtype without conversion or rescaling. The export also
    copies physical depths, detector metadata, and normalization provenance.
    All data comes from the point file; reconstruction is not repeated.

    Parameters
    ----------
    path : pathlib.Path or str
        A point file, such as ``points/<input stem>.h5`` of a scan directory.
    output_base : pathlib.Path or str
        Output filename prefix, not a directory. Existing files are replaced.

    Returns
    -------
    list of str
        The per-depth paths in depth order followed by the summary path.

    Raises
    ------
    InvalidScanFile
        If ``path`` is not a complete point file.
    OSError
        If the point file cannot be opened or an output cannot be written.
        Files already written are left in place, and the point file is
        unchanged.

    Notes
    -----
    The summary reports ``$executionTime`` as zero, because the export did not
    run reconstruction. ``$rows_at_one_time`` is the stripe height used
    during reconstruction.
    """
    output_base = str(output_base)
    with PointReader(path) as point:
        source = point._file
        settings = point.settings
        Path(output_base).parent.mkdir(parents=True, exist_ok=True)
        n_depths, rows, columns = point.shape
        paths = []
        for index in range(n_depths):
            target_path = f"{output_base}{index}.h5"
            with h5py.File(target_path, "w") as target:
                _copy_metadata(source, target, exclude_entry=("reconstruction", "depth"),
                               exclude_data=("depth",))
                for name in ("format", "version", "lauelab_version", "created", "default"):
                    if name in target.attrs:
                        del target.attrs[name]
                for name in list(target["entry1/data"].attrs):
                    if name in ("signal", "axes", "default", "auxiliary_signals") or name.endswith("_indices"):
                        del target["entry1/data"].attrs[name]
                target["entry1/data"].attrs["signal"] = "data"
                target["entry1/data"].attrs["axes"] = [".", "."]
                data = target["entry1/data"].create_dataset("data", data=point.frame(index))
                data.attrs["signal"] = np.int32(1)
                if "entry1/depth" in target:
                    depth = target["entry1/depth"]
                else:
                    depth = target["entry1"].create_dataset("depth", data=np.asarray([0.0]))
                    depth.attrs["units"] = "micron"
                depth[...] = np.asarray([point.depth_um[index]])
                exponent = settings["norm_exponent"]
                if settings["cosmic_filter"] or exponent is not None:
                    group = target["entry1"].require_group("microDiffraction")
                    if settings["cosmic_filter"]:
                        group.create_dataset("cosmic_filter", data=np.asarray([1], dtype=np.int32))
                    if exponent is not None:
                        group.create_dataset("norm_exponent", data=np.asarray([exponent]))
                        group.create_dataset("norm_threshold", data=np.asarray([point.norm_threshold]))
                        group.create_dataset("norm_rescale", data=np.asarray([point.norm_rescale]))
            paths.append(target_path)

        summary = f"{output_base}summary.txt"
        write_summary(
            summary, input_path=point.source_path, output_base=output_base,
            geometry_path=settings["geometry_path"] or "", detector=int(settings["detector"]),
            depth_um=point.depth_um, resolution=float(settings["resolution"]),
            wire_edge=_EDGE[settings["wire_edge"]], output_type=pixel_type(point.dtype),
            percent_brightest=float(settings["percent_brightest"]),
            memory_limit_mb=int(settings["memory_limit_mb"]),
            cosmic_filter=settings["cosmic_filter"],
            normalization=settings["normalization"],
            norm_exponent=exponent, norm_threshold=point.norm_threshold,
            norm_rescale=point.norm_rescale, scan_number=point.scan_number,
            sample_position=point.sample_position, energy_kev=point.energy_kev,
            image_geometry=ImageGeometry(*point.detector_size, point.start, point.group, rows, columns),
            rows_per_stripe=int(point._file["entry1/reconstruction/execution/rows_per_stripe"][()]), elapsed=0.0,
            depth_intensity=point.computed_depth_intensity(),
        )
        paths.append(summary)
    return paths
