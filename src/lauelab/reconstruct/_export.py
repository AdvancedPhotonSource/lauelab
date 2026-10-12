# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Export a reconstructed point file to per-depth HDF5 files."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from . import _scan_layout as layout
from ._reader import ImageGeometry
from ._scan_reader import PointReader
from ._writer import _copy_metadata, write_summary

_EDGE = {"leading": 1, "trailing": 0, "both": -1}


def export_per_depth(path, output_base) -> list[str]:
    """Write a point file as LaueGo-compatible per-depth files.

    Output follows the :meth:`Reconstructor.reconstruct` naming convention: ``<output_base><i>.h5`` for depth index ``i`` and
    ``<output_base>summary.txt``. Pixels are copied from the point file in
    their stored dtype without conversion or rescaling. Each file contains one
    frame and its physical depth, with the point file's acquisition metadata
    and ``entry1/microDiffraction`` normalization fields. All data comes from
    the point file; reconstruction is not repeated.

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
    The summary reports the ``$executionTime`` and ``$rows_at_one_time`` of
    the reconstruction that wrote the point file.
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
                # The point file already has the executable's layout; each
                # depth file contains one frame and its depth.
                _copy_metadata(source, target, exclude_entry=("reconstruction", "depth"))
                for name in ("format", "version", "lauelab_version", "created"):
                    del target.attrs[name]
                data = target["entry1/data"].create_dataset("data", data=point.frame(index))
                data.attrs["signal"] = np.int32(1)
                depth = target["entry1"].create_dataset("depth", data=point.depth_um[[index]])
                depth.attrs["units"] = "micron"
            paths.append(target_path)

        summary = f"{output_base}summary.txt"
        write_summary(
            summary, input_path=point.source_path, output_base=output_base,
            geometry_path=settings["geometry_path"], detector=settings["detector"],
            depth_um=point.depth_um, resolution=settings["resolution"],
            wire_edge=_EDGE[settings["wire_edge"]], output_type=settings["output_pixel_type"],
            percent_brightest=settings["percent_brightest"],
            memory_limit_mb=settings["memory_limit_mb"],
            cosmic_filter=settings["cosmic_filter"], frame_skip=settings["frame_skip"],
            wire_skip=settings["wire_skip"], normalization=settings["normalization"],
            norm_exponent=settings["norm_exponent"], norm_threshold=point.norm_threshold,
            norm_rescale=point.norm_rescale, scan_number=point.scan_number,
            sample_position=point.sample_position, energy_kev=point.energy_kev,
            image_geometry=ImageGeometry(*point.detector_size, point.start, point.group, rows, columns),
            rows_per_stripe=settings["rows_per_stripe"],
            elapsed=float(source[layout.RECONSTRUCTION + "elapsed_seconds"][0]),
            depth_intensity=point.computed_depth_intensity(),
        )
        paths.append(summary)
    return paths
