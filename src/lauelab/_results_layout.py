# Copyright 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Canonical dataset layout for indexing-results HDF5 files."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

import numpy as np

from ._hdf5 import UTF8, set_units

FORMAT = "lauelab-indexing-results"
VERSION = 1
SUPPORTED_VERSIONS = frozenset({VERSION})


@dataclass(frozen=True)
class DatasetSpec:
    """Storage convention for one results dataset."""

    dtype: object
    shape: tuple[int, ...] = ()
    units: str | None = None
    resizable: bool = True
    chunk_rows: int | None = 1024
    attrs: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", np.dtype(self.dtype))
        object.__setattr__(self, "attrs", MappingProxyType(dict(self.attrs)))


F4 = np.dtype("<f4")
F8 = np.dtype("<f8")
I2 = np.dtype("<i2")
I4 = np.dtype("<i4")
I8 = np.dtype("<i8")


def _spec(dtype, shape=(), units=None, *, resizable=True, chunk_rows=1024, **attrs):
    return DatasetSpec(
        dtype=dtype,
        shape=shape,
        units=units,
        resizable=resizable,
        chunk_rows=chunk_rows,
        attrs=attrs,
    )


DATASETS = MappingProxyType({
    "/crystal/lattice_parameters": _spec(
        F8, (6,), "nm", resizable=False, chunk_rows=None, angle_units="deg"
    ),
    "/crystal/atom_symbols": _spec(UTF8, resizable=False, chunk_rows=None),
    "/crystal/atom_labels": _spec(UTF8, resizable=False, chunk_rows=None),
    "/crystal/atom_positions": _spec(
        F8, (3,), "fractional", resizable=False, chunk_rows=None
    ),
    "/crystal/atom_occupancies": _spec(F4, resizable=False, chunk_rows=None),
    "/geometry/xml": _spec(UTF8, resizable=False, chunk_rows=None),
    "/frames/frame_ids": _spec(I4),
    "/frames/sample_positions": _spec(F8, (3,), "um"),
    "/frames/depths": _spec(F8, units="um"),
    "/frames/scan_numbers": _spec(I4),
    "/frames/energies_kev": _spec(F4, units="keV"),
    "/frames/detector_ids": _spec(UTF8),
    "/frames/input_images": _spec(UTF8),
    "/frames/titles": _spec(UTF8),
    "/frames/sample_names": _spec(UTF8),
    "/frames/user_names": _spec(UTF8),
    "/frames/beamlines": _spec(UTF8),
    "/frames/dates_exposed": _spec(UTF8),
    "/frames/ccd_shutters": _spec(UTF8),
    "/frames/mono_modes": _spec(UTF8),
    "/frames/exposure_seconds": _spec(F4, units="s"),
    "/frames/beam_bad": _spec(I4),
    "/frames/light_on": _spec(I4),
    "/frames/hutch_temperature": _spec(F4, units="unspecified"),
    "/frames/sample_distance": _spec(F4, units="unspecified"),
    "/frames/image_shapes": _spec(I4, (2,)),
    "/frames/roi_starts": _spec(I4, (2,)),
    "/frames/roi_groups": _spec(I4, (2,)),
    "/frames/n_peaks": _spec(I4),
    "/frames/n_patterns": _spec(I2),
    "/frames/threshold_used": _spec(F4),
    "/frames/threshold_ratio": _spec(F4),
    "/frames/total_sum": _spec(F8),
    "/frames/sum_above_threshold": _spec(F8),
    "/frames/num_above_threshold": _spec(I8),
    "/frames/peak_minwidth": _spec(F4),
    "/frames/peak_maxwidth": _spec(F4),
    "/frames/peak_max_cent_to_fit": _spec(F4),
    "/frames/peak_boxsize": _spec(I4),
    "/frames/peaksearch_seconds": _spec(F4, units="s"),
    "/frames/indexing_seconds": _spec(F4, units="s"),
    "/frames/peak_offsets": _spec(I8),
    "/frames/pattern_offsets": _spec(I8),
    "/peaks/fit_x": _spec(F4, units="pixel", chunk_rows=4096),
    "/peaks/fit_y": _spec(F4, units="pixel", chunk_rows=4096),
    "/peaks/intens": _spec(F4, chunk_rows=4096),
    "/peaks/integral": _spec(F4, chunk_rows=4096),
    "/peaks/hwhm_x": _spec(F4, units="pixel", chunk_rows=4096),
    "/peaks/hwhm_y": _spec(F4, units="pixel", chunk_rows=4096),
    "/peaks/tilt": _spec(F4, units="deg", chunk_rows=4096),
    "/peaks/chisq": _spec(F4, chunk_rows=4096),
    "/peaks/background": _spec(F4, chunk_rows=4096),
    "/peaks/qhat": _spec(F4, (3,), chunk_rows=4096),
    "/patterns/rank": _spec(I2),
    "/patterns/reciprocal": _spec(
        F8, (3, 3), "1/nm", rows="a*,b*,c*", includes_two_pi=True
    ),
    "/patterns/goodness": _spec(F4),
    "/patterns/rms_error_deg": _spec(F4, units="deg"),
    "/patterns/n_indexed": _spec(I4),
    "/patterns/assignment_offsets": _spec(I8),
    "/assignments/peak_index": _spec(I4, chunk_rows=4096),
    "/assignments/hkl": _spec(I2, (3,), chunk_rows=4096),
    "/assignments/error_deg": _spec(F4, units="deg", chunk_rows=4096),
    "/assignments/energy_kev": _spec(F4, units="keV", chunk_rows=4096),
    "/assignments/pred_intens": _spec(F4, chunk_rows=4096),
})

FRAME_IDS_STRING_SPEC = _spec(UTF8)


def set_attributes(dataset, spec):
    set_units(dataset, spec.units)
    for name, value in spec.attrs.items():
        dataset.attrs[name] = value


def write_dataset(target, path, values):
    spec = DATASETS[path]
    values = np.asarray(values, dtype=spec.dtype)
    if values.size == 0 and spec.shape:
        values = values.reshape((0,) + spec.shape)
    dataset = target.create_dataset(path, data=values, dtype=spec.dtype)
    set_attributes(dataset, spec)


def write_crystal(target, crystal):
    if crystal is None:
        return
    group = target.create_group("crystal")
    group.attrs["name"] = crystal.name
    group.attrs["space_group"] = crystal.space_group
    group.attrs["setting"] = crystal.setting or ""
    group.attrs["source"] = crystal.source or ""
    cell = crystal.cell.in_angstrom
    values = {
        "lattice_parameters": (cell.a / 10, cell.b / 10, cell.c / 10, cell.alpha, cell.beta, cell.gamma),
        "atom_symbols": [atom.symbol for atom in crystal.atoms],
        "atom_labels": [atom.label or "" for atom in crystal.atoms],
        "atom_positions": [atom.position for atom in crystal.atoms],
        "atom_occupancies": [atom.occupancy for atom in crystal.atoms],
    }
    for name, data in values.items():
        write_dataset(target, f"/crystal/{name}", data)
