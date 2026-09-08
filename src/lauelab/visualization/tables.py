# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Typed tabular views of normalized indexing results."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np

from .data import DataScope, ResultSet, VisualizationDataset


def _dataset(source):
    if isinstance(source, ResultSet):
        return source.to_visualization()
    if isinstance(source, VisualizationDataset):
        return source
    raise TypeError("source must be a ResultSet or VisualizationDataset")


@dataclass(frozen=True)
class Table:
    """Immutable named columns with direct pandas conversion."""

    columns: Mapping[str, np.ndarray]

    def __post_init__(self):
        normalized = {}
        length = None
        for name, values in self.columns.items():
            array = np.array(values, copy=True)
            if array.ndim != 1:
                raise ValueError(f"table column {name!r} must be one-dimensional")
            if length is None:
                length = len(array)
            elif len(array) != length:
                raise ValueError("table columns must have equal lengths")
            array.setflags(write=False)
            normalized[name] = array
        object.__setattr__(self, "columns", MappingProxyType(normalized))

    def __len__(self):
        return len(next(iter(self.columns.values()))) if self.columns else 0

    def __getitem__(self, name):
        return self.columns[name]

    def to_dataframe(self):
        """Return a new pandas DataFrame containing the table columns."""
        import pandas as pd

        return pd.DataFrame({name: values.copy() for name, values in self.columns.items()})

    def _repr_html_(self):
        """Return the pandas HTML representation used by notebook displays."""
        return self.to_dataframe()._repr_html_()


def _selected_pattern_rows(dataset, scope):
    return np.flatnonzero((scope or DataScope()).pattern_mask(dataset))


def _selected_frame_mask(dataset, scope):
    scope = scope or DataScope()
    if scope.patterns == "all_frames":
        mask = np.ones(dataset.n_frames, dtype=bool)
        if scope.min_detected is not None:
            mask &= dataset.frame_n_peaks >= scope.min_detected
        return mask
    rows = _selected_pattern_rows(dataset, scope)
    mask = np.zeros(dataset.n_frames, dtype=bool)
    mask[dataset.pattern_frame_indices[rows]] = True
    return mask


def _frame_columns(dataset, frame_indices):
    return {
        "frame_id": np.asarray(dataset.frame_ids, dtype=object)[np.asarray(frame_indices, dtype=int)],
        "scan_number": dataset.scan_numbers[frame_indices],
        "x_um": dataset.sample_positions[frame_indices, 0],
        "y_um": dataset.sample_positions[frame_indices, 1],
        "z_um": dataset.sample_positions[frame_indices, 2],
        "depth_um": dataset.depths[frame_indices],
        "frame_energy_kev": dataset.energies_kev[frame_indices],
    }


def peak_table(source, *, scope=None):
    """Return one row per detected peak in frames selected by ``scope``."""
    dataset = _dataset(source)
    selected = _selected_frame_mask(dataset, scope)
    rows = np.flatnonzero(selected[dataset.peak_frame_indices])
    frame_indices = dataset.peak_frame_indices[rows]
    columns = _frame_columns(dataset, frame_indices)
    columns.update({"peak_index": dataset.peak_indices[rows]})
    for name in dataset.peaks.dtype.names or ():
        values = dataset.peaks[name][rows]
        if values.ndim == 1:
            columns[name] = values
        else:
            for component in range(values.shape[1]):
                columns[f"{name}_{'xyz'[component]}"] = values[:, component]
    return Table(columns)


def pattern_table(source, *, scope=None):
    """Return one row per indexed pattern selected by ``scope``."""
    dataset = _dataset(source)
    rows = _selected_pattern_rows(dataset, scope)
    frame_indices = dataset.pattern_frame_indices[rows]
    columns = _frame_columns(dataset, frame_indices)
    columns.update({
        "pattern_index": dataset.pattern_indices[rows],
        "n_indexed": dataset.pattern_n_indexed[rows],
        "goodness": dataset.pattern_goodness[rows],
        "rms_error_deg": dataset.pattern_rms_error_deg[rows],
        "indexed_fraction": np.divide(
            dataset.pattern_n_indexed[rows],
            dataset.frame_n_peaks[frame_indices],
            out=np.zeros(len(rows), dtype=float),
            where=dataset.frame_n_peaks[frame_indices] > 0,
        ),
    })
    for prefix, values in (
        ("rotation", dataset.pattern_rotations[rows]),
        ("reciprocal", dataset.pattern_reciprocals[rows]),
    ):
        for row in range(3):
            for column in range(3):
                columns[f"{prefix}_{row}{column}"] = values[:, row, column]
    return Table(columns)


def _selected_assignment_rows(dataset, scope):
    """Return assignment rows for the scope with their pattern rows and frames."""
    pattern_rows = _selected_pattern_rows(dataset, scope)
    rows = np.flatnonzero(np.isin(dataset.assignment_pattern_rows, pattern_rows))
    selected_pattern_rows = dataset.assignment_pattern_rows[rows]
    return rows, selected_pattern_rows, dataset.pattern_frame_indices[selected_pattern_rows]


def _peak_rows(dataset, frame_indices, peak_indices):
    """Return rows of ``dataset.peaks`` for ``(frame, peak index)`` pairs.

    Rows are found with one sort of a composite key instead of a dictionary
    over every peak, which matters for scans with 10^5 frames.
    """
    frame_indices = np.asarray(frame_indices, dtype=np.int64)
    peak_indices = np.asarray(peak_indices, dtype=np.int64)
    if not len(dataset.peak_indices):
        if len(frame_indices):
            raise KeyError("assignment refers to a peak that is not in the dataset")
        return np.empty(0, dtype=int)
    stride = int(max(dataset.peak_indices.max(), peak_indices.max(initial=0))) + 1
    keys = dataset.peak_frame_indices.astype(np.int64) * stride + dataset.peak_indices
    order = np.argsort(keys, kind="stable")
    wanted = frame_indices * stride + peak_indices
    positions = np.searchsorted(keys[order], wanted)
    positions = np.minimum(positions, len(order) - 1)
    if np.any(keys[order[positions]] != wanted):
        raise KeyError("assignment refers to a peak that is not in the dataset")
    return order[positions]


def assignment_table(source, *, scope=None):
    """Return one row per selected pattern-to-peak assignment."""
    dataset = _dataset(source)
    rows, selected_pattern_rows, frame_indices = _selected_assignment_rows(dataset, scope)
    return Table(_assignment_columns(dataset, rows, selected_pattern_rows, frame_indices))


def _assignment_columns(dataset, rows, selected_pattern_rows, frame_indices):
    columns = _frame_columns(dataset, frame_indices)
    columns.update({
        "pattern_index": dataset.pattern_indices[selected_pattern_rows],
        "peak_index": dataset.assignment_peak_indices[rows],
        "h": dataset.assignment_hkl[rows, 0],
        "k": dataset.assignment_hkl[rows, 1],
        "l": dataset.assignment_hkl[rows, 2],
        "error_deg": dataset.assignment_error_deg[rows],
        "energy_kev": dataset.assignment_energy_kev[rows],
        "predicted_intensity": dataset.assignment_predicted_intensity[rows],
    })
    return columns


def indexed_peak_table(source, *, scope=None):
    """Return assignments joined with their detected peak and pattern values."""
    dataset = _dataset(source)
    assignment_rows, pattern_rows, frame_indices = _selected_assignment_rows(dataset, scope)
    columns = _assignment_columns(dataset, assignment_rows, pattern_rows, frame_indices)
    rows = _peak_rows(dataset, frame_indices, columns["peak_index"])
    for name in dataset.peaks.dtype.names or ():
        values = dataset.peaks[name][rows]
        if values.ndim == 1:
            columns[name] = values
        else:
            for component in range(values.shape[1]):
                columns[f"{name}_{'xyz'[component]}"] = values[:, component]
    columns.update({
        "goodness": dataset.pattern_goodness[pattern_rows],
        "rms_error_deg": dataset.pattern_rms_error_deg[pattern_rows],
        "pattern_n_indexed": dataset.pattern_n_indexed[pattern_rows],
    })
    return Table(columns)
