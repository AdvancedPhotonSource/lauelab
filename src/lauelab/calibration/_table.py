# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Measured reflections used to calibrate one detector."""

from __future__ import annotations

from dataclasses import dataclass, replace
from numbers import Real

import numpy as np

from lauelab._frame import roi_to_detector_pixels
from lauelab.indexing import InputError

# Two peaks closer than this (full-detector pixels) are the same peak when a
# table carries measured energies over to a new indexing.
_SAME_PEAK_PX = 0.01


@dataclass(frozen=True)
class CalibrationTable:
    """Reflections with known Miller indices and their measured values.

    Each row is one reflection. A row is used when it has a pixel position, a
    measured energy, or both, as in LaueGo's ``CalibrationList``.

    Parameters
    ----------
    hkl
        Integer Miller indices with shape ``(n, 3)``.
    pixels
        Zero-based full-detector ``(x, y)`` peak positions with shape
        ``(n, 2)``; ``NaN`` where the position was not measured.
    energy_kev
        Measured photon energies in keV with shape ``(n,)``; ``NaN`` where the
        energy was not measured.
    energy_source
        One free-text note per row, such as the scan in which the energy was
        measured; an empty string where there is none.
    peak_index
        Zero-based index of each row's peak in the peak array that was indexed,
        with shape ``(n,)``; ``-1`` for a row without a peak.

    Notes
    -----
    The arrays are read-only. :meth:`with_energy` and :meth:`with_hkl` return
    changed copies.
    """

    hkl: np.ndarray
    pixels: np.ndarray
    energy_kev: np.ndarray
    energy_source: tuple[str, ...]
    peak_index: np.ndarray

    def __post_init__(self) -> None:
        hkl = np.array(self.hkl, copy=True)
        if hkl.dtype.kind not in "iu" or hkl.ndim != 2 or hkl.shape[1] != 3:
            raise InputError("hkl must be an integer array with shape (n, 3)")
        if np.any(np.all(hkl == 0, axis=1)):
            raise InputError("hkl must not contain the zero reflection")
        n = len(hkl)
        pixels = np.array(self.pixels, dtype=np.float64, copy=True)
        energy = np.array(self.energy_kev, dtype=np.float64, copy=True)
        peak_index = np.array(self.peak_index, copy=True)
        if pixels.shape != (n, 2):
            raise InputError(f"pixels must have shape ({n}, 2)")
        if energy.shape != (n,):
            raise InputError(f"energy_kev must have shape ({n},)")
        if peak_index.shape != (n,) or (n and peak_index.dtype.kind not in "iu"):
            raise InputError(f"peak_index must be an integer array with shape ({n},)")
        if np.any(np.isinf(pixels)) or np.any(np.isinf(energy)):
            raise InputError("pixels and energy_kev must be finite or NaN")
        if np.any(energy <= 0):
            raise InputError("measured energies must be positive")
        sources = tuple(str(value) for value in self.energy_source)
        if len(sources) != n:
            raise InputError(f"energy_source must contain {n} entries")
        for name, value in (
            ("hkl", hkl.astype(np.int64)), ("pixels", pixels), ("energy_kev", energy),
            ("peak_index", peak_index.astype(np.int64)),
        ):
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "energy_source", sources)

    def __len__(self) -> int:
        return len(self.hkl)

    def __repr__(self) -> str:
        return f"CalibrationTable(n_rows={len(self)}, n_energies={self.n_energies})"

    @property
    def n_energies(self) -> int:
        """Number of rows with a measured energy."""
        return int(np.isfinite(self.energy_kev).sum())

    def with_energy(self, row: int, energy_kev: float | None, source: str = "") -> "CalibrationTable":
        """Return a copy with the measured energy of one row set or cleared.

        Parameters
        ----------
        row
            Zero-based row index.
        energy_kev
            Measured energy in keV, or `None` to clear it.
        source
            Free-text note on where the energy was measured.
        """
        row = self._row(row)
        if energy_kev is None:
            value = np.nan
        elif isinstance(energy_kev, Real) and np.isfinite(energy_kev) and energy_kev > 0:
            value = float(energy_kev)
        else:
            raise InputError(f"energy_kev must be a positive number or None; received {energy_kev!r}")
        energy = np.array(self.energy_kev)
        energy[row] = value
        sources = list(self.energy_source)
        sources[row] = "" if energy_kev is None else str(source)
        return replace(self, energy_kev=energy, energy_source=tuple(sources))

    def with_hkl(self, row: int, hkl) -> "CalibrationTable":
        """Return a copy with the Miller indices of one row changed.

        Use this to select another harmonic of a reflection whose measured
        energy belongs to a higher order, such as 008 instead of 004.
        """
        row = self._row(row)
        values = np.asarray(hkl)
        if values.shape != (3,) or values.dtype.kind not in "iu":
            raise InputError("hkl must be three integers")
        table = np.array(self.hkl)
        table[row] = values
        return replace(self, hkl=table)

    def _row(self, row) -> int:
        if isinstance(row, bool) or not isinstance(row, (int, np.integer)) or not 0 <= row < len(self):
            raise InputError(f"row must be an index from 0 to {len(self) - 1}; received {row!r}")
        return int(row)


def calibration_table(
    peaks,
    pattern,
    *,
    start: tuple[int, int] = (0, 0),
    group: tuple[int, int] = (1, 1),
    previous: CalibrationTable | None = None,
) -> CalibrationTable:
    """Return the calibration table of the peaks that a pattern indexed.

    This replaces LaueGo's ``MakeCalibData``, which matched each fitted peak to
    the nearest indexed peak. A :class:`~lauelab.indexing.Pattern` records the
    peak of each reflection in ``pk_index``, so no position matching is needed.

    Parameters
    ----------
    peaks
        The peaks whose scattering vectors were indexed, in the same order:
        frame ``(x, y)`` coordinates with shape ``(n, 2)``, or a structured
        array with ``fit_x`` and ``fit_y`` fields such as
        ``PeakSearch.peaks``.
    pattern
        One pattern returned by
        :func:`~lauelab.indexing.index_orientations`.
    start, group
        Full-detector origin and pixel grouping of the frame, as ``(x, y)``.
    previous
        An earlier table of the same frame. Measured energies, their sources,
        and Miller indices of rows with an energy are copied to the rows of the
        same peak (same full-detector position within 0.01 pixel).

    Returns
    -------
    CalibrationTable
        One row per indexed peak, in the order of ``pattern.pk_index``.
    """
    names = getattr(getattr(peaks, "dtype", None), "names", None)
    if names is not None:
        if not {"fit_x", "fit_y"} <= set(names):
            raise InputError("structured peaks need fit_x and fit_y fields")
        peaks = np.column_stack([peaks["fit_x"], peaks["fit_y"]])
    peaks = np.asarray(peaks, dtype=np.float64)
    if peaks.ndim != 2 or peaks.shape[1] != 2:
        raise InputError("peaks must have shape (n, 2)")
    index = np.asarray(pattern.pk_index, dtype=np.int64)
    if np.any(index < 0) or np.any(index >= len(peaks)):
        raise InputError("the pattern refers to peaks outside the peak array")
    pixels = roi_to_detector_pixels(peaks[index], start, group).astype(np.float64)
    hkl = np.asarray(pattern.hkl, dtype=np.int64)
    energy = np.full(len(index), np.nan)
    sources = [""] * len(index)
    if previous is not None:
        for old in np.flatnonzero(np.isfinite(previous.energy_kev)):
            distance = np.linalg.norm(pixels - previous.pixels[old], axis=1)
            matches = np.flatnonzero(distance < _SAME_PEAK_PX)
            if len(matches):
                row = matches[0]
                energy[row] = previous.energy_kev[old]
                sources[row] = previous.energy_source[old]
                hkl[row] = _same_order(previous.hkl[old], hkl[row])
    return CalibrationTable(hkl, pixels, energy, tuple(sources), index)


def _same_order(old_hkl: np.ndarray, new_hkl: np.ndarray) -> np.ndarray:
    """Return ``new_hkl`` scaled to the harmonic order of ``old_hkl``.

    A new indexing can label a reflection with a symmetry-equivalent hkl. The
    measured energy belongs to a harmonic order, which is kept.
    """
    old_order = int(np.gcd.reduce(np.abs(old_hkl)))
    new_order = int(np.gcd.reduce(np.abs(new_hkl)))
    return new_hkl // new_order * old_order
