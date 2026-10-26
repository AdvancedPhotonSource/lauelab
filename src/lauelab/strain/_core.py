# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
"""Independent deviatoric refinement with explicit scientific configuration."""

from __future__ import annotations

from numbers import Real

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from lauelab.indexing import Geometry, InputError
from ._inputs import (
    StrainReflections,
    _Source,
    _index,
    _json,
    _pixel_predictions,
    prepare_reflections,
)
from ._models import StrainResult
from ._tensor import (
    _canonical,
    _cell_parameters,
    _coordinates,
    _mask,
    _normalize,
    _orientation,
    _reference,
    _strain,
)


def _controls(max_nfev, condition_limit):
    max_nfev = _index(max_nfev, "max_nfev")
    if max_nfev == 0:
        raise InputError("max_nfev must be positive")
    if (
        isinstance(condition_limit, (bool, np.bool_))
        or not isinstance(condition_limit, Real)
        or not np.isfinite(condition_limit)
        or condition_limit <= 1
    ):
        raise InputError("condition_limit must be finite and greater than one")
    return max_nfev, float(condition_limit)


def _failure(
    data,
    ref,
    fit,
    coordinates,
    status,
    message,
    *,
    reference,
    max_nfev,
    condition_limit,
):
    mask, text = _mask(fit)
    n = len(data.hkl)
    return StrainResult(
        status,
        message,
        coordinates,
        text,
        np.full((3, 3), np.nan),
        np.full((3, 3), np.nan),
        ref,
        np.full((3, 3), np.nan),
        np.full(6, np.nan),
        np.full(6, np.nan),
        data,
        np.full((n, 3), np.nan),
        np.full((n, 2), np.nan),
        np.full(n, np.nan),
        3 + int(mask.sum()),
        0,
        np.inf,
        0,
        _json(
            {
                "measure": "deviatoric Biot stretch",
                "scale": "reference volume",
                "objective": "uniform unit-direction chord least squares",
                "reference_parameters_nm_deg": _cell_parameters(ref),
                "reference_source": getattr(reference, "source", None),
                "reference_name": getattr(reference, "name", None),
                "initial_reciprocal": data.initial_reciprocal,
                "fit_mask": text,
                "frozen_values": "reference, before volume normalization",
                "coordinates": coordinates,
                "optimizer": "scipy.optimize.least_squares/trf",
                "max_nfev": max_nfev,
                "condition_limit": condition_limit,
                "optimizer_run": False if status == "insufficient_support" else None,
                "empirical_igor_parity": "pending",
            }
        ),
    )


def refine_deviatoric(
    qhat,
    hkl=None,
    *,
    reference,
    initial_reciprocal=None,
    fit="110111",
    coordinates="beamline",
    max_nfev=1000,
    condition_limit=1e8,
):
    """Refine assigned directions with Igor's uniform chord objective.

    Parameters
    ----------
    qhat : numpy.ndarray or StrainReflections
        Measured directions, shape ``(n, 3)``, normalized internally, or a
        preparation-stage result supplying all measurement arguments.
    hkl : numpy.ndarray or None
        Aligned integer Miller indices, shape ``(n, 3)``. Omit for prepared data.
    reference : lauelab.indexing.Cell or lauelab.indexing.Crystal
        Explicit unstrained reference metric; independent of the initial fit.
    initial_reciprocal : numpy.ndarray or None
        Starting reciprocal rows, shape ``(3, 3)``, 1/nm including 2π.
        Required for array inputs; omit for prepared data.
    fit : str or tuple of str
        Six binary flags or unique parameter names in a,b,c,alpha,beta,gamma
        order. Default ``"110111"`` gives five shape and three rotational
        parameters. At least one length must be held. Excluded working-cell
        values come from reference before reference-volume normalization.
    coordinates : str
        ``"beamline"`` (default), ``"crystal"``, ``"xhf"`` or ``"sample"``.
    max_nfev : int
        Maximum optimizer function evaluations, default 1000.
    condition_limit : float
        Joint dimensionless Jacobian condition bound, default 1e8; must be
        finite and greater than one. This is not an experimental error bar.

    Returns
    -------
    StrainResult
        Owned fit evidence, including unsuccessful support/convergence
        outcomes with NaN tensors. No automatic reflection rejection,
        weighting or reassignment is performed.

    Notes
    -----
    Invalid inputs raise InputError or TypeError. Empirical Igor export
    comparisons remain a separate validation gate.
    """
    ref_parameters, ref = _reference(reference)
    bits, mask = _mask(fit)
    _coordinates(coordinates)
    max_nfev, condition_limit = _controls(max_nfev, condition_limit)
    if isinstance(qhat, StrainReflections):
        if hkl is not None or initial_reciprocal is not None:
            raise InputError(
                "prepared reflections already supply HKLs and initial lattice"
            )
        data = qhat
    else:
        data = StrainReflections(
            qhat,
            hkl,
            initial_reciprocal,
            provenance_json=_json({"source": {"kind": "directions"}}),
        )
    if data.initial_reciprocal is None:
        raise InputError("initial_reciprocal is required for measurement refinement")
    n = len(data.hkl)
    n_parameters = 3 + int(bits.sum())
    if 2 * n <= n_parameters:
        return _failure(
            data,
            ref,
            mask,
            coordinates,
            "insufficient_support",
            f"need 2*n_reflections > {n_parameters} free parameters; received {n}",
            reference=reference,
            max_nfev=max_nfev,
            condition_limit=condition_limit,
        )
    initial = _cell_parameters(data.initial_reciprocal)
    anchor = np.flatnonzero(~bits[:3])[0]
    initial[:3] *= ref_parameters[anchor] / initial[anchor]
    start = ref_parameters.copy()
    start[bits] = initial[bits]
    _canonical(start)  # Invalid projected starting models are explicit input errors.
    rotation0 = _orientation(data.initial_reciprocal)
    free = np.flatnonzero(bits)
    x0 = np.zeros(n_parameters)
    lower, upper = np.full(n_parameters, -np.inf), np.full(n_parameters, np.inf)
    for j, i in enumerate(free, 3):
        if i < 3:
            lower[j], upper[j] = (
                -20,
                20,
            )  # Numerical overflow guards, not a strain cutoff.
        else:
            lower[j] = np.radians(1e-6 - start[i])
            upper[j] = np.radians(180 - 1e-6 - start[i])

    def lattice(x):
        values = ref_parameters.copy()
        for j, i in enumerate(free, 3):
            values[i] = (
                start[i] * np.exp(x[j]) if i < 3 else start[i] + np.degrees(x[j])
            )
        rotation = Rotation.from_rotvec(x[:3]).as_matrix() @ rotation0
        return _canonical(values) @ rotation.T, rotation, values

    def residual(x):
        try:
            reciprocal, _, _ = lattice(x)
        except InputError:
            # A nonphysical trial cell is outside the model. A large residual
            # makes the trust region reject it; valid cells retain exactly the
            # unweighted direction-chord objective.
            return np.full(3 * n, 1e3)
        predicted = data.hkl @ reciprocal
        predicted /= np.linalg.norm(predicted, axis=1)[:, None]
        return (predicted - data.qhat).ravel()

    optimized = least_squares(
        residual,
        x0,
        bounds=(lower, upper),
        jac="3-point",
        method="trf",
        x_scale="jac",
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
        max_nfev=max_nfev,
    )
    candidate, rotation, working = lattice(optimized.x)
    reciprocal = _normalize(candidate, ref)
    predicted = data.hkl @ reciprocal
    predicted /= np.linalg.norm(predicted, axis=1)[:, None]
    angles = np.degrees(
        np.arctan2(
            np.linalg.norm(np.cross(predicted, data.qhat), axis=1),
            np.sum(predicted * data.qhat, axis=1),
        )
    )
    singular = np.linalg.svd(optimized.jac, compute_uv=False)
    rank = int(
        np.sum(singular > singular[0] * max(optimized.jac.shape) * np.finfo(float).eps)
    )
    condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else np.inf
    status = "converged"
    if not optimized.success:
        status = "nonconverged"
    elif rank < n_parameters:
        status = "insufficient_support"
    elif condition > condition_limit:
        status = "ill_conditioned"
    epsilon = (
        _strain(reciprocal, ref, coordinates)
        if status == "converged"
        else np.full((3, 3), np.nan)
    )
    provenance = {
        "measure": "deviatoric Biot stretch",
        "scale": "reference volume",
        "objective": "uniform unit-direction chord least squares",
        "reference_parameters_nm_deg": ref_parameters,
        "reference_source": reference.source if hasattr(reference, "source") else None,
        "reference_name": reference.name if hasattr(reference, "name") else None,
        "initial_reciprocal": data.initial_reciprocal,
        "effective_initial_parameters_nm_deg": start,
        "effective_initial_rotation": rotation0,
        "fit_mask": mask,
        "frozen_values": "reference, before volume normalization",
        "coordinates": coordinates,
        "optimizer": "scipy.optimize.least_squares/trf",
        "max_nfev": max_nfev,
        "condition_limit": condition_limit,
        "optimizer_success": bool(optimized.success),
        "optimizer_status": int(optimized.status),
        "optimality": float(optimized.optimality),
        "empirical_igor_parity": "pending",
    }
    return StrainResult(
        status,
        str(optimized.message),
        coordinates,
        mask,
        epsilon,
        reciprocal,
        ref,
        rotation,
        _cell_parameters(reciprocal),
        working,
        data,
        predicted,
        _pixel_predictions(data, reciprocal),
        angles,
        n_parameters,
        rank,
        condition,
        int(optimized.nfev),
        _json(provenance),
    )


class StrainRefiner:
    """Reusable strain configuration with independent results for each call.

    Parameters
    ----------
    reference : lauelab.indexing.Cell or lauelab.indexing.Crystal
        Explicit unstrained reference, independent of source indexing metadata.
    geometry : lauelab.indexing.Geometry or str or pathlib.Path or None
        Calibration for measured centroids; saved source geometry is used when
        None. Prepared directions retain their original fixed calibration.
    detector_index : int or None
        Physical detector slot, resolved from source metadata when omitted.
    fit : str or tuple of str
        Igor mask or free parameter names, default ``"110111"``.
    coordinates : str
        Result frame, default ``"beamline"``.
    max_nfev : int
        Maximum optimizer function evaluations, default 1000.
    condition_limit : float
        Numerical joint Jacobian condition bound, default 1e8.

    Notes
    -----
    Model, frames and numerical controls follow refine_deviatoric. Configuration
    is read-only; use replace to change it. No previous fit or active pattern
    is retained. Invalid configuration raises InputError, TypeError or ValueError.
    """

    def __init__(
        self,
        reference,
        *,
        geometry=None,
        detector_index=None,
        fit="110111",
        coordinates="beamline",
        max_nfev=1000,
        condition_limit=1e8,
    ):
        _, self._reference_reciprocal = _reference(reference)
        _, fit = _mask(fit)
        _coordinates(coordinates)
        max_nfev, condition_limit = _controls(max_nfev, condition_limit)
        self.reference = reference
        self.geometry = (
            geometry
            if geometry is None or isinstance(geometry, Geometry)
            else Geometry(geometry)
        )
        self.detector_index = (
            None if detector_index is None else _index(detector_index, "detector_index")
        )
        if self.geometry is not None and detector_index is not None:
            self.geometry.detector(self.detector_index)
        self.fit, self.coordinates = fit, coordinates
        self.max_nfev, self.condition_limit = max_nfev, condition_limit
        self._configured = True

    def __setattr__(self, name, value):
        if getattr(self, "_configured", False):
            raise AttributeError("strain configuration is read-only; use replace(...)")
        object.__setattr__(self, name, value)

    def __repr__(self):
        return f"StrainRefiner(fit={self.fit!r}, coordinates={self.coordinates!r})"

    def replace(self, **changes):
        """Return an independently validated refiner with changed configuration."""
        values = {
            name: getattr(self, name)
            for name in (
                "reference",
                "geometry",
                "detector_index",
                "fit",
                "coordinates",
                "max_nfev",
                "condition_limit",
            )
        }
        values.update(changes)
        return type(self)(**values)

    def refine(
        self, source, *, frame_id=None, pattern=0, peak_indices=None, depth=None
    ):
        """Return one result; malformed selected input raises an exception.

        Parameters
        ----------
        source : object
            Any source accepted by prepare_reflections, including prepared data.
        frame_id : str or int or None
            Source frame; required when the source contains multiple frames.
        pattern : int
            Nonnegative pattern rank, default zero.
        peak_indices : numpy.ndarray or None
            Original measured-peak IDs to select in source assignment order.
        depth : float or None
            Sample depth override in µm, including explicit zero.

        Returns
        -------
        StrainResult
            Independent fit result, including unsuccessful numerical outcomes.

        Notes
        -----
        Sources and selections follow prepare_reflections. Geometry is taken
        from this configuration, or from a saved source when not configured.
        The explicit reference is never replaced by source crystal metadata.
        """
        data = self._prepare(
            source,
            frame_id=frame_id,
            pattern=pattern,
            peak_indices=peak_indices,
            depth=depth,
        )
        return self._refine(data)

    def _prepare(self, source, **selection):
        raw = source.source if isinstance(source, _Source) else source
        # A preparation-stage output already fixes the measured directions and
        # calibration. Constructor defaults are not a request to reinterpret it.
        if isinstance(raw, StrainReflections):
            return prepare_reflections(source, **selection)
        return prepare_reflections(
            source,
            geometry=self.geometry,
            detector_index=self.detector_index,
            **selection,
        )

    def _refine(self, data):
        return refine_deviatoric(
            data,
            reference=self.reference,
            fit=self.fit,
            coordinates=self.coordinates,
            max_nfev=self.max_nfev,
            condition_limit=self.condition_limit,
        )

    def refine_many(self, source, *, frame_ids=None, patterns="all"):
        """Return ordered outcomes, fitting all patterns by default.

        Parameters
        ----------
        source : object
            Any source accepted by prepare_reflections. Saved files are read once.
        frame_ids : tuple or None
            Explicit frame IDs to include; None includes all frames. Results
            retain source order independently of filter order.
        patterns : str or tuple of int
            ``"all"`` (default), ``"best"`` (lowest rank per frame), or
            explicit nonnegative ranks to include.

        Returns
        -------
        list of StrainResult
            Independent results in source frame/pattern order.

        Notes
        -----
        Expected per-pattern input
        errors are recorded as invalid_input results; unexpected/internal or
        shared-source failures propagate. No support prefilter is applied.
        """
        context = _Source(source, self.geometry)
        if isinstance(patterns, str):
            if patterns not in ("all", "best"):
                raise InputError("patterns must be 'all', 'best' or explicit ranks")
        else:
            patterns = tuple(_index(p, "pattern") for p in patterns)
            if len(set(patterns)) != len(patterns):
                raise InputError("pattern selections must be unique")
        if frame_ids is not None:
            frame_ids = tuple(frame_ids)
            available = set(context.frame_ids)
            if not set(frame_ids) <= available:
                raise InputError("frame selection includes an unknown frame_id")
        keys = [key for key in context.keys if frame_ids is None or key[0] in frame_ids]
        if patterns == "best":
            best = {}
            for fid, p in keys:
                best[fid] = min(best.get(fid, p), p)
            keys = [key for key in keys if best[key[0]] == key[1]]
        elif patterns != "all":
            keys = [key for key in keys if key[1] in patterns]
        results = []
        for fid, pattern in keys:
            try:
                data = self._prepare(context, frame_id=fid, pattern=pattern)
            except (InputError, ValueError, TypeError) as error:
                data = StrainReflections(
                    np.empty((0, 3)),
                    np.empty((0, 3)),
                    None,
                    frame_id=fid,
                    pattern_index=pattern,
                    provenance_json=_json({"source": context.info}),
                )
                result = _failure(
                    data,
                    self._reference_reciprocal,
                    self.fit,
                    self.coordinates,
                    "invalid_input",
                    str(error),
                    reference=self.reference,
                    max_nfev=self.max_nfev,
                    condition_limit=self.condition_limit,
                )
            else:
                try:
                    result = self._refine(data)
                except InputError as error:
                    result = _failure(
                        data,
                        self._reference_reciprocal,
                        self.fit,
                        self.coordinates,
                        "invalid_input",
                        str(error),
                        reference=self.reference,
                        max_nfev=self.max_nfev,
                        condition_limit=self.condition_limit,
                    )
                # Unexpected solver ValueError/TypeError/RuntimeError propagate;
                # only our explicit scientific input exception is recoverable.
            results.append(result)
        return results
