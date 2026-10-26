# Copyright © 2026 UChicago Argonne, LLC. All rights reserved.
# Full license accessible at https://github.com/AdvancedPhotonSource/lauelab/blob/main/LICENSE
from dataclasses import fields

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from lauelab.indexing import Cell, InputError
from lauelab.strain import (
    StrainRefiner,
    StrainReflections,
    refine_deviatoric,
    strain_from_reciprocal,
)
from strain_cases import HKL, case, direct


@pytest.mark.parametrize(
    "cell",
    [
        Cell(0.35, 0.35, 0.35),
        Cell(0.29, 0.29, 0.46, 90, 90, 120),
        Cell(0.4, 0.5, 0.6, 78, 82, 104),
    ],
)
def test_known_deformation_recovered_from_assigned_directions(cell):
    ref, b, initial, q, expected = case(cell)
    result = refine_deviatoric(q, HKL, reference=ref, initial_reciprocal=initial)
    assert result.success, result.message
    assert result.rank == result.n_parameters == 8
    assert (
        result.nfev > 2
    )  # Regression: a tiny nonzero starting vector shrank the trust region.
    assert result.rms_error_deg < 1e-9
    np.testing.assert_allclose(
        result.tensor_in("crystal"), expected, atol=2e-10, rtol=0
    )
    np.testing.assert_allclose(
        result.reciprocal / np.linalg.norm(result.reciprocal),
        b / np.linalg.norm(b),
        atol=2e-10,
    )
    np.testing.assert_allclose(
        np.linalg.det(result.reciprocal),
        np.linalg.det(result.reference_reciprocal),
        rtol=2e-14,
    )


def test_noisy_recovery_and_independent_withheld_predictions():
    ref, b, initial, q, expected = case()
    noise = np.random.default_rng(42).normal(0, 2e-6, q.shape)
    noisy = q + noise
    noisy /= np.linalg.norm(noisy, axis=1)[:, None]
    result = refine_deviatoric(
        noisy[:8], HKL[:8], reference=ref, initial_reciprocal=initial
    )
    assert result.success
    assert result.rms_error_deg > 0
    np.testing.assert_allclose(result.tensor_in("crystal"), expected, atol=1e-5, rtol=0)
    predicted = HKL[8:] @ result.reciprocal
    predicted /= np.linalg.norm(predicted, axis=1)[:, None]
    np.testing.assert_allclose(predicted, q[8:], atol=1e-5, rtol=0)


def test_reduced_models_freeze_reference_values_without_a_public_source_switch():
    ref = Cell(0.4, 0.5, 0.6, 78, 82, 104)
    working = Cell(0.402, 0.498, 0.6, 78, 82, 104)
    rotated = Rotation.from_rotvec([0.4, -0.1, 0.2]).as_matrix()
    b = 2 * np.pi * np.linalg.inv(rotated @ direct(working))
    initial_cell = Cell(0.401, 0.499, 0.61, 79, 81, 103)
    initial = 2 * np.pi * np.linalg.inv(rotated @ direct(initial_cell))
    q = HKL @ b
    q /= np.linalg.norm(q, axis=1)[:, None]
    result = refine_deviatoric(
        q, HKL, reference=ref, initial_reciprocal=initial, fit=("a", "b")
    )
    assert result.success
    np.testing.assert_allclose(
        result.working_cell_parameters[2:], [0.6, 78, 82, 104], atol=1e-12
    )
    assert (
        result.cell_parameters[2] != 0.6
    )  # Final common volume normalization changes held lengths.
    np.testing.assert_allclose(
        result.tensor_in("crystal"),
        strain_from_reciprocal(b, reference=ref),
        atol=2e-10,
    )


@pytest.mark.parametrize("fit", ["110111", "101111", "011111"])
def test_any_held_length_can_anchor_the_general_shape_model(fit):
    ref, b, initial, q, expected = case()
    result = refine_deviatoric(
        q, HKL, reference=ref, initial_reciprocal=initial, fit=fit
    )
    assert result.success
    np.testing.assert_allclose(result.tensor_in("crystal"), expected, atol=2e-10)


@pytest.mark.parametrize(
    "fit", ["111111", "000000", "11011", "210111", ("a", "a"), ("aardvark",)]
)
def test_invalid_constraints_are_errors(fit):
    ref, _, initial, q, _ = case()
    with pytest.raises(InputError):
        refine_deviatoric(q, HKL, reference=ref, initial_reciprocal=initial, fit=fit)


def test_insufficient_observations_and_nonconvergence_are_not_zero_strain():
    ref, _, initial, q, _ = case()
    insufficient = refine_deviatoric(
        q[:4], HKL[:4], reference=ref, initial_reciprocal=initial
    )
    assert insufficient.status == "insufficient_support"
    assert np.isnan(insufficient.tensor).all()
    stopped = refine_deviatoric(
        q, HKL, reference=ref, initial_reciprocal=initial, max_nfev=1
    )
    assert stopped.status == "nonconverged"
    assert np.isnan(stopped.tensor).all()
    assert np.isfinite(stopped.reciprocal).all()


def test_support_failure_retains_the_effective_scientific_configuration():
    ref, _, initial, q, _ = case()
    result = refine_deviatoric(
        q[:3],
        HKL[:3],
        reference=ref,
        initial_reciprocal=initial,
        coordinates="sample",
        max_nfev=17,
        condition_limit=1e5,
    )
    assert result.status == "insufficient_support"
    assert result.provenance["max_nfev"] == 17
    assert result.provenance["condition_limit"] == 1e5
    assert result.provenance["coordinates"] == "sample"
    assert result.provenance["fit_mask"] == "110111"
    assert result.provenance["optimizer_run"] is False
    np.testing.assert_array_equal(result.provenance["initial_reciprocal"], initial)


def test_coplanar_reflections_do_not_pass_merely_because_there_are_many():
    ref, _, initial, _, _ = case()
    h = np.array([[1, i, 0] for i in range(-4, 5)])
    q = h @ initial
    q /= np.linalg.norm(q, axis=1)[:, None]
    result = refine_deviatoric(q, h, reference=ref, initial_reciprocal=initial)
    assert not result.success
    assert result.status in ("insufficient_support", "ill_conditioned")
    assert np.isnan(result.tensor).all()


def test_input_identity_and_harmonics_cannot_manufacture_support():
    ref, _, initial, q, _ = case()
    with pytest.raises(InputError, match="harmonics"):
        refine_deviatoric(
            q[:2], [[1, 1, 1], [2, 2, 2]], reference=ref, initial_reciprocal=initial
        )
    with pytest.raises(InputError, match="peak_indices"):
        StrainReflections(q, HKL, initial, peak_indices=np.zeros(len(q), dtype=int))


def test_results_own_arrays_and_provenance_and_preserve_igor_scalar():
    ref, _, initial, q, _ = case()
    result = refine_deviatoric(q, HKL, reference=ref, initial_reciprocal=initial)
    tensor = result.tensor.copy()
    q[:] = 0
    initial[:] = 0
    np.testing.assert_array_equal(result.tensor, tensor)
    for item in fields(result):
        value = getattr(result, item.name)
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable
    result.provenance["fit_mask"] = "broken"
    assert result.provenance["fit_mask"] == "110111"
    assert result.von_mises_strain == pytest.approx(np.sqrt(1.5 * np.sum(tensor**2)))


def test_configured_refiner_replacement_does_not_change_original():
    ref, _, _, _, _ = case()
    refiner = StrainRefiner(ref)
    changed = refiner.replace(fit=("a", "b"), coordinates="crystal")
    assert refiner.fit == "110111" and changed.fit == "110000"
    assert refiner.coordinates == "beamline" and changed.coordinates == "crystal"
    with pytest.raises(AttributeError, match="replace"):
        refiner.fit = "000000"
