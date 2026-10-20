"""NLopt's callback sign/order and honest stopping diagnostics."""

import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import Bounds

from rotating_polygons.nlopt_optimization import minimize_design_nlopt

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def nlopt_backend():
    """Core tests still run when the optional NLopt wheel is not installed."""
    return pytest.importorskip("nlopt")


def test_optional_dependency_is_lazy_with_actionable_install_message():
    script = textwrap.dedent("""
        import builtins
        import sys
        original_import = builtins.__import__
        def without_nlopt(name, *args, **kwargs):
            if name == "nlopt" or name.startswith("nlopt."):
                raise ModuleNotFoundError("NLopt deliberately unavailable", name="nlopt")
            return original_import(name, *args, **kwargs)
        builtins.__import__ = without_nlopt
        import rotating_polygons
        from rotating_polygons.nlopt_optimization import minimize_design_nlopt
        assert "nlopt" not in sys.modules
        try:
            minimize_design_nlopt(lambda parameters: parameters ** 2, 1.0)
        except ImportError as error:
            assert "rotating-polygons[nlopt]" in str(error)
        else:
            raise AssertionError("Expected an optional-dependency error")
    """)
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("algorithm", ["slsqp", "mma"])
def test_constrained_analytic_optimum_and_objective_derivative(nlopt_backend, algorithm):
    result = minimize_design_nlopt(
        lambda parameters: jnp.sum((parameters - 2.0) ** 2),
        jnp.array([0.2, 0.3]),
        bounds=[(0.0, 3.0), (0.0, 3.0)],
        constraints=[lambda parameters: 1.0 - jnp.sum(parameters)],
        algorithm=algorithm,
        maxeval=300,
        # MMA's approximate subproblems can terminate with a small active-
        # constraint residual; choose and verify an explicit absolute tolerance.
        constraint_tol=1e-6,
    )
    assert result.success, result.message
    assert result.feasible
    assert result.max_constraint_violation <= 1e-6
    np.testing.assert_allclose(result.x, [0.5, 0.5], atol=1e-4)
    np.testing.assert_allclose(result.jac, 2 * (result.x - 2.0), atol=1e-12)
    assert result.fun < result.initial_loss
    assert result.nit is None
    assert result.loss_history.size == result.nfev + 1


def test_nonlinear_vector_constraint_sign_and_rectangular_jacobian(nlopt_backend):
    # Maximize x subject to y >= x² and x + 2y <= 1. The unique x/y
    # solution is (1/2, 1/4); z is independent with optimum 0.7.
    # Two constraint rows and three variable columns catch a transpose error.
    result = minimize_design_nlopt(
        lambda p: -p[0] + (p[2] - 0.7) ** 2,
        jnp.array([0.2, 0.1, 0.3]),
        bounds=[(0, 2), (0, 2), (0, 2)],
        constraints=[
            lambda p: 1.0 - p[0] - 2 * p[1],
            lambda p: jnp.array([p[1] - p[0] ** 2]),
        ],
        maxeval=200,
        constraint_tol=1e-9,
    )
    assert result.success, result.message
    np.testing.assert_allclose(result.x, [0.5, 0.25, 0.7], atol=1e-5)
    np.testing.assert_allclose(result.constraint_margins, [0, 0], atol=1e-8)


def test_pytree_sorted_key_bounds_and_restored_shapes(nlopt_backend):
    # Deliberately insert z before a. JAX flattens the keys as a, z.
    initial = {"z": jnp.array([0.2, -0.2]), "a": 0}
    result = minimize_design_nlopt(
        lambda p: (p["a"] - 2) ** 2 + jnp.sum((p["z"] - jnp.array([1, -1])) ** 2),
        initial,
        bounds=Bounds([0, 0, -2], [0.5, 2, 0]),
    )
    assert result.success, result.message
    np.testing.assert_allclose(result.x, [0.5, 1, -1], atol=1e-6)
    assert np.asarray(result.parameters["z"]).shape == (2,)
    assert float(result.parameters["a"]) == pytest.approx(0.5)


def test_evaluation_limit_is_not_reported_as_convergence(nlopt_backend):
    result = minimize_design_nlopt(
        lambda p: jnp.sum((p - 2) ** 2),
        jnp.zeros(2),
        maxeval=1,
        ftol_rel=0,
        xtol_rel=0,
    )
    assert result.status == nlopt_backend.MAXEVAL_REACHED
    assert not result.success
    assert result.feasible
    assert result.nfev == 1
    assert "limit" in result.message.lower()


def test_time_limit_is_not_reported_as_convergence(nlopt_backend):
    result = minimize_design_nlopt(
        lambda p: jnp.sum((p - 2) ** 2),
        jnp.zeros(2),
        maxeval=10000,
        ftol_rel=0,
        xtol_rel=0,
        maxtime=1e-12,
    )
    assert result.status == nlopt_backend.MAXTIME_REACHED
    assert not result.success
    assert "time limit" in result.message.lower()


def test_infeasible_returned_candidate_is_a_failure(nlopt_backend):
    result = minimize_design_nlopt(
        lambda p: jnp.sum(p**2),
        jnp.ones(1),
        constraints=[lambda p: jnp.array([-1.0])],
        maxeval=10,
    )
    assert not result.success
    assert not result.feasible
    assert result.max_constraint_violation == pytest.approx(1.0)


def test_nonfinite_trial_callback_is_not_hidden_by_nlopt(nlopt_backend):
    with pytest.raises(FloatingPointError, match="nonfinite"):
        minimize_design_nlopt(
            lambda p: jnp.where(p[0] > 0, jnp.nan, (p[0] - 1) ** 2),
            jnp.zeros(1),
            bounds=[(0, 2)],
        )


@pytest.mark.parametrize(
    "settings",
    [
        {"algorithm": "cobyla"},
        {"maxeval": 0},
        {"maxeval": 1.5},
        {"ftol_rel": np.nan},
        {"xtol_rel": -1},
        {"constraint_tol": np.inf},
        {"maxtime": 0},
        {"bounds": [(np.nan, 2)]},
        {"bounds": [(1, 2)]},
    ],
)
def test_invalid_options_are_rejected(nlopt_backend, settings):
    with pytest.raises(ValueError):
        minimize_design_nlopt(lambda p: jnp.sum(p**2), jnp.zeros(1), **settings)
