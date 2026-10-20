"""Constrained shape optimization using JAX derivatives and SciPy SLSQP."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from scipy.optimize import Bounds, OptimizeResult, minimize


def minimize_design(
    loss: Callable,
    initial_parameters: Any,
    bounds: Bounds | Sequence | None = None,
    constraints: Sequence[Callable] = (),
    maxiter: int = 100,
    tol: float = 1e-6,
) -> OptimizeResult:
    """Minimize a scalar design objective with exact automatic derivatives.

    Parameters may be an array or a JAX PyTree. Each constraint callable accepts
    that representation and returns one or more margins that must be >= 0.
    Bounds follow SciPy's flattened parameter order; for dictionaries this is
    JAX's sorted-key PyTree order. Enable JAX's 64-bit mode for tight tolerances.

    The returned SciPy result retains its standard ``x`` and ``success`` fields
    and adds ``parameters`` (the original PyTree structure), ``initial_loss``,
    ``loss_history`` (initial and accepted-iterate values),
    ``constraint_margins``, ``max_constraint_violation``, and ``feasible``.
    A failed solver is returned as a failed result, including useful candidate
    parameters. Nonfinite objective, gradient, or constraint evaluations raise
    ``FloatingPointError`` instead of being silently accepted.
    """
    if maxiter < 1:
        raise ValueError("maxiter must be positive")
    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be finite and positive")
    initial_parameters = jax.tree_util.tree_map(
        lambda leaf: jnp.asarray(leaf, dtype=jnp.result_type(leaf, float)),
        initial_parameters,
    )
    flat_initial, unravel = ravel_pytree(initial_parameters)
    initial = np.asarray(flat_initial, dtype=float)
    if not initial.size or not np.all(np.isfinite(initial)):
        raise ValueError("initial_parameters must contain finite design variables")
    if bounds is not None:
        if isinstance(bounds, Bounds):
            try:
                lower = np.broadcast_to(np.asarray(bounds.lb, dtype=float), initial.shape)
                upper = np.broadcast_to(np.asarray(bounds.ub, dtype=float), initial.shape)
            except ValueError as error:
                raise ValueError("bounds must match the flattened parameter count") from error
        else:
            pairs = list(bounds)
            if len(pairs) != initial.size or any(len(pair) != 2 for pair in pairs):
                raise ValueError("provide one (lower, upper) bound per flattened parameter")
            lower = np.array(
                [-np.inf if pair[0] is None else pair[0] for pair in pairs], dtype=float
            )
            upper = np.array(
                [np.inf if pair[1] is None else pair[1] for pair in pairs], dtype=float
            )
        if np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
            raise ValueError("bounds must not contain NaN")
        if np.any(lower > upper):
            raise ValueError("each lower bound must be <= its upper bound")
        bounds = Bounds(lower, upper)
    constraint_functions = tuple(constraints)

    def flat_loss(flat):
        return loss(unravel(flat))

    value_and_gradient = jax.jit(jax.value_and_grad(flat_loss))
    objective_cache: dict = {}

    def evaluate_objective(flat):
        flat = np.asarray(flat, dtype=float)
        if "x" not in objective_cache or not np.array_equal(flat, objective_cache["x"]):
            value, gradient = value_and_gradient(jnp.asarray(flat))
            value = float(value)
            gradient = np.asarray(gradient, dtype=float)
            if not np.isfinite(value) or not np.all(np.isfinite(gradient)):
                raise FloatingPointError(
                    "The design objective or its gradient is nonfinite. "
                    "Check geometry validity and equilibrium convergence."
                )
            objective_cache.update(x=flat.copy(), value=value, gradient=gradient)
        return objective_cache["value"], objective_cache["gradient"]

    scipy_constraints = []
    constraint_cache: dict = {}
    if constraint_functions:

        def flat_constraints(flat):
            parameters = unravel(flat)
            return jnp.concatenate(
                [jnp.ravel(function(parameters)) for function in constraint_functions]
            )

        evaluate_constraints_jax = jax.jit(
            lambda flat: (flat_constraints(flat), jax.jacrev(flat_constraints)(flat))
        )

        def evaluate_constraints(flat):
            flat = np.asarray(flat, dtype=float)
            if "x" not in constraint_cache or not np.array_equal(flat, constraint_cache["x"]):
                margins, jacobian = evaluate_constraints_jax(jnp.asarray(flat))
                margins = np.asarray(margins, dtype=float)
                jacobian = np.asarray(jacobian, dtype=float)
                if not np.all(np.isfinite(margins)) or not np.all(np.isfinite(jacobian)):
                    raise FloatingPointError(
                        "Design constraints or their derivatives are nonfinite."
                    )
                constraint_cache.update(x=flat.copy(), margins=margins, jacobian=jacobian)
            return constraint_cache["margins"], constraint_cache["jacobian"]

        scipy_constraints.append(
            {
                "type": "ineq",
                "fun": lambda flat: evaluate_constraints(flat)[0],
                "jac": lambda flat: evaluate_constraints(flat)[1],
            }
        )

    initial_loss = evaluate_objective(initial)[0]
    history = [initial_loss]

    def record_iteration(flat):
        history.append(evaluate_objective(flat)[0])

    result = minimize(
        evaluate_objective,
        initial,
        jac=True,
        method="SLSQP",
        bounds=bounds,
        constraints=scipy_constraints,
        callback=record_iteration,
        options={"maxiter": int(maxiter), "ftol": tol},
    )
    final_loss = evaluate_objective(result.x)[0]
    if history[-1] != final_loss:
        history.append(final_loss)
    margins = (
        evaluate_constraints(result.x)[0] if constraint_functions else np.empty(0, dtype=float)
    )
    max_violation = max(0.0, -float(np.min(margins))) if margins.size else 0.0
    if bounds is not None:
        max_violation = max(
            max_violation,
            float(np.max(np.maximum(lower - result.x, 0.0))),
            float(np.max(np.maximum(result.x - upper, 0.0))),
        )
    feasible = bool(max_violation <= tol)
    result.parameters = unravel(jnp.asarray(result.x))
    result.initial_loss = initial_loss
    result.loss_history = np.asarray(history)
    result.constraint_margins = margins
    result.max_constraint_violation = max_violation
    result.feasible = feasible
    if result.success and not feasible:
        result.success = False
        result.message = f"{result.message}; final constraint violation exceeds {tol:g}."
    return result
