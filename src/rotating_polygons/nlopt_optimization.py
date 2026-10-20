"""Optional NLopt optimizers with exact JAX derivatives.

The Python callback conventions follow the official reference:
https://nlopt.readthedocs.io/en/latest/NLopt_Python_Reference/
NLopt itself is imported only when ``minimize_design_nlopt`` is called.
"""

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from scipy.optimize import Bounds, OptimizeResult


def _bounds_arrays(bounds, size):
    """Normalize bounds without silently clipping the supplied initial design."""
    if bounds is None:
        return np.full(size, -np.inf), np.full(size, np.inf)
    if isinstance(bounds, Bounds):
        try:
            lower = np.broadcast_to(np.asarray(bounds.lb, dtype=float), (size,))
            upper = np.broadcast_to(np.asarray(bounds.ub, dtype=float), (size,))
        except ValueError as error:
            raise ValueError("bounds must match the flattened parameter count") from error
    else:
        pairs = list(bounds)
        if len(pairs) != size or any(len(pair) != 2 for pair in pairs):
            raise ValueError("provide one (lower, upper) bound per flattened parameter")
        lower = np.array([-np.inf if pair[0] is None else pair[0] for pair in pairs], dtype=float)
        upper = np.array([np.inf if pair[1] is None else pair[1] for pair in pairs], dtype=float)
    if np.any(np.isnan(lower)) or np.any(np.isnan(upper)):
        raise ValueError("bounds must not contain NaN")
    if np.any(lower > upper):
        raise ValueError("each lower bound must be <= its upper bound")
    return lower, upper


def minimize_design_nlopt(
    loss: Callable,
    initial_parameters: Any,
    bounds: Bounds | Sequence | None = None,
    constraints: Sequence[Callable] = (),
    *,
    algorithm: str = "slsqp",
    maxeval: int = 100,
    ftol_rel: float = 1e-8,
    xtol_rel: float = 1e-8,
    constraint_tol: float = 1e-8,
    maxtime: float | None = None,
) -> OptimizeResult:
    """Minimize an array/PyTree design with NLopt SLSQP or MMA.

    ``constraints`` return margins >= 0, exactly as in ``minimize_design``.
    This adapter negates both the margins and their Jacobian for NLopt's <= 0
    convention. Only nonlinear inequalities and box bounds are supported.
    Bounds follow JAX's flattened PyTree order (sorted keys for dictionaries).
    The initial design must lie inside its box bounds.

    ``maxeval`` limits objective calls, not iterations. ``maxtime`` optionally
    limits NLopt's elapsed time in seconds; initial JAX compilation/validation
    occurs before that timer. Relative tolerances may be zero to disable them;
    ``constraint_tol`` is an absolute feasibility tolerance for each margin.

    The SciPy ``OptimizeResult`` contains flat ``x``, restored ``parameters``,
    ``fun``, ``jac``, raw NLopt ``status``/``nlopt_status``, ``nfev``,
    ``initial_loss``, ``loss_history``, ``constraint_margins``,
    ``max_constraint_violation``, and ``feasible``. ``loss_history`` records the
    initial preflight value followed by objective callback values, including
    rejected trial points and possible repeats; these are NOT iterations.
    NLopt does not expose an iteration count, so ``nit`` is ``None``.

    Only convergence statuses 1–4 combined with feasibility produce
    ``success=True``. Evaluation/time limits (5/6), roundoff limitation, and
    other optimizer failures return a failed result with the returned candidate.
    Exceptions raised by user callbacks, including nonfinite-value errors,
    propagate. Enable JAX 64-bit arithmetic before constructing the design.
    """
    try:
        import nlopt
    except ImportError as error:
        raise ImportError(
            "NLopt is optional. Install it with `pip install 'rotating-polygons[nlopt]'` "
            "or `pip install -e '.[nlopt]'` in the source checkout."
        ) from error

    algorithms = {"slsqp": nlopt.LD_SLSQP, "mma": nlopt.LD_MMA}
    if algorithm not in algorithms:
        raise ValueError("algorithm must be 'slsqp' or 'mma'")
    if isinstance(maxeval, bool) or not isinstance(maxeval, (int, np.integer)) or maxeval < 1:
        raise ValueError("maxeval must be a positive integer")
    for name, value in (
        ("ftol_rel", ftol_rel),
        ("xtol_rel", xtol_rel),
        ("constraint_tol", constraint_tol),
    ):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if maxtime is not None and (not np.isfinite(maxtime) or maxtime <= 0):
        raise ValueError("maxtime must be finite and positive")

    def real_leaf(leaf):
        if jnp.iscomplexobj(leaf):
            raise ValueError("design variables must be real")
        return jnp.asarray(leaf, dtype=jnp.result_type(leaf, float))

    initial_parameters = jax.tree_util.tree_map(real_leaf, initial_parameters)
    flat_initial, unravel = ravel_pytree(initial_parameters)
    initial = np.asarray(flat_initial, dtype=float)
    if not initial.size or not np.all(np.isfinite(initial)):
        raise ValueError("initial_parameters must contain finite design variables")
    lower, upper = _bounds_arrays(bounds, initial.size)
    if np.any(initial < lower) or np.any(initial > upper):
        raise ValueError("initial_parameters must satisfy the box bounds")

    value_and_gradient = jax.jit(jax.value_and_grad(lambda flat: loss(unravel(flat))))
    objective_cache = {}

    def evaluate_objective(flat):
        # NLopt owns its callback input buffer. A copy also keeps exception
        # tracebacks from retaining a view into memory released by the solver.
        flat = np.array(flat, dtype=float, copy=True)
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

    constraint_functions = tuple(constraints)
    constraint_cache = {}
    if constraint_functions:

        def flat_margins(flat):
            parameters = unravel(flat)
            return jnp.concatenate(
                [jnp.ravel(function(parameters)) for function in constraint_functions]
            )

        margins_and_jacobian = jax.jit(
            lambda flat: (flat_margins(flat), jax.jacrev(flat_margins)(flat))
        )

        def evaluate_constraints(flat):
            flat = np.array(flat, dtype=float, copy=True)
            if "x" not in constraint_cache or not np.array_equal(flat, constraint_cache["x"]):
                margins, jacobian = margins_and_jacobian(jnp.asarray(flat))
                margins = np.asarray(margins, dtype=float)
                jacobian = np.asarray(jacobian, dtype=float)
                if not np.all(np.isfinite(margins)) or not np.all(np.isfinite(jacobian)):
                    raise FloatingPointError(
                        "Design constraints or their derivatives are nonfinite."
                    )
                constraint_cache.update(x=flat.copy(), margins=margins, jacobian=jacobian)
            return constraint_cache["margins"], constraint_cache["jacobian"]

    # Preflight compiles both JAX paths before starting NLopt's own timer.
    initial_loss = evaluate_objective(initial)[0]
    initial_margins = evaluate_constraints(initial)[0] if constraint_functions else np.empty(0)
    history = [initial_loss]
    optimizer = nlopt.opt(algorithms[algorithm], initial.size)
    # Preserve NLopt's returned candidate even on roundoff/failure statuses.
    # Callback failures are transferred explicitly below: NLopt 2.11's SWIG
    # wrapper can otherwise replace the original exception with SystemError.
    optimizer.set_exceptions_enabled(False)
    optimizer.set_lower_bounds(lower)
    optimizer.set_upper_bounds(upper)
    optimizer.set_maxeval(int(maxeval))
    optimizer.set_ftol_rel(float(ftol_rel))
    optimizer.set_xtol_rel(float(xtol_rel))
    if maxtime is not None:
        optimizer.set_maxtime(float(maxtime))

    callback_failure = []

    def objective_callback(flat, gradient):
        try:
            value, derivative = evaluate_objective(flat)
        except BaseException as error:
            callback_failure.append(error)
            optimizer.force_stop()
            if gradient.size:
                gradient[:] = 0.0
            return float("inf")
        if gradient.size:
            gradient[:] = derivative  # NLopt requires an in-place write.
        history.append(value)
        return value

    optimizer.set_min_objective(objective_callback)
    if initial_margins.size:

        def constraint_callback(result, flat, gradient):
            try:
                margins, jacobian = evaluate_constraints(flat)
            except BaseException as error:
                callback_failure.append(error)
                optimizer.force_stop()
                result[:] = float("inf")
                if gradient.size:
                    gradient[:, :] = 0.0
                return
            result[:] = -margins
            if gradient.size:
                # Rows identify constraints; columns identify scalar variables.
                gradient[:, :] = -jacobian

        optimizer.add_inequality_mconstraint(
            constraint_callback, np.full(initial_margins.size, constraint_tol)
        )

    optimum = np.asarray(optimizer.optimize(initial.copy()), dtype=float)
    if callback_failure:
        raise callback_failure[0]
    status = int(optimizer.last_optimize_result())
    final_loss, final_gradient = evaluate_objective(optimum)
    final_margins = evaluate_constraints(optimum)[0] if constraint_functions else np.empty(0)
    max_violation = max(
        0.0,
        -float(np.min(final_margins)) if final_margins.size else 0.0,
        float(np.max(np.maximum(lower - optimum, 0.0))),
        float(np.max(np.maximum(optimum - upper, 0.0))),
    )
    feasible = bool(max_violation <= constraint_tol)
    messages = {
        1: "NLopt reported convergence.",
        2: "NLopt objective target reached.",
        3: "NLopt relative objective tolerance reached.",
        4: "NLopt relative parameter tolerance reached.",
        5: "NLopt evaluation limit reached; convergence was not established.",
        6: "NLopt time limit reached; convergence was not established.",
        -1: "NLopt reported a numerical failure.",
        -2: "NLopt rejected the optimization arguments.",
        -3: "NLopt ran out of memory.",
        -4: "NLopt stopped because roundoff limited progress.",
        -5: "NLopt was forcibly stopped.",
    }
    message = messages.get(status, f"NLopt returned unrecognized status {status}.")
    if not feasible:
        message += f" Final constraint violation exceeds {constraint_tol:g}."
    return OptimizeResult(
        x=optimum,
        parameters=unravel(jnp.asarray(optimum)),
        fun=final_loss,
        jac=final_gradient,
        success=bool(status in (1, 2, 3, 4) and feasible),
        status=status,
        nlopt_status=status,
        message=message,
        algorithm=algorithm,
        nfev=int(optimizer.get_numevals()),
        nit=None,
        initial_loss=initial_loss,
        loss_history=np.asarray(history),
        history_kind="initial preflight and objective evaluations",
        constraint_margins=final_margins,
        max_constraint_violation=max_violation,
        feasible=feasible,
    )
