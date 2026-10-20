"""Fit polygon shapes with NLopt while retaining the same JAX mechanics.

Install the optional backend and plotting support first:
    python -m pip install -e '.[nlopt,plot]'
    python examples/optimize_with_nlopt.py --output outputs

The default optimizes all 72 vertex coordinates independently. Use
``--two-parameters`` for the smaller repeated-offset teaching example.
"""

import argparse
from pathlib import Path

import jax
import numpy as np

from rotating_polygons.nlopt_optimization import minimize_design_nlopt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("slsqp", "mma"), default="slsqp")
    parser.add_argument("--two-parameters", action="store_true")
    parser.add_argument("--maxeval", type=int, default=300)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)

    if args.two_parameters:
        from optimize_square_grid import build_problem, plot_result
    else:
        from optimize_all_vertices import build_problem, plot_result

    problem = build_problem()
    # Geometry, energy, mechanical equilibrium, and the objective are unchanged.
    # Only this outer optimizer differs from the SciPy examples.
    curve_loss = getattr(problem, "curve_loss", problem.loss)
    initial_curve_loss = float(curve_loss(problem.initial_parameters))
    print(
        f"NLopt {args.algorithm}: {problem.initial_parameters.size} shape variables, "
        f"at most {args.maxeval} objective evaluations",
        flush=True,
    )
    result = minimize_design_nlopt(
        problem.loss,
        problem.initial_parameters,
        bounds=problem.bounds,
        constraints=[problem.constraint],  # Margins >= 0; the adapter flips the sign.
        algorithm=args.algorithm,
        maxeval=args.maxeval,
        ftol_rel=1e-8,
        xtol_rel=1e-7,
        constraint_tol=1e-8,
    )
    # A positive NLopt status can mean "evaluation/time limit reached".
    # The adapter's success flag deliberately excludes those cases.
    if not result.success or not result.feasible:
        raise RuntimeError(f"NLopt did not converge: {result.message}")
    final_curve_loss = float(curve_loss(result.parameters))
    if result.fun >= result.initial_loss or final_curve_loss >= initial_curve_loss:
        raise RuntimeError("The example did not improve its cost and stress-curve error")
    reference = problem.initialize_design(result.parameters)
    solution = problem.simulate(reference, problem.strains).check_convergence()
    print(f"Status {result.status}: {result.message}")
    print(f"Total cost: {result.initial_loss:.8g} -> {result.fun:.8g}")
    print(f"Curve MSE: {initial_curve_loss:.8g} -> {final_curve_loss:.8g}")
    print(
        f"Objective calls: {result.nfev}; maximum constraint violation: "
        f"{result.max_constraint_violation:.3g}"
    )
    print(f"Maximum equilibrium residual: {float(np.max(solution.residuals)):.3g}")

    # Existing plotting works because the adapter returns a SciPy-compatible
    # result with restored tensor/PyTree parameters and numerical diagnostics.
    args.output.mkdir(parents=True, exist_ok=True)
    label = "two_parameters" if args.two_parameters else "all_vertices"
    stem = f"nlopt_{args.algorithm}_{label}"
    figure = plot_result(problem, result)
    figure.savefig(args.output / f"{stem}.png", dpi=160)
    np.savez_compressed(
        args.output / f"{stem}.npz",
        parameters=np.asarray(result.parameters),
        vertices=np.asarray(reference.vertices),
        states=np.asarray(solution.states),
        strains=np.asarray(problem.strains),
        target_stress=np.asarray(problem.target_stress),
        optimized_stress=np.asarray(problem.measure_stress(solution.states, reference)),
        residuals=np.asarray(solution.residuals),
        converged=np.asarray(solution.converged),
        constraint_margins=result.constraint_margins,
        nlopt_status=result.status,
        objective_evaluations=result.nfev,
        loss_history=result.loss_history,
        initial_curve_loss=initial_curve_loss,
        final_curve_loss=final_curve_loss,
    )
    print(f"Saved NLopt results and figure to {args.output.resolve()}")


if __name__ == "__main__":
    main()
