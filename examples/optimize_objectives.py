"""Optimize shape for endpoint stiffness or uniform support forces.

Run after installing the package and plotting dependencies:
    python examples/optimize_objectives.py --objective all --output outputs

This reuses the geometry, boundary conditions, and two shape variables from
optimize_square_grid.py. Only the objective and the additional load restriction
change. Both objectives are dimensionless and are minimized.
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from optimize_square_grid import build_problem
from rotating_polygons import minimize_design, reaction_forces


def build_objectives(problem):
    """Construct two explicit costs and their common nonzero-load restriction."""
    centers = np.asarray(problem.geometry.reference_state)
    bottom = jnp.asarray(np.flatnonzero(np.isclose(centers[:, 1], centers[:, 1].min())))
    final_strain = float(problem.strains[-1])
    if final_strain <= 0:
        raise ValueError("Secant stiffness needs a positive final compression strain")

    def response(parameters):
        # Shape variables are local vertex displacements, in geometry length
        # units. Reinitialize so each candidate has its own rest configuration.
        reference = problem.initialize_design(parameters)
        solution = problem.simulate(reference, problem.strains)
        # Continue through all strain steps, but measure only the final state.
        stress = problem.measure_stress(solution.states[-1], reference)
        forces = reaction_forces(problem.potential, solution.states[-1], reference)[bottom, 1]
        # Bottom support forces are positive in compression with this boundary.
        return solution, stress / final_strain, forces

    initial_solution, initial_stiffness, initial_forces = response(problem.initial_parameters)
    initial_solution.check_convergence()
    initial_total_force = float(jnp.sum(initial_forces))
    if initial_total_force <= 0:
        raise ValueError("The starting design must carry a positive compression load")
    # A single secant stiffness is stress / strain, not the slope d(stress)/d(strain).
    # The synthetic target is attainable and uses the same reference cross-section.
    target_stiffness = float(problem.target_stress[-1] / final_strain)
    force_floor = initial_total_force / len(bottom) * 1e-12

    def stiffness_objective(parameters):
        solution, stiffness, _ = response(parameters)
        # Normalize by the target stiffness before squaring: a unitless cost.
        error = ((stiffness - target_stiffness) / target_stiffness) ** 2
        return jnp.where(jnp.all(solution.converged), error, jnp.inf)

    def uniformity_objective(parameters):
        solution, _, forces = response(parameters)
        mean_force = jnp.mean(forces)
        # Squared coefficient of variation: zero means equal support forces.
        # The floor prevents division by zero at a trial point; the separate
        # load constraint below excludes the trivial unloaded final design.
        spread = jnp.mean((forces - mean_force) ** 2)
        error = spread / jnp.maximum(mean_force**2, force_floor**2)
        return jnp.where(jnp.all(solution.converged), error, jnp.inf)

    def minimum_load_constraint(parameters):
        solution, _, forces = response(parameters)
        # All optimizer constraints are margins >= 0. Require at least half
        # the baseline total reaction, measured at the SAME imposed strain.
        margin = jnp.sum(forces) / initial_total_force - 0.5
        return jnp.where(jnp.all(solution.converged), margin, -jnp.inf)

    return SimpleNamespace(
        objectives={"stiffness": stiffness_objective, "uniformity": uniformity_objective},
        minimum_load_constraint=minimum_load_constraint,
        response=response,
        bottom=bottom,
        initial_forces=initial_forces,
        initial_stiffness=float(initial_stiffness),
        initial_total_force=initial_total_force,
        target_stiffness=target_stiffness,
    )


def optimize_objectives(problem, study, names=("stiffness", "uniformity"), max_iterations=60):
    """Solve independent designs from the same starting point, checking each."""
    results = {}
    for name in names:
        # SLSQP changes SHAPE. The inner BFGS solver still finds equilibrium.
        # Bounds limit each vertex offset; geometry margins preserve small
        # edges, convex corners, and nonzero reference ligament lengths.
        result = minimize_design(
            study.objectives[name],
            problem.initial_parameters,
            bounds=problem.bounds,
            constraints=[problem.constraint, study.minimum_load_constraint],
            maxiter=max_iterations,
            tol=1e-10,
        )
        if not result.success or not result.feasible:
            raise RuntimeError(f"{name} optimization failed: {result.message}")
        if not result.fun < result.initial_loss:
            raise RuntimeError(f"{name} objective did not improve")
        solution, _, forces = study.response(result.parameters)
        solution.check_convergence()
        if float(jnp.sum(forces)) < 0.5 * study.initial_total_force - 1e-9:
            raise RuntimeError(f"{name} design fails the minimum load requirement")
        results[name] = result
    return results


def plot_results(problem, study, results):
    """Compare costs, endpoint stiffness, and physical support-force patterns."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    names = ["Initial", *results]
    stiffnesses = [study.initial_stiffness]
    axes[1].plot(np.asarray(study.bottom), study.initial_forces, "o--", label="Initial")
    for name, result in results.items():
        solution, stiffness, forces = study.response(result.parameters)
        solution.check_convergence()
        stiffnesses.append(float(stiffness))
        # Different normalized costs have different meanings; each history
        # shows improvement of that objective, not a ranking of designs.
        axes[0].semilogy(np.maximum(result.loss_history, 1e-18), "o-", label=name)
        axes[1].plot(np.asarray(study.bottom), forces, "o-", label=name)
    axes[0].set(xlabel="Recorded design iteration", ylabel="Dimensionless objective")
    axes[0].legend(frameon=False)
    axes[1].set(xlabel="Bottom polygon index", ylabel="Vertical support force")
    axes[1].legend(frameon=False)
    axes[2].bar(names, stiffnesses, color=["#777777", "#147d92", "#bc6944"][: len(names)])
    axes[2].axhline(study.target_stiffness, color="black", ls="--", label="Stiffness target")
    axes[2].set(ylabel="Secant stiffness (stress / strain)")
    axes[2].legend(frameon=False)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objective", choices=("all", "stiffness", "uniformity"), default="all")
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--max-iterations", type=int, default=60)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    problem = build_problem()  # Explicit construction; importing the module runs no simulation.
    study = build_objectives(problem)
    names = tuple(study.objectives) if args.objective == "all" else (args.objective,)
    results = optimize_objectives(problem, study, names, args.max_iterations)

    print(f"Target secant stiffness: {study.target_stiffness:.8g}")
    print(f"Required total bottom force >= {0.5 * study.initial_total_force:.8g}")
    print("Objective     Initial loss   Final loss     Stiffness   Total force   Max residual")
    args.output.mkdir(parents=True, exist_ok=True)
    for name, result in results.items():
        solution, stiffness, forces = study.response(result.parameters)
        print(
            f"{name:12s} {result.initial_loss:12.5g} {result.fun:12.5g} "
            f"{float(stiffness):12.6g} {float(jnp.sum(forces)):12.6g} "
            f"{float(jnp.max(solution.residuals)):12.3g}"
        )
        np.savez_compressed(
            args.output / f"objective_{name}.npz",
            parameters=np.asarray(result.parameters),
            vertices=np.asarray(problem.vertices_from_parameters(result.parameters)),
            states=np.asarray(solution.states),
            strains=np.asarray(problem.strains),
            bottom_indices=np.asarray(study.bottom),
            bottom_forces=np.asarray(forces),
            secant_stiffness=np.asarray(stiffness),
            target_stiffness=np.asarray(study.target_stiffness),
            loss_history=result.loss_history,
            residuals=np.asarray(solution.residuals),
            constraint_margins=result.constraint_margins,
        )
    figure = plot_results(problem, study, results)
    figure.savefig(args.output / f"optimization_objectives_{args.objective}.png", dpi=160)
    print(f"Saved comparisons and numeric results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
