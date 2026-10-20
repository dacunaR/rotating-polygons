"""Fit a small grid's nominal stress curve by changing polygon vertices.

Run after ``pip install -e '.[plot]'``:
    python examples/optimize_square_grid.py --output outputs

The target is synthetic and attainable, generated from another admissible
shape. Replace the target with your measured/desired curve for a design task.
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from rotating_polygons import (
    PotentialParameters,
    QuasistaticOptions,
    compression_boundary,
    initialize,
    make_potential,
    make_quasistatic_solver,
    make_shape_constraint,
    make_stress_measurement,
    minimize_design,
    square_grid,
    stress_curve_loss,
)


def build_problem():
    """Create each interchangeable part of the example without optimizing."""
    # 1. Geometry, then prescribed degrees of freedom.
    # Centers are one length unit apart; a 0.2 gap leaves a finite ligament
    # between neighboring squares. The constructor makes the connectivity.
    geometry = square_grid(3, 3, spacing=1.0, hinge_gap=0.2)
    # Bottom and top polygon rows are clamped in x, y, and angle. Positive
    # strain moves the top down; the remaining polygons relax freely.
    boundary = compression_boundary(geometry)

    # 2. A repeated two-parameter shape family. The center and connectivity
    # stay fixed; moving the upper/right vertices changes each square's shape.
    def vertices_from_parameters(parameters):
        # These are the only two optimizer variables. Each row below is a
        # local vertex offset: right, up, left, down, in that order.
        offsets = jnp.zeros((4, 2))
        offsets = offsets.at[1, 0].set(parameters[0])  # upper vertex, horizontal
        offsets = offsets.at[0, 1].set(parameters[1])  # right vertex, vertical
        return geometry.vertices + jnp.tile(offsets, (geometry.n_polygons, 1))

    def initialize_design(parameters):
        # Recompute rest lengths and reference angles for EVERY candidate.
        # Reusing the old reference would introduce unintended prestress.
        return initialize(geometry, vertices_from_parameters(parameters))

    # 3. Material parameters, potential, loading path, and equilibrium method.
    coefficients = PotentialParameters(stretching=1.0, shearing=0.1, bending=0.02, contact=40.0)
    # These coefficients stay fixed during this shape optimization.
    potential = make_potential(geometry, coefficients, model="finite_hinge")
    # This INNER optimizer finds mechanical equilibrium, not a better shape.
    # Its tolerance is the norm of the residual on free degrees of freedom.
    options = QuasistaticOptions(method="bfgs", tolerance=1e-9, max_iterations=500)
    simulate = make_quasistatic_solver(geometry, boundary, potential, options)
    # Absolute positive compression strains, including the relaxed initial
    # state. Each solve starts from the previous equilibrium.
    strains = jnp.linspace(0.0, 0.04, 7)
    # Nominal stress is compressive grip force / (reference width * thickness).
    measure_stress = make_stress_measurement(potential, boundary, thickness=1.0)

    # 4. Reproducible target. Its parameters are never passed to the optimizer.
    target_reference = initialize_design(jnp.array([0.10, 0.07]))
    target_solution = simulate(target_reference, strains).check_convergence()
    target_stress = measure_stress(target_solution.states, target_reference)
    # Replace target_stress with experimental values on the SAME strain grid.
    # The helper divides errors by max(abs(target_stress)) before squaring,
    # giving a dimensionless mean-squared error.
    loss = stress_curve_loss(simulate, initialize_design, measure_stress, strains, target_stress)

    # 5. Bounds and manufacturing restrictions. Small deterministic offsets
    # avoid an exactly symmetric starting point with zero shape sensitivity.
    initial_parameters = jnp.array([0.025, -0.02])
    # One bound pair per scalar design variable, in geometry length units.
    bounds = [(-0.12, 0.12), (-0.12, 0.12)]
    # Each returned constraint margin must be >= 0. These apply to the
    # reference shape, not to collisions throughout its deformed trajectory.
    constraint = make_shape_constraint(
        geometry,
        vertices_from_parameters,
        min_edge_length=0.2,
        min_corner_sine=0.25,
        min_ligament_length=0.08,
    )
    return SimpleNamespace(
        geometry=geometry,
        boundary=boundary,
        potential=potential,
        vertices_from_parameters=vertices_from_parameters,
        initialize_design=initialize_design,
        simulate=simulate,
        strains=strains,
        measure_stress=measure_stress,
        target_stress=target_stress,
        initial_parameters=initial_parameters,
        bounds=bounds,
        constraint=constraint,
        loss=loss,
    )


def optimize(problem, max_iterations=40):
    """Run the outer shape optimizer and return its full diagnostic result."""
    # This OUTER SLSQP optimizer changes the two shape parameters. JAX
    # differentiates the objective through initialization and equilibrium.
    # maxiter limits design iterations; tol controls SLSQP termination and
    # final constraint feasibility, independently of the equilibrium tolerance.
    result = minimize_design(
        problem.loss,
        problem.initial_parameters,
        bounds=problem.bounds,
        constraints=[problem.constraint],
        maxiter=max_iterations,
        tol=1e-9,
    )
    if not result.success:
        # Keep failures visible instead of presenting the last iterate as an
        # optimized design. The result carries the solver's diagnostic message.
        raise RuntimeError(f"Design optimization did not converge: {result.message}")
    if result.fun >= result.initial_loss:
        raise RuntimeError("The example did not improve its initial stress-curve loss")
    return result


def plot_result(problem, result):
    """Plot curves and undeformed/deformed designs; return the figure."""
    import matplotlib.pyplot as plt

    from rotating_polygons.plotting import plot_geometry

    initial_reference = problem.initialize_design(problem.initial_parameters)
    optimized_reference = problem.initialize_design(result.parameters)
    initial = problem.simulate(initial_reference, problem.strains).check_convergence()
    optimized = problem.simulate(optimized_reference, problem.strains).check_convergence()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout="constrained")
    axes[0].plot(problem.strains, problem.target_stress, "ko", label="Target", markersize=5)
    axes[0].plot(
        problem.strains,
        problem.measure_stress(initial.states, initial_reference),
        "--",
        color="#bc6944",
        label="Initial",
    )
    axes[0].plot(
        problem.strains,
        problem.measure_stress(optimized.states, optimized_reference),
        color="#147d92",
        label="Optimized",
    )
    axes[0].set(xlabel="Compression strain", ylabel="Nominal stress", title="Shape optimization")
    axes[0].legend(frameon=False)
    plot_geometry(problem.geometry, vertices=initial_reference.vertices, ax=axes[1])
    axes[1].set_title("Initial design")
    plot_geometry(
        problem.geometry, optimized.states[-1], vertices=optimized_reference.vertices, ax=axes[2]
    )
    axes[2].set_title("Optimized, at 4% compression")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--max-iterations", type=int, default=40)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    problem = build_problem()
    result = optimize(problem, args.max_iterations)
    print(f"Normalized MSE: {result.initial_loss:.6g} -> {result.fun:.6g}")
    print(f"Shape parameters: {np.asarray(result.parameters)}")
    print(f"Maximum constraint violation: {result.max_constraint_violation:.3g}")
    args.output.mkdir(parents=True, exist_ok=True)
    figure = plot_result(problem, result)
    figure.savefig(args.output / "optimized_square_grid.png", dpi=160)
    optimized_reference = problem.initialize_design(result.parameters)
    solution = problem.simulate(optimized_reference, problem.strains).check_convergence()
    np.savez_compressed(
        args.output / "optimized_square_grid.npz",
        strains=np.asarray(problem.strains),
        target_stress=np.asarray(problem.target_stress),
        optimized_stress=np.asarray(problem.measure_stress(solution.states, optimized_reference)),
        parameters=np.asarray(result.parameters),
        vertices=np.asarray(optimized_reference.vertices),
        states=np.asarray(solution.states),
        loss_history=result.loss_history,
    )
    print(f"Saved figure and numeric results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
