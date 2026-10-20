"""Optimize every polygon vertex independently to match a stress–strain curve.

The default 3x3 grid has 9 * 4 * 2 = 72 independent shape variables, including
vertices on clamped polygons and outline vertices with no connecting ligament.
The nine free mechanical coordinates solved at each strain are a separate set.

Run after ``pip install -e '.[plot]'``:
    python examples/optimize_all_vertices.py --output outputs
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
    world_vertices,
)


def vertices_from_offsets(geometry, offsets):
    """Map ``(n_polygons, 4, 2)`` independent offsets to all local vertices.

    Vertex order within each polygon is right, up, left, down. Coordinate order
    is x, y. No tiling, tying, boundary mask, or averaging is applied.
    """
    offsets = jnp.asarray(offsets)
    expected_shape = (geometry.n_polygons, 4, 2)
    if offsets.shape != expected_shape:
        raise ValueError(f"offsets must have shape {expected_shape}")
    return geometry.vertices + offsets.reshape((-1, 2))


def build_problem(nx=3, ny=3, regularization_weight=1e-4):
    """Build a full-vertex design problem without running the outer optimizer."""
    if not np.isfinite(regularization_weight) or regularization_weight < 0:
        raise ValueError("regularization_weight must be finite and nonnegative")
    geometry = square_grid(nx, ny, spacing=1.0, hinge_gap=0.2)
    boundary = compression_boundary(geometry)
    parameter_shape = (geometry.n_polygons, 4, 2)
    offset_bound = 0.06  # Both coordinates of EVERY vertex get this bound.

    def vertices_from_parameters(parameters):
        return vertices_from_offsets(geometry, parameters)

    def initialize_design(parameters):
        # Rest lengths and angles belong to the current candidate, so recompute
        # them inside the differentiable objective before solving equilibrium.
        return initialize(geometry, vertices_from_parameters(parameters))

    coefficients = PotentialParameters(stretching=1.0, shearing=0.1, bending=0.02, contact=40.0)
    potential = make_potential(geometry, coefficients, model="finite_hinge")
    options = QuasistaticOptions(method="bfgs", tolerance=1e-9, max_iterations=750)
    simulate = make_quasistatic_solver(geometry, boundary, potential, options)
    strains = jnp.linspace(0.0, 0.04, 7)
    measure_stress = make_stress_measurement(potential, boundary, thickness=1.0)

    # Different seeds give independent, reproducible perturbations at all 72
    # entries. The optimizer receives the initial tensor, never the target one.
    initial_rng = np.random.default_rng(2027)
    target_rng = np.random.default_rng(2026)
    initial_parameters = jnp.asarray(initial_rng.uniform(-0.003, 0.003, parameter_shape))
    target_parameters = jnp.asarray(target_rng.uniform(-0.018, 0.018, parameter_shape))
    target_parameters = target_parameters.at[:, 1, 0].add(0.03)
    target_reference = initialize_design(target_parameters)
    target_solution = simulate(target_reference, strains).check_convergence()
    target_stress = measure_stress(target_solution.states, target_reference)
    curve_loss = stress_curve_loss(
        simulate, initialize_design, measure_stress, strains, target_stress
    )

    def loss(parameters):
        # Many different shapes can match seven stress samples. The optional
        # dimensionless penalty favors smaller displacements from the square
        # template, including passive outline corners, without freezing them.
        regularization = regularization_weight * jnp.mean((parameters / offset_bound) ** 2)
        return curve_loss(parameters) + regularization

    # SLSQP consumes one bound per flattened scalar; result.parameters restores
    # the full (n_polygons, 4, 2) tensor. Clamped polygons remain shape variables.
    bounds = [(-offset_bound, offset_bound)] * int(np.prod(parameter_shape))
    constraint = make_shape_constraint(
        geometry,
        vertices_from_parameters,
        min_edge_length=0.2,
        min_corner_sine=0.25,
        min_ligament_length=0.08,
    )
    if np.any(np.asarray(constraint(target_parameters)) < 0):
        raise RuntimeError("The synthetic target must satisfy the design restrictions")
    return SimpleNamespace(
        geometry=geometry,
        boundary=boundary,
        initial_parameters=initial_parameters,
        target_parameters=target_parameters,
        parameter_shape=parameter_shape,
        vertices_from_parameters=vertices_from_parameters,
        initialize_design=initialize_design,
        potential=potential,
        simulate=simulate,
        strains=strains,
        target_stress=target_stress,
        measure_stress=measure_stress,
        curve_loss=curve_loss,
        loss=loss,
        constraint=constraint,
        bounds=bounds,
        regularization_weight=regularization_weight,
        offset_bound=offset_bound,
    )


def optimize(problem, max_iterations=100):
    """Fit all independent offsets and verify both physics and feasibility."""
    initial_curve_loss = float(problem.curve_loss(problem.initial_parameters))
    result = minimize_design(
        problem.loss,
        problem.initial_parameters,
        bounds=problem.bounds,
        constraints=[problem.constraint],
        maxiter=max_iterations,
        tol=1e-10,
    )
    if not result.success or not result.feasible:
        raise RuntimeError(f"All-vertex optimization failed: {result.message}")
    # A small total cost alone is insufficient: check the actual curve error
    # separately from the regularization term.
    result.initial_curve_loss = initial_curve_loss
    result.final_curve_loss = float(problem.curve_loss(result.parameters))
    if not result.fun < result.initial_loss or not result.final_curve_loss < initial_curve_loss:
        raise RuntimeError("Optimization did not improve both the total cost and curve error")
    reference = problem.initialize_design(result.parameters)
    problem.simulate(reference, problem.strains).check_convergence()
    if np.any(np.asarray(problem.constraint(result.parameters)) < -1e-10):
        raise RuntimeError("The final design violates a geometry restriction")
    return result


def plot_result(problem, result):
    """Compare stress curves and shapes, separating design from deformation."""
    import matplotlib.pyplot as plt

    from rotating_polygons.plotting import plot_geometry

    initial_reference = problem.initialize_design(problem.initial_parameters)
    optimized_reference = problem.initialize_design(result.parameters)
    initial = problem.simulate(initial_reference, problem.strains).check_convergence()
    optimized = problem.simulate(optimized_reference, problem.strains).check_convergence()
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), layout="constrained")
    axes[0].plot(problem.strains, problem.target_stress, "ko", label="Target", markersize=5)
    axes[0].plot(
        problem.strains,
        problem.measure_stress(initial.states, initial_reference),
        "--",
        label="Initial",
    )
    axes[0].plot(
        problem.strains,
        problem.measure_stress(optimized.states, optimized_reference),
        "-",
        label="Optimized",
    )
    axes[0].set(
        xlabel="Compression strain",
        ylabel="Nominal stress",
        title=f"{problem.initial_parameters.size}-variable fit",
    )
    axes[0].legend(frameon=False)
    # Both design panels use IDENTICAL centers, angles, and axis limits. Their
    # differences therefore come from the optimized vertices, not compression.
    for ax, reference, title in (
        (axes[1], initial_reference, "Initial reference shape"),
        (axes[2], optimized_reference, "Optimized reference shape"),
    ):
        plot_geometry(
            problem.geometry, problem.geometry.reference_state, vertices=reference.vertices, ax=ax
        )
        ax.set_title(title)
    reference_points = np.concatenate(
        [
            np.asarray(
                world_vertices(
                    problem.geometry, problem.geometry.reference_state, reference.vertices
                )
            )
            for reference in (initial_reference, optimized_reference)
        ]
    )
    minimum = reference_points.min(axis=0) - 0.1
    maximum = reference_points.max(axis=0) + 0.1
    for ax in axes[1:3]:
        ax.set_xlim(minimum[0], maximum[0])
        ax.set_ylim(minimum[1], maximum[1])
    plot_geometry(
        problem.geometry, optimized.states[-1], vertices=optimized_reference.vertices, ax=axes[3]
    )
    axes[3].set_title("Optimized at 4% compression")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--regularization-weight", type=float, default=1e-4)
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    problem = build_problem(regularization_weight=args.regularization_weight)
    print(
        f"Independent shape variables: {problem.initial_parameters.size}; "
        f"free mechanical coordinates: {problem.boundary.free_indices.size}",
        flush=True,
    )
    result = optimize(problem, args.max_iterations)
    reference = problem.initialize_design(result.parameters)
    solution = problem.simulate(reference, problem.strains).check_convergence()
    print(f"Total cost: {result.initial_loss:.8g} -> {result.fun:.8g}")
    print(f"Curve MSE: {result.initial_curve_loss:.8g} -> {result.final_curve_loss:.8g}")
    print(
        f"Iterations: {result.nit}; max constraint violation: {result.max_constraint_violation:.3g}"
    )
    print(f"Max equilibrium residual: {float(jnp.max(solution.residuals)):.3g}")
    args.output.mkdir(parents=True, exist_ok=True)
    figure = plot_result(problem, result)
    figure.savefig(args.output / "optimized_all_vertices.png", dpi=160)
    np.savez_compressed(
        args.output / "optimized_all_vertices.npz",
        parameters=np.asarray(result.parameters),  # Full tensor, not tied coefficients.
        initial_parameters=np.asarray(problem.initial_parameters),
        target_parameters=np.asarray(problem.target_parameters),
        vertices=np.asarray(reference.vertices),
        initial_vertices=np.asarray(problem.vertices_from_parameters(problem.initial_parameters)),
        states=np.asarray(solution.states),
        strains=np.asarray(problem.strains),
        target_stress=np.asarray(problem.target_stress),
        optimized_stress=np.asarray(problem.measure_stress(solution.states, reference)),
        residuals=np.asarray(solution.residuals),
        converged=np.asarray(solution.converged),
        constraint_margins=result.constraint_margins,
        loss_history=result.loss_history,
        initial_curve_loss=result.initial_curve_loss,
        final_curve_loss=result.final_curve_loss,
        regularization_weight=problem.regularization_weight,
    )
    print(f"Saved all-vertex design and diagnostics to {args.output.resolve()}")


if __name__ == "__main__":
    main()
