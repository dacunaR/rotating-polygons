"""Optimize spring constants while keeping polygon shapes fixed.

Run either parameterization, or both:
    python examples/optimize_spring_constants.py --parameterization all --output outputs

``global`` fits one axial stiffness shared by every ligament.
``columns`` fits three axial stiffnesses, one per column of vertical ligaments;
the horizontal ligaments keep their original stiffness.
``bending`` fits one rotational spring constant while fixing axial stiffness.

The target consists of individual bottom-support forces at several strains.
Using several support measurements gives more information about stiffness
distribution than fitting only their sum. Synthetic target values make this a
reproducible tutorial; substitute measured forces for a calibration problem.
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from rotating_polygons import (
    MechanicalParameters,
    QuasistaticOptions,
    compression_boundary,
    initialize,
    make_parameterized_potential,
    make_quasistatic_solver,
    minimize_design,
    reaction_forces,
    square_grid,
)


def build_problem(parameterization="columns"):
    """Assemble the model, variable-to-material map, observations, and loss."""
    if parameterization not in ("global", "columns", "bending"):
        raise ValueError("parameterization must be 'global', 'columns', or 'bending'")

    geometry = square_grid(3, 3, hinge_gap=0.2)  # Nine polygons; 12 undirected hinges.
    boundary = compression_boundary(geometry)  # Bottom grip fixed; top grip driven down.
    offsets = jnp.zeros((4, 2)).at[1, 0].set(0.08).at[0, 1].set(0.04)
    vertices = geometry.vertices + jnp.tile(offsets, (geometry.n_polygons, 1))
    reference = initialize(geometry, vertices)  # This shape remains fixed during optimization.

    owners = geometry.vertex_polygons[geometry.bonds]  # Each bond's two polygon indices.
    centers = geometry.reference_state[owners, :2]  # Shape (n_bonds, 2 endpoints, 2 coordinates).
    vertical = jnp.abs(centers[:, 0, 1] - centers[:, 1, 1]) > 0.5  # Select vertical bonds.
    column_ids = jnp.rint(centers[:, 0, 0]).astype(int)  # x=0,1,2 identifies each column.
    bottom_ids = np.flatnonzero(np.isclose(np.asarray(geometry.reference_state[:, 1]), 0.0))

    def material_from_parameters(log_stiffness):
        """Turn unconstrained logarithms into physical positive stiffnesses."""
        stiffness = jnp.exp(log_stiffness)  # k=exp(z) guarantees k>0 for every trial.
        if parameterization == "global":
            axial_stiffness = stiffness[0]  # A scalar broadcasts to all 12 bonds.
        elif parameterization == "columns":
            # Indexing repeats each column variable on its two vertical bonds.
            # Horizontal bonds retain axial stiffness 1.0 and are not optimized.
            axial_stiffness = jnp.where(vertical, stiffness[column_ids], 1.0)
        else:
            axial_stiffness = 1.0  # Fix axial stiffness when the design variable is bending.
        bending_stiffness = stiffness[0] if parameterization == "bending" else 0.02
        return MechanicalParameters(
            reference=reference,  # Rest geometry is carried along with candidate coefficients.
            stretching=axial_stiffness,  # Scalar or per-bond axial constants.
            shearing=0.1,  # Fixed shear stiffness: omit it from the optimization vector.
            bending=bending_stiffness,  # Only active in the "bending" parameterization.
            contact=40.0,  # Fixed angular-contact penalty.
        )

    # This potential reads coefficients from MechanicalParameters at each call.
    # Construct the solver once; never rebuild it around traced coefficients.
    potential = make_parameterized_potential(geometry, model="finite_hinge")
    options = QuasistaticOptions(method="bfgs", tolerance=1e-9, max_iterations=500)
    simulate = make_quasistatic_solver(geometry, boundary, potential, options)
    strains = jnp.linspace(0.0, 0.03, 5)  # Absolute compression strains, including the rest state.

    def support_forces(states, material):
        reactions = reaction_forces(potential, states, material)  # dE/dq, not internal -dE/dq.
        return reactions[:, bottom_ids, 1]  # Vertical force on each bottom support, at each step.

    targets = {"global": [1.35], "columns": [0.7, 1.1, 1.5], "bending": [0.035]}
    target_constants = jnp.array(targets[parameterization])
    target_material = material_from_parameters(jnp.log(target_constants))
    target_solution = simulate(target_material, strains).check_convergence()
    target_forces = support_forces(target_solution.states, target_material)
    force_scale = float(jnp.max(jnp.abs(target_forces)))  # Fix a force scale before optimization.
    initial_constant = 0.02 if parameterization == "bending" else 1.0
    initial_parameters = jnp.full_like(target_constants, jnp.log(initial_constant))

    def loss(log_stiffness):
        material = material_from_parameters(log_stiffness)  # Candidate coefficients are explicit.
        solution = simulate(material, strains)  # Equilibria depend on those coefficients.
        predicted = support_forces(solution.states, material)  # Use the SAME material to measure.
        error = jnp.mean(((predicted - target_forces) / force_scale) ** 2)
        return jnp.where(jnp.all(solution.converged), error, jnp.inf)  # Reject failed equilibria.

    # Bounds live in the optimizer's coordinates z=log(k), not in physical k.
    lower, upper = (0.005, 0.1) if parameterization == "bending" else (0.4, 2.0)
    bounds = [(float(np.log(lower)), float(np.log(upper)))] * initial_parameters.size
    return SimpleNamespace(
        parameterization=parameterization,
        geometry=geometry,
        boundary=boundary,
        reference=reference,
        vertical=vertical,
        column_ids=column_ids,
        bottom_ids=bottom_ids,
        material_from_parameters=material_from_parameters,
        potential=potential,
        simulate=simulate,
        strains=strains,
        support_forces=support_forces,
        target_constants=target_constants,
        target_forces=target_forces,
        initial_parameters=initial_parameters,
        bounds=bounds,
        loss=loss,
    )


def optimize(problem):
    """Optimize logarithmic stiffnesses using exact derivatives through equilibrium."""
    result = minimize_design(
        problem.loss,  # A scalar objective; minimize_design differentiates it with JAX.
        problem.initial_parameters,  # Only these entries are changed by the optimizer.
        bounds=problem.bounds,  # Bounds on log(k), chosen for the active stiffness type.
        maxiter=60,  # Outer design iterations; separate from equilibrium max_iterations.
        tol=1e-10,  # Outer objective/optimality tolerance, not an equilibrium-force tolerance.
    )
    if not result.success:
        raise RuntimeError(f"Stiffness optimization failed: {result.message}")
    if result.fun >= result.initial_loss:
        raise RuntimeError("The stiffness fit did not improve its starting objective")
    return result


def plot_result(problem, result):
    """Show fitted physical constants and individual support-force histories."""
    import matplotlib.pyplot as plt

    material = problem.material_from_parameters(result.parameters)
    solution = problem.simulate(material, problem.strains).check_convergence()
    fitted_forces = problem.support_forces(solution.states, material)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    indices = np.arange(problem.initial_parameters.size)
    axes[0].bar(indices - 0.24, np.exp(problem.initial_parameters), 0.24, label="Initial")
    axes[0].bar(indices, problem.target_constants, 0.24, label="Target")
    axes[0].bar(indices + 0.24, np.exp(result.parameters), 0.24, label="Fitted")
    coefficient_name = "Bending" if problem.parameterization == "bending" else "Axial"
    axes[0].set(
        xlabel="Stiffness group", ylabel=f"{coefficient_name} spring constant", xticks=indices
    )
    axes[0].legend(frameon=False)
    for column in range(len(problem.bottom_ids)):
        axes[1].plot(
            problem.strains,
            problem.target_forces[:, column],
            "o",
            color=f"C{column}",
            label=f"Support {column} target",
        )
        axes[1].plot(
            problem.strains,
            fitted_forces[:, column],
            "-",
            color=f"C{column}",
            label=f"Support {column} fitted",
        )
    axes[1].set(xlabel="Compression strain", ylabel="Vertical support force")
    axes[1].legend(frameon=False, fontsize=8)
    return fig, solution, fitted_forces


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parameterization", choices=["global", "columns", "bending", "all"], default="all"
    )
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)  # Enable precision before creating model arrays.
    args.output.mkdir(parents=True, exist_ok=True)
    modes = (
        ["global", "columns", "bending"]
        if args.parameterization == "all"
        else [args.parameterization]
    )
    for mode in modes:
        problem = build_problem(mode)
        result = optimize(problem)
        figure, solution, forces = plot_result(problem, result)
        figure.savefig(args.output / f"spring_constants_{mode}.png", dpi=160)
        np.savez_compressed(
            args.output / f"spring_constants_{mode}.npz",
            stiffness=np.exp(np.asarray(result.parameters)),
            target_constants=np.asarray(problem.target_constants),
            strains=np.asarray(problem.strains),
            target_forces=np.asarray(problem.target_forces),
            fitted_forces=np.asarray(forces),
            states=np.asarray(solution.states),
            loss_history=result.loss_history,
        )
        print(f"{mode}: normalized force MSE {result.initial_loss:.6g} -> {result.fun:.6g}")
        print(f"  Fitted k: {np.exp(np.asarray(result.parameters))}")


if __name__ == "__main__":
    main()
