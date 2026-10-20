"""Compare prescribed degrees of freedom and compression/shear loading.

Install the package with plotting support, then run from the repository root:
    python examples/boundary_conditions.py --output outputs/boundaries

The accompanying 02_boundary_conditions.ipynb constructs the same boundaries
step by step. This script saves two figures and the complete numerical results.
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from rotating_polygons import (
    BoundaryConditions,
    PotentialParameters,
    QuasistaticOptions,
    compression_boundary,
    initialize,
    make_potential,
    make_quasistatic_solver,
    make_stress_measurement,
    square_grid,
)

CASE_LABELS = {
    "clamped": "Clamped compression",
    "sliding": "Sliding grips, free rotation",
    "center_angle": "Sliding + polygon 4 angle prescribed",
    "shear": "Simple shear",
}
CASE_COLORS = {
    "clamped": "#207b95",
    "sliding": "#d07837",
    "center_angle": "#8c58a3",
    "shear": "#3d8f65",
}


def build_geometry():
    """Create a slightly skewed 3 × 3 grid and its initialized reference.

    A small repeated vertex offset avoids an exactly symmetric stationary
    branch. Centers and bond topology remain those of the regular square grid.
    """
    geometry = square_grid(3, 3, spacing=1.0, hinge_gap=0.2)
    # Vertex order within each polygon is right, up, left, down.
    local_offsets = jnp.zeros((4, 2))
    local_offsets = local_offsets.at[1, 0].set(0.035)  # upper corner: shift right
    local_offsets = local_offsets.at[0, 1].set(0.020)  # right corner: shift up
    vertices = geometry.vertices + jnp.tile(local_offsets, (geometry.n_polygons, 1))
    # This defines the stress-free finite-ligament rest data for the skewed shape.
    reference = initialize(geometry, vertices)
    return geometry, reference


def make_boundaries(geometry, prescribed_center_angle=0.015):
    """Construct four independent choices of prescribed coordinates."""
    # q has shape (9, 3): q[polygon_id, 0/1/2] means x/y/angle in radians.
    clamped = compression_boundary(geometry)
    sliding = compression_boundary(
        geometry,
        clamp_horizontal=False,  # release grip x positions except one lower x pin
        clamp_rotations=False,  # all grip angles are solved by energy minimization
    )

    # Make a copy: changing this case must not change the sliding case.
    fixed = np.array(sliding.fixed_mask, copy=True)
    fixed[4, 2] = True  # prescribe ONLY polygon 4's angle; its x and y remain free
    target_state = np.array(sliding.reference_state, copy=True)
    target_state[4, 2] = prescribed_center_angle
    center_angle = BoundaryConditions(
        reference_state=target_state,  # boundary target; not the elastic rest state
        fixed_mask=fixed,
        strain_direction=sliding.strain_direction,  # keep the same compression drive
        height=sliding.height,
        width=sliding.width,
    )
    # A nonzero target angle can create reactions even at zero applied strain:
    # initialize(geometry, vertices) still describes the original rest angles.

    state = np.asarray(geometry.reference_state)
    bottom = np.isclose(state[:, 1], state[:, 1].min())  # IDs 0, 1, 2
    top = np.isclose(state[:, 1], state[:, 1].max())  # IDs 6, 7, 8
    fixed = np.zeros_like(state, dtype=bool)
    fixed[bottom, :] = True  # bottom x, y, and angle stay at their target values
    fixed[top, :] = True  # top y and angle stay fixed; top x will be prescribed
    shear_direction = np.zeros_like(state)
    shear_direction[top, 0] = clamped.height  # x_top(gamma) = x_top(0) + gamma * H
    shear = BoundaryConditions(
        reference_state=state,
        fixed_mask=fixed,
        strain_direction=shear_direction,
        height=clamped.height,
        width=clamped.width,
    )
    return {
        "clamped": clamped,
        "sliding": sliding,
        "center_angle": center_angle,
        "shear": shear,
    }


def assert_prescribed_coordinates(boundary, result, loading_values):
    """Verify every prescribed coordinate at every loading step."""
    expected = np.asarray(jax.vmap(boundary.state_at)(jnp.asarray(loading_values)))
    mask = np.asarray(boundary.fixed_mask)
    # Boolean indexing selects x/y/angle entries marked True for every step.
    np.testing.assert_allclose(
        np.asarray(result.states)[:, mask], expected[:, mask], atol=1e-12, rtol=0
    )


def run_cases(geometry, reference, boundaries, loading_values):
    """Solve each boundary case and measure its work-conjugate stress."""
    material = PotentialParameters(stretching=1.0, shearing=0.1, bending=0.02, contact=40.0)
    potential = make_potential(geometry, material, model="finite_hinge")
    options = QuasistaticOptions(method="bfgs", tolerance=1e-9, max_iterations=700)
    cases = {}
    for name, boundary in boundaries.items():
        simulate = make_quasistatic_solver(geometry, boundary, potential, options)
        # No failed equilibrium is silently included in a plotted curve.
        solution = simulate(reference, loading_values).check_convergence()
        assert_prescribed_coordinates(boundary, solution, loading_values)
        measure = make_stress_measurement(potential, boundary, thickness=1.0)
        # The shear case measures dE/dgamma / (H * W * t); the other three
        # measure dE/depsilon / (H * W * t), with positive compression epsilon.
        stress = measure(solution.states, reference)
        cases[name] = SimpleNamespace(boundary=boundary, solution=solution, stress=stress)
    return cases


def label_polygons(ax, state):
    """Put row-major polygon IDs at their centers."""
    for polygon_id, (x, y, _) in enumerate(np.asarray(state)):
        ax.text(
            x,
            y,
            str(polygon_id),
            ha="center",
            va="center",
            color="#172b35",
            fontsize=11,
            fontweight="bold",
        )


def plot_masks(geometry, reference, boundaries):
    """Display the geometry IDs and which individual coordinates are driven."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    from rotating_polygons.plotting import plot_geometry

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(15, 4.3),
        layout="constrained",
        gridspec_kw={"width_ratios": [2.2, 1, 1, 1, 1]},
    )
    plot_geometry(geometry, vertices=reference.vertices, ax=axes[0])
    label_polygons(axes[0], geometry.reference_state)
    axes[0].set_title("Polygon IDs\nx varies fastest")
    colors = ["#ecf0f2", "#7894aa", "#db8f46"]
    for ax, (name, boundary) in zip(axes[1:], boundaries.items()):
        mask = np.asarray(boundary.fixed_mask)
        # 0 = free; 1 = constant prescribed target; 2 = moving prescribed target.
        categories = mask.astype(int)
        categories[np.asarray(boundary.strain_direction) != 0] = 2
        ax.imshow(categories, cmap=ListedColormap(colors), vmin=0, vmax=2, aspect="auto")
        ax.set_xticks([0, 1, 2], ["x", "y", "θ"])
        ax.set_yticks(np.arange(geometry.n_polygons))
        ax.set_ylabel("Polygon ID")
        ax.set_title(
            {
                "clamped": "Clamped",
                "sliding": "Sliding",
                "center_angle": "Pin θ₄",
                "shear": "Shear",
            }[name]
        )
        # The symbol is redundant with color, keeping the mask readable in print.
        for row in range(geometry.n_polygons):
            for column in range(3):
                ax.text(
                    column,
                    row,
                    ["·", "F", "D"][categories[row, column]],
                    ha="center",
                    va="center",
                    color="#172b35",
                    fontsize=10,
                )
    fig.legend(
        handles=[
            Patch(facecolor=colors[0], label="· free"),
            Patch(facecolor=colors[1], label="F prescribed constant"),
            Patch(facecolor=colors[2], label="D prescribed moving"),
        ],
        loc="outside lower center",
        ncols=3,
        frameon=False,
    )
    return fig


def plot_responses(geometry, reference, cases, loading_values):
    """Show compression/shear responses and the effect of the interior angle pin."""
    import matplotlib.pyplot as plt

    from rotating_polygons.plotting import plot_geometry

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    for name in ("clamped", "sliding", "center_angle"):
        case = cases[name]
        axes[0, 0].plot(
            loading_values,
            case.stress,
            "-o",
            markersize=3,
            label=CASE_LABELS[name],
            color=CASE_COLORS[name],
        )
        axes[0, 1].plot(
            loading_values,
            case.solution.states[:, 4, 2],
            "-o",
            markersize=3,
            label=CASE_LABELS[name],
            color=CASE_COLORS[name],
        )
    axes[0, 0].set(
        title="Same shape, different compression supports",
        xlabel="Compression strain ε",
        ylabel="Nominal compression stress",
    )
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].set(
        title="The middle polygon's angular degree of freedom",
        xlabel="Compression strain ε",
        ylabel="Polygon 4 angle θ₄ [rad]",
    )
    axes[0, 1].legend(frameon=False, fontsize=8)
    shear = cases["shear"]
    axes[1, 0].plot(loading_values, shear.stress, "-o", color=CASE_COLORS["shear"])
    axes[1, 0].set(
        title="Upper grip translates horizontally",
        xlabel="Shear strain γ",
        ylabel="Nominal shear stress",
    )
    plot_geometry(geometry, shear.solution.states[-1], vertices=reference.vertices, ax=axes[1, 1])
    label_polygons(axes[1, 1], shear.solution.states[-1])
    axes[1, 1].set_title(f"Final shear configuration, γ = {float(loading_values[-1]):.3f}")
    for ax in axes.flat[:3]:
        ax.grid(alpha=0.2)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/boundaries"))
    args = parser.parse_args()
    # Tight force residuals require 64-bit arithmetic before arrays are created.
    jax.config.update("jax_enable_x64", True)
    geometry, reference = build_geometry()
    boundaries = make_boundaries(geometry)
    loading_values = jnp.linspace(0.0, 0.04, 9)
    cases = run_cases(geometry, reference, boundaries, loading_values)
    args.output.mkdir(parents=True, exist_ok=True)
    plot_masks(geometry, reference, boundaries).savefig(args.output / "boundary_masks.png", dpi=150)
    plot_responses(geometry, reference, cases, loading_values).savefig(
        args.output / "boundary_responses.png", dpi=150
    )
    arrays = {"loading_values": np.asarray(loading_values), "vertices": reference.vertices}
    for name, case in cases.items():
        arrays[f"{name}_states"] = np.asarray(case.solution.states)
        arrays[f"{name}_stress"] = np.asarray(case.stress)
        arrays[f"{name}_residuals"] = np.asarray(case.solution.residuals)
        arrays[f"{name}_fixed_mask"] = np.asarray(case.boundary.fixed_mask)
        arrays[f"{name}_strain_direction"] = np.asarray(case.boundary.strain_direction)
        print(
            f"{CASE_LABELS[name]}: "
            f"{case.boundary.free_indices.size} free coordinates; "
            f"maximum residual {float(jnp.max(case.solution.residuals)):.3g}"
        )
    np.savez_compressed(args.output / "boundary_conditions.npz", **arrays)
    print(f"Saved masks, responses, and numerical results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
