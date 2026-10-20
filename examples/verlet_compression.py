"""Compress a finite-hinge grid using damped velocity-Verlet dynamics."""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from rotating_polygons import (
    PotentialParameters,
    VerletOptions,
    compression_boundary,
    initialize,
    make_potential,
    make_verlet_solver,
    square_grid,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    import matplotlib.pyplot as plt

    from rotating_polygons.plotting import plot_geometry

    geometry = square_grid(3, 3)
    boundary = compression_boundary(geometry)
    reference = initialize(geometry)
    potential = make_potential(
        geometry, PotentialParameters(stretching=1, shearing=0.1, bending=0.02)
    )
    # Effective masses/inertias in consistent nondimensional units. The
    # timestep must be checked again when geometry or stiffnesses change.
    options = VerletOptions(time_step=0.002, damping=0.5)
    simulate = make_verlet_solver(
        geometry, boundary, potential, masses=1.0, inertias=0.1, options=options
    )
    times = jnp.arange(3001) * options.time_step
    ramp = jnp.minimum(times / 3.0, 1.0)
    strains = 0.04 * (3 * ramp**2 - 2 * ramp**3)  # smooth ramp, then hold
    result = simulate(reference, strains)
    if not np.all(np.isfinite(result.states)):
        raise RuntimeError("Nonfinite dynamic state; reduce the timestep")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    plot_geometry(geometry, result.states[-1], ax=axes[0])
    axes[0].set_title("Damped compression")
    axes[1].plot(result.times, result.potential_energies, label="Potential")
    axes[1].plot(result.times, result.kinetic_energies, label="Kinetic")
    axes[1].set(xlabel="Time", ylabel="Energy")
    axes[1].legend(frameon=False)
    args.output.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output / "verlet_compression.png", dpi=160)
    np.savez_compressed(
        args.output / "verlet_compression.npz",
        times=np.asarray(result.times),
        states=np.asarray(result.states),
        velocities=np.asarray(result.velocities),
        potential_energies=np.asarray(result.potential_energies),
        kinetic_energies=np.asarray(result.kinetic_energies),
    )
    print(f"Integrated {len(times) - 1} steps; saved results to {args.output.resolve()}")


if __name__ == "__main__":
    main()
