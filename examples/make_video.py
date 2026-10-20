"""Simulate a small compressed grid and export its deformation as a GIF or MP4.

Run after installing the package's video extra:
    python examples/make_video.py --output outputs/compression.gif
    python examples/make_video.py --output outputs/compression.mp4

MP4 requires ffmpeg. If it is unavailable, the default behavior writes a GIF
instead and reports its actual filename. Both formats use the exact simulated
states; no intermediate mechanical configurations are invented.
"""

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp

from rotating_polygons import (
    PotentialParameters,
    QuasistaticOptions,
    compression_boundary,
    initialize,
    make_potential,
    make_quasistatic_solver,
    square_grid,
)
from rotating_polygons.animation import save_video


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/compression.gif"))
    parser.add_argument("--fps", type=float, default=8, help="Playback frames per second.")
    parser.add_argument("--stride", type=int, default=1, help="Use every nth simulated state.")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)

    geometry = square_grid(3, 3, spacing=1.0, hinge_gap=0.2)
    boundary = compression_boundary(geometry)
    # A small shape asymmetry makes the rotational response visible. Passing
    # these vertices to the exporter is essential: they differ from the grid.
    offsets = jnp.zeros((4, 2)).at[1, 0].set(0.035).at[0, 1].set(0.02)
    vertices = geometry.vertices + jnp.tile(offsets, (geometry.n_polygons, 1))
    reference = initialize(geometry, vertices)
    potential = make_potential(
        geometry,
        PotentialParameters(stretching=1.0, shearing=0.1, bending=0.02, contact=40.0),
    )
    simulate = make_quasistatic_solver(
        geometry,
        boundary,
        potential,
        QuasistaticOptions(method="bfgs", tolerance=1e-9, max_iterations=700),
    )
    strains = jnp.linspace(0.0, 0.06, 17)
    solution = simulate(reference, strains).check_convergence()
    # Quasistatic continuation has no physical time. These labels display the
    # actual loading coordinate, while fps determines how quickly viewers see it.
    labels = [f"Compression strain = {float(strain):.3f}" for strain in strains]
    destination = save_video(
        geometry,
        solution.states,
        args.output,
        vertices=reference.vertices,
        frame_labels=labels,
        fps=args.fps,
        stride=args.stride,
        dpi=100,
    )
    print(f"Converged {len(strains)} equilibrium states.")
    print(f"Saved {len(range(0, len(strains), args.stride))} frames to {destination.resolve()}")


if __name__ == "__main__":
    main()
