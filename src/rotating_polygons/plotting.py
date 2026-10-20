"""Optional Matplotlib views; never mutate a geometry or simulation state."""

import numpy as np

from .geometry import world_vertices


def plot_geometry(
    geometry, state=None, *, vertices=None, ax=None, facecolor="#74b9c9", show_bonds=True
):
    """Draw polygon outlines and ligaments, returning the Matplotlib axes."""
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection

    if ax is None:
        _, ax = plt.subplots()
    state = geometry.reference_state if state is None else state
    points = np.asarray(world_vertices(geometry, state, vertices))
    outlines = [
        points[start:stop]
        for start, stop in zip(geometry.polygon_offsets[:-1], geometry.polygon_offsets[1:])
    ]
    ax.add_collection(
        PolyCollection(
            outlines, facecolors=facecolor, edgecolors="#163743", linewidths=1.0, alpha=0.85
        )
    )
    if show_bonds:
        bonds = points[np.asarray(geometry.bonds)].copy()
        # Use the same image convention as the potential.
        bonds[:, 1] -= np.asarray(geometry.image_shifts)
        ax.add_collection(LineCollection(bonds, colors="#d65e46", linewidths=1.5))
    ax.autoscale_view()
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return ax
