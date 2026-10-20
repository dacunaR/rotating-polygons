"""Draw the setup and independent vertex parameters used by the design example.

Run from the repository root after installing the plotting extra:
    python examples/draw_vertex_design.py --output docs/assets

The diagram is generated directly from the geometry with Matplotlib. Its SVG
remains editable, and the PNG is convenient for a README or notebook.
"""

import argparse
from pathlib import Path

import numpy as np

from rotating_polygons import square_grid, world_vertices


INK = "#203441"
MUTED = "#60737d"
TEAL = "#157f91"
ORANGE = "#d3753e"
BLUE = "#327aac"
PAPER = "#fcfcfa"


def _arrow(ax, start, end, color, *, linewidth=1.5, scale=12, style="-|>", zorder=6):
    """Draw a displacement arrow without shrinking either geometric endpoint."""
    from matplotlib.patches import FancyArrowPatch

    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=scale,
        linewidth=linewidth,
        color=color,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(arrow)
    return arrow


def _draw_specimen(ax, geometry):
    """Show all corners, bonds, mechanical supports, and measurement dimensions."""
    from matplotlib.collections import LineCollection, PolyCollection

    state = np.asarray(geometry.reference_state)
    points = np.asarray(world_vertices(geometry, state))
    bond_points = points[np.asarray(geometry.bonds)]
    gaps = np.linalg.norm(bond_points[:, 0] - bond_points[:, 1], axis=1)
    gap_label = (
        f"gap = {float(gaps[0]):.2g}"
        if len(gaps) and np.allclose(gaps, gaps[0])
        else "gaps vary by bond"
    )
    outlines = [
        points[start:stop]
        for start, stop in zip(geometry.polygon_offsets[:-1], geometry.polygon_offsets[1:])
    ]
    # Grip polygons and free polygons share the same vertex-marker color:
    # mechanical support does not remove a polygon's shape parameters.
    facecolors = ["#d9e7ee"] * 3 + ["#e1f1ed"] * 3 + ["#d9e7ee"] * 3
    ax.add_collection(
        PolyCollection(outlines, facecolors=facecolors, edgecolors="#547681", linewidths=1.2)
    )
    ax.add_collection(LineCollection(bond_points, colors="#a5afb2", linewidths=2))
    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=27,
        color=ORANGE,
        edgecolors="white",
        linewidths=0.55,
        zorder=4,
    )
    for polygon, (x, y, _) in enumerate(state):
        fixed = polygon < 3 or polygon >= 6
        ax.scatter(
            [x],
            [y],
            marker="s" if fixed else "o",
            s=205,
            facecolor="white",
            edgecolors=INK if fixed else TEAL,
            linewidths=1.0,
            zorder=5,
        )
        ax.text(
            x,
            y,
            str(polygon),
            ha="center",
            va="center",
            color=INK,
            fontsize=10,
            fontweight="bold",
            zorder=7,
        )

    # The displacement arrows refer to prescribed center motion, not vertex design.
    for x in state[6:9, 0]:
        _arrow(ax, (x, 2.88), (x, 2.52), BLUE, linewidth=1.7, scale=14)
    ax.text(
        1,
        3.08,
        r"Top centers: $x,\theta$ held; $y=y_0-\epsilon H$",
        ha="center",
        color=BLUE,
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        1,
        -1.28,
        r"Bottom centers: $x,y,\theta$ held",
        ha="center",
        color=INK,
        fontsize=11,
        fontweight="bold",
    )
    ax.text(
        3.08,
        1.10,
        "Middle row\n" + r"$x,y,\theta$ free",
        ha="center",
        va="center",
        color=TEAL,
        fontsize=11,
        linespacing=1.5,
    )
    _arrow(ax, (2.63, 1.0), (2.20, 1.0), TEAL, scale=10, linewidth=1.0)

    # Gauge height is between grip centers; nominal width spans the whole outline.
    height = float(np.ptp(state[:, 1]))
    left, right = float(points[:, 0].min()), float(points[:, 0].max())
    width = right - left
    ax.plot([-0.97, -0.53], [0, 0], color="#a2afb4", linewidth=0.8)
    ax.plot([-0.97, -0.53], [height, height], color="#a2afb4", linewidth=0.8)
    _arrow(ax, (-0.85, 0), (-0.85, height), MUTED, style="<->", linewidth=1, scale=11)
    ax.text(
        -1.08,
        height / 2,
        f"H = {height:g}\ngrip centers",
        rotation=90,
        ha="center",
        va="center",
        color=MUTED,
        fontsize=10,
    )
    ax.plot([left, left], [-0.55, -0.90], color="#a2afb4", linewidth=0.8)
    ax.plot([right, right], [-0.55, -0.90], color="#a2afb4", linewidth=0.8)
    _arrow(ax, (left, -0.76), (right, -0.76), MUTED, style="<->", linewidth=1, scale=11)
    ax.text(
        (left + right) / 2,
        -0.99,
        f"W = {width:.1f}  ·  reference outline width",
        ha="center",
        color=MUTED,
        fontsize=10,
    )
    ax.text(
        3.08,
        0.12,
        "Orange dots:\nall 36 design corners\n\n"
        f"{geometry.n_bonds} gray ligaments\ncenter spacing = 1\n{gap_label}",
        ha="center",
        va="center",
        fontsize=9,
        color=MUTED,
        linespacing=1.5,
    )
    ax.set(xlim=(-1.35, 3.95), ylim=(-1.48, 3.45), aspect="equal")
    ax.axis("off")


def _draw_corner_update(ax, geometry, parameters):
    """Enlarge polygon 4 and separate a design offset into its two components."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon

    polygon = 4
    start, stop = geometry.polygon_offsets[polygon : polygon + 2]
    reference = np.asarray(geometry.vertices[start:stop])
    candidate = reference + parameters[polygon]
    ax.add_patch(Polygon(candidate, facecolor="#dcefe9", edgecolor=TEAL, linewidth=1.8, zorder=2))
    ax.add_patch(
        Polygon(
            reference,
            facecolor="none",
            edgecolor="#778a94",
            linewidth=1.3,
            linestyle=(0, (4, 3)),
            zorder=3,
        )
    )
    ax.scatter(
        reference[:, 0],
        reference[:, 1],
        s=35,
        facecolors="white",
        edgecolors="#778a94",
        linewidths=1.1,
        zorder=5,
    )
    ax.scatter(
        candidate[:, 0],
        candidate[:, 1],
        s=34,
        color=ORANGE,
        edgecolors="white",
        linewidths=0.5,
        zorder=7,
    )
    ax.scatter([0], [0], s=40, color=INK, zorder=6)
    ax.text(
        0,
        -0.075,
        "local origin\npolygon 4 center",
        ha="center",
        va="top",
        color=MUTED,
        fontsize=10,
        linespacing=1.3,
    )

    for corner in (0, 2, 3):
        _arrow(ax, reference[corner], candidate[corner], ORANGE, scale=12, linewidth=1.7)
    # Corner 1 (the upper corner) shows both independent Cartesian components.
    original = reference[1]
    intermediate = original + np.array([parameters[polygon, 1, 0], 0.0])
    moved = candidate[1]
    _arrow(ax, original, intermediate, BLUE, scale=10, linewidth=1.8)
    _arrow(ax, intermediate, moved, ORANGE, scale=10, linewidth=1.8)
    ax.text(
        (original[0] + intermediate[0]) / 2,
        original[1] + 0.048,
        r"$\delta_x$",
        color=BLUE,
        ha="center",
        fontsize=12,
    )
    ax.text(
        intermediate[0] + 0.038,
        (intermediate[1] + moved[1]) / 2,
        r"$\delta_y$",
        color=ORANGE,
        va="center",
        fontsize=12,
    )
    # Local basis is an explanatory overlay, separate from equilibrium rotation.
    _arrow(ax, (-0.56, -0.58), (-0.34, -0.58), MUTED, linewidth=1.0, scale=10)
    _arrow(ax, (-0.56, -0.58), (-0.56, -0.36), MUTED, linewidth=1.0, scale=10)
    ax.text(-0.31, -0.58, "x", color=MUTED, va="center", fontsize=10)
    ax.text(-0.56, -0.33, "y", color=MUTED, ha="center", fontsize=10)

    corner_labels = [
        (0.57, 0.015, "c = 0\nright"),
        (-0.11, 0.59, "c = 1 · up"),
        (-0.58, 0.08, "c = 2\nleft"),
        (0.04, -0.57, "c = 3 · down"),
    ]
    for x, y, label in corner_labels:
        ax.text(x, y, label, ha="center", va="center", color=INK, fontsize=10)
    ax.text(
        0,
        -0.70,
        "The center is unchanged by the design offsets.",
        ha="center",
        color=MUTED,
        fontsize=9,
    )
    ax.text(
        0,
        0.80,
        r"$v'_{p,c}=v^0_{p,c}+\delta_{p,c}$",
        color=INK,
        ha="center",
        fontsize=18,
    )
    ax.text(
        0,
        0.69,
        "Each corner gets its own two-component offset",
        color=MUTED,
        ha="center",
        fontsize=10,
    )
    ax.legend(
        handles=[
            Line2D([0], [0], color="#778a94", linestyle="--", label="Reference outline"),
            Line2D([0], [0], color=TEAL, linewidth=2, label="Candidate outline"),
            Line2D(
                [0],
                [0],
                color=ORANGE,
                marker="o",
                markersize=4,
                label="Independent corner movement",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncols=1,
        frameon=False,
        fontsize=9,
        handlelength=2,
    )
    ax.set(xlim=(-0.78, 0.80), ylim=(-0.95, 0.91), aspect="equal")
    ax.axis("off")


def _draw_index_guide(ax):
    """Map tensor indices to geometry and separate design from mechanical motion."""
    from matplotlib.patches import FancyBboxPatch

    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0.005, 0.025),
            0.99,
            0.94,
            boxstyle="round,pad=0.008,rounding_size=0.025",
            facecolor="#f0f4f3",
            edgecolor="none",
        )
    )
    ax.text(
        0.025,
        0.84,
        "C  Every vertex has its own design variables",
        fontsize=13,
        fontweight="bold",
        color=INK,
    )
    ax.plot([0.33, 0.33], [0.13, 0.70], color="#cbd8d9", linewidth=1)
    ax.plot([0.66, 0.66], [0.13, 0.70], color="#cbd8d9", linewidth=1)
    ax.text(
        0.025,
        0.61,
        "delta.shape = (9, 4, 2)",
        family="monospace",
        fontsize=13,
        color=TEAL,
    )
    ax.text(
        0.025,
        0.42,
        "9 polygons × 4 corners × 2 coordinates",
        fontsize=10,
        color=INK,
    )
    ax.text(
        0.025,
        0.22,
        "72 independent scalars",
        fontsize=15,
        fontweight="bold",
        color=TEAL,
    )
    ax.text(
        0.355,
        0.61,
        "delta[4, 1, 0]",
        family="monospace",
        fontsize=13,
        color=ORANGE,
    )
    ax.text(
        0.355,
        0.42,
        "polygon 4  →  upper corner  →  x offset",
        fontsize=10,
        color=INK,
    )
    ax.text(
        0.355,
        0.22,
        "No shared or tiled design variables.",
        fontsize=10,
        color=MUTED,
    )
    ax.text(
        0.685,
        0.60,
        r"$p_{p,c}=c_p+R(\theta_p)\,v'_{p,c}$",
        fontsize=17,
        color=INK,
    )
    ax.text(
        0.685,
        0.39,
        r"$c_p,\theta_p$: center motion and equilibrium rotation",
        fontsize=10,
        color=MUTED,
    )
    ax.text(
        0.685,
        0.20,
        r"$\delta_{p,c}$: shape change in the polygon's body frame",
        fontsize=10,
        color=MUTED,
    )


def plot_setup(geometry=None, parameters=None):
    """Return an explanatory figure for the nine-polygon independent-vertex design.

    ``parameters`` contains Cartesian local-vertex offsets and may have shape
    ``(9, 4, 2)``, ``(36, 2)``, or ``(72,)``. Polygon 4 is enlarged to explain its
    offsets. If parameters are omitted, a small illustrative design is shown.
    The default geometry is the 3 × 3 grid with spacing 1 and hinge gap 0.2.
    Supplied geometry must use the same row-major unit-spaced centers and
    right/up/left/down local corner order for this explanatory layout.

    The boundary panel always shows the undeformed reference geometry, so its
    support labels describe the setup independently of the candidate shape.
    """
    import matplotlib.pyplot as plt

    geometry = square_grid(3, 3, spacing=1.0, hinge_gap=0.2) if geometry is None else geometry
    if geometry.n_polygons != 9 or np.any(np.diff(geometry.polygon_offsets) != 4):
        raise ValueError("This explanatory diagram requires nine four-vertex polygons.")
    expected_centers = np.asarray([(x, y) for y in range(3) for x in range(3)])
    if not np.allclose(np.asarray(geometry.reference_state)[:, :2], expected_centers):
        raise ValueError("The diagram requires a row-major 3 × 3 grid with unit center spacing.")
    if parameters is None:
        parameters = np.zeros((9, 4, 2))
        parameters[4] = [[0.06, 0.03], [0.05, -0.04], [-0.04, 0.02], [0.02, -0.06]]
    else:
        parameters = np.asarray(parameters, dtype=float)
        if parameters.shape not in ((9, 4, 2), (36, 2), (72,)):
            raise ValueError("parameters must have shape (9, 4, 2), (36, 2), or (72,).")
        parameters = parameters.reshape((9, 4, 2))
        if not np.all(np.isfinite(parameters)):
            raise ValueError("parameters must be finite.")

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "text.color": INK,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "svg.fonttype": "none",  # Preserve editable text in the SVG export.
        }
    ):
        fig = plt.figure(figsize=(15, 10), facecolor=PAPER)
        grid = fig.add_gridspec(
            2,
            2,
            height_ratios=[3.5, 1.05],
            width_ratios=[1.12, 1.0],
            left=0.035,
            right=0.98,
            bottom=0.075,
            top=0.87,
            hspace=0.10,
            wspace=0.06,
        )
        specimen = fig.add_subplot(grid[0, 0])
        corners = fig.add_subplot(grid[0, 1])
        index = fig.add_subplot(grid[1, :])
        specimen.set_title(
            "A  Compression setup", loc="left", fontsize=15, fontweight="bold", pad=12
        )
        corners.set_title(
            "B  Change one polygon's shape",
            loc="left",
            fontsize=15,
            fontweight="bold",
            pad=12,
        )
        _draw_specimen(specimen, geometry)
        _draw_corner_update(corners, geometry, parameters)
        _draw_index_guide(index)
        fig.text(
            0.035,
            0.958,
            "Optimize every corner independently",
            fontsize=24,
            fontweight="bold",
            color=INK,
        )
        fig.text(
            0.035,
            0.919,
            "Mechanical supports constrain center motion and rotation; "
            "design offsets change the polygon outlines.",
            fontsize=12,
            color=MUTED,
        )
        fig.text(
            0.5,
            0.035,
            "All corners remain design variables — including every corner "
            "on the clamped and driven grip polygons.",
            ha="center",
            fontsize=12,
            color=TEAL,
            fontweight="bold",
        )
    return fig


def main():
    import matplotlib as mpl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/assets"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    figure = plot_setup()
    with mpl.rc_context({"svg.fonttype": "none"}):
        for extension in ("svg", "png"):
            destination = args.output / f"vertex_design_setup.{extension}"
            figure.savefig(destination, dpi=160, facecolor=figure.get_facecolor())
            print(f"Saved {destination}")


if __name__ == "__main__":
    main()
