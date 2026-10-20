"""Animate simulated polygon states and export GIF or MP4 movies.

Matplotlib is imported only when an animation is requested. Frames are sampled
from the supplied states without interpolation. Playback fps is independent
of the physical loading rate or the timestep used by a dynamic simulation.
"""

from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
import warnings

import numpy as np


@dataclass(frozen=True)
class _FrameData:
    """Concrete plotting data; retain the trajectory without making a full copy."""

    states: object
    vertices: np.ndarray
    owners: np.ndarray
    offsets: tuple
    bonds: np.ndarray
    image_shifts: np.ndarray
    indices: range
    labels: tuple | None

    def points(self, index):
        state = np.asarray(self.states[index], dtype=float)[self.owners]
        angle = state[:, 2]
        cosine, sine = np.cos(angle), np.sin(angle)
        x, y = self.vertices.T
        return state[:, :2] + np.column_stack((cosine * x - sine * y, sine * x + cosine * y))

    def segments(self, points):
        # Match the potential's convention: gap = endpoint_i - endpoint_j + shift.
        return np.stack(
            (points[self.bonds[:, 0]], points[self.bonds[:, 1]] - self.image_shifts),
            axis=1,
        )

    def label(self, index):
        return (
            self.labels[index]
            if self.labels is not None
            else f"Frame {index + 1} / {self.states.shape[0]}"
        )


def _positive_real(value, name):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a positive finite number.")
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number.")


def _prepare_frames(geometry, states, vertices, frame_labels, fps, stride):
    _positive_real(fps, "fps")
    if isinstance(stride, bool) or not isinstance(stride, Integral) or stride < 1:
        raise ValueError("stride must be a positive integer.")
    # JAX arrays and NumPy memory maps already support shape/indexing. Keeping
    # them avoids copying an entire long trajectory just to plot selected frames.
    if not hasattr(states, "shape"):
        states = np.asarray(states)
    expected = (geometry.n_polygons, 3)
    if len(states.shape) != 3 or states.shape[1:] != expected or states.shape[0] == 0:
        raise ValueError(
            f"states must have shape (n_frames, {geometry.n_polygons}, 3), with n_frames > 0."
        )
    # Validate even frames skipped by stride, with bounded temporary storage.
    for start in range(0, states.shape[0], 64):
        try:
            chunk = np.asarray(states[start : start + 64], dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("states must contain finite numeric coordinates.") from error
        if not np.all(np.isfinite(chunk)):
            raise ValueError("states must contain finite numeric coordinates.")
    try:
        vertices = np.asarray(geometry.vertices if vertices is None else vertices, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("vertices must contain finite numeric coordinates.") from error
    if vertices.shape != (geometry.n_vertices, 2):
        raise ValueError("vertices must have the same shape as geometry.vertices.")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("vertices must contain finite numeric coordinates.")
    labels = None
    if frame_labels is not None:
        if isinstance(frame_labels, (str, bytes)):
            raise ValueError("frame_labels must contain one label per original frame.")
        labels = tuple(str(label) for label in frame_labels)
        if len(labels) != states.shape[0]:
            raise ValueError("frame_labels must contain one label per original frame.")
    return _FrameData(
        states=states,
        vertices=vertices,
        owners=np.asarray(geometry.vertex_polygons),
        offsets=geometry.polygon_offsets,
        bonds=np.asarray(geometry.bonds),
        image_shifts=np.asarray(geometry.image_shifts),
        indices=range(0, states.shape[0], int(stride)),
        labels=labels,
    )


def _fixed_limits(data, padding, show_bonds):
    if not isinstance(padding, Real) or not np.isfinite(padding) or padding < 0:
        raise ValueError("padding must be finite and nonnegative.")
    lower, upper = np.full(2, np.inf), np.full(2, -np.inf)
    # Two passes over selected states cost time, but avoid an array containing
    # every world vertex at every time sample. Only one frame exists at a time.
    for index in data.indices:
        points = data.points(index)
        if not np.all(np.isfinite(points)):
            raise ValueError("World vertex positions overflow the plotting coordinate range.")
        lower = np.minimum(lower, points.min(axis=0))
        upper = np.maximum(upper, points.max(axis=0))
        if show_bonds and len(data.bonds):
            segments = data.segments(points)
            lower = np.minimum(lower, segments.min(axis=(0, 1)))
            upper = np.maximum(upper, segments.max(axis=(0, 1)))
    span = upper - lower
    common_span = float(max(span.max(), 1e-12))
    margins = np.full(2, padding * common_span)
    # A collapsed candidate is still useful to inspect; avoid singular limits.
    margins[span == 0] = max(common_span / 2, padding * common_span)
    return lower - margins, upper + margins


def create_animation(
    geometry,
    states,
    *,
    vertices=None,
    frame_labels=None,
    fps=20,
    stride=1,
    ax=None,
    figsize=(6, 6),
    padding=0.05,
    show_bonds=True,
):
    """Return a Matplotlib ``FuncAnimation`` for a simulated trajectory.

    Parameters
    ----------
    states
        Array shaped ``(n_frames, n_polygons, 3)``, for example ``result.states``
        from either the quasistatic or Verlet solver.
    vertices
        Candidate local vertices. Supply these when animating an optimized
        shape; omission uses ``geometry.vertices``.
    frame_labels
        Optional labels for *all original frames*, such as strain or time.
    fps, stride
        Display ``states[::stride]`` at ``fps`` frames per second. States are
        never interpolated. The last original frame is included only if the
        selected stride reaches it. For uniform dynamic samples of spacing dt,
        physical speed playback requires ``fps = 1 / (stride * dt)``. A
        quasistatic trajectory has no inherent physical time.

    Limits are fixed across all displayed frames, including periodic-image
    bond endpoints. Frame artists are updated in place and frame caching is
    disabled. Keep the returned animation alive until displayed or saved, and
    do not modify the input trajectory while it is in use.
    """
    data = _prepare_frames(geometry, states, vertices, frame_labels, fps, stride)
    return _create_from_frames(
        data, fps=fps, ax=ax, figsize=figsize, padding=padding, show_bonds=show_bonds
    )


def _create_from_frames(data, *, fps, ax, figsize, padding, show_bonds):
    """Build artists from validated frames without copying or revalidating states."""
    lower, upper = _fixed_limits(data, padding, show_bonds)
    size = np.asarray(figsize, dtype=float)
    if size.shape != (2,) or not np.all(np.isfinite(size)) or np.any(size <= 0):
        raise ValueError("figsize must contain two positive finite lengths.")

    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.collections import LineCollection, PolyCollection

    if ax is None:
        _, ax = plt.subplots(figsize=tuple(size), layout="constrained")
    polygons = PolyCollection(
        [], facecolors="#74b9c9", edgecolors="#163743", linewidths=1, alpha=0.85
    )
    ax.add_collection(polygons, autolim=False)
    bonds = None
    if show_bonds:
        bonds = LineCollection([], colors="#d65e46", linewidths=1.5)
        ax.add_collection(bonds, autolim=False)
    title = ax.set_title("")
    ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), xlabel="x", ylabel="y")
    ax.set_aspect("equal", adjustable="box")
    artists = (polygons, title) if bonds is None else (polygons, bonds, title)

    def update(index):
        points = data.points(index)
        polygons.set_verts(
            [points[start:stop] for start, stop in zip(data.offsets[:-1], data.offsets[1:])]
        )
        if bonds is not None:
            bonds.set_segments(data.segments(points))
        title.set_text(data.label(index))
        return artists

    # Populate the first frame immediately so the returned figure is also
    # meaningful as a static preview before the animation starts.
    update(data.indices[0])
    # Resolve spacing once. Changing labels must not move the axes rectangle
    # from frame to frame, even when the figure initially uses constrained layout.
    ax.figure.canvas.draw()
    ax.figure.set_layout_engine(None)
    return FuncAnimation(
        ax.figure,
        update,
        frames=data.indices,
        init_func=lambda: update(data.indices[0]),
        interval=1000 / float(fps),
        repeat=False,
        blit=False,
        cache_frame_data=False,
    )


def save_video(
    geometry,
    states,
    filename,
    *,
    vertices=None,
    frame_labels=None,
    fps=20,
    stride=1,
    dpi=120,
    fallback_to_gif=True,
    gif_memory_limit_mb=256,
    figsize=(6, 6),
    padding=0.05,
    show_bonds=True,
):
    """Save an MP4 or GIF and return the path actually written.

    The filename suffix selects the writer: ``.mp4`` uses an external ffmpeg
    executable; ``.gif`` uses Pillow. If ffmpeg is unavailable and
    ``fallback_to_gif=True``, emit a warning and write a same-stem GIF.
    Set it to ``False`` to require MP4 output.

    MP4 frames are streamed to ffmpeg. Pillow buffers GIF images, so GIF export
    estimates its RGBA frame storage before encoding and rejects estimates
    above ``gif_memory_limit_mb`` (default 256 MiB). Use a larger stride,
    smaller image, or MP4 for long trajectories. This estimate is not a strict
    process-memory cap. Passing ``None`` explicitly disables that safeguard.

    ``fps`` controls playback, not the simulation's timestep. See
    :func:`create_animation` for stride and physical-time conventions.
    Parent directories are created if needed.
    """
    destination = Path(filename)
    suffix = destination.suffix.lower()
    if suffix not in (".mp4", ".gif"):
        raise ValueError("filename must end in .mp4 or .gif.")
    _positive_real(dpi, "dpi")
    if gif_memory_limit_mb is not None:
        _positive_real(gif_memory_limit_mb, "gif_memory_limit_mb")

    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, PillowWriter, writers

    if suffix == ".mp4" and not writers.is_available("ffmpeg"):
        if not fallback_to_gif:
            raise RuntimeError("MP4 export requires ffmpeg; install it or choose a .gif filename.")
        destination = destination.with_suffix(".gif")
        suffix = ".gif"
        warnings.warn(
            f"ffmpeg is unavailable; saving GIF to {destination} instead.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Construct validation and frame metadata before creating a figure.
    data = _prepare_frames(geometry, states, vertices, frame_labels, fps, stride)
    size = np.asarray(figsize, dtype=float)
    if size.shape != (2,) or not np.all(np.isfinite(size)) or np.any(size <= 0):
        raise ValueError("figsize must contain two positive finite lengths.")
    if suffix == ".gif" and gif_memory_limit_mb is not None:
        pixels_per_frame = np.prod(np.ceil(size * dpi))
        estimated_mb = len(data.indices) * pixels_per_frame * 4 / 1024**2
        if estimated_mb > gif_memory_limit_mb:
            raise ValueError(
                f"GIF frame storage is approximately {estimated_mb:.1f} MiB, exceeding "
                f"gif_memory_limit_mb={gif_memory_limit_mb}. Increase stride, reduce "
                "figsize/dpi, or use MP4."
            )
    writer = (
        PillowWriter(fps=float(fps))
        if suffix == ".gif"
        else FFMpegWriter(
            fps=float(fps),
            codec="libx264",
            extra_args=["-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"],
        )
    )
    figure, axis = plt.subplots(figsize=tuple(size), layout="constrained")
    try:
        animation = _create_from_frames(
            data,
            fps=fps,
            ax=axis,
            figsize=figsize,
            padding=padding,
            show_bonds=show_bonds,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        animation.save(str(destination), writer=writer, dpi=dpi)
    finally:
        plt.close(figure)
    return destination
