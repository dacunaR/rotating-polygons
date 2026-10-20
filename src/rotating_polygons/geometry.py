"""Polygon geometry and bond topology, independent of the mechanical model.

Each rigid polygon has three state variables: ``(center_x, center_y, angle)``.
Its vertices are local arm vectors, measured from its center at zero angle.
Vertices are stored consecutively for each polygon in counterclockwise order.
A bond connects two *vertex indices*, so polygons may have different numbers
of vertices without padding the arrays used by JAX.
"""

from dataclasses import dataclass, field
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike


def _integer_array(values: ArrayLike, name: str) -> np.ndarray:
    """Reject fractional topology indices instead of silently truncating them."""
    array = np.asarray(values)
    if not np.issubdtype(array.dtype, np.integer):
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError(f"{name} must contain integer indices.")
        if not np.all(np.isfinite(array)) or not np.all(array == np.floor(array)):
            raise ValueError(f"{name} must contain integer indices.")
    if np.any(array < np.iinfo(np.int32).min) or np.any(array > np.iinfo(np.int32).max):
        raise ValueError(f"{name} indices must fit in a signed 32-bit integer.")
    return np.asarray(array, dtype=np.int64)


@dataclass(frozen=True, eq=False)
class Geometry:
    """Validated reference geometry with immutable topology.

    Parameters
    ----------
    reference_state
        Array of shape ``(n_polygons, 3)`` containing centers and angles.
    vertices
        Array of shape ``(n_vertices, 2)`` of local vertex arm vectors.
    vertex_polygons
        Owner polygon of each vertex, shape ``(n_vertices,)``.
    polygon_offsets
        Start offsets of the contiguous polygon vertex lists, including their
        final endpoint; shape ``(n_polygons + 1,)``.
    bonds
        Pairs of global vertex indices, shape ``(n_bonds, 2)``.
    image_shifts
        Optional periodic shifts of shape ``(n_bonds, 2)``. The shift is added
        to ``world_vertex[first] - world_vertex[second]``. These are fixed
        lattice-image choices, not a dynamic nearest-image operation.
    cell_size
        Optional positive ``(length_x, length_y)`` periodic cell dimensions.

    Construct geometry outside JAX transformations. During differentiation,
    pass candidate ``vertices`` separately to the potential or simulator.
    """

    reference_state: jax.Array
    vertices: jax.Array
    vertex_polygons: jax.Array
    polygon_offsets: tuple[int, ...]
    bonds: jax.Array
    image_shifts: jax.Array | None = None
    cell_size: tuple[float, float] | None = None
    previous_vertices: jax.Array = field(init=False)
    next_vertices: jax.Array = field(init=False)

    def __post_init__(self) -> None:
        owners = _integer_array(self.vertex_polygons, "vertex_polygons")
        offsets = _integer_array(self.polygon_offsets, "polygon_offsets")
        bonds = _integer_array(self.bonds, "bonds")
        if bonds.size == 0:
            bonds = bonds.reshape((0, 2))
        if bonds.ndim != 2 or bonds.shape[1] != 2:
            raise ValueError("bonds must have shape (n_bonds, 2).")
        if offsets.ndim != 1:
            raise ValueError("polygon_offsets must be a one-dimensional array.")
        # Use JAX's configured floating precision; importing this module does
        # not change global precision or select a particular accelerator.
        object.__setattr__(self, "reference_state", jnp.asarray(self.reference_state, dtype=float))
        object.__setattr__(self, "vertices", jnp.asarray(self.vertices, dtype=float))
        object.__setattr__(self, "vertex_polygons", jnp.asarray(owners, dtype=jnp.int32))
        object.__setattr__(self, "polygon_offsets", tuple(int(i) for i in offsets))
        object.__setattr__(self, "bonds", jnp.asarray(bonds, dtype=jnp.int32))
        shifts = np.zeros((len(bonds), 2)) if self.image_shifts is None else self.image_shifts
        object.__setattr__(self, "image_shifts", jnp.asarray(shifts, dtype=float))
        if self.cell_size is not None:
            object.__setattr__(self, "cell_size", tuple(float(x) for x in self.cell_size))
        self.validate()
        previous = np.empty(self.n_vertices, dtype=np.int32)
        following = np.empty(self.n_vertices, dtype=np.int32)
        for start, stop in zip(self.polygon_offsets[:-1], self.polygon_offsets[1:]):
            indices = np.arange(start, stop)
            previous[start:stop] = np.roll(indices, 1)
            following[start:stop] = np.roll(indices, -1)
        object.__setattr__(self, "previous_vertices", jnp.asarray(previous))
        object.__setattr__(self, "next_vertices", jnp.asarray(following))

    @property
    def n_polygons(self) -> int:
        return self.reference_state.shape[0]

    @property
    def n_vertices(self) -> int:
        return self.vertices.shape[0]

    @property
    def n_bonds(self) -> int:
        return self.bonds.shape[0]

    def validate(self) -> None:
        """Raise ``ValueError`` for inconsistent or degenerate input geometry."""
        state = np.asarray(self.reference_state)
        vertices = np.asarray(self.vertices)
        owners = np.asarray(self.vertex_polygons)
        offsets = np.asarray(self.polygon_offsets)
        bonds = np.asarray(self.bonds)
        shifts = np.asarray(self.image_shifts)
        if state.ndim != 2 or state.shape[1] != 3 or len(state) == 0:
            raise ValueError(
                "reference_state must have shape (n_polygons, 3), with n_polygons > 0."
            )
        if vertices.ndim != 2 or vertices.shape[1] != 2:
            raise ValueError("vertices must have shape (n_vertices, 2).")
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(vertices)):
            raise ValueError("Polygon centers, angles, and vertices must be finite.")
        if offsets.shape != (len(state) + 1,):
            raise ValueError("polygon_offsets must contain n_polygons + 1 entries.")
        if offsets[0] != 0 or offsets[-1] != len(vertices) or np.any(np.diff(offsets) < 3):
            raise ValueError("Each polygon must have at least three consecutive vertices.")
        expected_owners = np.repeat(np.arange(len(state)), np.diff(offsets))
        if owners.shape != (len(vertices),) or not np.array_equal(owners, expected_owners):
            raise ValueError("vertex_polygons must agree with the contiguous polygon_offsets.")
        if bonds.ndim != 2 or bonds.shape[1] != 2:
            raise ValueError("bonds must have shape (n_bonds, 2).")
        if np.any(bonds < 0) or np.any(bonds >= len(vertices)):
            raise ValueError("Bond vertex indices are outside the vertex array.")
        if len(bonds) and np.any(owners[bonds[:, 0]] == owners[bonds[:, 1]]):
            raise ValueError("Every bond must connect vertices of different polygons.")
        if len(bonds) != len(np.unique(np.sort(bonds, axis=1), axis=0)):
            raise ValueError("Duplicate bonds connect the same pair of vertices.")
        if shifts.shape != (len(bonds), 2) or not np.all(np.isfinite(shifts)):
            raise ValueError("image_shifts must be finite with shape (n_bonds, 2).")
        if self.cell_size is not None:
            cell = np.asarray(self.cell_size)
            if cell.shape != (2,) or not np.all(np.isfinite(cell)) or np.any(cell <= 0):
                raise ValueError("cell_size must contain two finite positive lengths.")
        elif np.any(shifts != 0):
            raise ValueError("Nonzero image_shifts require cell_size.")
        for polygon, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
            points = vertices[start:stop]
            following = np.roll(points, -1, axis=0)
            twice_area = np.sum(points[:, 0] * following[:, 1] - points[:, 1] * following[:, 0])
            if twice_area <= 0:
                raise ValueError(
                    f"Polygon {polygon} must have positive area and counterclockwise vertices."
                )
            if np.any(np.all(points == following, axis=1)):
                raise ValueError(f"Polygon {polygon} contains a zero-length edge.")


def from_polygons(
    centers: ArrayLike,
    polygons: Sequence[ArrayLike],
    bonds: ArrayLike,
    *,
    angles: ArrayLike | None = None,
    image_shifts: ArrayLike | None = None,
    cell_size: tuple[float, float] | None = None,
) -> Geometry:
    """Build geometry from local vertex lists of arbitrary polygon sizes.

    ``polygons[i]`` contains polygon ``i``'s local vertices in counterclockwise
    order. Global bond indices refer to the concatenation of these lists.
    Angles are in radians; omitted angles are zero.
    """
    centers = np.asarray(centers, dtype=float)
    if centers.ndim != 2 or centers.shape[1] != 2 or len(centers) == 0:
        raise ValueError("centers must have shape (n_polygons, 2), with n_polygons > 0.")
    local_vertices = [np.asarray(polygon, dtype=float) for polygon in polygons]
    if len(local_vertices) != len(centers):
        raise ValueError("Provide one local vertex list per polygon center.")
    if any(points.ndim != 2 or points.shape[1] != 2 for points in local_vertices):
        raise ValueError("Each polygon vertex list must have shape (n_polygon_vertices, 2).")
    angles = np.zeros(len(centers)) if angles is None else np.asarray(angles, dtype=float)
    if angles.shape != (len(centers),):
        raise ValueError("angles must contain one angle per polygon.")
    counts = np.asarray([len(points) for points in local_vertices])
    offsets = np.concatenate(([0], np.cumsum(counts)))
    return Geometry(
        reference_state=np.column_stack((centers, angles)),
        vertices=np.concatenate(local_vertices),
        vertex_polygons=np.repeat(np.arange(len(centers)), counts),
        polygon_offsets=tuple(offsets),
        bonds=bonds,
        image_shifts=image_shifts,
        cell_size=cell_size,
    )


def square_grid(nx: int, ny: int, spacing: float = 1.0, hinge_gap: float = 0.2) -> Geometry:
    """Create a finite rectangular grid of diamond-oriented squares.

    Centers lie on a regular lattice; polygon indices run left to right, then
    bottom to top. Each square's vertices are ordered right, up, left, down.
    ``hinge_gap`` is the distance between the two vertices of each bond in
    the reference geometry, and must satisfy ``0 <= hinge_gap < spacing``.
    No bonds wrap across the grid boundaries.
    """
    if any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1 for n in (nx, ny)):
        raise ValueError("nx and ny must be positive integers.")
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError("spacing must be finite and positive.")
    if not np.isfinite(hinge_gap) or not 0 <= hinge_gap < spacing:
        raise ValueError("hinge_gap must satisfy 0 <= hinge_gap < spacing.")
    centers = np.asarray([(x * spacing, y * spacing) for y in range(ny) for x in range(nx)])
    radius = (spacing - hinge_gap) / 2
    square = radius * np.asarray([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=float)
    bonds = []
    for y in range(ny):
        for x in range(nx):
            polygon = y * nx + x
            if x + 1 < nx:
                bonds.append((4 * polygon, 4 * (polygon + 1) + 2))
            if y + 1 < ny:
                bonds.append((4 * polygon + 1, 4 * (polygon + nx) + 3))
    return from_polygons(centers, [square] * (nx * ny), bonds)


def rotate_vectors(vectors: ArrayLike, angles: ArrayLike) -> jax.Array:
    """Rotate 2D vectors counterclockwise by broadcast-compatible angles."""
    vectors = jnp.asarray(vectors)
    angles = jnp.asarray(angles)
    cosine, sine = jnp.cos(angles), jnp.sin(angles)
    x, y = vectors[..., 0], vectors[..., 1]
    return jnp.stack((cosine * x - sine * y, sine * x + cosine * y), axis=-1)


def world_vertices(
    geometry: Geometry,
    state: ArrayLike,
    vertices: ArrayLike | None = None,
) -> jax.Array:
    """Map local vertex arms to world coordinates for a ``(n_polygons, 3)`` state."""
    state = jnp.asarray(state)
    arms = geometry.vertices if vertices is None else jnp.asarray(vertices)
    owners = geometry.vertex_polygons
    return state[owners, :2] + rotate_vectors(arms, state[owners, 2])
