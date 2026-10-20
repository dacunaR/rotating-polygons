"""Prescribed degrees of freedom, independent of geometry and energy choices."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class BoundaryConditions:
    """An affine displacement path on selected coordinates.

    States have shape ``(n_polygons, 3)`` with columns ``x, y, angle``.
    ``fixed_mask=True`` means prescribed, including moving grips.
    ``strain_direction`` is displacement per unit engineering strain.
    """

    reference_state: object
    fixed_mask: object
    strain_direction: object
    height: float = 1.0
    width: float = 1.0

    def __post_init__(self):
        state = np.asarray(self.reference_state, dtype=float)
        mask = np.asarray(self.fixed_mask, dtype=bool)
        direction = np.asarray(self.strain_direction, dtype=float)
        if state.ndim != 2 or state.shape[1] != 3:
            raise ValueError("reference_state must have shape (n_polygons, 3)")
        if mask.shape != state.shape or direction.shape != state.shape:
            raise ValueError("fixed_mask and strain_direction must match the state")
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(direction)):
            raise ValueError("Boundary coordinates must be finite")
        if np.any(direction[~mask] != 0):
            raise ValueError("Only prescribed degrees of freedom may be driven")
        if not np.isfinite(self.height) or not np.isfinite(self.width):
            raise ValueError("height and width must be finite")
        if self.height <= 0 or self.width <= 0:
            raise ValueError("height and width must be positive")
        object.__setattr__(self, "reference_state", jnp.asarray(state))
        object.__setattr__(self, "fixed_mask", jnp.asarray(mask))
        object.__setattr__(self, "strain_direction", jnp.asarray(direction))
        object.__setattr__(self, "free_indices", jnp.asarray(np.flatnonzero(~mask.ravel())))
        object.__setattr__(self, "fixed_indices", jnp.asarray(np.flatnonzero(mask.ravel())))

    def state_at(self, strain):
        """Reference coordinates with prescribed motion applied."""
        return self.reference_state + strain * self.strain_direction

    def assemble(self, free_coordinates, strain):
        """Insert free coordinates into the prescribed state."""
        return (
            self.state_at(strain)
            .ravel()
            .at[self.free_indices]
            .set(free_coordinates)
            .reshape(self.reference_state.shape)
        )

    def project(self, state, strain):
        """Enforce prescribed coordinates, retaining the free coordinates."""
        return jnp.where(self.fixed_mask, self.state_at(strain), state)


def compression_boundary(geometry, *, clamp_rotations=True, clamp_horizontal=True, width=None):
    """Fix the bottom row and drive the top down with positive compression.

    The gauge height is the distance between grip centers. The default nominal
    width is the initial polygon bounding-box width. Lateral grip motion and
    rotation can be released independently; one bottom x coordinate remains
    pinned when horizontal sliding is allowed, removing rigid translation.
    """
    from .geometry import world_vertices

    state = np.asarray(geometry.reference_state)
    y = state[:, 1]
    bottom = np.isclose(y, y.min())
    top = np.isclose(y, y.max())
    height = float(np.ptp(y))
    if height <= 0:
        raise ValueError("Compression requires at least two distinct rows")
    mask = np.zeros_like(state, dtype=bool)
    mask[bottom | top, 1] = True
    if clamp_horizontal:
        mask[bottom | top, 0] = True
    else:
        mask[np.flatnonzero(bottom)[0], 0] = True
    if clamp_rotations:
        mask[bottom | top, 2] = True
    direction = np.zeros_like(state)
    direction[top, 1] = -height
    if width is None:
        width = float(np.ptp(np.asarray(world_vertices(geometry, state))[:, 0]))
    return BoundaryConditions(state, mask, direction, height, width)
