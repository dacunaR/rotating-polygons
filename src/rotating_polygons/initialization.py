"""Differentiable construction of the undeformed ligament reference data."""

from typing import NamedTuple

import jax.numpy as jnp

from .geometry import rotate_vectors


class InitializedGeometry(NamedTuple):
    vertices: object
    rest_lengths: object
    rest_shear_angles: object
    contact_angles: object
    rest_rotations: object


def cross_2d(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def angle_between(a, b):
    """Signed angle, with a finite fallback for unused zero-length directions."""
    valid = (jnp.sum(a * a, axis=-1) > 1e-24) & (jnp.sum(b * b, axis=-1) > 1e-24)
    cross = jnp.where(valid, cross_2d(a, b), 0.0)
    dot = jnp.where(valid, jnp.sum(a * b, axis=-1), 1.0)
    return jnp.arctan2(cross, dot)


def wrap_angle(angle):
    """Shortest angular difference in [-pi, pi]."""
    return jnp.arctan2(jnp.sin(angle), jnp.cos(angle))


def vector_length(vector):
    """Smooth norm; the 1e-12 length regularizer avoids undefined derivatives."""
    return jnp.sqrt(jnp.sum(vector * vector, axis=-1) + 1e-24)


def initialize(geometry, vertices=None):
    """Recompute reference lengths and angles for each candidate shape.

    ``vertices`` are body-frame arms, not global positions. Reinitialization
    makes every finite-hinge design stress-free under the elastic terms.
    Point hinges additionally require coincident endpoints. Contact may still
    be active for an initially overlapping design.
    """
    vertices = geometry.vertices if vertices is None else jnp.asarray(vertices)
    if vertices.shape != geometry.vertices.shape:
        raise ValueError("vertices must have the same shape as geometry.vertices")
    state = geometry.reference_state
    owners = geometry.vertex_polygons
    arms = rotate_vectors(vertices, state[owners, 2])
    endpoints = state[owners, :2] + arms
    first, second = geometry.bonds.T
    gap = endpoints[first] - endpoints[second] + geometry.image_shifts
    shear = jnp.stack(
        (angle_between(-gap, arms[first]), angle_between(gap, arms[second])),
        axis=-1,
    )
    previous, following = geometry.previous_vertices, geometry.next_vertices
    # The two openings adjacent to each connected corner, in CCW order.
    beta_1 = angle_between(
        arms[previous[second]] - arms[second],
        arms[following[first]] - arms[first],
    )
    beta_2 = angle_between(
        arms[previous[first]] - arms[first],
        arms[following[second]] - arms[second],
    )
    beta = jnp.stack((beta_1, beta_2), axis=-1)
    beta = (beta + jnp.pi / 2) % (2 * jnp.pi) - jnp.pi / 2
    rest_rotations = state[owners[first], 2] - state[owners[second], 2]
    return InitializedGeometry(vertices, vector_length(gap), shear, beta, rest_rotations)
