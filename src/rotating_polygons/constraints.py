"""Differentiable manufacturing restrictions, expressed as margins >= 0."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def _safe_length(vectors: jax.Array) -> jax.Array:
    """Keep derivatives finite even at an infeasible zero-length edge."""
    return jnp.sqrt(jnp.sum(jnp.square(vectors), axis=-1) + 1e-24)


def edge_lengths(geometry: Any, vertices: jax.Array) -> jax.Array:
    """Return the length of every polygon edge, in vertex order."""
    vertices = jnp.asarray(vertices)
    return _safe_length(vertices[geometry.next_vertices] - vertices)


def corner_sines(geometry: Any, vertices: jax.Array) -> jax.Array:
    """Return signed turning-angle sines for counterclockwise polygons.

    Positive values indicate convex corners. A positive minimum excludes both
    nearly collinear and reflex corners; for example ``sin(20 degrees)`` keeps
    interior angles between 20 and 160 degrees. This is a local convexity
    restriction, not a general polygon-intersection detector.
    """
    vertices = jnp.asarray(vertices)
    incoming = vertices - vertices[geometry.previous_vertices]
    outgoing = vertices[geometry.next_vertices] - vertices
    cross = incoming[:, 0] * outgoing[:, 1] - incoming[:, 1] * outgoing[:, 0]
    return cross / (_safe_length(incoming) * _safe_length(outgoing))


def ligament_lengths(geometry: Any, vertices: jax.Array) -> jax.Array:
    """Return distances between connected vertices in the reference state.

    Local arm vectors are rotated by each polygon's reference angle before
    adding its center. The bond's periodic image shift is included.
    """
    vertices = jnp.asarray(vertices)
    states = geometry.reference_state[geometry.vertex_polygons]
    cosine = jnp.cos(states[:, 2])
    sine = jnp.sin(states[:, 2])
    rotated = jnp.stack(
        (
            cosine * vertices[:, 0] - sine * vertices[:, 1],
            sine * vertices[:, 0] + cosine * vertices[:, 1],
        ),
        axis=1,
    )
    endpoints = states[:, :2] + rotated
    gaps = endpoints[geometry.bonds[:, 0]] - endpoints[geometry.bonds[:, 1]] + geometry.image_shifts
    return _safe_length(gaps)


def shape_constraints(
    geometry: Any,
    vertices: jax.Array,
    min_edge_length: float = 0.05,
    min_corner_sine: float = 0.05,
    min_ligament_length: float = 0.0,
) -> jax.Array:
    """Concatenate edge, convex-corner, and optional ligament margins.

    All entries must be nonnegative. Length limits use the same units as the
    geometry; corner limits are dimensionless. A zero ligament limit disables
    that restriction, allowing the shared-corner model's coincident endpoints.
    Restrictions apply to the reference design, not every deformed state.
    """
    if (
        not np.isfinite(min_edge_length)
        or not np.isfinite(min_ligament_length)
        or min_edge_length < 0
        or min_ligament_length < 0
    ):
        raise ValueError("minimum lengths must be finite and nonnegative")
    if not 0 <= min_corner_sine < 1:
        raise ValueError("min_corner_sine must lie in [0, 1)")
    vertices = jnp.asarray(vertices)
    if vertices.shape != geometry.vertices.shape:
        raise ValueError("vertices must have the same shape as geometry.vertices")
    margins = [
        edge_lengths(geometry, vertices) - min_edge_length,
        corner_sines(geometry, vertices) - min_corner_sine,
    ]
    if min_ligament_length > 0:
        margins.append(ligament_lengths(geometry, vertices) - min_ligament_length)
    return jnp.concatenate(margins)


def make_shape_constraint(
    geometry: Any,
    vertices_from_parameters: Callable,
    *,
    min_edge_length: float = 0.05,
    min_corner_sine: float = 0.05,
    min_ligament_length: float = 0.0,
) -> Callable:
    """Adapt shape restrictions to an optimizer's parameter representation."""

    def constraint(parameters):
        return shape_constraints(
            geometry,
            vertices_from_parameters(parameters),
            min_edge_length,
            min_corner_sine,
            min_ligament_length,
        )

    return constraint
