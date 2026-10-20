"""Elastic ligament energies and local angular contact."""

from dataclasses import dataclass
from typing import Literal, NamedTuple

import jax.numpy as jnp
import numpy as np

from .geometry import rotate_vectors
from .initialization import (
    InitializedGeometry,
    angle_between,
    cross_2d,
    vector_length,
    wrap_angle,
)


@dataclass(frozen=True)
class PotentialParameters:
    """Scalar or per-bond elastic coefficients; all stiffnesses are nonnegative.

    Stretching, shearing: energy / length². Bending and contact: energy / rad².
    Positive ``contact_angle_offset`` permits additional corner closure.
    """

    stretching: object = 1.0
    shearing: object = 1e-4
    bending: object = 0.005
    contact: object = 40.0
    contact_angle_offset: float = 0.0

    def __post_init__(self):
        for name in ("stretching", "shearing", "bending", "contact"):
            value = np.asarray(getattr(self, name))
            if not np.all(np.isfinite(value)) or np.any(value < 0):
                raise ValueError(f"{name} must be finite and nonnegative")
        if not np.isfinite(self.contact_angle_offset):
            raise ValueError("contact_angle_offset must be finite")


class MechanicalParameters(NamedTuple):
    """Differentiable geometry and material data passed explicitly to a solver.

    This NamedTuple is a JAX PyTree. Each coefficient can be a scalar or an
    array of shape ``(n_bonds,)``; ``reference`` comes from :func:`initialize`.
    Stretching and shearing have units energy / length²; bending and contact
    have units energy / rad². Positive contact-angle offsets permit additional
    corner closure.

    Values are not converted to NumPy or checked during construction, so fields
    may be tracers inside ``jit`` and ``grad``. The potential checks array shapes.
    The caller must keep coefficients finite and stiffnesses nonnegative, using
    optimizer bounds or a positive parameterization such as ``exp(log_K)``.
    Reinitialize the reference geometry for each candidate in joint shape and
    material optimization.
    """

    reference: InitializedGeometry
    stretching: object = 1.0
    shearing: object = 1e-4
    bending: object = 0.005
    contact: object = 40.0
    contact_angle_offset: object = 0.0


def _coefficient_arrays(parameters, n_bonds):
    """Validate static shapes without inspecting possibly traced values."""
    values = []
    for name in ("stretching", "shearing", "bending", "contact", "contact_angle_offset"):
        value = jnp.asarray(getattr(parameters, name))
        if value.ndim > 1 or (value.ndim == 1 and value.shape != (n_bonds,)):
            raise ValueError(f"{name} must be scalar or have shape (n_bonds,)")
        values.append(value)
    return tuple(values)


def _make_energy_evaluator(geometry, model):
    """Build the single constitutive implementation used by both public APIs."""
    if model not in ("finite_hinge", "point_hinge"):
        raise ValueError("model must be 'finite_hinge' or 'point_hinge'")
    first, second = geometry.bonds.T
    owners = geometry.vertex_polygons
    polygon_i, polygon_j = owners[first], owners[second]

    def energy(state, reference, stretching, shearing, bending, contact, contact_angle_offset):
        arms = rotate_vectors(reference.vertices, state[owners, 2])
        center_separation = state[polygon_i, :2] - state[polygon_j, :2] + geometry.image_shifts
        gap = center_separation + arms[first] - arms[second]
        rotation = state[polygon_i, 2] - state[polygon_j, 2] - reference.rest_rotations
        if model == "finite_hinge":
            extension = vector_length(gap) - reference.rest_lengths
            angles = jnp.stack(
                (angle_between(-gap, arms[first]), angle_between(gap, arms[second])),
                axis=-1,
            )
            shear_angles = wrap_angle(angles - reference.rest_shear_angles)
            stretch_energy = stretching * extension**2
            shear_energy = shearing * reference.rest_lengths**2 * jnp.sum(shear_angles**2, axis=-1)
        else:
            direction = center_separation / vector_length(center_separation)[:, None]
            stretch_energy = stretching * jnp.sum(gap * direction, axis=-1) ** 2
            shear_energy = shearing * cross_2d(direction, gap) ** 2
        bending_energy = bending * rotation**2
        openings = (
            reference.contact_angles
            + jnp.stack((rotation, -rotation), axis=-1)
            + contact_angle_offset[..., None]
        )
        contact_energy = contact * jnp.sum(jnp.minimum(openings, 0.0) ** 2, axis=-1)
        return 0.5 * jnp.sum(stretch_energy + shear_energy + bending_energy + contact_energy)

    return energy


def make_potential(
    geometry, parameters=None, *, model: Literal["finite_hinge", "point_hinge"] = "finite_hinge"
):
    """Return ``energy(state, reference)`` with fixed material coefficients.

    ``finite_hinge`` uses extension and changes of the two endpoint angles.
    ``point_hinge`` uses longitudinal/transverse mismatch of coincident arms.
    Both include relative-rotation bending and the same angular contact proxy.
    Each undirected bond is counted once with the conventional factor 1/2.

    Pass validated :class:`PotentialParameters` for constant coefficients.
    To optimize material coefficients, use :func:`make_parameterized_potential`
    and pass :class:`MechanicalParameters` explicitly to the simulator.
    """
    parameters = parameters or PotentialParameters()
    evaluator = _make_energy_evaluator(geometry, model)
    coefficients = _coefficient_arrays(parameters, geometry.n_bonds)

    def energy(state, reference):
        return evaluator(state, reference, *coefficients)

    return energy


def make_parameterized_potential(
    geometry, model: Literal["finite_hinge", "point_hinge"] = "finite_hinge"
):
    """Return ``energy(state, MechanicalParameters)`` for material design.

    The topology and constitutive model stay fixed; geometry reference data and
    material coefficients are explicit differentiable inputs. Use the returned
    potential with either integrator and call
    ``simulate(mechanical_parameters, strains)``. In particular, passing
    coefficients this way lets quasistatic implicit differentiation account for
    their effect on the equilibrium; do not capture traced stiffnesses in a new
    potential or solver closure inside the objective.

    All coefficient fields accept scalars or ``(n_bonds,)`` arrays. Shape
    validation is compatible with JIT. Keep stiffness values finite and
    nonnegative through design bounds or a positive parameterization.
    """
    evaluator = _make_energy_evaluator(geometry, model)

    def energy(state, parameters: MechanicalParameters):
        coefficients = _coefficient_arrays(parameters, geometry.n_bonds)
        return evaluator(state, parameters.reference, *coefficients)

    return energy
