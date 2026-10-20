"""Reaction measurements and reusable design objectives.

Compression is positive. The reported nominal stress is the force conjugate to
the imposed engineering strain, divided by the reference specimen volume.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def reaction_forces(
    potential: Callable,
    states: jax.Array,
    reference: Any,
) -> jax.Array:
    """Return ``dE/dq`` for a state or a sequence of states.

    At constrained degrees of freedom these are the forces and torques the
    supports exert on the specimen. At free degrees of freedom they are the
    equilibrium residuals. The internal mechanical force is their negative.
    """
    states = jnp.asarray(states)
    gradient = jax.grad(potential, argnums=0)
    if states.ndim == 2:
        return gradient(states, reference)
    if states.ndim == 3:
        return jax.vmap(gradient, in_axes=(0, None))(states, reference)
    raise ValueError("states must have shape (n_polygons, 3) or (n_steps, n_polygons, 3)")


def nominal_stress(
    potential: Callable,
    boundary: Any,
    states: jax.Array,
    reference: Any,
    thickness: float = 1.0,
) -> jax.Array:
    """Measure nominal stress from the imposed displacement field.

    For compression strain ``epsilon``, this evaluates
    ``sum(dE/dq * dq_boundary/depsilon) / (height * width * thickness)``.
    No additional factor of one half belongs in this expression: the spring
    energy already contains its conventional one-half factor.

    ``height`` and ``width`` refer to the undeformed specimen. The default
    thickness of one gives a two-dimensional force-per-width convention.
    """
    if not np.isfinite(thickness) or thickness <= 0:
        raise ValueError("thickness must be finite and positive")
    if boundary.height <= 0 or boundary.width <= 0:
        raise ValueError("reference height and width must be positive")
    reactions = reaction_forces(potential, states, reference)
    volume = boundary.height * boundary.width * thickness
    return jnp.sum(reactions * boundary.strain_direction, axis=(-2, -1)) / volume


def make_stress_measurement(
    potential: Callable,
    boundary: Any,
    thickness: float = 1.0,
) -> Callable:
    """Return a measurement callable with signature ``stress(states, reference)``."""
    if not np.isfinite(thickness) or thickness <= 0:
        raise ValueError("thickness must be finite and positive")

    def measure(states, reference):
        return nominal_stress(potential, boundary, states, reference, thickness)

    return measure


def stress_curve_loss(
    simulate: Callable,
    initialize_design: Callable,
    stress_function: Callable,
    strains: jax.Array,
    target_stress: jax.Array,
    scale: float | None = None,
) -> Callable:
    """Build a differentiable mean-squared stress-curve error.

    ``initialize_design(parameters)`` constructs initialized geometry or a
    ``MechanicalParameters`` PyTree with active material coefficients;
    ``simulate(reference, strains)`` returns an object with a ``states`` field.
    ``stress_function(states, reference)`` returns one stress per strain.
    The returned callable accepts the same parameter array or PyTree as
    ``initialize_design``.

    Errors are divided by ``scale`` before squaring. By default, the largest
    absolute target stress is used, with a scale of one for an all-zero target.
    This changes the optimization scale without changing the target curve.
    An unconverged simulation returns infinity so the optimizer cannot silently
    accept a stress curve obtained from a nonequilibrium state.
    """
    strain_values = np.asarray(strains)
    target_values = np.asarray(target_stress)
    if strain_values.ndim != 1 or not strain_values.size:
        raise ValueError("strains must be a nonempty one-dimensional array")
    if target_values.shape != strain_values.shape:
        raise ValueError("target_stress must have one value per strain")
    if not np.all(np.isfinite(strain_values)) or not np.all(np.isfinite(target_values)):
        raise ValueError("strains and target_stress must be finite")
    if scale is None:
        largest_target = float(np.max(np.abs(target_values)))
        scale = largest_target if largest_target > 0 else 1.0
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    strain_array = jnp.asarray(strain_values)
    target_array = jnp.asarray(target_values)

    def loss(parameters):
        reference = initialize_design(parameters)
        result = simulate(reference, strain_array)
        stress = stress_function(result.states, reference)
        if stress.shape != target_array.shape:
            raise ValueError("stress_function must return one value per strain")
        error = jnp.mean(jnp.square((stress - target_array) / scale))
        return jnp.where(jnp.all(result.converged), error, jnp.inf)

    return loss
