"""Differentiable quasistatic continuation and velocity-Verlet dynamics."""

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jaxopt
import numpy as np


class SimulationResult(NamedTuple):
    states: object
    energies: object
    residuals: object
    iterations: object
    converged: object

    def check_convergence(self):
        """Raise outside JIT if any equilibrium failed to meet the tolerance."""
        failed = np.flatnonzero(~np.asarray(self.converged))
        if failed.size:
            raise RuntimeError(
                f"Equilibrium failed at step(s) {failed.tolist()}; "
                f"maximum free-force residual={float(np.max(self.residuals)):.3g}"
            )
        return self


@dataclass(frozen=True)
class QuasistaticOptions:
    method: str = "lbfgs"
    tolerance: float = 1e-8
    max_iterations: int = 1000
    max_line_search_iterations: int = 50
    perturbation_amplitude: float = 0.0
    seed: int = 0

    def __post_init__(self):
        if self.method not in ("bfgs", "lbfgs"):
            raise ValueError("method must be 'bfgs' or 'lbfgs'")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0:
            raise ValueError("tolerance must be positive and finite")
        if self.max_iterations < 1 or self.max_line_search_iterations < 1:
            raise ValueError("iteration limits must be positive")
        if not np.isfinite(self.perturbation_amplitude) or self.perturbation_amplitude < 0:
            raise ValueError("perturbation_amplitude must be finite and nonnegative")
        if not isinstance(self.seed, (int, np.integer)):
            raise ValueError("seed must be an integer")


def make_quasistatic_solver(geometry, boundary, potential, options=None):
    """Return ``simulate(reference, strains, initial_state=None)``.

    Strains are absolute values and may describe loading/unloading cycles.
    JAXopt implicit differentiation uses stationarity of each equilibrium;
    derivatives require a converged, nonsingular local solution and do not
    differentiate discrete choices between buckling branches.
    Optional seeded uniform perturbations are added only to free coordinates
    before each solve, with range [-amplitude/2, amplitude/2]. The amplitude
    applies to both lengths and angles in the chosen coordinate units.
    """
    options = options or QuasistaticOptions()
    if geometry.reference_state.shape != boundary.reference_state.shape:
        raise ValueError("geometry and boundary state shapes must match")

    def free_energy(free_coordinates, strain, reference):
        return potential(boundary.assemble(free_coordinates, strain), reference)

    optimizer_class = jaxopt.LBFGS if options.method == "lbfgs" else jaxopt.BFGS
    optimizer = optimizer_class(
        free_energy,
        tol=options.tolerance,
        maxiter=options.max_iterations,
        maxls=options.max_line_search_iterations,
        implicit_diff=True,
    )
    # JAXopt 0.8.3 BFGS passes verbose=-1 to its line search when silent.
    # Correct the owned solver instance before tracing; no global logging edits.
    optimizer.linesearch_solver.verbose = False
    gradient = jax.grad(free_energy)

    def simulate(reference, strains, initial_state=None):
        strains = jnp.asarray(strains)
        if strains.ndim != 1 or strains.shape[0] == 0:
            raise ValueError("strains must be a nonempty one-dimensional array")
        initial_state = (
            boundary.reference_state if initial_state is None else jnp.asarray(initial_state)
        )
        if initial_state.shape != boundary.reference_state.shape:
            raise ValueError("initial_state must have shape (n_polygons, 3)")
        initial_free = initial_state.ravel()[boundary.free_indices]
        perturbations = jnp.zeros((strains.size, initial_free.size))
        if options.perturbation_amplitude:
            perturbations = options.perturbation_amplitude * (
                jax.random.uniform(jax.random.key(options.seed), perturbations.shape) - 0.5
            )

        def step(free_coordinates, data):
            strain, perturbation = data
            if boundary.free_indices.size:
                free_coordinates = free_coordinates + perturbation
                solution = optimizer.run(free_coordinates, strain, reference)
                free_coordinates = solution.params
                iterations = solution.state.iter_num
                residual = jnp.linalg.norm(gradient(free_coordinates, strain, reference))
            else:
                iterations = jnp.asarray(0)
                residual = jnp.asarray(0.0)
            state = boundary.assemble(free_coordinates, strain)
            energy = potential(state, reference)
            converged = (
                jnp.isfinite(energy)
                & jnp.all(jnp.isfinite(state))
                & jnp.isfinite(residual)
                & (residual <= options.tolerance)
            )
            return free_coordinates, (state, energy, residual, iterations, converged)

        _, values = jax.lax.scan(step, initial_free, (strains, perturbations))
        return SimulationResult(*values)

    return simulate


class DynamicResult(NamedTuple):
    times: object
    states: object
    velocities: object
    potential_energies: object
    kinetic_energies: object


@dataclass(frozen=True)
class VerletOptions:
    time_step: float = 1e-3
    damping: float = 0.0

    def __post_init__(self):
        if not np.isfinite(self.time_step) or self.time_step <= 0:
            raise ValueError("time_step must be positive and finite")
        if not np.isfinite(self.damping) or self.damping < 0:
            raise ValueError("damping must be nonnegative and finite")


def make_verlet_solver(
    geometry, boundary, potential, *, masses=1.0, inertias=1.0, options=None, external_force=None
):
    """Return ``simulate(reference, strains, initial_state=None, initial_velocity=None)``.

    One strain value is prescribed at every uniform time sample, including
    t=0. Free translation and rotation use velocity Verlet. Optional linear
    drag ``dv/dt=-damping*v`` is split into exact exponential half steps.
    ``external_force(state, time)`` returns generalized forces of shape (N, 3).
    A stable timestep must resolve the fastest translational/rotational mode.
    """
    options = options or VerletOptions()
    if geometry.reference_state.shape != boundary.reference_state.shape:
        raise ValueError("geometry and boundary state shapes must match")
    mass = np.broadcast_to(np.asarray(masses, dtype=float), (geometry.n_polygons,))
    inertia = np.broadcast_to(np.asarray(inertias, dtype=float), (geometry.n_polygons,))
    if np.any(~np.isfinite(mass)) or np.any(~np.isfinite(inertia)):
        raise ValueError("masses and inertias must be finite")
    if np.any(mass <= 0) or np.any(inertia <= 0):
        raise ValueError("masses and inertias must be positive")
    generalized_mass = jnp.asarray(np.column_stack((mass, mass, inertia)))
    dt = options.time_step
    decay = jnp.exp(-0.5 * dt * options.damping)
    free_mask = ~boundary.fixed_mask
    gradient = jax.grad(potential)

    def acceleration(state, reference, time):
        force = -gradient(state, reference)
        if external_force is not None:
            applied_force = jnp.asarray(external_force(state, time))
            if applied_force.shape != state.shape:
                raise ValueError("external_force must return shape (n_polygons, 3)")
            force = force + applied_force
        return jnp.where(free_mask, force / generalized_mass, 0.0)

    def simulate(reference, strains, initial_state=None, initial_velocity=None):
        strains = jnp.asarray(strains)
        if strains.ndim != 1 or strains.size < 2:
            raise ValueError("Verlet requires at least two strain samples")
        state = boundary.reference_state if initial_state is None else jnp.asarray(initial_state)
        velocity = (
            jnp.zeros_like(state) if initial_velocity is None else jnp.asarray(initial_velocity)
        )
        if state.shape != boundary.reference_state.shape or velocity.shape != state.shape:
            raise ValueError("initial_state and initial_velocity must have shape (N, 3)")
        state = boundary.project(state, strains[0])
        prescribed_velocities = (
            jnp.concatenate((jnp.diff(strains)[:1], jnp.diff(strains)))[:, None, None]
            * boundary.strain_direction
            / dt
        )
        velocity = jnp.where(free_mask, velocity, prescribed_velocities[0])
        times = dt * jnp.arange(strains.size)

        def step(carry, data):
            q, v, time = carry
            strain, next_time, grip_velocity = data
            half_velocity = decay * v + 0.5 * dt * acceleration(q, reference, time)
            next_q = boundary.project(q + dt * half_velocity, strain)
            next_v = decay * (half_velocity + 0.5 * dt * acceleration(next_q, reference, next_time))
            next_v = jnp.where(free_mask, next_v, grip_velocity)
            return (next_q, next_v, next_time), (next_q, next_v)

        _, (states, velocities) = jax.lax.scan(
            step,
            (state, velocity, times[0]),
            (strains[1:], times[1:], prescribed_velocities[1:]),
        )
        states = jnp.concatenate((state[None], states))
        velocities = jnp.concatenate((velocity[None], velocities))
        energies = jax.vmap(potential, in_axes=(0, None))(states, reference)
        kinetic = 0.5 * jnp.sum(generalized_mass * velocities**2, axis=(1, 2))
        return DynamicResult(times, states, velocities, energies, kinetic)

    return simulate
