"""Analytic equilibrium and oscillator checks, including implicit derivatives."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from rotating_polygons.boundary import BoundaryConditions, compression_boundary
from rotating_polygons.energy import PotentialParameters, make_potential
from rotating_polygons.geometry import square_grid
from rotating_polygons.initialization import initialize
from rotating_polygons.integrators import (
    QuasistaticOptions,
    VerletOptions,
    make_quasistatic_solver,
    make_verlet_solver,
)

jax.config.update("jax_enable_x64", True)


def vertical_spring():
    """One free vertical coordinate connected to a fully fixed polygon."""
    geometry = square_grid(1, 2, hinge_gap=0.4)
    mask = np.ones((2, 3), dtype=bool)
    mask[1, 1] = False
    boundary = BoundaryConditions(geometry.reference_state, mask, np.zeros((2, 3)))
    potential = make_potential(
        geometry,
        PotentialParameters(stretching=4.0, shearing=0.0, bending=0.0, contact=0.0),
    )
    return geometry, boundary, potential, initialize(geometry)


class QuasistaticTests(unittest.TestCase):
    def test_seeded_perturbations_preserve_grips_and_are_reproducible(self):
        geometry, boundary, potential, reference = vertical_spring()
        solve = make_quasistatic_solver(
            geometry,
            boundary,
            potential,
            QuasistaticOptions(tolerance=1e-10, perturbation_amplitude=0.02, seed=17),
        )
        initial = geometry.reference_state.at[1, 1].add(0.08)
        first = solve(reference, jnp.array([0.0, 0.02]), initial).check_convergence()
        second = solve(reference, jnp.array([0.0, 0.02]), initial).check_convergence()
        np.testing.assert_array_equal(first.states, second.states)
        np.testing.assert_allclose(
            first.states,
            np.broadcast_to(geometry.reference_state, first.states.shape),
            atol=1e-10,
        )

    def test_two_polygon_spring_relaxes_to_analytic_equilibrium(self):
        geometry, boundary, potential, reference = vertical_spring()
        perturbed = np.asarray(boundary.reference_state).copy()
        perturbed[1, 1] += 0.08
        boundary = BoundaryConditions(perturbed, boundary.fixed_mask, boundary.strain_direction)
        for method in ("lbfgs", "bfgs"):
            with self.subTest(method=method):
                solve = make_quasistatic_solver(
                    geometry,
                    boundary,
                    potential,
                    QuasistaticOptions(method=method, tolerance=1e-10, max_iterations=100),
                )
                result = solve(reference, jnp.array([0.0, 0.02]))
                result.check_convergence()
                np.testing.assert_allclose(
                    result.states,
                    np.broadcast_to(geometry.reference_state, (2, 2, 3)),
                    atol=1e-10,
                )
                np.testing.assert_allclose(result.energies, 0.0, atol=1e-18)

    def test_clamped_compression_has_correct_reaction_sign_and_energy(self):
        geometry, _, potential, reference = vertical_spring()
        boundary = compression_boundary(geometry, width=1.0)
        strains = jnp.array([0.0, 0.03, 0.06, 0.02])
        result = make_quasistatic_solver(geometry, boundary, potential)(reference, strains)
        result.check_convergence()
        expected_states = jax.vmap(boundary.state_at)(strains)
        np.testing.assert_array_equal(result.states, expected_states)
        np.testing.assert_allclose(result.energies, 0.5 * 4.0 * strains**2, atol=1e-14)
        reactions = jax.vmap(jax.grad(potential), in_axes=(0, None))(result.states, reference)
        np.testing.assert_allclose(reactions[:, 0, 1], 4.0 * strains, atol=1e-12)
        np.testing.assert_allclose(reactions[:, 1, 1], -4.0 * strains, atol=1e-12)
        np.testing.assert_array_equal(result.iterations, np.zeros(len(strains), dtype=int))

    def test_serial_springs_equilibrate_and_keep_all_prescribed_coordinates(self):
        geometry = square_grid(1, 3, hinge_gap=0.4)
        boundary = compression_boundary(geometry, width=1.0)
        mask = np.asarray(boundary.fixed_mask).copy()
        mask[:, [0, 2]] = True
        boundary = BoundaryConditions(
            boundary.reference_state,
            mask,
            boundary.strain_direction,
            boundary.height,
            boundary.width,
        )
        potential = make_potential(
            geometry,
            PotentialParameters(
                stretching=np.array([2.0, 6.0]), shearing=0.0, bending=0.0, contact=0.0
            ),
        )
        strains = jnp.array([0.0, 0.02, 0.04, 0.01])
        result = make_quasistatic_solver(
            geometry,
            boundary,
            potential,
            QuasistaticOptions(tolerance=1e-10, max_iterations=100),
        )(initialize(geometry), strains)
        result.check_convergence()
        total_displacement = boundary.height * strains
        expected_middle_y = 1.0 - (6.0 / (2.0 + 6.0)) * total_displacement
        np.testing.assert_allclose(result.states[:, 1, 1], expected_middle_y, atol=1e-10)
        prescribed = jax.vmap(boundary.state_at)(strains)
        np.testing.assert_array_equal(
            np.asarray(result.states)[:, mask], np.asarray(prescribed)[:, mask]
        )
        # Series spring stiffness: k1*k2/(k1+k2) = 1.5.
        np.testing.assert_allclose(result.energies, 0.75 * total_displacement**2, atol=1e-12)

    def test_implicit_shape_gradient_matches_resolved_finite_difference(self):
        geometry = square_grid(1, 3, hinge_gap=0.4)
        boundary = compression_boundary(geometry, width=1.0)
        mask = np.asarray(boundary.fixed_mask).copy()
        mask[:, [0, 2]] = True
        boundary = BoundaryConditions(
            boundary.reference_state,
            mask,
            boundary.strain_direction,
            boundary.height,
            boundary.width,
        )
        potential = make_potential(
            geometry,
            PotentialParameters(
                stretching=np.array([2.0, 6.0]), shearing=0.2, bending=0.01, contact=0.0
            ),
        )
        solve = make_quasistatic_solver(
            geometry,
            boundary,
            potential,
            QuasistaticOptions(tolerance=1e-11, max_iterations=100),
        )

        def response(offset):
            # Move one endpoint sideways; its rest length and endpoint shear
            # angles must be reinitialized for every shape candidate.
            vertices = geometry.vertices.at[1, 0].add(offset)
            result = solve(initialize(geometry, vertices), jnp.array([0.035]))
            return result.states[-1, 1, 1]

        offset, step = 0.12, 1e-5
        analytic = jax.grad(response)(offset)
        numerical = (response(offset + step) - response(offset - step)) / (2 * step)
        self.assertGreater(abs(float(analytic)), 1e-4)
        np.testing.assert_allclose(analytic, numerical, rtol=2e-4, atol=2e-7)


class VerletTests(unittest.TestCase):
    def test_external_force_requires_full_coordinate_shape(self):
        geometry, boundary, potential, reference = vertical_spring()
        for wrong_shape in ((), (2, 1)):
            with self.subTest(shape=wrong_shape):
                solve = make_verlet_solver(
                    geometry,
                    boundary,
                    potential,
                    external_force=lambda state, time: jnp.ones(wrong_shape),
                )
                with self.assertRaises(ValueError):
                    solve(reference, jnp.zeros(2))

    def test_boundary_must_match_geometry_shape(self):
        geometry, _, potential, _ = vertical_spring()
        boundary = BoundaryConditions(
            jnp.zeros((3, 3)),
            jnp.ones((3, 3), dtype=bool),
            jnp.zeros((3, 3)),
        )
        with self.assertRaises(ValueError):
            make_verlet_solver(geometry, boundary, potential)

    def oscillator(self, time_step, damping=0.0):
        geometry, boundary, potential, reference = vertical_spring()
        solve = make_verlet_solver(
            geometry,
            boundary,
            potential,
            masses=2.0,
            options=VerletOptions(time_step=time_step, damping=damping),
        )
        state = geometry.reference_state.at[1, 1].add(0.07)
        velocity = jnp.zeros_like(state).at[1, 1].set(0.06)
        samples = round(4.0 / time_step) + 1
        return solve(reference, jnp.zeros(samples), state, velocity), boundary

    def test_harmonic_accuracy_and_energy_error_decrease_quadratically(self):
        errors, energy_errors = [], []
        for time_step in (0.04, 0.02):
            result, boundary = self.oscillator(time_step)
            omega = np.sqrt(4.0 / 2.0)
            times = np.asarray(result.times)
            exact = 1.0 + 0.07 * np.cos(omega * times) + 0.06 / omega * np.sin(omega * times)
            errors.append(float(np.max(np.abs(np.asarray(result.states[:, 1, 1]) - exact))))
            total = np.asarray(result.potential_energies + result.kinetic_energies)
            energy_errors.append(float(np.max(np.abs(total - total[0]))))
            mask = np.asarray(boundary.fixed_mask)
            expected = np.broadcast_to(boundary.reference_state, result.states.shape)
            np.testing.assert_array_equal(np.asarray(result.states)[:, mask], expected[:, mask])
            np.testing.assert_array_equal(np.asarray(result.velocities)[:, mask], 0.0)
        self.assertLess(errors[1], errors[0] / 3.5)
        self.assertLess(energy_errors[1], energy_errors[0] / 3.5)
        self.assertLess(energy_errors[1], 5e-6)

    def test_damping_dissipates_energy(self):
        result, _ = self.oscillator(0.005, damping=0.8)
        total = np.asarray(result.potential_energies + result.kinetic_energies)
        self.assertLess(total[-1], 0.1 * total[0])
        # Splitting and Verlet have small integration errors; large positive
        # increments would indicate damping has the wrong sign or application.
        self.assertLess(float(np.max(np.diff(total))), 1e-9)

    def test_uniformly_driven_grip_and_free_body_translate_together(self):
        geometry, boundary, potential, reference = vertical_spring()
        direction = np.zeros((2, 3))
        direction[0, 1] = 0.3
        boundary = BoundaryConditions(boundary.reference_state, boundary.fixed_mask, direction)
        dt = 0.01
        times = jnp.arange(51) * dt
        speed = 0.3 * 0.4
        velocity = jnp.zeros((2, 3)).at[:, 1].set(speed)
        solve = make_verlet_solver(
            geometry,
            boundary,
            potential,
            options=VerletOptions(time_step=dt),
        )
        result = solve(reference, 0.4 * times, initial_velocity=velocity)
        expected = np.asarray(geometry.reference_state)[None] + np.asarray(times)[
            :, None, None
        ] * np.array([[0.0, speed, 0.0], [0.0, speed, 0.0]])
        np.testing.assert_allclose(result.states, expected, atol=2e-13)
        np.testing.assert_allclose(result.velocities[:, :, 1], speed, atol=2e-13)
        np.testing.assert_allclose(result.potential_energies, 0.0, atol=1e-24)

    def test_nonuniform_drive_prescribes_positions_and_sampled_velocities(self):
        geometry, boundary, potential, reference = vertical_spring()
        direction = np.zeros((2, 3))
        direction[0] = [0.1, 0.3, -0.2]
        boundary = BoundaryConditions(boundary.reference_state, boundary.fixed_mask, direction)
        dt = 0.02
        strains = jnp.array([0.01, 0.02, -0.01, 0.03, 0.025])
        solve = make_verlet_solver(
            geometry,
            boundary,
            potential,
            options=VerletOptions(time_step=dt, damping=0.2),
        )
        corrupted_fixed_state = geometry.reference_state.at[0].add(2.0)
        result = solve(reference, strains, initial_state=corrupted_fixed_state)
        expected = np.asarray(jax.vmap(boundary.state_at)(strains))
        mask = np.asarray(boundary.fixed_mask)
        np.testing.assert_array_equal(np.asarray(result.states)[:, mask], expected[:, mask])
        rates = np.r_[np.diff(np.asarray(strains))[:1], np.diff(np.asarray(strains))] / dt
        expected_velocity = rates[:, None, None] * direction
        np.testing.assert_allclose(
            np.asarray(result.velocities)[:, mask], expected_velocity[:, mask], atol=1e-14
        )


if __name__ == "__main__":
    unittest.main()
