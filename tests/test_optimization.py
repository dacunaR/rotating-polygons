"""Analytical checks of measurement, restrictions, and optimization adapters."""

import unittest
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np

from rotating_polygons.boundary import BoundaryConditions
from rotating_polygons.constraints import (
    corner_sines,
    ligament_lengths,
    shape_constraints,
)
from rotating_polygons.geometry import square_grid
from rotating_polygons.objectives import (
    make_stress_measurement,
    nominal_stress,
    reaction_forces,
    stress_curve_loss,
)
from rotating_polygons.optimization import minimize_design

jax.config.update("jax_enable_x64", True)


class MeasurementTests(unittest.TestCase):
    def setUp(self):
        reference = np.array([[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        direction = np.zeros_like(reference)
        direction[1, 1] = -2.0
        self.boundary = BoundaryConditions(
            reference, np.ones_like(reference, dtype=bool), direction, height=2.0, width=3.0
        )
        self.strains = jnp.array([0.0, 0.1, 0.2])
        self.states = jax.vmap(self.boundary.state_at)(self.strains)
        self.energy = lambda state, stiffness: stiffness * (state[1, 1] - 2.0) ** 2 / 2

    def test_reaction_sign_and_nominal_stress_units(self):
        reactions = reaction_forces(self.energy, self.states, 5.0)
        np.testing.assert_allclose(reactions[:, 1, 1], -10 * self.strains, atol=1e-14)
        stress = nominal_stress(self.energy, self.boundary, self.states, 5.0, thickness=2)
        np.testing.assert_allclose(stress, 5 * self.strains / 3, atol=1e-14)
        scalar = nominal_stress(self.energy, self.boundary, self.states[1], 5.0, thickness=2)
        self.assertAlmostEqual(float(scalar), float(stress[1]))

    def test_stress_matches_work_conjugate_derivative(self):
        strain = 0.1
        step = 1e-6

        def energy(epsilon):
            return self.energy(self.boundary.state_at(epsilon), 5.0)

        difference = (energy(strain + step) - energy(strain - step)) / (2 * step)
        stress = nominal_stress(self.energy, self.boundary, self.boundary.state_at(strain), 5.0)
        self.assertAlmostEqual(float(stress), float(difference / 6), places=9)

    def make_loss(self, converged=True):
        def simulate(stiffness, strains):
            return SimpleNamespace(
                states=jax.vmap(self.boundary.state_at)(strains),
                converged=jnp.full(strains.shape, converged),
            )

        return stress_curve_loss(
            simulate,
            lambda stiffness: stiffness,
            make_stress_measurement(self.energy, self.boundary, thickness=2),
            self.strains,
            2 * self.strains,
            scale=1,
        )

    def test_curve_loss_has_correct_design_derivative(self):
        loss = self.make_loss()
        stiffness = 3.0
        expected = jnp.mean(((stiffness / 3 - 2) * self.strains) ** 2)
        expected_gradient = jnp.mean(2 * (stiffness / 3 - 2) * self.strains**2 / 3)
        self.assertAlmostEqual(float(loss(stiffness)), float(expected))
        self.assertAlmostEqual(float(jax.grad(loss)(stiffness)), float(expected_gradient))

    def test_unconverged_curve_is_rejected(self):
        loss = self.make_loss(converged=False)
        self.assertTrue(np.isinf(loss(3.0)))
        with self.assertRaises(FloatingPointError):
            minimize_design(loss, 3.0)

    def test_invalid_measurement_inputs(self):
        for thickness in (0, -1, np.nan, np.inf):
            with self.subTest(thickness=thickness):
                with self.assertRaises(ValueError):
                    make_stress_measurement(self.energy, self.boundary, thickness=thickness)
                with self.assertRaises(ValueError):
                    nominal_stress(
                        self.energy, self.boundary, self.states, 5.0, thickness=thickness
                    )
        with self.assertRaises(ValueError):
            stress_curve_loss(None, None, None, [0.0, 0.1], [0.0])
        with self.assertRaises(ValueError):
            stress_curve_loss(None, None, None, [0.0], [0.0], scale=0)


class OptimizationTests(unittest.TestCase):
    def test_invalid_bounds_are_rejected_before_optimization(self):
        for bounds in (
            [(np.nan, np.nan)],
            [(0.0, np.nan)],
            [(2.0, 1.0)],
            [(0, 1), (0, 1)],
        ):
            with self.subTest(bounds=bounds):
                with self.assertRaises(ValueError):
                    minimize_design(
                        lambda x: jnp.sum((x - 1) ** 2), jnp.array([0.5]), bounds=bounds
                    )

    def test_constrained_quadratic_optimum(self):
        result = minimize_design(
            lambda parameters: jnp.sum((parameters - 2.0) ** 2),
            jnp.array([0.2, 0.3]),
            bounds=[(0.0, 3.0), (0.0, 3.0)],
            constraints=(lambda parameters: 1.0 - jnp.sum(parameters),),
            tol=1e-9,
        )
        self.assertTrue(result.success, result.message)
        self.assertTrue(result.feasible)
        np.testing.assert_allclose(result.x, [0.5, 0.5], atol=1e-7)
        self.assertLess(result.fun, result.initial_loss)
        self.assertGreaterEqual(len(result.loss_history), 2)
        self.assertLessEqual(result.max_constraint_violation, 1e-9)

    def test_pytree_parameters_are_restored(self):
        result = minimize_design(
            lambda parameters: (
                (parameters["stiffness"] - 2.0) ** 2
                + jnp.sum((parameters["shape"] - jnp.array([1.0, -1.0])) ** 2)
            ),
            {"stiffness": 1, "shape": jnp.zeros(2)},
        )
        self.assertTrue(result.success, result.message)
        self.assertAlmostEqual(float(result.parameters["stiffness"]), 2.0)
        np.testing.assert_allclose(result.parameters["shape"], [1.0, -1.0], atol=1e-9)

    def test_infeasibility_is_reported(self):
        result = minimize_design(
            lambda parameters: jnp.sum(parameters**2),
            jnp.ones(1),
            constraints=(lambda parameters: jnp.array([-1.0]),),
            maxiter=5,
        )
        self.assertFalse(result.success)
        self.assertFalse(result.feasible)
        self.assertEqual(result.max_constraint_violation, 1.0)

    def test_nonfinite_objective_is_an_error(self):
        with self.assertRaises(FloatingPointError):
            minimize_design(lambda parameters: jnp.nan * parameters.sum(), jnp.zeros(1))

    def test_invalid_settings(self):
        for options in ({"maxiter": 0}, {"tol": 0.0}, {"tol": np.nan}):
            with self.assertRaises(ValueError):
                minimize_design(lambda parameters: parameters.sum(), jnp.zeros(1), **options)


class ShapeConstraintTests(unittest.TestCase):
    def test_invalid_minimum_lengths_cannot_disable_constraints(self):
        geometry = square_grid(2, 2)
        for name in ("min_edge_length", "min_ligament_length"):
            for value in (-1, np.nan, np.inf):
                with self.subTest(parameter=name, value=value):
                    with self.assertRaises(ValueError):
                        shape_constraints(geometry, geometry.vertices, **{name: value})

    def test_square_geometry_margins_and_gap(self):
        geometry = square_grid(2, 2, hinge_gap=0.2)
        np.testing.assert_allclose(corner_sines(geometry, geometry.vertices), 1.0)
        np.testing.assert_allclose(ligament_lengths(geometry, geometry.vertices), 0.2)
        margins = shape_constraints(geometry, geometry.vertices, min_ligament_length=0.1)
        self.assertTrue(np.all(margins > 0))
        gradient = jax.jacrev(lambda vertices: shape_constraints(geometry, vertices))(
            geometry.vertices
        )
        self.assertTrue(np.all(np.isfinite(gradient)))

    def test_degenerate_candidate_has_finite_restrictions(self):
        geometry = square_grid(1, 1)
        collapsed = jnp.zeros_like(geometry.vertices)
        margins = shape_constraints(geometry, collapsed)
        derivative = jax.jacrev(lambda vertices: shape_constraints(geometry, vertices))(collapsed)
        self.assertTrue(np.all(margins < 0))
        self.assertTrue(np.all(np.isfinite(derivative)))


if __name__ == "__main__":
    unittest.main()
