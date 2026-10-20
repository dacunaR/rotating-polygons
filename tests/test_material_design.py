"""Material coefficients remain explicit differentiable solver arguments."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from rotating_polygons import (
    BoundaryConditions,
    MechanicalParameters,
    PotentialParameters,
    QuasistaticOptions,
    VerletOptions,
    initialize,
    make_parameterized_potential,
    make_potential,
    make_quasistatic_solver,
    make_verlet_solver,
    square_grid,
)

jax.config.update("jax_enable_x64", True)


def rotating_pair():
    """Prescribe transverse displacement; let the second polygon rotate."""
    geometry = square_grid(2, 1, hinge_gap=0.35)
    mask = np.ones((2, 3), dtype=bool)
    mask[1, 2] = False
    direction = np.zeros((2, 3))
    direction[1, 1] = 0.2
    boundary = BoundaryConditions(geometry.reference_state, mask, direction)
    potential = make_parameterized_potential(geometry)
    solve = make_quasistatic_solver(
        geometry,
        boundary,
        potential,
        QuasistaticOptions(tolerance=1e-11, max_iterations=100),
    )
    return geometry, boundary, potential, solve


class MaterialDesignTests(unittest.TestCase):
    def test_parameterized_energy_matches_fixed_material_api(self):
        geometry = square_grid(2, 2, hinge_gap=0.3)
        reference = initialize(geometry)
        state = (
            geometry.reference_state.at[:, 2]
            .set(jnp.array([-1.8, 0.11, -0.17, 0.08]))
            .at[0, :2]
            .add(jnp.array([0.02, 0.01]))
        )
        for model in ("finite_hinge", "point_hinge"):
            for per_bond in (False, True):
                with self.subTest(model=model, per_bond=per_bond):
                    scale = np.arange(1, geometry.n_bonds + 1) if per_bond else 1.0
                    coefficients = dict(
                        stretching=2.0 * scale,
                        shearing=0.3 * scale,
                        bending=0.07 * scale,
                        contact=4.0 * scale,
                        contact_angle_offset=0.02,
                    )
                    fixed = make_potential(
                        geometry, PotentialParameters(**coefficients), model=model
                    )
                    parameterized = make_parameterized_potential(geometry, model=model)
                    material = MechanicalParameters(reference, **coefficients)
                    np.testing.assert_allclose(
                        jax.jit(parameterized)(state, material),
                        fixed(state, reference),
                        rtol=1e-13,
                        atol=1e-13,
                    )
                    np.testing.assert_allclose(
                        jax.grad(parameterized)(state, material),
                        jax.grad(fixed)(state, reference),
                        rtol=1e-13,
                        atol=1e-13,
                    )

    def test_jitted_bending_gradient_matches_direct_energy_derivative(self):
        geometry = square_grid(2, 2, hinge_gap=0.3)
        reference = initialize(geometry)
        potential = make_parameterized_potential(geometry)
        state = geometry.reference_state.at[:, 2].set(jnp.array([0.1, -0.2, 0.3, -0.1]))

        def energy(bending):
            parameters = MechanicalParameters(
                reference, stretching=2.0, shearing=0.4, bending=bending, contact=0.0
            )
            return potential(state, parameters)

        stiffnesses = jnp.array([0.04, 0.07, 0.11, 0.15])
        value, derivative = jax.jit(jax.value_and_grad(energy))(stiffnesses)
        polygon_pairs = np.asarray(geometry.vertex_polygons)[np.asarray(geometry.bonds)]
        angles = np.asarray(state)[:, 2]
        expected = 0.5 * (angles[polygon_pairs[:, 0]] - angles[polygon_pairs[:, 1]]) ** 2
        self.assertTrue(np.isfinite(float(value)))
        np.testing.assert_allclose(derivative, expected, rtol=1e-13, atol=1e-13)
        step = 1e-6
        direction = jnp.array([0.5, -0.2, 0.7, -0.3])
        finite_difference = (
            energy(stiffnesses + step * direction) - energy(stiffnesses - step * direction)
        ) / (2 * step)
        np.testing.assert_allclose(
            jnp.dot(derivative, direction), finite_difference, rtol=1e-8, atol=1e-10
        )

    def test_implicit_bending_gradient_includes_equilibrium_response(self):
        geometry, _, _, solve = rotating_pair()
        reference = initialize(geometry)

        def response(bending):
            parameters = MechanicalParameters(
                reference, stretching=2.0, shearing=0.3, bending=bending, contact=0.0
            )
            return solve(parameters, jnp.array([0.25])).states[-1, 1, 2]

        stiffnesses = jnp.array([0.12])
        parameters = MechanicalParameters(
            reference, stretching=2.0, shearing=0.3, bending=stiffnesses, contact=0.0
        )
        solve(parameters, jnp.array([0.25])).check_convergence()
        value, derivative = jax.jit(jax.value_and_grad(response))(stiffnesses)
        step = 1e-5
        finite_difference = (response(stiffnesses + step) - response(stiffnesses - step)) / (
            2 * step
        )
        self.assertGreater(abs(float(value)), 1e-3)
        self.assertGreater(abs(float(derivative[0])), 1e-3)
        np.testing.assert_allclose(derivative[0], finite_difference, rtol=2e-5, atol=2e-8)

    def test_joint_shape_and_positive_material_design_through_equilibrium(self):
        geometry, _, _, solve = rotating_pair()

        def response(design):
            vertices = geometry.vertices.at[0, 1].add(design["shape_offset"])
            parameters = MechanicalParameters(
                initialize(geometry, vertices),
                stretching=2.0,
                shearing=0.3,
                bending=jnp.exp(design["log_bending"]),
                contact=0.0,
            )
            return solve(parameters, jnp.array([0.25])).states[-1, 1, 2]

        design = {"shape_offset": jnp.array(0.03), "log_bending": jnp.log(0.12)}
        value, derivative = jax.jit(jax.value_and_grad(response))(design)
        self.assertTrue(np.isfinite(float(value)))
        step = 1e-5
        for name in design:
            with self.subTest(parameter=name):
                forward = dict(design)
                backward = dict(design)
                forward[name] += step
                backward[name] -= step
                finite_difference = (response(forward) - response(backward)) / (2 * step)
                self.assertGreater(abs(float(derivative[name])), 1e-4)
                np.testing.assert_allclose(
                    derivative[name], finite_difference, rtol=2e-4, atol=2e-8
                )

    def test_parameterized_material_is_compatible_with_verlet(self):
        geometry = square_grid(1, 3, hinge_gap=0.4)
        mask = np.ones((3, 3), dtype=bool)
        mask[1, 1] = False
        boundary = BoundaryConditions(geometry.reference_state, mask, np.zeros((3, 3)))
        reference = initialize(geometry)
        coefficients = dict(
            stretching=jnp.array([2.0, 6.0]),
            shearing=0.0,
            bending=jnp.array([0.01, 0.03]),
            contact=0.0,
        )
        fixed = make_verlet_solver(
            geometry,
            boundary,
            make_potential(geometry, PotentialParameters(**coefficients)),
            options=VerletOptions(time_step=0.01),
        )
        parameterized = make_verlet_solver(
            geometry,
            boundary,
            make_parameterized_potential(geometry),
            options=VerletOptions(time_step=0.01),
        )
        initial_state = geometry.reference_state.at[1, 1].add(0.03)
        strains = jnp.zeros(31)
        expected = fixed(reference, strains, initial_state)
        actual = jax.jit(parameterized)(
            MechanicalParameters(reference, **coefficients), strains, initial_state
        )
        for actual_field, expected_field in zip(actual, expected):
            np.testing.assert_allclose(actual_field, expected_field, rtol=1e-13, atol=1e-13)

    def test_runtime_coefficient_shapes_are_checked_without_value_conversion(self):
        geometry = square_grid(2, 2, hinge_gap=0.3)
        reference = initialize(geometry)
        potential = make_parameterized_potential(geometry)
        parameters = MechanicalParameters(reference)
        for name in ("stretching", "shearing", "bending", "contact", "contact_angle_offset"):
            for shape in ((3,), (4, 1)):
                with self.subTest(parameter=name, shape=shape):
                    malformed = parameters._replace(**{name: jnp.ones(shape)})
                    with self.assertRaisesRegex(ValueError, name):
                        jax.jit(potential)(geometry.reference_state, malformed)

        # All fields may be traced per-bond arrays, including contact offsets.
        def energy(scale):
            dynamic = MechanicalParameters(
                reference,
                stretching=jnp.ones(4) * scale,
                shearing=jnp.ones(4) * scale,
                bending=jnp.ones(4) * scale,
                contact=jnp.ones(4) * scale,
                contact_angle_offset=jnp.ones(4) * scale,
            )
            return potential(geometry.reference_state, dynamic)

        value, derivative = jax.jit(jax.value_and_grad(energy))(0.1)
        self.assertTrue(np.isfinite(float(value)))
        self.assertTrue(np.isfinite(float(derivative)))


if __name__ == "__main__":
    unittest.main()
