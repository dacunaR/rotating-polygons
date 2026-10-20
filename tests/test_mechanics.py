"""Physical and independent numerical checks of the extracted energy models."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from rotating_polygons.energy import PotentialParameters, make_potential
from rotating_polygons.geometry import from_polygons, square_grid
from rotating_polygons.initialization import initialize

jax.config.update("jax_enable_x64", True)


def numpy_rotation(vector, angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([cosine * vector[0] - sine * vector[1], sine * vector[0] + cosine * vector[1]])


def numpy_angle(first, second):
    return np.arctan2(
        first[0] * second[1] - first[1] * second[0],
        np.dot(first, second),
    )


def notebook_bond_energies(geometry, vertices, state, coefficients):
    """Independent scalar-loop transcription of Giovanni2 at zero rest angles.

    This deliberately does not call package angle, rotation, or initialization
    helpers. The selected test state stays away from angular branch cuts.
    """
    centers = np.asarray(geometry.reference_state)[:, :2]
    owners = np.asarray(geometry.vertex_polygons)
    vertices = np.asarray(vertices)
    state = np.asarray(state)
    totals, contacts = [], []
    for bond, (first, second) in enumerate(np.asarray(geometry.bonds)):
        polygon_i, polygon_j = owners[[first, second]]
        arm_i, arm_j = vertices[[first, second]]
        rest_gap = centers[polygon_i] - centers[polygon_j] + arm_i - arm_j
        rest_length = np.linalg.norm(rest_gap)
        rest_angles = np.array([numpy_angle(-rest_gap, arm_i), numpy_angle(rest_gap, arm_j)])
        opening_1 = numpy_angle(
            vertices[int(geometry.previous_vertices[second])] - arm_j,
            vertices[int(geometry.next_vertices[first])] - arm_i,
        )
        opening_2 = numpy_angle(
            vertices[int(geometry.previous_vertices[first])] - arm_i,
            vertices[int(geometry.next_vertices[second])] - arm_j,
        )
        openings = (np.array([opening_1, opening_2]) + np.pi / 2) % (2 * np.pi) - np.pi / 2
        theta_i, theta_j = state[[polygon_i, polygon_j], 2]
        rotated_i = numpy_rotation(arm_i, theta_i)
        rotated_j = numpy_rotation(arm_j, theta_j)
        gap = state[polygon_i, :2] - state[polygon_j, :2] + rotated_i - rotated_j
        shear = np.array([numpy_angle(-gap, rotated_i), numpy_angle(gap, rotated_j)]) - rest_angles
        if np.any(np.abs(shear) >= np.pi):
            raise AssertionError("The reference comparison crossed an angle branch cut")
        relative_angle = theta_i - theta_j
        openings += np.array([relative_angle, -relative_angle])
        openings += coefficients.contact_angle_offset
        contact = np.asarray(coefficients.contact)[bond] * np.sum(np.minimum(openings, 0.0) ** 2)
        total = (
            np.asarray(coefficients.stretching)[bond] * (np.linalg.norm(gap) - rest_length) ** 2
            + np.asarray(coefficients.shearing)[bond] * rest_length**2 * np.dot(shear, shear)
            + np.asarray(coefficients.bending)[bond] * relative_angle**2
            + contact
        ) / 2
        totals.append(total)
        contacts.append(contact / 2)
    return np.asarray(totals), np.asarray(contacts)


class MechanicsTests(unittest.TestCase):
    def test_reference_is_stress_free_with_nonzero_initial_rotations(self):
        base = square_grid(2, 2, hinge_gap=0.35)
        polygons = [
            np.asarray(base.vertices)[start:stop]
            for start, stop in zip(base.polygon_offsets[:-1], base.polygon_offsets[1:])
        ]
        geometry = from_polygons(
            np.asarray(base.reference_state)[:, :2],
            polygons,
            base.bonds,
            angles=[0.15, -0.12, 0.07, 0.21],
        )
        reference = initialize(geometry)
        energy = make_potential(
            geometry, PotentialParameters(stretching=2.3, shearing=0.4, bending=0.17)
        )
        self.assertAlmostEqual(float(energy(geometry.reference_state, reference)), 0.0, places=13)
        np.testing.assert_allclose(
            jax.grad(energy)(geometry.reference_state, reference), 0.0, atol=1e-12
        )

    def test_energy_is_invariant_under_common_rigid_motion(self):
        geometry = square_grid(2, 2, hinge_gap=0.3)
        reference = initialize(geometry)
        energy = make_potential(
            geometry, PotentialParameters(stretching=2.0, shearing=0.7, bending=0.2)
        )
        state = geometry.reference_state + jnp.asarray(
            [[0.03, -0.02, 0.14], [-0.02, 0.04, -0.08], [0.01, -0.03, 0.09], [0.03, 0.01, -0.12]]
        )
        angle = 2.9  # Also moves several endpoint angles across atan2's cut.
        rotated_centers = np.stack(
            [numpy_rotation(center, angle) for center in np.asarray(state[:, :2])]
        )
        transformed = state.at[:, :2].set(rotated_centers + [3.2, -4.7])
        transformed = transformed.at[:, 2].add(angle)
        np.testing.assert_allclose(
            energy(transformed, reference),
            energy(state, reference),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_finite_hinge_matches_latest_notebook_energy_with_contact_once(self):
        geometry = square_grid(2, 2, hinge_gap=0.3)
        vertices = geometry.vertices.at[0].add(jnp.array([0.03, 0.01]))
        reference = initialize(geometry, vertices)
        parameters = PotentialParameters(
            stretching=np.array([2.1, 2.4, 2.7, 3.0]),
            shearing=np.array([0.3, 0.5, 0.7, 0.9]),
            bending=np.array([0.11, 0.13, 0.17, 0.19]),
            contact=np.array([3.0, 4.0, 5.0, 6.0]),
            contact_angle_offset=0.02,
        )
        state = (
            geometry.reference_state.at[:, 2]
            .set(jnp.array([-1.8, 0.11, -0.17, 0.08]))
            .at[0, :2]
            .add(jnp.array([0.02, 0.01]))
        )
        expected, contacts = notebook_bond_energies(geometry, vertices, state, parameters)
        self.assertGreater(float(contacts.sum()), 0.01)
        actual = make_potential(geometry, parameters)(state, reference)
        np.testing.assert_allclose(actual, expected.sum(), rtol=1e-12, atol=1e-12)
        # Independently isolate each bond through per-bond coefficients.
        for index, expected_bond in enumerate(expected):
            mask = np.arange(geometry.n_bonds) == index
            isolated = PotentialParameters(
                stretching=parameters.stretching * mask,
                shearing=parameters.shearing * mask,
                bending=parameters.bending * mask,
                contact=parameters.contact * mask,
                contact_angle_offset=parameters.contact_angle_offset,
            )
            np.testing.assert_allclose(
                make_potential(geometry, isolated)(state, reference),
                expected_bond,
                rtol=1e-12,
                atol=1e-12,
            )

    def test_force_and_shape_derivatives_agree_with_finite_differences(self):
        geometry = square_grid(2, 1, hinge_gap=0.3)
        energy = make_potential(
            geometry, PotentialParameters(stretching=2.0, shearing=0.6, bending=0.13)
        )
        state = geometry.reference_state + jnp.array([[0.03, -0.02, 0.09], [-0.04, 0.03, -0.07]])
        reference = initialize(geometry)
        direction = jnp.array([[0.2, -0.3, 0.5], [-0.4, 0.7, -0.1]])
        step = 1e-6
        analytic = jnp.sum(jax.grad(energy)(state, reference) * direction)
        numerical = (
            energy(state + step * direction, reference)
            - energy(state - step * direction, reference)
        ) / (2 * step)
        np.testing.assert_allclose(analytic, numerical, rtol=1e-7, atol=1e-9)

        def shape_energy(offset):
            vertices = geometry.vertices.at[0, 1].add(offset)
            return energy(state, initialize(geometry, vertices))

        analytic = jax.grad(shape_energy)(0.04)
        numerical = (shape_energy(0.04 + step) - shape_energy(0.04 - step)) / (2 * step)
        self.assertGreater(abs(float(analytic)), 1e-4)
        np.testing.assert_allclose(analytic, numerical, rtol=1e-7, atol=1e-9)

    def test_point_hinge_has_finite_derivatives_at_coincident_endpoints(self):
        geometry = square_grid(2, 1, hinge_gap=0.0)
        reference = initialize(geometry)
        energy = make_potential(
            geometry,
            PotentialParameters(stretching=2.0, shearing=0.8),
            model="point_hinge",
        )
        state = geometry.reference_state
        self.assertAlmostEqual(float(energy(state, reference)), 0.0, places=13)
        gradient = jax.grad(energy)(state, reference)
        hessian = jax.jacfwd(jax.grad(energy))(state, reference)
        self.assertTrue(np.all(np.isfinite(hessian)))
        np.testing.assert_allclose(gradient, 0.0, atol=1e-13)
        displaced = state.at[1, 0].add(0.1)
        # One connector: E = k_stretch * displacement² / 2.
        self.assertAlmostEqual(float(energy(displaced, reference)), 0.01, places=12)
        np.testing.assert_allclose(
            jax.grad(energy)(displaced, reference)[:, 0], [-0.2, 0.2], atol=1e-12
        )


if __name__ == "__main__":
    unittest.main()
