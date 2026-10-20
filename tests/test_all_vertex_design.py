"""The full-design example must not tie or freeze any vertex coordinate."""

import importlib.util
from pathlib import Path
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from rotating_polygons import compression_boundary, square_grid

jax.config.update("jax_enable_x64", True)

# Import the example's pure shape mapping without constructing its simulation.
example_path = Path(__file__).resolve().parents[1] / "examples" / "optimize_all_vertices.py"
spec = importlib.util.spec_from_file_location("all_vertex_example", example_path)
example = importlib.util.module_from_spec(spec)
spec.loader.exec_module(example)


class AllVertexDesignTests(unittest.TestCase):
    def test_full_design_jacobian_is_identity(self):
        geometry = square_grid(3, 3)
        offsets = jnp.zeros((9, 4, 2))
        jacobian = jax.jacrev(
            lambda parameters: example.vertices_from_offsets(geometry, parameters).ravel()
        )(offsets).reshape((72, 72))
        # Identity proves every input changes exactly its own x/y coordinate:
        # full rank 72, no symmetry sharing, and no deleted design variables.
        np.testing.assert_array_equal(jacobian, np.eye(72))

    def test_single_offsets_include_clamped_and_unconnected_corners(self):
        geometry = square_grid(3, 3)
        boundary = compression_boundary(geometry)
        self.assertTrue(np.all(boundary.fixed_mask[0]))
        self.assertTrue(np.all(boundary.fixed_mask[8]))
        self.assertNotIn(4 * 0 + 3, np.asarray(geometry.bonds).ravel())
        self.assertNotIn(4 * 8 + 1, np.asarray(geometry.bonds).ravel())
        for polygon, vertex, coordinate in ((0, 3, 1), (8, 1, 0), (4, 0, 1)):
            with self.subTest(polygon=polygon, vertex=vertex, coordinate=coordinate):
                offsets = jnp.zeros((9, 4, 2)).at[polygon, vertex, coordinate].set(0.017)
                actual = example.vertices_from_offsets(geometry, offsets) - geometry.vertices
                expected = np.zeros((36, 2))
                expected[4 * polygon + vertex, coordinate] = 0.017
                np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0)


if __name__ == "__main__":
    unittest.main()
