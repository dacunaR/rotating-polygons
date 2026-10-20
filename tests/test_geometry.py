"""Geometry regressions using generated fixtures, independent of old notebooks."""

from pathlib import Path
import tempfile
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from rotating_polygons.geometry import (
    Geometry,
    from_polygons,
    rotate_vectors,
    square_grid,
    world_vertices,
)
from rotating_polygons.io import load_geometry, read_legacy_dat, save_geometry


def torus_text(nx=3, ny=3):
    """An Auxetic-format lattice, including bonds across the periodic cell."""
    rows = [f"{nx} {ny} -1.1 -1.1"]
    for y in range(ny):
        for x in range(nx):
            neighbors = [
                y * nx + (x + 1) % nx,
                ((y + 1) % ny) * nx + x,
                y * nx + (x - 1) % nx,
                ((y - 1) % ny) * nx + x,
            ]
            rows.append(" ".join(map(str, [x, y, *neighbors, -1, -1])))
    return "\n".join(rows) + "\n"


class GeometryTests(unittest.TestCase):
    def test_grid_vertex_positions_and_hinge_gap(self):
        geometry = square_grid(3, 2, spacing=2.0, hinge_gap=0.4)
        self.assertEqual((geometry.n_polygons, geometry.n_vertices, geometry.n_bonds), (6, 24, 7))
        points = np.asarray(world_vertices(geometry, geometry.reference_state))
        np.testing.assert_allclose(points[:4], [[0.8, 0], [0, 0.8], [-0.8, 0], [0, -0.8]])
        np.testing.assert_allclose(geometry.reference_state[-1], [4, 2, 0])
        separation = (
            points[np.asarray(geometry.bonds)[:, 0]] - points[np.asarray(geometry.bonds)[:, 1]]
        )
        np.testing.assert_allclose(np.linalg.norm(separation, axis=1), 0.4, rtol=1e-6)

    def test_mixed_polygons_preserve_local_adjacency(self):
        triangle = [[0, 1], [-1, -1], [1, -1]]
        square = [[1, 0], [0, 1], [-1, 0], [0, -1]]
        geometry = from_polygons([[0, 0], [3, 0]], [triangle, square], [[2, 5]])
        self.assertEqual(geometry.polygon_offsets, (0, 3, 7))
        np.testing.assert_array_equal(geometry.previous_vertices, [2, 0, 1, 6, 3, 4, 5])
        np.testing.assert_array_equal(geometry.next_vertices, [1, 2, 0, 4, 5, 6, 3])
        np.testing.assert_array_equal(geometry.vertex_polygons, [0, 0, 0, 1, 1, 1, 1])

    def test_rotation_and_world_vertices_are_differentiable(self):
        np.testing.assert_allclose(rotate_vectors([1.0, 0.0], np.pi / 2), [0, 1], atol=1e-6)
        geometry = square_grid(1, 1)
        state = jnp.array([[2.0, 3.0, np.pi / 2]])
        transformed = jax.jit(lambda q: world_vertices(geometry, q))(state)
        np.testing.assert_allclose(transformed[0], [2, 3.4], atol=1e-6)
        # A rotation preserves squared arm lengths; its derivative is 2 * arm.
        gradient = jax.grad(
            lambda arms: jnp.sum((world_vertices(geometry, state, arms) - state[0, :2]) ** 2)
        )(geometry.vertices)
        np.testing.assert_allclose(gradient, 2 * geometry.vertices, atol=1e-6)
        self.assertEqual(geometry.bonds.shape, (0, 2))

    def test_invalid_inputs_fail_explicitly(self):
        for parameters in [(0, 2), (1.5, 2), (True, 2)]:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                square_grid(*parameters)
        with self.assertRaises(ValueError):
            square_grid(2, 2, hinge_gap=1)
        with self.assertRaisesRegex(ValueError, "counterclockwise"):
            from_polygons([[0, 0]], [[[0, 1], [1, -1], [-1, -1]]], [])
        with self.assertRaisesRegex(ValueError, "finite"):
            from_polygons([[np.nan, 0]], [[[0, 1], [-1, -1], [1, -1]]], [])
        geometry = square_grid(2, 1)
        values = dict(
            reference_state=geometry.reference_state,
            vertices=geometry.vertices,
            vertex_polygons=geometry.vertex_polygons,
            polygon_offsets=geometry.polygon_offsets,
            bonds=geometry.bonds,
        )
        for bonds, message in [
            ([[0, 8]], "outside"),
            ([[0, 1]], "different polygons"),
            ([[0.5, 4]], "integer"),
            ([[0, 4], [4, 0]], "Duplicate"),
            ([[0, 2**32 + 4]], "32-bit"),
        ]:
            with self.subTest(bonds=bonds), self.assertRaisesRegex(ValueError, message):
                Geometry(**{**values, "bonds": bonds})


class GeometryIOTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "fixture.dat"
        self.path.write_text(torus_text(), encoding="utf-8")

    def test_cutting_bonds_retains_complete_polygon_outlines(self):
        for periodic, n_bonds in [((False, False), 12), ((True, False), 15), ((True, True), 18)]:
            with self.subTest(periodic=periodic):
                geometry = read_legacy_dat(self.path, periodic=periodic)
                self.assertEqual(geometry.n_polygons, 9)
                self.assertEqual(geometry.n_vertices, 36)
                self.assertEqual(geometry.n_bonds, n_bonds)
                points = np.asarray(world_vertices(geometry, geometry.reference_state))
                separation = (
                    points[np.asarray(geometry.bonds)[:, 0]]
                    - points[np.asarray(geometry.bonds)[:, 1]]
                    + geometry.image_shifts
                )
                np.testing.assert_allclose(np.linalg.norm(separation, axis=1), 0.2, atol=1e-6)
                if not any(periodic):
                    np.testing.assert_array_equal(geometry.image_shifts, 0)

    def test_malformed_adjacency_is_rejected(self):
        rows = torus_text().splitlines()
        rows[1] = "0 0 4 3 2 6 -1 -1"
        self.path.write_text("\n".join(rows), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "reciprocal"):
            read_legacy_dat(self.path)
        rows[1] = "0 0 99 3 2 6 -1 -1"
        self.path.write_text("\n".join(rows), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            read_legacy_dat(self.path)
        self.path.write_text(torus_text(nx=2), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate neighbor"):
            read_legacy_dat(self.path)

    def test_portable_archive_roundtrip(self):
        for geometry in [
            square_grid(1, 1),
            read_legacy_dat(self.path, periodic=(True, False)),
        ]:
            with self.subTest(cell_size=geometry.cell_size):
                archive_path = Path(self.directory.name) / "geometry_without_extension"
                save_geometry(archive_path, geometry)
                self.assertTrue(archive_path.exists())
                restored = load_geometry(archive_path)
                self.assertEqual(restored.cell_size, geometry.cell_size)
                self.assertEqual(restored.polygon_offsets, geometry.polygon_offsets)
                for name in ["reference_state", "vertices", "bonds", "image_shifts"]:
                    np.testing.assert_array_equal(getattr(restored, name), getattr(geometry, name))
                with np.load(archive_path, allow_pickle=False) as archive:
                    self.assertTrue(all(archive[key].dtype != object for key in archive.files))

    def test_unknown_archive_version_is_rejected(self):
        path = Path(self.directory.name) / "unsupported.npz"
        np.savez(path, format_version=1.5)
        with self.assertRaisesRegex(ValueError, "version"):
            load_geometry(path)


if __name__ == "__main__":
    unittest.main()
