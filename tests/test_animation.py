"""Animation regressions for geometry, exact frame selection, and file export."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import jax
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from rotating_polygons.animation import create_animation, save_video
from rotating_polygons.geometry import Geometry, square_grid, world_vertices


class AnimationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        jax.config.update("jax_enable_x64", True)

    def setUp(self):
        self.geometry = square_grid(2, 1)
        self.states = np.repeat(np.asarray(self.geometry.reference_state)[None], 5, axis=0)
        self.states[:, :, 0] += np.linspace(0, 0.3, 5)[:, None]
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.addCleanup(plt.close, "all")

    def test_candidate_vertices_periodic_bonds_and_fixed_limits(self):
        geometry = Geometry(
            reference_state=self.geometry.reference_state,
            vertices=self.geometry.vertices,
            vertex_polygons=self.geometry.vertex_polygons,
            polygon_offsets=self.geometry.polygon_offsets,
            bonds=self.geometry.bonds,
            image_shifts=[[2.0, 0.0]],
            cell_size=(2.0, 1.0),
        )
        vertices = np.asarray(geometry.vertices) * 0.75
        figure, axis = plt.subplots()
        animation = create_animation(
            geometry,
            self.states,
            vertices=vertices,
            ax=axis,
            frame_labels=[f"sample {i}" for i in range(5)],
            stride=2,
        )
        figure.canvas.draw()  # Start drawing so Matplotlib knows it was rendered.
        first_limits = (axis.get_xlim(), axis.get_ylim())
        first_points = np.asarray(world_vertices(geometry, self.states[0], vertices))
        np.testing.assert_allclose(
            axis.collections[0].get_paths()[0].vertices[:4], first_points[:4]
        )
        segment = axis.collections[1].get_segments()[0]
        bond = np.asarray(geometry.bonds)[0]
        np.testing.assert_allclose(segment[0], first_points[bond[0]])
        np.testing.assert_allclose(segment[1], first_points[bond[1]] - [2, 0])
        self.assertLess(first_limits[0][0], segment[1, 0])
        self.assertEqual(list(animation.new_frame_seq()), [0, 2, 4])
        # Render the last selected physical state, rather than an interpolated one.
        animation._func(4)
        self.assertEqual((axis.get_xlim(), axis.get_ylim()), first_limits)
        self.assertEqual(axis.get_title(), "sample 4")
        final_points = np.asarray(world_vertices(geometry, self.states[4], vertices))
        np.testing.assert_allclose(
            axis.collections[0].get_paths()[0].vertices[:4], final_points[:4]
        )

    def test_save_gif_contains_selected_frames_and_playback_duration(self):
        path = Path(self.directory.name) / "nested" / "trajectory.gif"
        written = save_video(
            self.geometry, self.states, path, fps=5, stride=2, figsize=(2, 2), dpi=50
        )
        self.assertEqual(written, path)
        with Image.open(path) as movie:
            self.assertEqual(movie.n_frames, 3)
            self.assertEqual(movie.size, (100, 100))
            self.assertEqual(movie.info["duration"], 200)
            movie.seek(2)
            last_frame = np.asarray(movie.convert("RGB"))
            movie.seek(0)
            self.assertFalse(np.array_equal(last_frame, np.asarray(movie.convert("RGB"))))

    def test_missing_ffmpeg_falls_back_or_fails_explicitly(self):
        path = Path(self.directory.name) / "trajectory.mp4"
        with patch("matplotlib.animation.writers.is_available", return_value=False):
            with self.assertWarnsRegex(RuntimeWarning, "saving GIF"):
                written = save_video(self.geometry, self.states, path, figsize=(2, 2), dpi=40)
            self.assertEqual(written, path.with_suffix(".gif"))
            self.assertTrue(written.is_file())
            self.assertFalse(path.exists())
            with self.assertRaisesRegex(RuntimeError, "ffmpeg"):
                save_video(self.geometry, self.states, path, fallback_to_gif=False)

    def test_invalid_trajectory_and_options(self):
        for states in (self.states[0], self.states[:, :1], self.states[:0]):
            with self.subTest(shape=states.shape), self.assertRaisesRegex(ValueError, "shape"):
                create_animation(self.geometry, states)
        invalid = self.states.copy()
        invalid[1, 0, 0] = np.nan
        # A skipped frame is still part of the input and must be valid.
        with self.assertRaisesRegex(ValueError, "finite"):
            create_animation(self.geometry, invalid, stride=2)
        for options in (
            {"fps": 0},
            {"fps": float("nan")},
            {"stride": 0},
            {"stride": 1.5},
            {"stride": True},
            {"padding": -1},
            {"figsize": (0, 3)},
            {"vertices": np.zeros((2, 2))},
            {"frame_labels": ["only one"]},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                create_animation(self.geometry, self.states, **options)
        with self.assertRaisesRegex(ValueError, ".mp4 or .gif"):
            save_video(self.geometry, self.states, "unsupported.avi")

    def test_gif_memory_guard_prevents_large_frame_buffer(self):
        path = Path(self.directory.name) / "too_large.gif"
        with self.assertRaisesRegex(ValueError, "Increase stride"):
            save_video(
                self.geometry,
                self.states,
                path,
                dpi=200,
                figsize=(10, 10),
                gif_memory_limit_mb=1,
            )
        self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
