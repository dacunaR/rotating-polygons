# Validation and reproducibility

Validation performed on 20 October 2026, using Python 3.12 on Linux with CPU JAX
and 64-bit arithmetic. The package was tested in an isolated virtual environment;
the existing research environment was not modified.

## Reproduce the checks

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-tested.txt
python -m pip install -e ".[dev,nlopt,video]"
JAX_PLATFORMS=cpu pytest -q
ruff check src tests examples
ruff format --check src tests examples
MPLBACKEND=Agg python examples/optimize_square_grid.py --output outputs
MPLBACKEND=Agg python examples/boundary_conditions.py --output outputs/boundaries
MPLBACKEND=Agg python examples/optimize_objectives.py --objective all --output outputs
MPLBACKEND=Agg python examples/optimize_spring_constants.py --parameterization all --output outputs
MPLBACKEND=Agg python examples/optimize_all_vertices.py --output outputs
MPLBACKEND=Agg python examples/draw_vertex_design.py --output docs/assets
MPLBACKEND=Agg python examples/optimize_with_nlopt.py --output outputs
MPLBACKEND=Agg python examples/make_video.py --output outputs/compression.gif
MPLBACKEND=Agg python examples/make_video.py --output outputs/compression.mp4
MPLBACKEND=Agg python examples/verlet_compression.py --output outputs
python -m build
```

The tested numerical dependencies are pinned in
[`requirements-tested.txt`](../requirements-tested.txt). The broader dependency
ranges in `pyproject.toml` describe the intended installation range; the exact
versions in the pins are the versions used for these results.

## What the tests establish

| Area | Checks |
| --- | --- |
| Geometry and I/O | Mixed polygon sizes, connectivity, minimum-image import, cut-boundary vertices, malformed adjacency, numeric archive round trips |
| Energy | Independent NumPy evaluation of the selected notebook equation, one contact contribution, zero reference energy, finite point-hinge derivatives |
| Objectivity | Rigid translation/rotation invariance, angle-branch crossing, nonzero initial polygon rotations |
| Derivatives | Finite-difference forces, shape energy gradients, and implicit shape response after resolved equilibria |
| Quasistatics | Analytic spring and series-spring equilibria, BFGS and L-BFGS, reactions, exact prescribed coordinates, reproducible seeded perturbations |
| Dynamics | Harmonic-oscillator accuracy, second-order timestep error, bounded conservative energy error, damping dissipation, moving grips |
| Optimization | Analytic constrained optimum, PyTree parameters, feasibility diagnostics, nonfinite settings, stress normalization, failed-equilibrium rejection |
| Material design | Scalar/per-bond coefficient equivalence, JIT material gradients, implicitly differentiated bending response, joint shape/material derivatives, parameterized Verlet |
| Independent vertex design | Full-rank identity map from all 72 offsets to vertex coordinates; single-corner updates including clamped polygons and unconnected outline corners |
| NLopt | SLSQP/MMA analytic optima, nonlinear inequality signs and rectangular Jacobians, PyTree ordering, optional dependency handling, feasibility and evaluation/time-limit diagnostics, callback exceptions |
| Animation | Encoded GIF frames, candidate vertices, periodic image convention, fixed camera limits, subsampling, labels, missing-ffmpeg fallback, invalid-input and GIF-memory checks |

The full test suite passes: **69 tests and 62 subtests**. Static linting and
formatting checks also pass.
The direct energy comparison checks the finite-hinge formula independently of
the vectorized implementation, including an active contact case. It does not
execute an entire original notebook.

## End-to-end example

`examples/optimize_square_grid.py` and the notebook both optimize the same 3×3
grid with two repeated shape variables. On the tested environment:

| Quantity | Result |
| --- | --- |
| Initial normalized mean squared stress error | `0.08970797036` |
| Final normalized mean squared stress error | `1.7092111e-10` |
| Optimized vertex offsets | approximately `[0.09995045, 0.06951482]` |
| Maximum final constraint violation | `0` |
| Optimizer status | Successful |

The target is a synthetic curve produced by a feasible shape with offsets
`[0.10, 0.07]`. The optimizer receives only the target samples, not those
parameters. Agreement measures curve fitting, not unique identification of
geometry or experimental validity.

The shape notebook was executed from a fresh kernel: all 22 cells completed
with three rendered figures and no cell errors. Its saved outputs are included.
The dynamics script completed 3,000 integration steps and saved finite states,
energies, and a figure. Both scripts write numeric archives without pickle.

Applicable examples in the module documentation were also executed, including
partial boundary constraints, a custom potential, shape restrictions,
quasistatic loading, and Verlet dynamics.

## Boundary, objective, and material tutorials

Three additional notebooks and companion Python scripts demonstrate concrete
changes to the model and optimization. Their calculations use the same
finite-hinge equations as the first tutorial.

The boundary tutorial compares clamped compression, sliding/free-rotation
grips, one prescribed interior angle, and simple shear. Every prescribed
coordinate is checked throughout each trajectory. All four cases converge,
with the largest free-coordinate residual below `1e-9`. The notebook also
runs from either the repository root or its `examples/` directory.

The alternative-objective tutorial executes two independent shape
optimizations, both with geometric restrictions and an explicit minimum
total-support-force requirement:

| Objective | Initial cost | Final cost |
| --- | --- | --- |
| Target secant stiffness | `0.2635913` | `1.48e-16` |
| Support-force uniformity (squared coefficient of variation) | `0.00171796` | `6.13e-12` |

These are different dimensionless objectives, so their magnitudes do not rank
the two designs against each other. Both optimizers succeeded, their final
constraints were satisfied, and all equilibrium residuals were below `1e-9`.

The material tutorial holds polygon shapes fixed and fits individual support
forces at several strains. The optimizer receives logarithms of positive
spring constants; the potential and simulator receive the resulting
`MechanicalParameters` explicitly.

| Active material variables | Synthetic target | Fitted values | Final normalized force error |
| --- | --- | --- | --- |
| One common axial constant | `1.35` | `1.34999542` | `1.33e-12` |
| Axial constants of three vertical columns | `[0.7, 1.1, 1.5]` | `[0.70000084, 1.10000286, 1.49999850]` | `3.19e-13` |
| One common bending constant | `0.035` | `0.03500015` | `1.33e-14` |

The tutorial also evaluates a per-bond bending parameterization. It does not
claim unique identification of independent constants from one load case.
The joint shape/bending example in the module guide was checked through
initial feasibility, target convergence, and finite nonzero gradients; its
optional outer optimization snippet was not run.

## Independent vertices on the whole square grid

Tutorial 05 and `examples/optimize_all_vertices.py` optimize all 36 vertices
of a 3×3 grid. The `(9, 4, 2)` tensor contains 72 independent scalar offsets,
including all corners of the clamped polygons. The new mapping tests verify
an identity 72×72 Jacobian and individual updates on both clamped and
unconnected outline corners.

The script and notebook produce the same results with the default seeds and
regularization weight `1e-4`:

| Quantity | Result |
| --- | --- |
| Initial normalized mean squared stress error | `0.0012174394` |
| Final normalized mean squared stress error | `2.2281194e-8` |
| Initial / final total cost, including shape penalty | `0.0012175296` / `7.0275133e-7` |
| SLSQP iterations / status | `49` / successful |
| Maximum final constraint violation | `0` |
| Maximum equilibrium residual | `7.48e-10` |
| Maximum absolute vertex offset | `0.0214651`, within the `0.06` bound |

All 72 coordinates changed from their initial values. This does not establish
unique recovery of the synthetic target shape: one stress curve cannot identify
all coordinates, and several corners are insensitive to this loading when
contact is inactive. A small squared-offset penalty favors smaller changes
while retaining independent variables.

The notebook's 25 cells executed from a fresh kernel with three rendered
figures, an exported and embedded GIF, and no cell errors. Initial and optimized reference-shape plots use
identical centers, angles, and axis limits; compression is shown separately.
The annotated setup diagram was generated from the geometry, visually
inspected, and exported as editable SVG and PNG.

## NLopt optimization and video export

The optional NLopt backend was tested with NLopt 2.11.0. It applies the same
JAX shape objective, independent 72-variable map, and geometric restrictions.
The default `examples/optimize_with_nlopt.py` run uses SLSQP with a maximum
of 300 objective evaluations:

| Quantity | Result |
| --- | --- |
| Initial / final normalized stress-curve error | `0.0012174394` / `2.3106517e-8` |
| Initial / final total cost | `0.0012175296` / `6.7968888e-7` |
| NLopt status / objective calls | `3` (objective tolerance reached) / `266` |
| Maximum constraint violation | `0` |
| Maximum equilibrium residual | `6.27e-10` |

MMA is covered by an independent constrained analytic test. The complete
72-variable simulation fit above was run with SLSQP. The adapter explicitly
reports evaluation/time limits as unsuccessful; feasible partial results are
not presented as converged designs.

The video script converged all 17 quasistatic states and exported both GIF and
MP4 at 600×600 pixels. Pillow confirmed 17 GIF frames; `ffprobe` confirmed
17 MP4 frames and 8 fps. An extracted final frame was visually inspected.
The GIF is included in the README. MP4 requires system ffmpeg; the automated
tests also cover the GIF fallback when ffmpeg is unavailable.

## Packaging and original files

Both a wheel and a source distribution build successfully. The standalone
package imports no modules or data from the parent research folder. Source
archives include documentation, examples, tests, and the MIT license.

SHA-256 fingerprints of all 118 original top-level Python files, notebooks,
and network files were compared before and after the work: all were unchanged.
Selected source fingerprints are recorded in the
[version audit](version-audit.md).

## What remains a scientific modeling question

These checks do not validate the constitutive model against experiments or
establish equality with complete saved legacy trajectories. In particular:

- The angular contact term is a connected-corner approximation, not general
  collision detection.
- A small equilibrium residual does not prove that a stationary state is the
  desired stable branch. Buckling, hysteresis, and branch selection need
  loading-step and perturbation studies.
- Implicit gradients can become unreliable near singular equilibria or
  changes of branch; finite differences should follow the same branch.
- Dynamic predictions require appropriate effective masses, inertias,
  loading rates, and timestep-convergence checks.
- Only the CPU configuration above was validated locally. The GitHub Actions
  workflow additionally specifies Python 3.11 and 3.12 for future CI runs.
