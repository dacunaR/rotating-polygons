# Modules and extension points

The package separates topology, constraints on motion, reference
initialization, constitutive energy, simulation, and design optimization.
Geometry and solver configuration are constructed once. Candidate shapes and
material coefficients remain JAX arrays passed through initialization and
simulation, so the derivatives can follow the entire design calculation.

For a complete optimization and an executable notebook, start with the
[README](../README.md) and the `examples/` directory. This guide explains
where to make changes for a different system.

## Geometry and file input

`geometry.py` owns `Geometry`, `square_grid`, `from_polygons`, and coordinate
transformations. It contains no material coefficients or boundary choices.

```python
import jax
import jax.numpy as jnp
import numpy as np
import rotating_polygons as rp

jax.config.update("jax_enable_x64", True)
geometry = rp.square_grid(4, 4, spacing=1.0, hinge_gap=0.2)
```

The package does not enable precision or select a device during import.
Enable 64-bit arithmetic before constructing arrays when using tight
equilibrium tolerances.

The main arrays are:

- `reference_state`: one `(x, y, angle)` row per polygon.
- `vertices`: local two-dimensional arm vectors, concatenated by polygon.
- `vertex_polygons`: the owner of each vertex.
- `polygon_offsets`: starts of the polygon vertex lists, plus their final end.
- `bonds`: pairs of global vertex indices.
- `previous_vertices` and `next_vertices`: adjacency around each outline.
- `image_shifts`: fixed periodic shifts for the bond endpoint differences.

`n_polygons`, `n_vertices`, and `n_bonds` expose the corresponding counts.
Construction validates array shapes, finite values, counterclockwise
positive-area outlines, and bond indices. It does not perform a general
intersection test.

Polygons need not all have the same number of vertices:

```python
mixed = rp.from_polygons(
    centers=[[0.0, 0.0], [1.2, 0.0]],
    polygons=[
        [[-0.3, -0.3], [0.3, -0.3], [0.0, 0.3]],  # triangle: 0, 1, 2
        [[0.4, 0.0], [0.0, 0.4], [-0.4, 0.0], [0.0, -0.4]],  # square: 3..6
    ],
    bonds=[[1, 5]],
    angles=[0.0, 0.0],
)
points = rp.world_vertices(mixed, mixed.reference_state)
```

The `polygons` argument contains **local** arm vectors, not world coordinates.
For nonzero reference angles, transform a desired world-coordinate outline
back to the local frame before supplying it.

`io.py` reads the legacy neighbor-list format and saves independent geometry
archives:

```python
geometry_from_file = rp.read_legacy_dat(
    "my_network.dat", shrink=0.8, periodic=(False, False)
)
rp.save_geometry("geometry.npz", geometry_from_file)
restored = rp.load_geometry("geometry.npz")
```

A `.dat` header begins with the two cell dimensions. Each following row gives
a polygon center, its zero-based neighbor indices, and optional `-1` padding.
The loader validates reciprocal adjacency, forms the arm vectors from
minimum-image neighbor separations, and sorts them counterclockwise.
`shrink=0.8` scales those half-distance arms to 80% of their original size.
Cutting a boundary removes connections, not outline vertices.

Duplicate neighbor IDs cannot identify which periodic image they refer to;
the loader rejects them. This affects legacy cells only two polygons wide.
Use `square_grid(2, ny)` for a finite grid or supply explicit geometry and
image shifts. `.npz` archives contain numeric arrays and are loaded with
`allow_pickle=False`; they do not store material parameters or solver results.

## Prescribed and free degrees of freedom

`boundary.py` defines `BoundaryConditions`. The convenience constructor
clamps the lower and upper rows and drives the upper row downward:

```python
boundary = rp.compression_boundary(
    geometry,
    clamp_rotations=True,
    clamp_horizontal=True,
)
```

Release horizontal grip motion or grip rotation using the corresponding
boolean. With horizontal sliding allowed, one lower horizontal coordinate
remains pinned to remove rigid translation.

For independent control over each degree of freedom, construct the mask and
driving array directly. This example drives only the vertical coordinates
of the grip rows, pins one horizontal coordinate, and leaves rotation free:

```python
state = np.asarray(geometry.reference_state)
bottom = np.isclose(state[:, 1], state[:, 1].min())
top = np.isclose(state[:, 1], state[:, 1].max())
height = float(np.ptp(state[:, 1]))
width = float(np.ptp(np.asarray(rp.world_vertices(geometry, state))[:, 0]))

fixed = np.zeros_like(state, dtype=bool)
fixed[bottom | top, 1] = True
fixed[np.flatnonzero(bottom)[0], 0] = True
direction = np.zeros_like(state)
direction[top, 1] = -height

sliding_boundary = rp.BoundaryConditions(
    reference_state=state,
    fixed_mask=fixed,
    strain_direction=direction,
    height=height,
    width=width,
)
```

`True` means prescribed, including moving coordinates. A nonzero entry in
`strain_direction` is allowed only at a prescribed coordinate. `assemble`
inserts a vector of free coordinates into the driven state; `project` enforces
the prescribed part of an existing full state. Custom supports must remove
unwanted rigid-body modes.

## Shape parameters and initialization

`initialization.py` converts candidate local vertices to `InitializedGeometry`:

```python
reference = rp.initialize(geometry)
```

The returned tuple contains `vertices`, `rest_lengths`,
`rest_shear_angles`, `contact_angles`, and `rest_rotations`. It is a JAX
PyTree, suitable for passing through compiled simulations.

The optimizer does not impose a shape parameterization. A direct per-vertex
displacement parameterization can be as simple as:

```python
def vertices_from_parameters(displacements):
    return geometry.vertices + displacements

initial_parameters = jnp.zeros_like(geometry.vertices)

def initialize_design(parameters):
    return rp.initialize(geometry, vertices_from_parameters(parameters))
```

This allows independent motion of every local vertex. Lower-dimensional
choices can enforce symmetries, share parameters between unit cells, or keep
selected vertices unchanged. The mapping should use JAX operations and return
the original `(n_vertices, 2)` shape.

Constructing a new `Geometry` inside a JAX-transformed objective is not the
intended workflow: its topology checks operate on concrete NumPy values.
Keep topology fixed and pass the differentiable candidate vertices through
`initialize`. Reference geometry validation does not automatically run on
every candidate, so supply shape restrictions to the outer optimizer.

### Shared corners for the point-hinge model

The point-hinge energy assumes initially coincident connected vertices.
Moving both endpoints of a bond by the **same world-coordinate offset**
preserves that condition. On a zero-angle square grid, the local and world
offsets agree:

```python
point_geometry = rp.square_grid(4, 4, hinge_gap=0.0)
first, second = point_geometry.bonds.T
point_initial = jnp.zeros((point_geometry.n_bonds, 2))

def shared_corner_vertices(offsets):
    return (
        point_geometry.vertices
        .at[first].add(offsets)
        .at[second].add(offsets)
    )

def initialize_point_design(offsets):
    return rp.initialize(point_geometry, shared_corner_vertices(offsets))
```

This parameterization leaves unconnected boundary vertices unchanged. For
nonzero reference body angles, convert each shared world offset into its
own polygon's local frame:

```python
from rotating_polygons.geometry import rotate_vectors

def shared_corner_vertices_rotated(offsets):
    owners = point_geometry.vertex_polygons
    angles = point_geometry.reference_state[:, 2]
    first_offsets = rotate_vectors(offsets, -angles[owners[first]])
    second_offsets = rotate_vectors(offsets, -angles[owners[second]])
    return (
        point_geometry.vertices
        .at[first].add(first_offsets)
        .at[second].add(second_offsets)
    )
```

The starting geometry must already have coincident reference endpoints.
These bond-wise updates assume each connected vertex belongs to one bond,
as in the generated grid. For a vertex shared among several connections,
parameterize the common junction once and distribute its offset to every
incident vertex.

## Potential and simulation method

For fixed material coefficients, `energy.py` supplies `PotentialParameters`
and `make_potential`:

```python
parameters = rp.PotentialParameters(
    stretching=1.0,
    shearing=1e-4,
    bending=0.005,
    contact=40.0,
)
potential = rp.make_potential(geometry, parameters, model="finite_hinge")
```

Stiffness coefficients can be scalars or per-bond arrays. [The model
guide](model.md) defines the equations, signs, and applicability of the two
available models.

For material optimization, construct the potential once with
`make_parameterized_potential`. Its second argument is a
`MechanicalParameters` PyTree containing both reference geometry and
coefficients:

```python
material_potential = rp.make_parameterized_potential(geometry, model="finite_hinge")
mechanical = rp.MechanicalParameters(
    reference=rp.initialize(geometry),
    stretching=1.0,
    shearing=0.1,
    bending=jnp.full(geometry.n_bonds, 0.02),
    contact=40.0,
)
energy = material_potential(geometry.reference_state, mechanical)
```

Each coefficient field accepts a scalar shared by every bond or an array
of shape `(geometry.n_bonds,)`. The same applies to `contact_angle_offset`.
The `MechanicalParameters` constructor accepts traced JAX values; the
potential checks their shapes. Keep stiffnesses nonnegative and all values
finite through bounds or a positive mapping such as `bending=exp(log_bending)`.
`PotentialParameters` instead validates concrete, fixed coefficients.

Both integrators accept this PyTree as their first simulation argument:
`simulate(mechanical, strains)`. Measurements accept it as well. Passing
coefficients explicitly lets implicit differentiation account for their
effect on equilibrium. Construct the potential and solver outside the
objective; do not create a solver closure around a traced stiffness inside
the objective. A complete joint design example appears below.

A custom potential only needs the signature
`energy(state, reference) -> scalar`. For example, add horizontal foundation
springs without changing the simulator:

```python
base_potential = potential

def potential_with_foundation(state, reference):
    displacement = state[:, 0] - geometry.reference_state[:, 0]
    foundation_energy = 0.5 * 0.02 * jnp.sum(displacement**2)
    return base_potential(state, reference) + foundation_energy
```

Use JAX operations and retain a scalar output. A replacement reference
representation can also be a JAX PyTree, provided the custom initializer and
potential agree on its meaning.

`integrators.py` supplies both quasistatic and dynamic solvers:

```python
simulate = rp.make_quasistatic_solver(
    geometry,
    boundary,
    potential,
    rp.QuasistaticOptions(method="lbfgs", tolerance=1e-8, max_iterations=1000),
)
strains = jnp.linspace(0.0, 0.05, 11)
result = simulate(rp.initialize(geometry), strains)
result.check_convergence()
```

`SimulationResult` contains `states`, `energies`, `residuals`, `iterations`,
and `converged`. Strain values are absolute, so the first value need not be
zero. Changing `method` to `"bfgs"` selects full BFGS.
`QuasistaticOptions.tolerance` is the norm tolerance on free-coordinate energy
gradients at each equilibrium, and `max_iterations` limits each inner solve.
These differ from the outer optimizer's `tol` and `maxiter`, which control
SLSQP's design search.

For dynamics, the strain array specifies the boundary position at each time:

```python
dynamic_simulate = rp.make_verlet_solver(
    geometry,
    boundary,
    potential,
    masses=1.0,
    inertias=0.1,
    options=rp.VerletOptions(time_step=1e-3, damping=0.2),
)
time_samples = jnp.arange(1001) * 1e-3
dynamic_strains = 0.01 * time_samples / time_samples[-1]
dynamic_result = dynamic_simulate(rp.initialize(geometry), dynamic_strains)
```

Masses and inertias may instead contain one value per polygon. The solver
also accepts `external_force(state, time)` and optional `initial_state`
and `initial_velocity` arguments. `DynamicResult` contains times, states,
velocities, potential energies, and kinetic energies. Choose and verify
the timestep for the actual stiffness, mass, inertia, and loading timescale.

## Measurements, objectives, and restrictions

`objectives.py` implements support reactions, nominal stress, and a normalized
stress-curve loss:

```python
measure_stress = rp.make_stress_measurement(potential, boundary, thickness=1.0)
target_stress = 0.1 * strains
curve_loss = rp.stress_curve_loss(
    simulate,
    initialize_design,
    measure_stress,
    strains,
    target_stress,
)
```

The returned function accepts the design parameters. By default the squared
error is normalized by the largest absolute target stress. You can supply a
positive `scale` explicitly. A trajectory containing a failed equilibrium
produces an infinite objective.
The initializer may return either `InitializedGeometry` for fixed material
coefficients or `MechanicalParameters` for joint shape and material design;
the simulator and measurement must use the matching potential.

A custom scalar objective can combine this curve error with another
differentiable quantity:

```python
def regularized_loss(displacements):
    return curve_loss(displacements) + 1e-3 * jnp.mean(displacements**2)
```

For an entirely different objective, call `initialize_design` and `simulate`
inside your function, then measure the desired state or reaction quantities.
Preserve the convergence checks. `stress_curve_loss` expects quasistatic
results with `converged` flags; a dynamic design objective should be written
directly against `DynamicResult`.

`constraints.py` expresses restrictions as margins that must be
**greater than or equal to zero**:

```python
shape_restriction = rp.make_shape_constraint(
    geometry,
    vertices_from_parameters,
    min_edge_length=0.1,
    min_corner_sine=0.1,
    min_ligament_length=0.05,
)

def displacement_restriction(displacements):
    # Keep each vertex within a disk of radius 0.04 around its original arm.
    return 0.04**2 - jnp.sum(displacements**2, axis=1)
```

`shape_constraints` combines minimum polygon-edge length, positive
turning-angle sine, and optional minimum ligament length. Positive corner
sines restrict local convexity and avoid nearly collinear corners.
The low-level `edge_lengths`, `corner_sines`, and `ligament_lengths`
functions are available from `rotating_polygons.constraints`.

For shared-corner point hinges, leave `min_ligament_length=0.0`, which disables
that restriction. All these margins describe the reference design; none
constitutes a general collision check over the loading trajectory.

## Optimization

`optimization.py` bridges JAX objective/constraint derivatives to SciPy SLSQP:

```python
optimized = rp.minimize_design(
    regularized_loss,
    initial_parameters,
    bounds=[(-0.04, 0.04)] * initial_parameters.size,
    constraints=(shape_restriction, displacement_restriction),
    maxiter=100,
    tol=1e-6,
)
print(optimized.success, optimized.message)
print(optimized.feasible, optimized.max_constraint_violation)
candidate_vertices = vertices_from_parameters(optimized.parameters)
```

The parameter representation can be an array or a PyTree. Bounds follow the
flattened JAX PyTree order; dictionary keys are sorted. The result retains
SciPy's fields and adds the reconstructed `parameters`, `initial_loss`,
`loss_history`, final `constraint_margins`, `max_constraint_violation`, and
`feasible`. Inspect `success` as well as feasibility: a feasible candidate
does not establish convergence or a global optimum.

Use physically useful bounds to avoid extreme trial designs. SLSQP may
evaluate intermediate candidates that violate nonlinear constraints. If
an equilibrium or derivative becomes nonfinite, the optimizer raises an
error rather than accepting the candidate.

### NLopt backend

Install `pip install -e ".[nlopt]"` to use `nlopt_optimization.py`.
`minimize_design_nlopt` preserves the same parameter representation, bounds
ordering, objective signature, and nonnegative constraint margins:

```python
optimized = rp.minimize_design_nlopt(
    regularized_loss,
    initial_parameters,
    bounds=[(-0.04, 0.04)] * initial_parameters.size,
    constraints=(shape_restriction, displacement_restriction),
    algorithm="slsqp",  # NLopt SLSQP; "mma" selects moving asymptotes.
    maxeval=300,  # Maximum objective evaluations, not outer iterations.
    ftol_rel=1e-8,  # Relative objective-change stopping tolerance.
    xtol_rel=1e-7,  # Relative parameter-change stopping tolerance.
    constraint_tol=1e-8,  # Tolerance for the final constraint margins.
)
if not optimized.success:
    raise RuntimeError(optimized.message)
```

The wrapper converts each margin `m(parameters) >= 0` to NLopt's
`-m(parameters) <= 0`, including the sign of its JAX Jacobian. Both supported
algorithms use gradients. Constraints remain inequalities; equality constraints
and other NLopt algorithms are not exposed by this interface.

Results retain `parameters`, `initial_loss`, `constraint_margins`,
`max_constraint_violation`, and `feasible`. `loss_history` contains evaluated
objective values, which may include rejected trials. It is not an accepted-step
history or proof of monotonic descent. The raw NLopt termination status is
available in `status`; reaching `maxeval` or `maxtime` does not set `success`.
The optional `maxtime` argument limits optimizer wall-clock time in seconds.

The [NLopt example](../examples/optimize_with_nlopt.py) runs a complete shape
fit with convergence and feasibility checks.

## Joint shape and bending-stiffness optimization

Design variables may be a dictionary. For independent motion of every corner
and one shared bending stiffness, the mapping is:

```python
g = rp.square_grid(2, 3, hinge_gap=0.3)
per_vertex_start = {
    "shape": jnp.zeros_like(g.vertices),
    "log_bending": jnp.log(0.02),
}

def initialize_per_vertex(p):
    return rp.MechanicalParameters(
        reference=rp.initialize(g, g.vertices + p["shape"]),
        stretching=1.0,
        shearing=0.1,
        bending=jnp.exp(p["log_bending"]),
        contact=40.0,
    )
```

Replacing the scalar `log_bending` with
`jnp.full(g.n_bonds, jnp.log(0.02))` gives one positive bending stiffness per
bond. Exponentiation enforces positivity; additional limits are useful to
avoid extremely soft or stiff trial materials.

The following setup uses **three design variables** to keep the example small:
vertical arm scaling, horizontal shear of the local outline, and one shared
logarithmic bending stiffness. It uses the imports and 64-bit configuration
from the beginning of this guide.

```python
joint_boundary = rp.compression_boundary(g)
joint_potential = rp.make_parameterized_potential(g, model="finite_hinge")
simulate_joint = rp.make_quasistatic_solver(
    g,
    joint_boundary,
    joint_potential,
    rp.QuasistaticOptions(tolerance=1e-8, max_iterations=300),
)
measure_joint_stress = rp.make_stress_measurement(joint_potential, joint_boundary)
joint_strains = jnp.linspace(0.0, 0.02, 5)

def joint_vertices(p):
    vertical_scale, horizontal_shear = p["shape"]
    # Both modes preserve the number and ordering of the polygon vertices.
    x = g.vertices[:, 0] + horizontal_shear * g.vertices[:, 1]
    y = (1.0 + vertical_scale) * g.vertices[:, 1]
    return jnp.stack((x, y), axis=1)

def initialize_model(p):
    # Shape-dependent rest lengths and angles are recomputed for every design.
    return rp.MechanicalParameters(
        reference=rp.initialize(g, joint_vertices(p)),
        stretching=1.0,
        shearing=0.1,
        bending=jnp.exp(p["log_bending"]),
        contact=40.0,
    )

initial_joint = {
    "shape": jnp.array([0.01, 0.01]),
    "log_bending": jnp.log(0.02),
}

# Generate a reproducible synthetic target; replace these values with
# measured stresses at joint_strains for an experimental design task.
target_design = {
    "shape": jnp.array([0.035, 0.02]),
    "log_bending": jnp.log(0.035),
}
target_model = initialize_model(target_design)
target_result = simulate_joint(target_model, joint_strains).check_convergence()
joint_target = measure_joint_stress(target_result.states, target_model)

joint_loss = rp.stress_curve_loss(
    simulate_joint,
    initialize_model,
    measure_joint_stress,
    joint_strains,
    joint_target,
)

joint_shape_restriction = rp.make_shape_constraint(
    g,
    joint_vertices,                  # Accepts the whole parameter dictionary.
    min_edge_length=0.1,
    min_corner_sine=0.1,
    min_ligament_length=0.08,
)

def joint_parameter_limits(p):
    # All margins must be >= 0. These impose |shape_i| <= 0.08 and
    # 0.002 <= bending <= 0.2 without flattening the dictionary.
    return jnp.concatenate((
        0.08**2 - p["shape"]**2,
        jnp.array([
            p["log_bending"] - jnp.log(0.002),
            jnp.log(0.2) - p["log_bending"],
        ]),
    ))

value, gradient = jax.jit(jax.value_and_grad(joint_loss))(initial_joint)
print(value, gradient["shape"], gradient["log_bending"])
```

The gradient has the same dictionary structure as the design. It includes
changes in reference geometry, material energy, equilibrium configuration,
and measured stress. The synthetic curve checks the workflow; one short
stress–strain curve does not generally identify a unique shape and stiffness.

To run the outer optimization after inspecting the initial simulation and
gradient:

```python
joint_optimized = rp.minimize_design(
    joint_loss,
    initial_joint,
    constraints=(joint_shape_restriction, joint_parameter_limits),
    maxiter=40,
    tol=1e-7,
)
print(joint_optimized.success, joint_optimized.message)
print(joint_optimized.feasible, joint_optimized.max_constraint_violation)
best_joint = initialize_model(joint_optimized.parameters)
best_joint_result = simulate_joint(best_joint, joint_strains).check_convergence()
best_bending = jnp.exp(joint_optimized.parameters["log_bending"])
```

This uses named constraints instead of positional box bounds. If box bounds
are preferred, use `jax.flatten_util.ravel_pytree` on lower and upper PyTrees
with the same structure as `initial_joint`, then pass their arrays in
`scipy.optimize.Bounds`. JAX sorts dictionary keys, so the flattened order
here is `log_bending`, followed by the two `shape` entries; a bounds list in
the dictionary's insertion order would constrain the wrong variables.

The `maxiter=40` and `tol=1e-7` above control the outer design optimization.
They do not change the equilibrium force tolerance of `1e-8` established
when constructing `simulate_joint`. Inspect both outer optimization status
and the final simulation's convergence.

## Visualization

`plotting.py` is an optional Matplotlib layer:

```python
from rotating_polygons.plotting import plot_geometry

ax = plot_geometry(geometry, vertices=candidate_vertices)
```

To plot a deformed design, pass both the corresponding state and its candidate
vertices. The plotting functions do not change geometry or simulation data.

`animation.py` renders a trajectory using the same local vertices and periodic
image convention. Pass `SimulationResult.states` or `DynamicResult.states`
with shape `(frames, n_polygons, 3)`:

```python
animation = rp.create_animation(
    geometry,
    solution.states,
    vertices=reference.vertices,  # The actual candidate shape.
    fps=20,
    stride=1,  # Select states[::stride]; no intermediate states are synthesized.
)
# In Jupyter:
# from IPython.display import HTML
# HTML(animation.to_jshtml())

saved_path = rp.save_video(
    geometry,
    solution.states,
    "outputs/compression.mp4",
    vertices=reference.vertices,
    fps=20,
    frame_labels=[f"Frame {i}" for i in range(len(solution.states))],
)
```

Limits are fixed across the sampled trajectory. `frame_labels` has one string
per original input frame and is sampled alongside the states. `fps` is playback
speed, not a simulation timestep. For uniformly saved dynamic states separated
by `dt_saved`, real-time playback after subsampling would use
`fps = 1 / (stride * dt_saved)`; other values slow or accelerate the display.
Quasistatic steps have no physical duration.

MP4 requires a system `ffmpeg` executable. GIF uses Pillow and needs no external
encoder. If ffmpeg is missing, `save_video` warns and returns the `.gif` path
unless `fallback_to_gif=False`. Long trajectories are best exported to MP4;
Pillow buffers GIF frames in memory, so use `stride` to reduce their count.
The default `gif_memory_limit_mb=256` rejects exports whose estimated RGBA
frame storage exceeds 256 MiB; set another limit or `None` explicitly to override it.
The [video example](../examples/make_video.py) demonstrates both formats.
