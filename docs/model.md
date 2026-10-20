# Mechanical model and numerical conventions

This package represents rigid polygons connected by compliant ligaments. The
polygon outlines define the connection positions and local contact angles.
Deformation comes from translation and rotation of the polygons and deformation
of the ligaments; the polygons themselves do not stretch.

The implementation provides two distinct ligament models. Choose one explicitly
when interpreting or comparing results:

```python
potential = make_potential(geometry, parameters, model="finite_hinge")
# Or, for connected vertices that coincide in the reference geometry:
potential = make_potential(geometry, parameters, model="point_hinge")
```

The finite-ligament model is the default. The point-hinge option represents the
shared-corner model used in earlier versions. Its coefficients should not be
assumed to reproduce a finite ligament's response.

## Coordinates and reference geometry

Polygon \(i\) has state

\[
q_i=(x_i,y_i,\theta_i), \qquad c_i=(x_i,y_i).
\]

Angles are in radians and positive counterclockwise. A vertex's local arm vector
\(v_a\) belongs to polygon \(i(a)\). Its rotated arm and world position are

\[
a_a(q)=R(\theta_{i(a)})v_a,\qquad
p_a(q)=c_{i(a)}+a_a(q).
\]

Local arms are defined at **zero body angle**. They are not displacement vectors
relative to the polygon's reference angle. Consequently, a nonzero angle in
`geometry.reference_state` already rotates the supplied local vertices.

Vertices are stored counterclockwise, consecutively within each polygon.
`geometry.bonds[b] = (a, d)` connects global vertex indices \(a\) and \(d\);
the corresponding polygon owners are \(i\) and \(j\). Every undirected bond
appears once.

For this bond, define the endpoint separation

\[
g_b=p_a-p_d+s_b,
\]

where \(s_b\) is `geometry.image_shifts[b]`. It is zero for a finite specimen.

`initialize(geometry, vertices)` computes the reference ligament lengths,
endpoint angles, contact openings, and relative polygon rotations. Call it
again inside the design objective for **each candidate shape**. Otherwise
the optimizer changes the shape while retaining rest data belonging to a
different design.

## Finite-ligament energy

Let \(\ell=\lVert g_b\rVert\), and let \(\ell_0\) denote its reference value.
For the signed angle

\[
\operatorname{angle}(u,v)
=\operatorname{atan2}(u_xv_y-u_yv_x,u\cdot v),
\]

the endpoint angles and their changes are

\[
\alpha_i=\operatorname{angle}(-g_b,a_a),\qquad
\alpha_j=\operatorname{angle}(g_b,a_d),\qquad
\Delta\alpha_k=\operatorname{wrap}(\alpha_k-\alpha_{k,0}).
\]

`wrap` gives the shortest angular difference in \([-\pi,\pi]\).
The relative rotation change is

\[
\Delta\theta=(\theta_i-\theta_j)-(\theta_{i,0}-\theta_{j,0}).
\]

The elastic energy of one finite ligament is

\[
E_b^{\mathrm{elastic}}=
\frac12\left[
k_l(\ell-\ell_0)^2
+k_s\ell_0^2\left((\Delta\alpha_i)^2+(\Delta\alpha_j)^2\right)
+k_\theta(\Delta\theta)^2
\right].
\]

The names in `PotentialParameters` are `stretching`, `shearing`, and `bending`,
respectively. Each coefficient may be a nonnegative scalar or an array with
one entry per bond. Reinitialization makes these elastic terms zero at the
candidate's reference state.

Use a positive minimum ligament length when optimizing finite ligaments:

```python
restriction = make_shape_constraint(
    geometry,
    vertices_from_parameters,
    min_edge_length=0.1,
    min_corner_sine=0.1,
    min_ligament_length=0.05,
)
```

Choose these numbers in the geometry's units. A finite ligament needs a
well-defined direction; collapsing it into a point changes the mechanical
model. Internally, norms use
\(\sqrt{u\cdot u+10^{-24}}\), corresponding to a length regularizer of
\(10^{-12}\), and undefined angle directions have a finite fallback. These
numerical safeguards do not supply physical behavior for a collapsed ligament.
The reference constraint also does not prevent a ligament from collapsing
during loading.

## Point-hinge energy

For the point-hinge model, the reference endpoints should coincide:
\(g_{b,0}=0\). Define the current center direction

\[
n=\frac{c_i-c_j+s_b}{\lVert c_i-c_j+s_b\rVert}.
\]

The elastic bond energy is

\[
E_b^{\mathrm{elastic}}=
\frac12\left[
k_l(g_b\cdot n)^2
+k_s(n_xg_{b,y}-n_yg_{b,x})^2
+k_\theta(\Delta\theta)^2
\right].
\]

There is no subtracted reference endpoint gap in this model. Supplying
separated reference endpoints generally introduces prestress, even after
`initialize`. Start with `square_grid(..., hinge_gap=0.0)` and preserve shared
corner positions while changing shapes; [the module guide](modules.md) shows
how to do that. Polygon centers must remain distinct.

## Local angular contact

Both models add the same contact penalty. Initialization obtains two reference
openings \(\beta_1,\beta_2\) from the edges adjacent to the connected vertices.
Each signed opening is represented in \([-\pi/2,3\pi/2)\). During deformation,

\[
o_1=\beta_1+\Delta\theta+\eta,\qquad
o_2=\beta_2-\Delta\theta+\eta,
\]

\[
E_b^{\mathrm{contact}}=
\frac{k_c}{2}
\left[\min(o_1,0)^2+\min(o_2,0)^2\right].
\]

Here `contact` is \(k_c\), and `contact_angle_offset` is \(\eta\).
A positive offset permits additional closure before the penalty becomes active.

This is a **local angular contact approximation**, inherited from the original
connected-corner formulation. It does not search for intersections between
arbitrary polygons, detect collisions between unconnected polygons, or impose
friction. Translation of connected vertices is not part of the opening-angle
test. A valid reference outline and nonnegative shape-constraint margins
therefore do not establish that an entire deformed trajectory is free of
overlap. Inspect the deformed geometry when interpreting large deformations.

The penalty has a continuous first derivative at activation, but its second
derivative changes there. Contact activation can affect equilibrium-gradient
accuracy.

## Boundaries, reaction signs, and units

`BoundaryConditions.fixed_mask` marks prescribed coordinates with `True`.
Prescribed coordinates may move: their path is

\[
q_{\mathrm{prescribed}}(\epsilon)
=q_{\mathrm{reference}}+\epsilon\,d,
\]

where \(d\) is `strain_direction`. Free coordinates are determined by the
chosen solver.

For `compression_boundary`, positive engineering strain moves the upper grip
downward by \(\epsilon H\). The lower grip remains at its reference height.
\(H\) is the initial distance between the upper and lower polygon-center rows.
The default width \(W\) is the reference polygon bounding-box width; it can be
overridden explicitly.

`reaction_forces` returns \(r=\partial E/\partial q\). At equilibrium,
prescribed entries are the forces or torques exerted by the supports on the
specimen. The internal mechanical force is \(-r\). Free entries should be
approximately zero.

The nominal stress measurement is

\[
\sigma_{\mathrm{nominal}}
=\frac{\sum r\odot d}{H W t}.
\]

For uniaxial compression this gives positive compression stress, equal to the
compressive grip force divided by \(W t\). Do not take the absolute value of
every reaction or divide the reaction by two. Each bond already carries its
usual one-half factor in the energy.

The package does not impose a unit system. With energy measured in joules and
length in meters, `stretching` and `shearing` have units of energy/length²;
`bending` and `contact` have units of energy/radian². Translational reactions
have units of force, and rotational reactions have units of torque. Supply a
physical `thickness` when interpreting stress as force/area. The default
thickness of one is commonly used for a two-dimensional force-per-width
convention.

For a custom boundary path that drives several loading modes or angles,
`nominal_stress` remains the normalized quantity conjugate to that path
parameter. It is not automatically a particular tensor component of stress.

## Periodic images

The `.dat` importer constructs local arms using minimum-image neighbor
displacements. With `periodic=(False, False)`, it removes bonds crossing the
cell boundaries while preserving their polygon vertices.

For retained periodic connections, the importer records the shift added to
the first-minus-second endpoint separation. That shift remains fixed
throughout the simulation. The solver does not switch nearest images,
deform the periodic cell, replicate cells, or impose affine periodic
displacements. A prescribed change in the cell itself requires extending the
model and its boundary treatment.

## Quasistatic continuation and derivatives

`make_quasistatic_solver` minimizes the energy over free coordinates at each
specified absolute strain, using BFGS or L-BFGS. The previous equilibrium
initializes the next solve. Strain arrays may include unloading and cyclic
paths.

Every result contains the free-coordinate gradient norm, iteration count, and
a convergence flag. Call `result.check_convergence()` outside JIT before
accepting a simulation. `stress_curve_loss` rejects trajectories containing
unconverged steps, and the design optimizer raises an error on nonfinite
objective or gradient evaluations.

Implicit differentiation uses stationarity of the converged solution. The
derivative is meaningful for a sufficiently converged local equilibrium with
a nonsingular free-coordinate Hessian. It does not differentiate a discrete
choice between buckling branches. Near bifurcation, snap-through, contact
activation, or a remaining rigid mode, gradients may be ill-conditioned or
branch dependent. A symmetric initial configuration may also remain on a
symmetric stationary branch.

For reproducible symmetry breaking, set
`QuasistaticOptions(perturbation_amplitude=1e-5, seed=0)`. A seeded uniform
perturbation is added only to free coordinates before each equilibrium solve.
Its amplitude spans both positional and angular coordinates in their chosen
units. Alternatively, supply `initial_state=` to seed the first equilibrium
with a particular deformation. These choices can change the branch reached;
they do not make branch switching differentiable.

Check sensitivity to equilibrium tolerance and strain-step spacing. For a
design of interest, compare selected automatic derivatives with finite
differences while ensuring both perturbed solves follow the same branch.
The residual combines translation and rotation derivatives, so its numerical
tolerance must be interpreted with the chosen units and coordinate scales.

## Velocity-Verlet dynamics

`make_verlet_solver` integrates translations and rotations using positive
polygon masses \(m_i\) and moments of inertia \(I_i\):

\[
m_i\ddot c_i=-\partial E/\partial c_i+f_i,\qquad
I_i\ddot\theta_i=-\partial E/\partial\theta_i+\tau_i.
\]

The generalized external-force callback returns an array shaped
`(n_polygons, 3)`, with the last column containing torques. Prescribed
coordinates are projected onto their boundary path at each time sample;
only free coordinates obey the acceleration update.

The strain array contains one value per uniform time sample, including the
initial state at time zero. `VerletOptions.time_step` sets the spacing between
samples. Optional damping applies the velocity-decay law
\(\dot v=-\gamma v\) through exponential half steps, with \(\gamma\) measured
in inverse time. The mass and inertia defaults are convenient numerical
values; set them from the intended physical system for quantitative dynamics.

Choose a timestep that resolves the fastest translational and rotational
modes. Halve the timestep while keeping the total duration and boundary
history fixed, then compare trajectories and energies. For an undamped system
with stationary boundaries and no external forcing, monitor total energy.
Moving grips and external forces do work; damping removes energy, so
conservation is not expected in those cases. Reported kinetic energy includes
the prescribed coordinates as well as the free ones.

Verlet dynamics retains inertia and may oscillate. It is not an equilibrium
minimizer. Approximating a quasistatic response through slow, damped dynamics
requires separate checks of loading-rate, damping, and timestep dependence.
