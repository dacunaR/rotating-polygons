# Source audit and model provenance

This package reorganizes research code for rigid polygons connected at their
corners. The original research files were not edited. Selected notebooks,
`AuxeticLib2Dv2.py`, and their input networks have byte-identical archival copies
in `originalFiles/` in the GitHub repository; these are not runtime dependencies.
This audit uses **zero-based notebook cell
indices**, independent of the notebooks' saved execution counters.

## Selected reference

The reference for the finite-length connector model is
`singleModel3.ipynb`, cell 0, specifically `make_initialization3` and
`makePotentialGiovanni2`. Cells 1–3 select and exercise this pair on
`square10x10_h.dat`, shrink the polygon arms to 80%, fix the top and bottom
polygons, and prescribe vertical compression.

This is a selection of the most useful implemented model and architecture, not
a claim that the entire notebook is correct. Its mechanics need the corrections
and qualifications below.

| Source | Assessment |
| --- | --- |
| `singleModel3.ipynb` | Best modular reference among the three `singleModel` notebooks. Separates initialization, potential, equilibrium solution, measurement, loss, and constraints into factory functions, although they share a large mutable class. The active finite-length model includes endpoint-angle shear. |
| `singleModel2.ipynb` | Same mechanics as `singleModel3`. The substantive code addition in version 3 is `plotK`; that function accepts `K` but does not use it. Version 3 is not evidence of improved mechanical correctness over version 2. |
| `singleModel.ipynb` | Earlier modular experiment. Does not contain `make_initialization3` or the two `Giovanni` potentials. Its finite-length `makePotential2` ignores the supplied per-bond `K` and uses `self.kt`; it also duplicates the contact contribution. Saved outputs include errors in cells 11 and 17. |
| `AuxeticLib2Dv2.py` | Useful legacy network reader and geometry conventions. Builds each corner arm halfway toward a periodic nearest-image neighbor and sorts arms by their initial polar angle. It mixes geometry, plotting, mutable graph editing, and serialization, so it is not a suitable runtime dependency for the new package. |

The older Pareto notebooks are another research lineage, with a different
emphasis on stiffness, force distribution, and parameter tying. Their
zero-length corner connectors are a separate constitutive choice from the
finite-length connectors selected above.

### Older stress-matching and Pareto lineage

| Source | Useful contribution and limitations |
| --- | --- |
| `meta2DJax27StressMatch.ipynb`, cells 1 and 6 | A direct square-grid stress-curve optimization example. Shape variables add the same displacement to both arms meeting at each bond, preserving initially coincident endpoints. Mechanics are embedded in one large closure; optimization is unconstrained. |
| `meta2DJax33StressMatch.ipynb`, cells 1 and 6 | Vectorizes bond energies and reactions with `vmap`, replacing version 27's sequential energy loop. This is the clearest older reference for the shared-corner model, but its energy normalization changed without consistently changing force measurement. |
| `meta2DJax50Pareto2StiffnessvsSpread.ipynb`, cells 1, 9, 13, 19 | Contains full-curve fitting and useful edge/corner restriction ideas. The saved code mixes per-bond bending stiffness with an earlier four-coefficient parameter layout; its setup and optimization cells are not mutually consistent. |
| `meta2DJax51Pareto2StiffnessvsSpreadnovs.ipynb`, cells 1, 3, 8, 23 | Separates geometry from stiffness and fixes the integer reshape dimension. It optimizes stiffness with geometry held fixed, rather than solving the requested shape-design example. Older incompatible shape-constraint and plotting cells remain. |
| `paretoFunc.py` | Extracts shared-corner mechanics and explicit JAX derivatives for NLopt. Importing the module runs a simulation. Its objective rewards stiffness with a negative term and penalizes force spread with a positive term, but `myFunc` selects maximization, reversing both apparent intended goals. |
| `paretoSingleModel.py` | Carries the modular finite-length model and endpoint-angle shear into a script, but still runs setup and simulation at import. Its `makeloss2` computes the material proxy from global `vs` instead of the current `args["vs"]`, so that term has no design gradient. |

Specific inconsistencies checked in these sources:

- In version 50, cell 1 sets `NumStiffness = Nb` and
  `StiffPerBond = NumStiffness / Nb`, then uses the resulting float in a reshape
  dimension. The model removes `Nb` trailing variables as stiffness, while
  cells 4, 13, 18, and 19 still assume four trailing coefficients. The geometry
  restrictions in the optimization cell are commented out.
- Version 51 changes that reshape dimension to an integer and accepts
  `model(K, vs, key)`, with periodic repetition of `K`. The old restriction
  functions still slice four entries off the geometry vector, and later cells
  contain calls with the earlier model signature.
- Versions 27 and 50, version 51, and `paretoFunc.py` put `1/2` in the energy and
  divide the resulting reaction gradient by another `2`. Version 33 removes the
  energy factor but retains the reaction division. In each case the published
  measurement is not the derivative of the implemented energy. The new package
  uses `grad(E)` with no extra factor, and normalizes nominal stress explicitly.
- Versions 27/33 compute contact opening from the currently rotated polygon
  edges and subtract `t0`. Versions 50/51 precompute reference opening angles
  and add `t0` after the relative rotation. The sign of this contact offset
  changed; its physical meaning must be specified when transferring parameters.
- Older edge constraints use `0.05 - edge_length**2 <= 0`, meaning a minimum
  edge length of `sqrt(0.05)`, while the newer modular constraint uses
  `0.05 - edge_length <= 0`. These are different dimensional conventions.
- The earlier negative quartic bending coefficient
  `gamma = -0.00088` makes the isolated polynomial
  `kt * delta_theta**2 + gamma * delta_theta**4` unbounded below as rotations
  increase. The selected modular endpoint-angle model does not use that term.
- The four inspected older notebooks contain inherited scratch cells, undefined
  variables, model redefinitions, saved errors, and a literal unfinished
  `for i in` statement (cell 132 in versions 27/33, 170 in version 50, and 174
  in version 51). None is a clean, sequentially executable package entry point.

The reusable choices are therefore the modular separation and finite-length
endpoint-angle model from `singleModel3`, the shared-corner formulation and
vectorization from version 33, and explicit shape restrictions from the Pareto
work. No complete older notebook is adopted without correction.

## Mechanical interpretation

Every polygon is rigid. Its state is
`q_i = (x_i, y_i, theta_i)`; changing its local corner arms changes the shape,
while translating and rotating the polygon changes its simulated state.
For a connector between polygon corners, write

```text
a_i = R(theta_i) a_i_local
a_j = R(theta_j) a_j_local
d   = c_i - c_j + a_i - a_j
l   = ||d||
```

`d` points from the endpoint on polygon `j` to the endpoint on polygon `i`.
The active notebook model computes the rest length `l0` from the designed,
undeformed geometry, then uses the following energy for each undirected bond:

```text
E_b = 1/2 [
    k_stretch (l - l0)^2
  + k_shear l0^2 (delta_psi_i^2 + delta_psi_j^2)
  + k_bend (theta_i - theta_j)^2
  + k_contact (min(gap_1, 0)^2 + min(gap_2, 0)^2)
]

psi_i = angle(-d, a_i)
psi_j = angle( d, a_j)
delta_psi = psi - psi_reference

gap_1 = beta_1 + theta_i - theta_j + contact_offset
gap_2 = beta_2 - theta_i + theta_j + contact_offset
```

`angle(u, v)` is the signed angle computed with `atan2(cross(u, v), dot(u, v))`.
`beta_1` and `beta_2` are reference angles between the adjacent polygon edges
on either side of a connected corner pair. The notebook uses the interval
`[-pi/2, 3*pi/2)` when constructing these opening angles.

The package's finite-length formulation makes reference rotations explicit,
subtracts the reference relative rotation in bending, and wraps the change of
endpoint angle. These changes allow nonzero reference polygon orientations and
avoid artificial jumps when an endpoint angle crosses the `atan2` branch cut.
They are intentional corrections, not exact transcription.

The zero-length model is a separate constitutive choice: penalize the components
of `d` along and across the center-to-center direction, plus bending and optional
angular contact. It does not require a ligament direction at `d = 0`.
Finite-length endpoint-angle shear should not be evaluated at a vanishing
ligament length.

The factor `1/2` is the usual quadratic spring-energy convention. Bonds are
already enumerated once; it is not a correction for duplicate bond counting.

## Confirmed issues and consequential limitations

### Notebook execution and configuration

- **Out-of-order state:** `singleModel3`, cell 5, uses `bot_boundary` and
  `target_stress` before cell 6 defines them. A clean sequential execution does
  not reproduce the saved interactive session. Version 2 has the same ordering.
- **Overwritten settings:** cell 2 assigns `coeffs` twice; only the second
  dictionary is used. Several older options remain in cell 0 even though their
  arguments do not match the selected initialization.
- **Mutable closures:** the potential factories in cell 0 write their
  coefficients onto `self`. Creating another potential on the same instance can
  change the coefficients read by an existing closure. JIT compilation makes
  this especially confusing because compiled traces may retain earlier values.
- **Randomness and precision:** cell 2 seeds JAX from NumPy's global random state;
  no reproducible seed or explicit precision policy is established. The
  `1e-14` perturbation is generally below single-precision resolution for
  coordinates of order one.
- **Plotting mutates geometry:** `plotvs2`, `plotK`, and the video routines
  overwrite polygon arms and positions in `self.Ar`. Plotting is therefore not
  a read-only operation in the source notebooks.

### Geometry, boundary conditions, and initialization

- **Fixed degrees of freedom alter connectivity:** `setfixedDoF(...,
  usepoliId=True)` removes bonds between any two fully fixed polygons. This is
  a physical modeling choice, not a necessary step for imposing Dirichlet
  conditions. Partially fixed degrees of freedom follow different behavior.
  Geometry/topology and constraints should be independently selectable.
- **Fragile repeated-cell indexing:** `make_poli_connectivity` selects bonds
  with `j % P > i % P`, assumes `N // P` equivalent copies, and tiles design
  indices by bond order. There is no validation of divisibility or correspondence
  between those copies. Same-residue connections can be omitted. This is not a
  general periodic-network implementation.
- **Broken legacy neighbor indices:** in the version 2/3
  `make_poli_connectivity`, `Nb` is incremented before assigning
  `pata.id = [Nb, ...]`; invalid bonds reuse the current valid-bond count.
  The older `make_initialization` then uses those IDs to index bond pairs.
  The active `make_initialization3` uses vertex indices instead and avoids this
  particular route.
- **Incomplete periodic initialization:** `make_initialization2` and
  `make_initialization3` compute reference lengths from raw center differences,
  while the potential's `PBC=True` applies a nearest-image correction in **x
  only**. The reference and deformed configurations can therefore disagree for
  a crossing bond. The demonstrated notebook cuts both periodic boundaries and
  uses `PBC=False`, avoiding that particular mismatch.
- **Initial rotation assumptions:** reference lengths and endpoint angles are
  computed using unrotated local arms, and bending assumes zero reference
  relative rotation. These assumptions hold for the demonstrated initial
  geometry but not for an arbitrary initial state.

### Potential and equilibrium solution

- **Duplicated contact:** `makePotential` in all three notebooks adds contact
  twice. The original `singleModel` also does so in `makePotential2`. The active
  version 2/3 `makePotentialGiovanni2` adds contact once. The package uses one
  contribution.
- **Earlier finite-length shear is inadequate:** version 2/3
  `makePotential2` multiplies both stretch and shear by `(l - l0)^2`. A
  connector can change direction at unchanged length without paying its nominal
  shear energy.
- **First Giovanni model is not rotation invariant:** `makePotentialGiovanni`
  uses `angle(center_difference, d) - (theta_i + theta_j)/2`. Under a common
  rigid rotation of centers and polygons, the first term is unchanged while
  the second changes. The endpoint-angle formulation in
  `makePotentialGiovanni2` fixes that issue.
- **Unwrapped angle differences:** the active `Giovanni2` model subtracts two
  principal endpoint angles directly. Crossing a branch cut can create an
  artificial change close to `2*pi`.
- **Degenerate connectors:** normalized directions, `norm`, and `atan2`
  derivatives can be undefined at coincident endpoints or degenerate arms.
  Finite-length models need nonzero reference geometry and design constraints.
- **No convergence diagnostics:** `makeIntegrator` discards the state returned
  by `jaxopt.LBFGS`. It does not report residuals, line-search failures, or
  iteration limits. Outer optimization may then use implicit gradients from
  states that are not stationary.
- **Continuation is path dependent:** each imposed-displacement step starts
  from the previous equilibrium. Near buckling, multiple equilibria and
  singular Hessians can make both branch selection and gradients sensitive to
  step size, perturbations, tolerances, and precision.
- **Contact is local:** the angular penalty is associated with connected
  corners. It is not polygon–polygon collision detection and does not exclude
  every possible overlap elsewhere in the network.

### Measurements and optimization

- **Force is labeled stress:** `makePressureMeasurement` returns selected
  components of `grad(E)`. Cell 6 sums them as `stress` without dividing by
  specimen width or thickness. These values are support forces. A stress
  convention must state its reference cross-section.
- **Strain length is ambiguous:** in cell 2 the displacement uses
  `meta.Ly = 10` for the 10-by-10 data file, while the fixed polygon-center rows
  lie at `y = 0` and `y = 9`. A nominal 20% displacement based on the periodic
  cell length corresponds to about 22.2% of the boundary-center separation.
  Either convention can be defined, but it must be explicit.
- **The “weight” term is a proxy:** `makeWeightMeasurement` returns connector
  rest lengths. `makeloss2` uses their mean as a material proxy, not polygon
  area or physical mass. The printed score in cells 15–16 has a different
  factor of two from the material term optimized in cell 6.
- **Constraints do not prove valid polygons:** cell 7 enforces minimum edge
  length, a minimum projected connector gap, and a minimum angle. These do not
  establish polygon simplicity, convexity, or absence of inter-polygon overlap.
- **Tolerances can weaken geometric limits:** cell 8 uses inequality tolerances
  of `0.02` for some limits and `0.01` for the connector-gap limit itself.
  Optimization termination within such tolerances is not a strict positive-gap
  certificate.
- **Stiffness bounds:** cell 8 permits `K >= -1e-10`, a numerical accommodation
  rather than an explicit nonnegative physical stiffness convention.

## Scope of verification

This audit is based on source inspection and the saved notebook metadata and
outputs. It does **not** establish numerical equivalence of full original and
packaged stress–strain trajectories, nor reproduce the saved optimization
results. Correcting contact multiplicity, reference orientations, angle wrapping,
normalization, convergence handling, or randomness can change those results.

The package's [validation checks](validation.md) cover rigid-motion invariance,
zero reference energy where intended, finite-difference force and design-gradient
checks away from instabilities, exact prescribed degrees of freedom, equilibrium
residuals, and energy behavior for conservative dynamics. They support code
correctness for their tested cases; they do not identify constitutive coefficients
or validate the physical model against experiment.

The original `.dat` reader treats the first two header values as periodic cell
lengths and each following row as `x y neighbor_id ...`, omitting `-1`
placeholders. Neighbor IDs are zero-based row indices. Importing this format
does not by itself specify which boundaries should remain periodic.

## Source fingerprints

SHA-256 hashes recorded during the read-only audit:

```text
ed96a92d058629f91a088a753b4b37522aecb51c1ebb13a7b5979287c92001ca  singleModel3.ipynb
8decc04e15db2b36d87f925063a2d782a0eb8feea1b4c4645fe240580b74b581  singleModel2.ipynb
cd01d8a9bad8f04f0759f45bb3c5e0ddcd20e150f4afbd1692d3aeddc5abe170  singleModel.ipynb
0696a43c8d0a404f6ac07b2ea075e91b8f9d0368cda21f9bf2fd5c08f7634821  AuxeticLib2Dv2.py
```
