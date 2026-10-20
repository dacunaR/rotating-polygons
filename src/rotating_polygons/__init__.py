"""Differentiable rotating-polygon mechanics without import-time configuration.

Enable JAX 64-bit mode explicitly in your application before constructing
geometry when using the default tight equilibrium tolerances.
"""

from .boundary import BoundaryConditions, compression_boundary
from .animation import create_animation, save_video
from .constraints import make_shape_constraint, shape_constraints
from .energy import (
    MechanicalParameters,
    PotentialParameters,
    make_parameterized_potential,
    make_potential,
)
from .geometry import Geometry, from_polygons, square_grid, world_vertices
from .initialization import InitializedGeometry, initialize
from .integrators import (
    DynamicResult,
    QuasistaticOptions,
    SimulationResult,
    VerletOptions,
    make_quasistatic_solver,
    make_verlet_solver,
)
from .io import load_geometry, read_legacy_dat, save_geometry
from .objectives import (
    make_stress_measurement,
    nominal_stress,
    reaction_forces,
    stress_curve_loss,
)
from .optimization import minimize_design
from .nlopt_optimization import minimize_design_nlopt

__version__ = "0.1.0"
__all__ = [
    "BoundaryConditions",
    "DynamicResult",
    "Geometry",
    "InitializedGeometry",
    "MechanicalParameters",
    "PotentialParameters",
    "QuasistaticOptions",
    "SimulationResult",
    "VerletOptions",
    "compression_boundary",
    "create_animation",
    "from_polygons",
    "initialize",
    "load_geometry",
    "make_parameterized_potential",
    "make_potential",
    "make_quasistatic_solver",
    "make_shape_constraint",
    "make_stress_measurement",
    "make_verlet_solver",
    "minimize_design",
    "minimize_design_nlopt",
    "nominal_stress",
    "reaction_forces",
    "read_legacy_dat",
    "save_geometry",
    "save_video",
    "shape_constraints",
    "square_grid",
    "stress_curve_loss",
    "world_vertices",
]
