"""Import legacy Auxetic networks and store portable geometry archives."""

from pathlib import Path

import numpy as np

from .geometry import Geometry, from_polygons


def read_legacy_dat(
    path: str | Path,
    shrink: float = 0.8,
    periodic: tuple[bool, bool] = (False, False),
) -> Geometry:
    """Read the center/neighbor format used by ``AuxeticLib2Dv2``.

    The header begins with ``length_x length_y``. Each remaining nonempty
    line contains ``center_x center_y neighbor_id ...``, with zero-based
    neighbor indices and optional ``-1`` padding. A local vertex points
    halfway to each neighbor's minimum-image center, multiplied by ``shrink``.
    The loader sorts vertices counterclockwise and pairs reciprocal neighbors.

    Both cell dimensions are used when constructing the original polygon
    outlines. Bonds crossing a dimension are removed unless its corresponding
    ``periodic`` flag is true. Vertices belonging to removed bonds are kept:
    they still define polygon boundaries and contact angles.

    The format contains no image identifiers. Duplicate neighbor indices
    (such as a periodic grid only two cells wide) are ambiguous and rejected.
    Use :func:`~rotating_polygons.geometry.square_grid` for such finite grids.
    """
    if not np.isfinite(shrink) or not 0 < shrink <= 1:
        raise ValueError("shrink must satisfy 0 < shrink <= 1.")
    if len(periodic) != 2 or any(not isinstance(flag, (bool, np.bool_)) for flag in periodic):
        raise ValueError("periodic must be a pair of boolean flags.")
    source = Path(path)
    with source.open(encoding="utf-8") as stream:
        lines = [(number, line.split()) for number, line in enumerate(stream, 1) if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"{source}: expected a cell-size header and polygon rows.")
    try:
        cell = np.asarray([float(value) for value in lines[0][1][:2]])
    except ValueError as error:
        raise ValueError(f"{source}: invalid numeric cell-size header.") from error
    if cell.shape != (2,) or not np.all(np.isfinite(cell)) or np.any(cell <= 0):
        raise ValueError(f"{source}: cell sizes must be finite and positive.")

    centers = []
    neighbors = []
    for number, fields in lines[1:]:
        if len(fields) < 5:
            raise ValueError(f"{source}:{number}: expected a center and at least three neighbors.")
        try:
            center = [float(value) for value in fields[:2]]
            indices = np.asarray([float(value) for value in fields[2:]])
        except ValueError as error:
            raise ValueError(f"{source}:{number}: nonnumeric polygon row.") from error
        if not np.all(np.isfinite(indices)) or np.any(indices != np.floor(indices)):
            raise ValueError(f"{source}:{number}: neighbor indices must be finite integers.")
        indices = indices[indices != -1].astype(np.int64)
        if len(indices) < 3:
            raise ValueError(
                f"{source}:{number}: a polygon needs at least three neighbors/vertices."
            )
        if len(set(indices)) != len(indices):
            raise ValueError(
                f"{source}:{number}: duplicate neighbor indices cannot identify periodic images; "
                "use square_grid for a finite grid, or supply explicit vertices and bonds."
            )
        centers.append(center)
        neighbors.append(indices)
    centers = np.asarray(centers)
    if not np.all(np.isfinite(centers)):
        raise ValueError(f"{source}: polygon centers must be finite.")
    count = len(centers)
    for polygon, indices in enumerate(neighbors):
        if np.any(indices < 0) or np.any(indices >= count):
            raise ValueError(f"{source}: polygon {polygon} has an out-of-range neighbor index.")
        if polygon in indices:
            raise ValueError(f"{source}: polygon {polygon} lists itself as a neighbor.")
        for neighbor in indices:
            if polygon not in neighbors[neighbor]:
                raise ValueError(f"{source}: adjacency {polygon} -> {neighbor} is not reciprocal.")

    polygons = []
    vertex_lookup = {}
    offset = 0
    for polygon, indices in enumerate(neighbors):
        displacement = centers[indices] - centers[polygon]
        displacement -= cell * np.round(displacement / cell)
        if np.any(np.all(displacement == 0, axis=1)):
            raise ValueError(f"{source}: polygon {polygon} has a coincident neighbor image.")
        arms = displacement * (shrink / 2)
        order = np.argsort(np.arctan2(arms[:, 1], arms[:, 0]))
        polygons.append(arms[order])
        for local_vertex, neighbor in enumerate(indices[order]):
            vertex_lookup[polygon, int(neighbor)] = offset + local_vertex
        offset += len(indices)

    bonds = []
    shifts = []
    for polygon, indices in enumerate(neighbors):
        for neighbor in indices:
            if neighbor <= polygon:
                continue
            displacement = centers[polygon] - centers[neighbor]
            image_shift = -cell * np.round(displacement / cell)
            crosses_boundary = image_shift != 0
            if np.any(crosses_boundary & ~np.asarray(periodic)):
                continue
            bonds.append(
                (vertex_lookup[polygon, int(neighbor)], vertex_lookup[int(neighbor), polygon])
            )
            shifts.append(image_shift)
    return from_polygons(
        centers,
        polygons,
        bonds,
        image_shifts=np.asarray(shifts).reshape((-1, 2)),
        cell_size=tuple(cell),
    )


def save_geometry(path: str | Path, geometry: Geometry) -> None:
    """Write a compressed NumPy archive containing only numeric arrays.

    The exact path is used, without automatically appending an extension.
    Optimization results should be stored separately from this reference
    geometry (or used to construct a new geometry explicitly).
    """
    geometry.validate()
    with Path(path).open("wb") as stream:
        np.savez_compressed(
            stream,
            format_version=np.asarray(1, dtype=np.int32),
            reference_state=np.asarray(geometry.reference_state),
            vertices=np.asarray(geometry.vertices),
            vertex_polygons=np.asarray(geometry.vertex_polygons),
            polygon_offsets=np.asarray(geometry.polygon_offsets),
            bonds=np.asarray(geometry.bonds),
            image_shifts=np.asarray(geometry.image_shifts),
            cell_size=np.asarray(geometry.cell_size if geometry.cell_size is not None else []),
        )


def load_geometry(path: str | Path) -> Geometry:
    """Load and validate an archive produced by :func:`save_geometry`.

    Pickled Python objects are never enabled.
    """
    with np.load(path, allow_pickle=False) as archive:
        if "format_version" not in archive or archive["format_version"].shape != ():
            raise ValueError("Geometry archive has no scalar format_version.")
        if archive["format_version"].item() != 1:
            raise ValueError("Unsupported geometry archive version.")
        names = (
            "reference_state",
            "vertices",
            "vertex_polygons",
            "polygon_offsets",
            "bonds",
            "image_shifts",
            "cell_size",
        )
        missing = set(names) - set(archive.files)
        if missing:
            raise ValueError(f"Geometry archive is missing arrays: {', '.join(sorted(missing))}.")
        values = {name: archive[name] for name in names}
    cell = values.pop("cell_size")
    return Geometry(**values, cell_size=None if cell.size == 0 else tuple(cell))
