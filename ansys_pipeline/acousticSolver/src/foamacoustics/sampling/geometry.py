"""Analytic closed surfaces for volume-field sampling."""

from __future__ import annotations

import math
import operator
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, cast

import torch

from ..utils.validation import positive_float


Vector3 = tuple[float, float, float]


def _finite_vector3(value: Sequence[float], name: str) -> Vector3:
    try:
        vector = tuple(float(component) for component in value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain three finite coordinates.") from error
    if len(vector) != 3 or not all(math.isfinite(component) for component in vector):
        raise ValueError(f"{name} must contain three finite coordinates.")
    return vector[0], vector[1], vector[2]


def _integer_at_least(value: int, minimum: int, name: str) -> int:
    try:
        integer = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer of at least {minimum}.") from error
    if integer < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}.")
    return integer


def _unit_vector3(value: Sequence[float], name: str) -> Vector3:
    vector = _finite_vector3(value, name)
    magnitude = math.sqrt(sum(component * component for component in vector))
    if not math.isfinite(magnitude) or magnitude == 0.0:
        raise ValueError(f"{name} must have a finite, non-zero magnitude.")
    return (
        vector[0] / magnitude,
        vector[1] / magnitude,
        vector[2] / magnitude,
    )


def _mesh_tensors(
    vertices_m: torch.Tensor,
    triangles: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize and validate the shape and values of mesh tensors."""

    vertices = torch.as_tensor(vertices_m, dtype=torch.float64).detach().cpu()
    faces = torch.as_tensor(triangles, dtype=torch.int64).detach().cpu()
    vertices = vertices.clone().contiguous()
    faces = faces.clone().contiguous()
    if vertices.ndim != 2 or vertices.shape[1:] != (3,) or len(vertices) < 4:
        raise ValueError(
            "vertices_m must have shape (n_vertices, 3) with at least four vertices."
        )
    if faces.ndim != 2 or faces.shape[1:] != (3,) or len(faces) < 4:
        raise ValueError(
            "triangles must have shape (n_triangles, 3) with at least four faces."
        )
    if not torch.isfinite(vertices).all().item():
        raise ValueError("vertices_m must contain only finite coordinates.")
    if torch.any(faces < 0).item() or torch.any(faces >= len(vertices)).item():
        raise ValueError("triangles contain an out-of-range vertex index.")
    if torch.any(
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 2] == faces[:, 0])
    ).item():
        raise ValueError("A triangle cannot repeat a vertex index.")
    return vertices, faces


def _validate_closed_topology(triangles: torch.Tensor, vertex_count: int) -> None:
    """Reject duplicate faces, unused vertices, and invalid shared edges."""

    canonical_faces: set[tuple[int, int, int]] = set()
    edge_counts: dict[tuple[int, int], int] = {}
    edge_balance: dict[tuple[int, int], int] = {}
    referenced_vertices: set[int] = set()
    for face in triangles.tolist():
        canonical_face = tuple(sorted(face))
        if canonical_face in canonical_faces:
            raise ValueError("The mesh contains a duplicate triangle.")
        canonical_faces.add(canonical_face)
        referenced_vertices.update(face)
        for start, end in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            edge = (min(start, end), max(start, end))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
            direction = 1 if (start, end) == edge else -1
            edge_balance[edge] = edge_balance.get(edge, 0) + direction

    if len(referenced_vertices) != vertex_count:
        raise ValueError("Every mesh vertex must be referenced by a triangle.")
    if any(count != 2 for count in edge_counts.values()):
        raise ValueError("The triangle mesh is not watertight and two-manifold.")
    if any(balance != 0 for balance in edge_balance.values()):
        raise ValueError(
            "Adjacent triangles must traverse their shared edge in opposite directions."
        )


def _signed_volume(vertices_m: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    """Compute translation-stable signed volume for a triangular mesh."""

    relative_vertices = vertices_m - vertices_m.mean(dim=0)
    triangle_vertices = relative_vertices[triangles]
    return (
        torch.sum(
            torch.einsum(
                "ij,ij->i",
                triangle_vertices[:, 0],
                torch.linalg.cross(
                    triangle_vertices[:, 1],
                    triangle_vertices[:, 2],
                ),
            )
        )
        / 6.0
    )


@dataclass(frozen=True, slots=True, eq=False)
class TriangleMesh:
    """A validated, closed, outward-oriented triangular surface mesh.

    Vertices are stored as CPU ``float64`` coordinates in metres and triangle
    indices as CPU ``int64`` values. Construction rejects degenerate faces,
    open or non-manifold edges, inconsistent winding, and inward global
    orientation.

    Attributes:
        vertices_m: CPU ``float64`` coordinates with shape ``(vertices, 3)``.
        triangles: CPU ``int64`` vertex indices with shape ``(faces, 3)``.
    """

    vertices_m: torch.Tensor
    triangles: torch.Tensor

    def __post_init__(self) -> None:
        vertices, triangles = _mesh_tensors(self.vertices_m, self.triangles)
        _validate_closed_topology(triangles, len(vertices))

        triangle_vertices = vertices[triangles]
        twice_area_vectors = torch.linalg.cross(
            triangle_vertices[:, 1] - triangle_vertices[:, 0],
            triangle_vertices[:, 2] - triangle_vertices[:, 0],
        )
        twice_areas = torch.linalg.vector_norm(twice_area_vectors, dim=1)
        if (
            not torch.isfinite(twice_areas).all().item()
            or torch.any(twice_areas <= 0.0).item()
        ):
            raise ValueError("Every triangle must have a finite, positive area.")

        signed_volume_m3 = _signed_volume(vertices, triangles)
        if not torch.isfinite(signed_volume_m3).item() or signed_volume_m3 <= 0.0:
            raise ValueError(
                "The triangle mesh must enclose positive volume with outward winding."
            )

        object.__setattr__(self, "vertices_m", vertices)
        object.__setattr__(self, "triangles", triangles)

    @property
    def triangle_vertices_m(self) -> torch.Tensor:
        """Triangle vertex coordinates with shape ``(faces, 3, 3)``."""

        return self.vertices_m[self.triangles]

    @property
    def face_centers_m(self) -> torch.Tensor:
        """Arithmetic triangle centres with shape ``(faces, 3)``."""

        return self.triangle_vertices_m.mean(dim=1)

    @property
    def area_vectors_m2(self) -> torch.Tensor:
        """Outward-oriented triangle area vectors."""

        vertices = self.triangle_vertices_m
        return cast(
            torch.Tensor,
            0.5
            * torch.linalg.cross(
                vertices[:, 1] - vertices[:, 0],
                vertices[:, 2] - vertices[:, 0],
            ),
        )

    @property
    def face_areas_m2(self) -> torch.Tensor:
        """Positive triangle areas."""

        return cast(
            torch.Tensor,
            torch.linalg.vector_norm(self.area_vectors_m2, dim=1),
        )

    @property
    def face_normals(self) -> torch.Tensor:
        """Unit outward triangle normals."""

        area_vectors = self.area_vectors_m2
        return cast(
            torch.Tensor,
            area_vectors / torch.linalg.vector_norm(area_vectors, dim=1)[:, None],
        )

    @property
    def signed_volume_m3(self) -> float:
        """Signed enclosed volume, positive for the required outward winding."""

        return float(_signed_volume(self.vertices_m, self.triangles))


def _outward_triangles(
    vertices_m: Sequence[Sequence[float]],
    triangles: Sequence[Sequence[int]],
    interior_point_m: Sequence[float],
) -> torch.Tensor:
    vertices = torch.tensor(vertices_m, dtype=torch.float64)
    oriented = torch.tensor(triangles, dtype=torch.int64)
    triangle_vertices = vertices[oriented]
    area_directions = torch.linalg.cross(
        triangle_vertices[:, 1] - triangle_vertices[:, 0],
        triangle_vertices[:, 2] - triangle_vertices[:, 0],
    )
    interior = torch.tensor(interior_point_m, dtype=torch.float64)
    outward_directions = triangle_vertices.mean(dim=1) - interior
    orientation = torch.einsum("ij,ij->i", area_directions, outward_directions)
    if torch.any(orientation == 0.0).item():
        raise ValueError("Generated a triangle whose outward direction is ambiguous.")
    inward = orientation < 0.0
    if torch.any(inward).item():
        old_second = oriented[inward, 1].clone()
        oriented[inward, 1] = oriented[inward, 2]
        oriented[inward, 2] = old_second
    return oriented


@dataclass(frozen=True, slots=True)
class SphereSurface:
    """Closed triangulated sphere centred at ``center_m``.

    Attributes:
        radius_m: Positive sphere radius in metres.
        center_m: Sphere centre in metres.
        polar_divisions: Number of pole-to-pole angular intervals, at least two.
        azimuthal_divisions: Number of circumferential intervals, at least three.
    """

    radius_m: float
    center_m: Vector3 = (0.0, 0.0, 0.0)
    polar_divisions: int = 32
    azimuthal_divisions: int = 64

    def __post_init__(self) -> None:
        object.__setattr__(self, "radius_m", positive_float(self.radius_m, "radius_m"))
        object.__setattr__(self, "center_m", _finite_vector3(self.center_m, "center_m"))
        object.__setattr__(
            self,
            "polar_divisions",
            _integer_at_least(self.polar_divisions, 2, "polar_divisions"),
        )
        object.__setattr__(
            self,
            "azimuthal_divisions",
            _integer_at_least(self.azimuthal_divisions, 3, "azimuthal_divisions"),
        )

    def triangulate(self) -> TriangleMesh:
        """Return a deterministic, outward-oriented triangular sphere mesh.

        Returns:
            A validated closed mesh with the configured angular resolution.
        """

        cx, cy, cz = self.center_m
        vertices: list[Vector3] = [(cx, cy, cz + self.radius_m)]
        for polar_index in range(1, self.polar_divisions):
            polar_angle = math.pi * polar_index / self.polar_divisions
            radial = self.radius_m * math.sin(polar_angle)
            z = cz + self.radius_m * math.cos(polar_angle)
            for azimuthal_index in range(self.azimuthal_divisions):
                azimuth = 2.0 * math.pi * azimuthal_index / self.azimuthal_divisions
                vertices.append(
                    (
                        cx + radial * math.cos(azimuth),
                        cy + radial * math.sin(azimuth),
                        z,
                    )
                )
        south_pole = len(vertices)
        vertices.append((cx, cy, cz - self.radius_m))

        triangles: list[tuple[int, int, int]] = []
        first_ring = 1
        for azimuthal_index in range(self.azimuthal_divisions):
            next_index = (azimuthal_index + 1) % self.azimuthal_divisions
            triangles.append((0, first_ring + azimuthal_index, first_ring + next_index))

        for polar_index in range(self.polar_divisions - 2):
            upper_start = 1 + polar_index * self.azimuthal_divisions
            lower_start = upper_start + self.azimuthal_divisions
            for azimuthal_index in range(self.azimuthal_divisions):
                next_index = (azimuthal_index + 1) % self.azimuthal_divisions
                upper = upper_start + azimuthal_index
                upper_next = upper_start + next_index
                lower = lower_start + azimuthal_index
                lower_next = lower_start + next_index
                triangles.append((upper, lower, lower_next))
                triangles.append((upper, lower_next, upper_next))

        last_ring = 1 + (self.polar_divisions - 2) * self.azimuthal_divisions
        for azimuthal_index in range(self.azimuthal_divisions):
            next_index = (azimuthal_index + 1) % self.azimuthal_divisions
            triangles.append(
                (south_pole, last_ring + next_index, last_ring + azimuthal_index)
            )

        oriented = _outward_triangles(vertices, triangles, self.center_m)
        return TriangleMesh(torch.tensor(vertices, dtype=torch.float64), oriented)


@dataclass(frozen=True, slots=True)
class CylinderSurface:
    """Closed capped cylinder defined by its centre, axis, and full length.

    Attributes:
        radius_m: Positive cylinder radius in metres.
        length_m: Positive full cap-to-cap length in metres.
        center_m: Midpoint between the two cap centres.
        axis: Unit vector from the lower cap towards the upper cap.
        axial_divisions: Number of intervals along the side wall.
        azimuthal_divisions: Number of circumferential intervals.
        cap_radial_divisions: Number of radial intervals on each cap.
    """

    radius_m: float
    length_m: float
    center_m: Vector3 = (0.0, 0.0, 0.0)
    axis: Vector3 = (0.0, 0.0, 1.0)
    axial_divisions: int = 16
    azimuthal_divisions: int = 64
    cap_radial_divisions: int = 8

    def __post_init__(self) -> None:
        object.__setattr__(self, "radius_m", positive_float(self.radius_m, "radius_m"))
        object.__setattr__(self, "length_m", positive_float(self.length_m, "length_m"))
        object.__setattr__(self, "center_m", _finite_vector3(self.center_m, "center_m"))
        object.__setattr__(self, "axis", _unit_vector3(self.axis, "axis"))
        object.__setattr__(
            self,
            "axial_divisions",
            _integer_at_least(self.axial_divisions, 1, "axial_divisions"),
        )
        object.__setattr__(
            self,
            "azimuthal_divisions",
            _integer_at_least(self.azimuthal_divisions, 3, "azimuthal_divisions"),
        )
        object.__setattr__(
            self,
            "cap_radial_divisions",
            _integer_at_least(self.cap_radial_divisions, 1, "cap_radial_divisions"),
        )

    @classmethod
    def from_endpoints(
        cls,
        start_m: Sequence[float],
        end_m: Sequence[float],
        *,
        radius_m: float,
        axial_divisions: int = 16,
        azimuthal_divisions: int = 64,
        cap_radial_divisions: int = 8,
    ) -> CylinderSurface:
        """Construct a cylinder whose cap centres are ``start_m`` and ``end_m``.

        Args:
            start_m: Centre of the lower cap in metres.
            end_m: Centre of the upper cap in metres.
            radius_m: Positive cylinder radius in metres.
            axial_divisions: Number of side-wall intervals along the axis.
            azimuthal_divisions: Number of circumferential intervals.
            cap_radial_divisions: Number of radial intervals on each cap.

        Returns:
            A cylinder centred between the endpoints and aligned from start to end.

        Raises:
            ValueError: If either point is invalid or the endpoints coincide.
        """

        start = _finite_vector3(start_m, "start_m")
        end = _finite_vector3(end_m, "end_m")
        displacement: Vector3 = (
            end[0] - start[0],
            end[1] - start[1],
            end[2] - start[2],
        )
        length = math.sqrt(sum(component * component for component in displacement))
        if not math.isfinite(length) or length == 0.0:
            raise ValueError("start_m and end_m must be distinct finite points.")
        center: Vector3 = (
            0.5 * (start[0] + end[0]),
            0.5 * (start[1] + end[1]),
            0.5 * (start[2] + end[2]),
        )
        return cls(
            radius_m=radius_m,
            length_m=length,
            center_m=center,
            axis=displacement,
            axial_divisions=axial_divisions,
            azimuthal_divisions=azimuthal_divisions,
            cap_radial_divisions=cap_radial_divisions,
        )

    def triangulate(self) -> TriangleMesh:
        """Return a deterministic, outward-oriented capped-cylinder mesh.

        Returns:
            A validated closed mesh with independently resolved wall and caps.
        """

        center = torch.tensor(self.center_m, dtype=torch.float64)
        axis = torch.tensor(self.axis, dtype=torch.float64)
        reference = torch.zeros(3, dtype=torch.float64)
        reference[int(torch.argmin(torch.abs(axis)))] = 1.0
        radial_u = reference - torch.dot(reference, axis) * axis
        radial_u /= torch.linalg.vector_norm(radial_u)
        radial_v = torch.linalg.cross(axis, radial_u)

        azimuths = (
            2.0 * math.pi * index / self.azimuthal_divisions
            for index in range(self.azimuthal_divisions)
        )
        radial_directions = torch.stack(
            [
                math.cos(azimuth) * radial_u + math.sin(azimuth) * radial_v
                for azimuth in azimuths
            ]
        )
        side_rings = [
            center
            + (axial_index / self.axial_divisions - 0.5) * self.length_m * axis
            + self.radius_m * radial_directions
            for axial_index in range(self.axial_divisions + 1)
        ]
        vertex_blocks = [torch.cat(side_rings, dim=0)]
        vertex_count = vertex_blocks[0].shape[0]

        bottom_cap_rings: list[int] = []
        top_cap_rings: list[int] = []
        bottom_cap_center = center - 0.5 * self.length_m * axis
        top_cap_center = center + 0.5 * self.length_m * axis
        for radial_index in range(1, self.cap_radial_divisions):
            radial_fraction = radial_index / self.cap_radial_divisions
            radial_offsets = radial_fraction * self.radius_m * radial_directions
            bottom_cap_rings.append(vertex_count)
            vertex_blocks.append(bottom_cap_center + radial_offsets)
            vertex_count += self.azimuthal_divisions
            top_cap_rings.append(vertex_count)
            vertex_blocks.append(top_cap_center + radial_offsets)
            vertex_count += self.azimuthal_divisions

        bottom_center = vertex_count
        vertex_blocks.append(bottom_cap_center[None, :])
        top_center = vertex_count + 1
        vertex_blocks.append(top_cap_center[None, :])
        vertices = torch.cat(vertex_blocks, dim=0)

        triangles: list[tuple[int, int, int]] = []
        for axial_index in range(self.axial_divisions):
            lower_start = axial_index * self.azimuthal_divisions
            upper_start = lower_start + self.azimuthal_divisions
            for azimuthal_index in range(self.azimuthal_divisions):
                next_index = (azimuthal_index + 1) % self.azimuthal_divisions
                lower = lower_start + azimuthal_index
                lower_next = lower_start + next_index
                upper = upper_start + azimuthal_index
                upper_next = upper_start + next_index
                triangles.append((lower, lower_next, upper_next))
                triangles.append((lower, upper_next, upper))

        top_ring = self.axial_divisions * self.azimuthal_divisions
        bottom_cap_rings.append(0)
        top_cap_rings.append(top_ring)
        for cap_center, cap_rings in (
            (bottom_center, bottom_cap_rings),
            (top_center, top_cap_rings),
        ):
            inner_ring = cap_rings[0]
            for azimuthal_index in range(self.azimuthal_divisions):
                next_index = (azimuthal_index + 1) % self.azimuthal_divisions
                triangles.append(
                    (cap_center, inner_ring + azimuthal_index, inner_ring + next_index)
                )

            for inner_ring, outer_ring in zip(cap_rings, cap_rings[1:]):
                for azimuthal_index in range(self.azimuthal_divisions):
                    next_index = (azimuthal_index + 1) % self.azimuthal_divisions
                    triangles.append(
                        (
                            inner_ring + azimuthal_index,
                            outer_ring + azimuthal_index,
                            outer_ring + next_index,
                        )
                    )
                    triangles.append(
                        (
                            inner_ring + azimuthal_index,
                            outer_ring + next_index,
                            inner_ring + next_index,
                        )
                    )

        oriented = _outward_triangles(vertices.tolist(), triangles, self.center_m)
        return TriangleMesh(vertices, oriented)


SurfaceGeometry = SphereSurface | CylinderSurface


def _stl_float(value: float) -> str:
    if value == 0.0:
        value = 0.0
    return f"{value:.17g}"


def write_ascii_stl(
    mesh: TriangleMesh,
    path: str | Path,
    *,
    solid_name: str = "surface",
) -> Path:
    """Write a validated triangle mesh as deterministic ASCII STL.

    Args:
        mesh: Closed triangular mesh to serialize.
        path: Destination file. Its parent directory must already exist.
        solid_name: Non-empty ASCII name written to the STL header and footer.

    Returns:
        The normalized destination path.

    Raises:
        ValueError: If ``solid_name`` is empty, non-ASCII, or contains a newline.
    """

    solid_name = str(solid_name)
    if not solid_name.strip() or "\n" in solid_name or "\r" in solid_name:
        raise ValueError("solid_name must be non-empty and cannot contain a newline.")
    try:
        solid_name.encode("ascii")
    except UnicodeEncodeError as error:
        raise ValueError("solid_name must contain only ASCII characters.") from error

    path = Path(path)
    normals = mesh.face_normals.tolist()
    triangle_vertices = mesh.triangle_vertices_m.tolist()
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(f"solid {solid_name}\n")
        for normal, vertices in zip(normals, triangle_vertices, strict=True):
            stream.write(
                "  facet normal "
                + " ".join(_stl_float(value) for value in normal)
                + "\n"
            )
            stream.write("    outer loop\n")
            for vertex in vertices:
                stream.write(
                    "      vertex "
                    + " ".join(_stl_float(value) for value in vertex)
                    + "\n"
                )
            stream.write("    endloop\n")
            stream.write("  endfacet\n")
        stream.write(f"endsolid {solid_name}\n")
    return path
