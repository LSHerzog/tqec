"""Write a PrismGraph to a Collada DAE file."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import BinaryIO

import collada
import collada.source
import numpy as np
import numpy.typing as npt

from prism import BasisPrism, Port, ZXPrism
from prism_graph import PrismGraph
from pipe_prism import PrismPipe, PrismPipeKind
from _geometry_prism import (
    PrismColor,
    PrismGeometry,
    PipeGeometry,
    TriangleFace,
    get_prism_geometry,
    get_horizontal_pipe_geometry,
    get_vertical_pipe_geometry,
)

# ---------------------------------------------------------------------------
# Collada asset metadata
# ---------------------------------------------------------------------------

_ASSET_AUTHOR = "TQEC Community"
_ASSET_AUTHORING_TOOL = "prism_graph_writer"
_ASSET_UNIT_NAME = "inch"
_ASSET_UNIT_METER = 0.02539999969303608
_MATERIAL_SYMBOL = "MaterialSymbol"


# ---------------------------------------------------------------------------
# Coordinate conversion: Position3DHex → Euclidean 3D
# ---------------------------------------------------------------------------

def _hex_to_euclidean(x: int, y: int, z: int, z_spacing: float) -> tuple[float, float, float]:
    """Convert mhwombat triangular grid (x, y, z) to Euclidean cell_origin (cx, cy, cz).

    Returns the bottom-left corner of the unit triangle template for tile (x, y).
    Valid for both ▲ (even x) and ▽ (odd x) — no orientation offset needed.

    The formula places rows y=2k and y=2k+1 in a horizontal band at height k*h,
    with a half-unit horizontal stagger per completed row-pair.

    Even x → upward triangle ▲
    Odd  x → downward triangle ▽
    """
    h = np.sqrt(3) / 2
    cx = x / 2.0 + (y // 2) * 0.5
    cy = (y // 2) * h
    cz = z * z_spacing
    return cx, cy, cz


# ---------------------------------------------------------------------------
# Library key: identifies a unique prism shape (kind + orientation)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _PrismLibraryKey:
    """Unique identifier for a prism geometry in the Collada library.

    Attributes:
        kind_str:    String representation of the ZXPrism kind, e.g. "XZ".
        pointing_up: True for ▲, False for ▽.
    """
    kind_str: str
    pointing_up: bool

    def __str__(self) -> str:
        orientation = "up" if self.pointing_up else "down"
        return f"prism_{self.kind_str}_{orientation}"


@dataclass(frozen=True)
class _PipeLibraryKey:
    """Unique identifier for a pipe geometry in the Collada library.

    Vertical pipes are identified by basis + orientation and are placed via
    a translation matrix (geometry in local space).
    Horizontal pipes are identified by basis + edge direction and are placed
    via an identity matrix (geometry already in world space per instance).

    Attributes:
        basis:       "X" or "Z".
        pipe_kind:   "vertical_up", "vertical_down", "horizontal_bottom",
                     "horizontal_right", or "horizontal_left".
    """
    basis: str
    pipe_kind: str

    def __str__(self) -> str:
        return f"pipe_{self.basis}_{self.pipe_kind}"


# ---------------------------------------------------------------------------
# _BasePrismColladaData: owns the collada.Collada mesh + all library nodes
# ---------------------------------------------------------------------------

class _BasePrismColladaData:
    """Builds and owns the Collada scene for a PrismGraph.

    Analogous to _BaseColladaData in read_write.py.

    Workflow:
        1. __init__ sets up the mesh, scene, materials.
        2. add_prism_instance() is called once per prism in the graph.
        3. mesh.write() serialises everything to a file.
    """

    def __init__(self) -> None:
        self.mesh = collada.Collada()

        # maps PrismColor → collada Material
        self.materials: dict[PrismColor, collada.material.Material] = {}

        # maps _PrismLibraryKey → collada Node (reusable prism template)
        self.prism_library: dict[_PrismLibraryKey, collada.scene.Node] = {}

        # maps _PipeLibraryKey → collada Node (reusable pipe template)
        # Note: horizontal pipes are NOT cached here because their geometry is
        # instance-specific (world-space coordinates baked in).  Vertical pipes
        # ARE cached because their geometry is in local space and reused.
        self.pipe_library: dict[_PipeLibraryKey, collada.scene.Node] = {}

        # root scene node — all instances hang off this
        self.root_node = collada.scene.Node("SketchUp", name="SketchUp")

        self._num_instances: int = 0
        self._num_geometries: int = 0

        self._setup_scene()
        self._add_asset_info()
        self._add_materials()

    # ------------------------------------------------------------------
    # Scene / asset / material setup
    # ------------------------------------------------------------------

    def _setup_scene(self) -> None:
        scene = collada.scene.Scene("scene", [self.root_node])
        self.mesh.scenes.append(scene)
        self.mesh.scene = scene

    def _add_asset_info(self) -> None:
        if self.mesh.assetInfo is None:
            return
        self.mesh.assetInfo.contributors.append(
            collada.asset.Contributor(
                author=_ASSET_AUTHOR,
                authoring_tool=_ASSET_AUTHORING_TOOL,
            )
        )
        self.mesh.assetInfo.unitmeter = _ASSET_UNIT_METER
        self.mesh.assetInfo.unitname = _ASSET_UNIT_NAME
        self.mesh.assetInfo.upaxis = collada.asset.UP_AXIS.Z_UP

    def _add_materials(self) -> None:
        """Create one Lambert material per PrismColor."""
        for color in PrismColor:
            r, g, b, a = color.rgba
            rgba = (r, g, b, a)
            effect = collada.material.Effect(
                f"{color.value}_effect",
                [],
                "lambert",
                diffuse=rgba,
                emission=None,
                specular=None,
                transparent=rgba,
                transparency=a,
                ambient=None,
                reflective=None,
                double_sided=True,
            )
            self.mesh.effects.append(effect)
            material = collada.material.Material(
                f"{color.value}_material",
                f"{color.value}_material",
                effect,
            )
            self.mesh.materials.append(material)
            self.materials[color] = material

    # ------------------------------------------------------------------
    # Face geometry (triangles only — quads are pre-split)
    # ------------------------------------------------------------------

    def _add_triangle_face_geometry_node(
        self, face: TriangleFace
    ) -> collada.scene.GeometryNode:
        """Register a TriangleFace as a Collada geometry and return its node."""
        gid = f"FaceID{self._num_geometries}"
        self._num_geometries += 1

        # vertices: shape (3, 3) → flatten to (9,)
        verts_flat = face.vertices.flatten().astype(np.float64)

        # normal: same for all 3 vertices → tile to (9,)
        normal = face.get_normal()
        norm_len = np.linalg.norm(normal)
        if norm_len > 1e-10:
            normal = normal / norm_len
        normals_flat = np.tile(normal, 3).astype(np.float64)

        positions_src = collada.source.FloatSource(
            gid + "_positions", verts_flat, ("X", "Y", "Z")
        )
        normals_src = collada.source.FloatSource(
            gid + "_normals", normals_flat, ("X", "Y", "Z")
        )

        geom = collada.geometry.Geometry(
            self.mesh, gid, gid, [positions_src, normals_src]
        )
        input_list = collada.source.InputList()
        input_list.addInput(0, "VERTEX", "#" + positions_src.id)
        input_list.addInput(1, "NORMAL", "#" + normals_src.id)

        # one triangle: indices [0, 1, 2]
        indices = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        triset = geom.createTriangleSet(indices, input_list, _MATERIAL_SYMBOL)
        geom.primitives.append(triset)
        self.mesh.geometries.append(geom)

        material = self.materials[face.color]
        geom_node = collada.scene.GeometryNode(
            geom,
            [collada.scene.MaterialNode(_MATERIAL_SYMBOL, material, [("UVSET0", "TEXCOORD", "0")])],
        )
        return geom_node

    # ------------------------------------------------------------------
    # Prism library node (reusable template per kind+orientation)
    # ------------------------------------------------------------------

    def _get_or_create_prism_library_node(
        self, key: _PrismLibraryKey, geometry: PrismGeometry
    ) -> collada.scene.Node:
        """Return (creating if needed) the library node for this prism shape."""
        if key in self.prism_library:
            return self.prism_library[key]

        # collect geometry nodes for every triangle face of this prism
        children: list[collada.scene.GeometryNode] = []
        for tri in geometry.all_triangle_faces():
            children.append(self._add_triangle_face_geometry_node(tri))

        node = collada.scene.Node(str(key), children, name=str(key))
        self.mesh.nodes.append(node)
        self.prism_library[key] = node
        return node

    # ------------------------------------------------------------------
    # Pipe library node (cached for vertical; unique per instance for horizontal)
    # ------------------------------------------------------------------

    def _build_pipe_node(
        self, node_id: str, geometry: PipeGeometry
    ) -> collada.scene.Node:
        """Build a Collada node from a PipeGeometry (not cached)."""
        children: list[collada.scene.GeometryNode] = []
        for tri in geometry.all_triangle_faces():
            children.append(self._add_triangle_face_geometry_node(tri))
        node = collada.scene.Node(node_id, children, name=node_id)
        self.mesh.nodes.append(node)
        return node

    def _get_or_create_vertical_pipe_library_node(
        self, key: _PipeLibraryKey, geometry: PipeGeometry
    ) -> collada.scene.Node:
        """Return (creating if needed) the library node for a vertical pipe shape."""
        if key in self.pipe_library:
            return self.pipe_library[key]
        node = self._build_pipe_node(str(key), geometry)
        self.pipe_library[key] = node
        return node

    # ------------------------------------------------------------------
    # Public API: place one prism instance in the scene
    # ------------------------------------------------------------------

    def add_prism_instance(
        self,
        transform_matrix: npt.NDArray[np.float32],
        key: _PrismLibraryKey,
        geometry: PrismGeometry,
    ) -> None:
        """Add one prism instance to the scene at the given transform."""
        library_node = self._get_or_create_prism_library_node(key, geometry)
        self._add_instance_node(library_node, transform_matrix)

    def add_vertical_pipe_instance(
        self,
        transform_matrix: npt.NDArray[np.float32],
        key: _PipeLibraryKey,
        geometry: PipeGeometry,
    ) -> None:
        """Add one vertical pipe instance (geometry in local space, translated)."""
        library_node = self._get_or_create_vertical_pipe_library_node(key, geometry)
        self._add_instance_node(library_node, transform_matrix)

    def add_horizontal_pipe_instance(
        self,
        geometry: PipeGeometry,
    ) -> None:
        """Add one horizontal pipe instance (geometry already in world space)."""
        # Each horizontal pipe has unique world-space geometry, so no library caching.
        node_id = f"hpipe_{self._num_instances}"
        pipe_node = self._build_pipe_node(node_id, geometry)
        identity = np.eye(4, dtype=np.float32)
        self._add_instance_node(pipe_node, identity)

    def _add_instance_node(
        self,
        library_node: collada.scene.Node,
        transform_matrix: npt.NDArray[np.float32],
    ) -> None:
        """Place a library node into the scene with the given transform."""
        instance_id = f"ID{self._num_instances}"
        child_node = collada.scene.Node(
            instance_id,
            name=f"instance_{self._num_instances}",
            transforms=[collada.scene.MatrixTransform(transform_matrix.flatten())],
        )
        child_node.children.append(collada.scene.NodeNode(library_node))
        self.root_node.children.append(child_node)
        self._num_instances += 1


# ---------------------------------------------------------------------------
# Helper: basis enum → string for geometry lookup
# ---------------------------------------------------------------------------

def _basis_str(b: BasisPrism) -> str:
    return b.value  # "X", "Z", or "N"


def _scaled_cell_origin(x: int, y: int, scale: float) -> tuple[float, float]:
    """Return the scaled cell_origin for triangle (x, y).

    Scaling is applied to the centroid (not the cell_origin corner) so that
    centroid-to-centroid distances expand uniformly in all directions.
    The cell_origin is back-computed from the scaled centroid.

      ▲ (even x): centroid offset from cell_origin = (+0.5, +h/3)
      ▽ (odd  x): centroid offset from cell_origin = (+0.5, +2h/3)
    """
    h = np.sqrt(3) / 2
    pointing_up = (x % 2 == 0)
    cy_offset = h / 3 if pointing_up else 2 * h / 3

    ox = x / 2.0 + (y // 2) * 0.5
    oy = (y // 2) * h

    cx = (ox + 0.5) * scale
    cy = (oy + cy_offset) * scale

    return cx - 0.5, cy - cy_offset


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def write_prism_graph_to_dae_file(
    prism_graph: PrismGraph,
    file_like: str | pathlib.Path | BinaryIO,
    spacing: float = 2.0,
) -> None:
    """Write a PrismGraph to a Collada DAE file.

    Each prism is placed at its Euclidean position derived from its
    Position3DHex(x, y, z) using the mhwombat triangular coordinate scheme.
    Port prisms are skipped (they are invisible boundary markers).

    The spacing parameter adds a uniform gap between all prisms in every
    direction (x, y, and z), so the visual separation is consistent.
    A value of 0.0 means prisms are placed edge-to-edge.

    Args:
        prism_graph: The prism graph to export.
        file_like:   Output file path or binary file-like object.
        spacing:     Extra gap between adjacent prisms in all directions.
                     Default is 2.0.
    """
    base = _BasePrismColladaData()

    h = np.sqrt(3) / 2  # height of unit equilateral triangle

    # spacing > 0 adds a uniform gap between adjacent prisms in all directions.
    # xy scaling: centroid-to-centroid = scale = (1 + spacing), so the gap between
    # parallel faces = spacing / sqrt(3) (since centroid-to-face = 1/(2*sqrt(3))).
    # z scaling: to match the horizontal gap, we use z_step = 1 + spacing/sqrt(3),
    # so the z-gap between prisms also equals spacing / sqrt(3).
    scale  = 1.0 + spacing                    # xy centroid scale factor
    z_step = 1.0 + spacing / np.sqrt(3)       # z distance between prism origins

    for position, attrs in prism_graph._graph.nodes(data=True):
        prism = attrs[prism_graph._NODE_DATA_KEY]

        if prism.is_port:
            continue
        if not prism.is_zx_prism:
            continue

        kind: ZXPrism = prism.kind  # type: ignore[assignment]
        x, y, z = position.x, position.y, position.z

        pointing_up = (x % 2 == 0)

        cell_origin_x, cell_origin_y = _scaled_cell_origin(x, y, scale)
        cell_origin_z = z * z_step

        matrix = np.eye(4, dtype=np.float32)
        matrix[0, 3] = float(cell_origin_x)
        matrix[1, 3] = float(cell_origin_y)
        matrix[2, 3] = float(cell_origin_z)

        geometry = get_prism_geometry(
            prep=_basis_str(kind.prep),
            meas=_basis_str(kind.meas),
            pointing_up=pointing_up,
        )

        key = _PrismLibraryKey(
            kind_str=str(kind),
            pointing_up=pointing_up,
        )

        base.add_prism_instance(matrix, key, geometry)

    # ------------------------------------------------------------------
    # Edge loop: render pipes between adjacent prisms
    # ------------------------------------------------------------------

    for pos1, pos2, edge_attrs in prism_graph._graph.edges(data=True):
        pipe = edge_attrs.get(prism_graph._EDGE_DATA_KEY)
        if pipe is None:
            continue

        pipe_kind: PrismPipeKind = pipe.kind
        hor = str(pipe_kind.hor) if pipe_kind.hor is not None else "N"
        ver = str(pipe_kind.ver) if pipe_kind.ver is not None else "N"

        if pos1.x == pos2.x and pos1.y == pos2.y:
            # ----------------------------------------------------------
            # VERTICAL pipe: same (x,y), adjacent z levels
            # Geometry is in local space (same as the prism template).
            # We reuse the same library node for all vertical pipes of
            # the same basis + orientation, translated to position.
            # ----------------------------------------------------------
            z_low = min(pos1.z, pos2.z)
            x, y  = pos1.x, pos1.y
            pointing_up = (x % 2 == 0)

            # Gap spans from top of lower prism to bottom of upper prism
            z_pipe_bottom = z_low * z_step + 1.0
            z_pipe_top    = (z_low + 1) * z_step

            geometry = get_vertical_pipe_geometry(
                hor=hor,
                ver=ver,
                pointing_up=pointing_up,
                z_bottom=z_pipe_bottom,
                z_top=z_pipe_top,
            )

            ox_scaled, oy_scaled = _scaled_cell_origin(x, y, scale)

            matrix = np.eye(4, dtype=np.float32)
            matrix[0, 3] = float(ox_scaled)
            matrix[1, 3] = float(oy_scaled)
            matrix[2, 3] = 0.0  # z already baked into geometry

            pipe_key = _PipeLibraryKey(
                basis=f"{hor}_{ver}",
                pipe_kind=f"vertical_{'up' if pointing_up else 'down'}_z{z_low}",
            )
            base.add_vertical_pipe_instance(matrix, pipe_key, geometry)

        else:
            # ----------------------------------------------------------
            # HORIZONTAL pipe: adjacent (x,y), same z
            # Geometry is built in world space (no translation needed).
            # ----------------------------------------------------------

            # Work from the ▲ prism's perspective to name the edge direction
            if pos1.x % 2 == 0:
                tri_up, tri_dn = pos1, pos2
            else:
                tri_up, tri_dn = pos2, pos1

            dx = tri_dn.x - tri_up.x
            dy = tri_dn.y - tri_up.y

            # Map relative position to edge name (from ▲'s perspective)
            if   (dx, dy) == (+1, +1):
                edge_name = "right"
            elif (dx, dy) == (-1, +1):
                edge_name = "left"
            else:  # (dx, dy) == (+1, -1)
                edge_name = "bottom"

            # Cell origin of the ▲ prism in world (scaled) space
            x, y, z = tri_up.x, tri_up.y, tri_up.z

            ox_scaled, oy_scaled = _scaled_cell_origin(x, y, scale)

            z_bottom = z * z_step
            z_top    = z * z_step + 1.0

            geometry = get_horizontal_pipe_geometry(
                hor=hor,
                ver=ver,
                edge=edge_name,
                ox=ox_scaled,
                oy=oy_scaled,
                depth=spacing / np.sqrt(3),   # gap between faces = spacing/sqrt(3)
                z_bottom=z_bottom,
                z_top=z_top,
            )

            base.add_horizontal_pipe_instance(geometry)

    base.mesh.write(file_like)