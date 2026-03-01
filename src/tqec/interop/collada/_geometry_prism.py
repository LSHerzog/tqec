"""Define the face geometries for triangular prism blocks and pipes used in COLLADA models.

A ZXPrism is a triangular prism standing upright in the z-direction:
  - Bottom face (prep basis): equilateral triangle at z=0
  - Top face (meas basis):    equilateral triangle at z=1
  - 3 rectangular side faces: connecting bottom to top, always transparent

A PrismPipe connects two adjacent prisms. There are two kinds:
  - Vertical pipe:   same (x,y), adjacent z — triangular cross-section, fills the z-gap
  - Horizontal pipe: adjacent (x,y), same z — rectangular box filling the xy-gap

Triangle orientation follows the mhwombat coordinate scheme:
  - Even x-coordinate → upward triangle   ▲
  - Odd  x-coordinate → downward triangle ▽
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------------------
# Color definitions
# ---------------------------------------------------------------------------

class PrismColor(Enum):
    """Colors used for prism faces."""
    X    = "X"      # X basis → red  (matching tqec convention)
    Z    = "Z"      # Z basis → blue
    N    = "N"      # No basis → fully transparent
    SIDE = "SIDE"   # Side faces → always fully transparent
    PORT = "PORT"   # Port → fully transparent

    @property
    def rgba(self) -> tuple[float, float, float, float]:
        return {
            "X":    (0.8, 0.2, 0.2, 1.0),   # red,  fully opaque
            "Z":    (0.2, 0.2, 0.8, 1.0),   # blue, fully opaque
            "N":    (0.0, 0.0, 0.0, 0.0),   # fully transparent
            "SIDE": (0.0, 0.0, 0.0, 0.0),   # fully transparent
            "PORT": (0.0, 0.0, 0.0, 0.0),   # fully transparent
        }[self.value]

    @property
    def is_transparent(self) -> bool:
        return self.rgba[3] == 0.0


# ---------------------------------------------------------------------------
# Low-level mesh primitives
# ---------------------------------------------------------------------------

@dataclass
class TriangleFace:
    """A triangular face defined by 3 vertices and a color.

    Vertices are in counter-clockwise order when viewed from outside.
    """
    vertices: npt.NDArray[np.float64]   # shape (3, 3) — three xyz points
    color: PrismColor

    def get_normal(self) -> npt.NDArray[np.float64]:
        """Compute the face normal via cross product."""
        v0, v1, v2 = self.vertices
        return np.cross(v1 - v0, v2 - v0)


@dataclass
class QuadFace:
    """A rectangular (quad) face defined by 4 vertices and a color.

    Vertices are in counter-clockwise order: bottom-left, bottom-right,
    top-right, top-left when viewed from outside.
    """
    vertices: npt.NDArray[np.float64]   # shape (4, 3) — four xyz points
    color: PrismColor

    def get_normal(self) -> npt.NDArray[np.float64]:
        """Compute the face normal via cross product of first triangle."""
        v0, v1, v2, _ = self.vertices
        return np.cross(v1 - v0, v2 - v0)

    def to_triangles(self) -> tuple[TriangleFace, TriangleFace]:
        """Split quad into two triangles for collada export."""
        v = self.vertices
        t1 = TriangleFace(np.array([v[0], v[1], v[2]]), self.color)
        t2 = TriangleFace(np.array([v[0], v[2], v[3]]), self.color)
        return t1, t2

    def to_triangles_double_sided(self) -> list[TriangleFace]:
        """Split quad into four triangles (front + back winding) for double-sided rendering.

        Emits each triangle twice with reversed winding so the face is visible
        from both sides regardless of the renderer's backface culling or
        transparent depth-sorting behaviour.
        """
        v = self.vertices
        t1f = TriangleFace(np.array([v[0], v[1], v[2]]), self.color)
        t2f = TriangleFace(np.array([v[0], v[2], v[3]]), self.color)
        t1b = TriangleFace(np.array([v[2], v[1], v[0]]), self.color)
        t2b = TriangleFace(np.array([v[3], v[2], v[0]]), self.color)
        return [t1f, t2f, t1b, t2b]


# ---------------------------------------------------------------------------
# Triangle vertex helpers
# ---------------------------------------------------------------------------

def _triangle_vertices_2d(pointing_up: bool) -> npt.NDArray[np.float64]:
    """Return the 2D (x, y) vertices of a unit equilateral triangle.

    Args:
        pointing_up: True for ▲, False for ▽.

    Returns:
        Array of shape (3, 2) with the triangle vertices.
    """
    h = np.sqrt(3) / 2  # height of equilateral triangle with side 1
    if pointing_up:
        return np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, h  ],
        ], dtype=np.float64)
    else:
        return np.array([
            [0.0, h  ],
            [1.0, h  ],
            [0.5, 0.0],
        ], dtype=np.float64)


def _make_horizontal_triangle(
    verts_2d: npt.NDArray[np.float64],
    z: float,
    color: PrismColor,
    flip_normal: bool = False,
) -> TriangleFace:
    """Build a horizontal (z = const) triangular face from 2D vertices."""
    verts_3d = np.column_stack([verts_2d, np.full(3, z)])
    if flip_normal:
        verts_3d = verts_3d[[0, 2, 1]]  # reverse winding for downward normal
    return TriangleFace(verts_3d, color)


def _make_side_quad(
    v0_2d: npt.NDArray[np.float64],
    v1_2d: npt.NDArray[np.float64],
    z_bottom: float,
    z_top: float,
    color: PrismColor,
) -> QuadFace:
    """Build a rectangular side face between two 2D edge vertices.

    Args:
        v0_2d: First edge vertex (x, y).
        v1_2d: Second edge vertex (x, y).
        z_bottom: z coordinate of the bottom edge.
        z_top:    z coordinate of the top edge.
        color:    Face color.
    """
    verts = np.array([
        [v0_2d[0], v0_2d[1], z_bottom],  # bottom-left
        [v1_2d[0], v1_2d[1], z_bottom],  # bottom-right
        [v1_2d[0], v1_2d[1], z_top   ],  # top-right
        [v0_2d[0], v0_2d[1], z_top   ],  # top-left
    ], dtype=np.float64)
    return QuadFace(verts, color)


# ---------------------------------------------------------------------------
# PrismGeometry: all faces for one prism kind + orientation
# ---------------------------------------------------------------------------

@dataclass
class PrismGeometry:
    """All faces making up a single triangular prism instance.

    Attributes:
        bottom_face:  The triangular prep face (z=0).
        top_face:     The triangular meas face (z=1).
        side_faces:   The 3 rectangular lateral faces.
    """
    bottom_face: TriangleFace
    top_face: TriangleFace
    side_faces: list[QuadFace]

    def all_triangle_faces(self) -> list[TriangleFace]:
        """Return all faces as triangles (splitting quads)."""
        faces: list[TriangleFace] = [self.bottom_face, self.top_face]
        for quad in self.side_faces:
            t1, t2 = quad.to_triangles()
            faces.extend([t1, t2])
        return faces


# ---------------------------------------------------------------------------
# Factory: build PrismGeometry from ZXPrism kind + position parity
# ---------------------------------------------------------------------------

def build_prism_geometry(
    prep_color: PrismColor,
    meas_color: PrismColor,
    pointing_up: bool,
    z_bottom: float = 0.0,
    z_top: float = 1.0,
) -> PrismGeometry:
    """Build the geometry for a triangular prism.

    Side faces are always fully transparent — the prism shape is conveyed
    by black edges in the visualizer, not filled side faces.

    Args:
        prep_color:  Color of the bottom (prep) triangular face.
        meas_color:  Color of the top (meas) triangular face.
        pointing_up: True if the triangle points up ▲ (even x), False for ▽.
        z_bottom:    z coordinate of the bottom face. Default 0.0.
        z_top:       z coordinate of the top face. Default 1.0.

    Returns:
        A :py:class:`PrismGeometry` with all 5 faces defined.
    """
    verts_2d = _triangle_vertices_2d(pointing_up)

    bottom = _make_horizontal_triangle(verts_2d, z_bottom, prep_color, flip_normal=True)
    top = _make_horizontal_triangle(verts_2d, z_top, meas_color, flip_normal=False)

    # side faces always transparent — shape shown via black edges only
    side_faces: list[QuadFace] = []
    n = len(verts_2d)
    for i in range(n):
        v0 = verts_2d[i]
        v1 = verts_2d[(i + 1) % n]
        side_faces.append(_make_side_quad(v0, v1, z_bottom, z_top, PrismColor.SIDE))

    return PrismGeometry(bottom, top, side_faces)


# ---------------------------------------------------------------------------
# PrismGeometries: registry for all ZXPrism kinds
# ---------------------------------------------------------------------------

def _basis_to_color(basis_str: str) -> PrismColor:
    return {
        "X": PrismColor.X,
        "Z": PrismColor.Z,
        "N": PrismColor.N,
    }[basis_str.upper()]



def get_prism_geometry(
    prep: str,
    meas: str,
    pointing_up: bool,
    z_bottom: float = 0.0,
    z_top: float = 1.0,
) -> PrismGeometry:
    """Convenience function: get prism geometry from basis strings.

    Args:
        prep:        Prep basis as string: "X", "Z", or "N".
        meas:        Meas basis as string: "X", "Z", or "N".
        pointing_up: True for ▲ (even x), False for ▽ (odd x).
        z_bottom:    Bottom z coordinate.
        z_top:       Top z coordinate.
    """
    return build_prism_geometry(
        prep_color=_basis_to_color(prep),
        meas_color=_basis_to_color(meas),
        pointing_up=pointing_up,
        z_bottom=z_bottom,
        z_top=z_top,
    )


# ---------------------------------------------------------------------------
# PipeGeometry: geometry for a pipe connecting two adjacent prisms
# ---------------------------------------------------------------------------

@dataclass
class PipeGeometry:
    """All faces making up a single pipe between two adjacent prisms.

    A pipe is a box (rectangular prism) that fills the gap between two prisms.
    It has:
      - 2 large colored faces (the visible sides of the pipe)
      - 2 transparent end-cap faces (the z=bottom and z=top quads, hidden inside prisms)
      - 2 transparent inner/outer faces for horizontal pipes (flush with the prism faces)

    Attributes:
        faces: All quad faces of the pipe (colored and transparent).
    """
    faces: list[QuadFace]

    def all_triangle_faces(self) -> list[TriangleFace]:
        """Return all faces split into triangles for Collada export.

        Colored faces are emitted double-sided (front + back winding) so they
        are visible from both sides in the renderer.
        Transparent faces use single-sided winding only.
        """
        result: list[TriangleFace] = []
        for quad in self.faces:
            if quad.color.is_transparent:
                result.extend(quad.to_triangles())
            else:
                result.extend(quad.to_triangles_double_sided())
        return result


# ---------------------------------------------------------------------------
# Horizontal pipe: fills the xy-gap between two adjacent triangles (same z)
# ---------------------------------------------------------------------------

# For each edge of a ▲ triangle at cell_origin (ox, oy), the shared edge
# endpoints and outward perpendicular direction (unit vector pointing away
# from the ▲ towards the neighbouring ▽) are:
#
#   "bottom": edge (ox,oy)→(ox+1,oy),          perpendicular (0, -1)
#   "right":  edge (ox+1,oy)→(ox+0.5,oy+h),    perpendicular (√3/2, 1/2)
#   "left":   edge (ox+0.5,oy+h)→(ox,oy),      perpendicular (-√3/2, 1/2)
#
# The pipe box occupies the gap of width `depth` between the two triangles,
# centred on the shared edge. Its four 2D corners are:
#   inner side (at triangle edge): vA, vB
#   outer side (depth away):       vA + depth*perp, vB + depth*perp

_H = np.sqrt(3) / 2  # height of unit equilateral triangle

_EDGE_DATA: dict[str, tuple[
    tuple[float, float],  # vertex A offset from (ox, oy)
    tuple[float, float],  # vertex B offset from (ox, oy)
    tuple[float, float],  # outward perpendicular (unit vector)
]] = {
    "bottom": ((0.0, 0.0), (1.0, 0.0),        (0.0,  -1.0)),
    "right":  ((1.0, 0.0), (0.5, _H),          (_H,    0.5)),
    "left":   ((0.5, _H),  (0.0, 0.0),         (-_H,   0.5)),
}


def get_horizontal_pipe_geometry(
    hor: str | None,
    ver: str | None,
    edge: str,
    ox: float,
    oy: float,
    depth: float,
    z_bottom: float,
    z_top: float,
) -> PipeGeometry:
    """Build the geometry for a horizontal pipe connecting two adjacent triangles.

    The pipe is a rectangular box filling the gap between two adjacent prisms.
    It is specified from the perspective of the ▲ prism.

    Face coloring follows the hor/ver convention:
      - Inner and outer faces (running in z, perpendicular to the triangle plane):
        colored with `ver` basis — these are the "vertically oriented" faces of the box.
      - End-cap faces at z_bottom and z_top (the horizontal slabs):
        colored with `hor` basis — these are the "horizontally oriented" faces.
      - If hor/ver is None or "N", the corresponding faces are transparent.

    Args:
        hor:      Horizontal basis: "X", "Z", or None/("N") → transparent end-caps.
        ver:      Vertical basis:   "X", "Z", or None/("N") → transparent side faces.
        edge:     Which edge of the ▲ is shared: "bottom", "right", or "left".
        ox:       cell_origin x of the ▲ prism (in world/scaled coordinates).
        oy:       cell_origin y of the ▲ prism (in world/scaled coordinates).
        depth:    Thickness of the pipe = spacing (gap between the two prisms).
        z_bottom: Bottom z of the pipe (= bottom of both prisms, in world coords).
        z_top:    Top z of the pipe (= top of both prisms, in world coords).

    Returns:
        A :py:class:`PipeGeometry` with 4 quad faces.
    """
    transparent = PrismColor.SIDE
    ver_color = _basis_to_color(ver) if ver and ver != "N" else transparent
    hor_color = _basis_to_color(hor) if hor and hor != "N" else transparent

    (da_x, da_y), (db_x, db_y), (px, py) = _EDGE_DATA[edge]

    # Edge endpoints in world 2D space
    ax, ay = ox + da_x, oy + da_y
    bx, by = ox + db_x, oy + db_y

    # Outer edge: shifted outward by depth in the perpendicular direction
    ax2, ay2 = ax + depth * px, ay + depth * py
    bx2, by2 = bx + depth * px, by + depth * py

    va  = np.array([ax,  ay ])
    vb  = np.array([bx,  by ])
    va2 = np.array([ax2, ay2])
    vb2 = np.array([bx2, by2])

    # inner/outer flush with prism faces → transparent
    inner = _make_side_quad(va,  vb,  z_bottom, z_top, transparent)
    outer = _make_side_quad(va2, vb2, z_bottom, z_top, transparent)
    # top/bottom slabs → hor
    cap_bottom = QuadFace(np.array([
        [ax,  ay,  z_bottom],
        [bx,  by,  z_bottom],
        [bx2, by2, z_bottom],
        [ax2, ay2, z_bottom],
    ], dtype=np.float64), hor_color)
    cap_top = QuadFace(np.array([
        [ax,  ay,  z_top],
        [bx,  by,  z_top],
        [bx2, by2, z_top],
        [ax2, ay2, z_top],
    ], dtype=np.float64), hor_color)
    # end_a/end_b (visible side walls) → ver
    end_a = QuadFace(np.array([
        [ax,  ay,  z_bottom],
        [ax2, ay2, z_bottom],
        [ax2, ay2, z_top   ],
        [ax,  ay,  z_top   ],
    ], dtype=np.float64), ver_color)
    end_b = QuadFace(np.array([
        [bx,  by,  z_bottom],
        [bx2, by2, z_bottom],
        [bx2, by2, z_top   ],
        [bx,  by,  z_top   ],
    ], dtype=np.float64), ver_color)

    return PipeGeometry(faces=[inner, outer, cap_bottom, cap_top, end_a, end_b])


# ---------------------------------------------------------------------------
# Vertical pipe: fills the z-gap between two prisms at the same (x, y)
# ---------------------------------------------------------------------------

def get_vertical_pipe_geometry(
    hor: str | None,
    ver: str | None,
    pointing_up: bool,
    z_bottom: float,
    z_top: float,
) -> PipeGeometry:
    """Build the geometry for a vertical pipe connecting two z-adjacent prisms.

    The pipe has the same triangular cross-section as the prisms it connects
    and spans the z-gap introduced by spacing.

    Face coloring follows the hor/ver convention:
      - The 3 rectangular side faces (running in z) → colored with `ver` basis.
      - The triangular caps at z_bottom/z_top sit inside the prisms and are
        always transparent (they are covered by the prism top/bottom faces).
      - If ver is None or "N", side faces are transparent (temporal pipe).

    The geometry is in local space (cell_origin = (0,0)) — the caller applies
    the same translation matrix used for the prisms at that position.

    Args:
        hor:         Horizontal basis (unused for coloring — caps are always
                     transparent since they sit inside prisms). Kept for symmetry.
        ver:         Vertical basis: "X", "Z", or None/("N") → transparent sides.
        pointing_up: Triangle orientation: True for ▲, False for ▽.
        z_bottom:    Bottom z of the pipe (= scaled top of lower prism).
        z_top:       Top z of the pipe (= scaled bottom of upper prism).

    Returns:
        A :py:class:`PipeGeometry` with 3 quad faces (side walls only).
    """
    transparent = PrismColor.SIDE
    ver_color = _basis_to_color(ver) if ver and ver != "N" else transparent

    verts_2d = _triangle_vertices_2d(pointing_up)
    n = len(verts_2d)

    faces: list[QuadFace] = []

    # 3 rectangular side faces running in z — colored with ver
    for i in range(n):
        v0 = verts_2d[i]
        v1 = verts_2d[(i + 1) % n]
        faces.append(_make_side_quad(v0, v1, z_bottom, z_top, ver_color))

    # Triangular caps at z_bottom and z_top are omitted — they sit flush
    # against the prism top/bottom faces which already cover them.

    return PipeGeometry(faces=faces)