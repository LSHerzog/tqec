"""Utility functions for PyZX interop."""

from fractions import Fraction

from pyzx.graph.graph_s import GraphS
from pyzx.utils import EdgeType, FractionLike, VertexType, vertex_is_zx

from tqec.computation.cube import CubeKind, Port, ZXCube
from tqec.computation.pipe_prism import PrismPipe
from tqec.computation.prism import BasisPrism, PrismKind, ZXPrism
from tqec.utils.enums import Basis, Pauli
from tqec.utils.exceptions import TQECError


def is_zx_no_phase(g: GraphS, v: int) -> bool:
    """Check if a vertex in a PyZX graph is a Z/X spider with phase 0."""
    return vertex_is_zx(g.type(v)) and g.phase(v) == 0


def is_z_no_phase(g: GraphS, v: int) -> bool:
    """Check if a vertex in a PyZX graph is a Z spider with phase 0."""
    return g.type(v) is VertexType.Z and g.phase(v) == 0


def is_x_no_phase(g: GraphS, v: int) -> bool:
    """Check if a vertex in a PyZX graph is a X spider with phase 0."""
    return g.type(v) is VertexType.X and g.phase(v) == 0


def is_boundary(g: GraphS, v: int) -> bool:
    """Check if a vertex in a PyZX graph is a boundary type spider."""
    return g.type(v) is VertexType.BOUNDARY


def is_s(g: GraphS, v: int) -> bool:
    """Check if a vertex in a PyZX graph is a S node."""
    return g.type(v) is VertexType.Z and g.phase(v) == Fraction(1, 2)


def is_hadamard(g: GraphS, edge: tuple[int, int]) -> bool:
    """Check if an edge in a PyZX graph is a Hadamard edge."""
    return g.edge_type(edge) is EdgeType.HADAMARD

def prism_kind_to_zx(kind: PrismKind, neighbor_pipes: list[PrismPipe]) -> tuple[VertexType, FractionLike]:
    """Convert a Prism to corresponding PyZX vertex type and phase.

    Since prisms have no ZX type by themselfes, one needs the corresponding neighboring pipes
    because the pipe colors determine the zx type of the node
    """
    if isinstance(kind, ZXPrism):
        neighbor_pipes_temporal = [pipe for pipe in neighbor_pipes if pipe.kind.is_temporal]
        neighbor_pipes_spatial = [pipe for pipe in neighbor_pipes if pipe.kind.is_spatial]#filter spatial pipes, because this is only relevant for spatial pipes

        if len(neighbor_pipes) != 0:
            list_ver = [pipe.kind.ver for pipe in neighbor_pipes]
            list_hor = [pipe.kind.hor for pipe in neighbor_pipes]
        else:
            raise TQECError("We do not consider sole prisms for zx diagrams.")

        if len(neighbor_pipes_spatial) == 0 and len(neighbor_pipes_temporal) >= 2:
            #! TEMPORARY, if no spatial pipe, then just choose some color, but more detailed rules apply here.
            return VertexType.Z, 0
        elif len(neighbor_pipes_spatial) == 0 and len(neighbor_pipes_temporal) == 1:
            #this is a boundary prism, i.e. determine the node color depending on its prep or meas face
            if kind.meas is BasisPrism.X or kind.prep is BasisPrism.X: #other possibilities already ruled out in construction of PipeGraph
                return VertexType.Z, 0
            if kind.meas is BasisPrism.Z or kind.prep is BasisPrism.Z: #other possibilities already ruled out in construction of PipeGraph
                return VertexType.X, 0

        #remove the BasisPrism.N entries since they do not lead to inconsistency
        list_ver = [el for el in list_ver if el is not BasisPrism.N]
        list_hor = [el for el in list_hor if el is not BasisPrism.N]

        if len(set(list_ver)) > 1:
            raise TQECError("Inconsistent `ver` values of pipes entering the same prism.")
        if len(set(list_hor)) > 1:
            raise TQECError("Inconsistent `hor` values of pipes entering the same prism.")

        if list_hor[0] == BasisPrism.X and list_ver[0] == BasisPrism.Z:
            return VertexType.Z, 0
        elif list_hor[0] == BasisPrism.Z and list_ver[0] == BasisPrism.X:
            return VertexType.X, 0
        else:
            raise TQECError("Invalid Pipe Configuration.")
    #import here to avoid overlapping Port imports for block and prism
    from tqec.computation.prism import Port  # noqa: PLC0415
    if isinstance(kind, Port):
        return VertexType.BOUNDARY, 0
    else:
        raise NotImplementedError(f"type {kind} not implemented yet.")

def cube_kind_to_zx(kind: CubeKind) -> tuple[VertexType, FractionLike]:
    """Convert the cube kind to the corresponding PyZX vertex type and phase.

    The conversion is as follows:

    - Port -> BOUNDARY spider with phase 0.
    - YHalfCube -> Z spider with phase 1/2.
    - ZXCube -> Z spider with phase 0 if it has only one Z basis boundary,
        otherwise X spider with phase 0.

    Args:
        kind: The cube kind to be converted.

    Returns:
        A tuple of vertex type and spider phase.

    """
    if isinstance(kind, ZXCube):
        if sum(basis == Basis.Z for basis in kind.as_tuple()) == 1:
            return VertexType.Z, 0
        return VertexType.X, 0
    if isinstance(kind, Port):
        return VertexType.BOUNDARY, 0
    else:  # isinstance(kind, YHalfCube)
        return VertexType.Z, Fraction(1, 2)


def zx_to_pauli(g: GraphS, v: int) -> Pauli:
    """Convert a PyZX vertex to the corresponding Pauli operator.

    Args:
        g: The PyZX graph.
        v: The vertex id.

    Raises:
        ValueError: If the vertex is not a Clifford or a boundary.

    Returns:
        The corresponding Pauli operator.

    """
    return vertex_type_to_pauli(g.type(v), g.phase(v))


def vertex_type_to_pauli(vertex_type: VertexType, phase: FractionLike = 0) -> Pauli:
    """Convert a PyZX vertex type to the corresponding Pauli operator.

    Args:
        vertex_type: The PyZX vertex type.
        phase: The phase of the vertex. Default is 0.

    Raises:
        TQECError: If the vertex type and phase do not correspond to a Pauli operator.

    Returns:
        The corresponding Pauli operator.

    """
    match vertex_type, phase:
        case VertexType.X, 0:
            return Pauli.X
        case VertexType.Z, 0:
            return Pauli.Z
        case VertexType.Z, Fraction(numerator=1, denominator=2):
            return Pauli.Y
        case VertexType.BOUNDARY, _:
            return Pauli.I
        case _:
            raise TQECError(
                f"Cannot convert vertex type {vertex_type} and phase {phase} to Pauli operator."
            )


def zx_to_basis(g: GraphS, v: int) -> Basis:
    """Convert a PyZX vertex to the corresponding Basis.

    Args:
        g: The PyZX graph.
        v: The vertex id.

    Raises:
        ValueError: If the vertex is not a Clifford or a boundary.

    Returns:
        The corresponding Basis.

    """
    return vertex_type_to_basis(g.type(v), g.phase(v))


def vertex_type_to_basis(vertex_type: VertexType, phase: FractionLike = 0) -> Basis:
    """Convert a PyZX vertex type to the corresponding Basis.

    Args:
        vertex_type: The PyZX vertex type.
        phase: The phase of the vertex. Default is 0.

    Raises:
        TQECError: If the vertex type and phase do not correspond to a Basis.

    Returns:
        The corresponding Basis.

    """
    try:
        return vertex_type_to_pauli(vertex_type, phase).to_basis()
    except TQECError:
        raise TQECError(f"Cannot convert vertex type {vertex_type} and phase {phase} to Basis.")
