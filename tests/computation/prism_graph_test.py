import os
import pickle

import pytest

from tqec.computation.pipe_prism import PrismPipe, PrismPipeKind
from tqec.computation.prism import BasisPrism, Position3DHex, ZXPrism
from tqec.computation.prism_graph import PrismGraph
from tqec.utils.enums import Basis
from tqec.utils.exceptions import TQECError

# ==============================================================================
# Simple Fake Classes to avoid using unittest.mock
# ==============================================================================


class FakeZXGraph:
    def __init__(self, positions: dict[int, Position3DHex]):  # noqa: D107
        self._positions = positions


class FakeNode:
    def __init__(self, node_id: int, basis: Basis):  # noqa: D107
        self.id = node_id
        self.basis = basis


class FakeEdge:
    def __init__(self, u_node: FakeNode, v_node: FakeNode):  # noqa: D107
        self.u = u_node
        self.v = v_node


class FakeCorrelationSurface:
    def __init__(self, span_edges: list[FakeEdge]):  # noqa: D107
        self.span = span_edges


def test_prism_graph_add_prism() -> None:
    """Test adding basic prisms, ports, string kind casting, and duplication guardrails."""
    g = PrismGraph("Test Graph")
    assert len(g.prisms) == 0

    pos1 = Position3DHex(0, 0, 0)
    kind1 = ZXPrism.from_str("XX")
    g.add_prism(pos1, kind1, label="p1")

    assert len(g.prisms) == 1
    assert g[pos1].kind == kind1
    assert g[pos1].label == "p1"

    pos2 = Position3DHex(1, 1, 0)
    g.add_prism(pos2, "ZX", label="p2")
    assert len(g.prisms) == 2
    assert isinstance(g[pos2].kind, ZXPrism)
    assert g[pos2].kind.prep == BasisPrism.Z

    pos_port = Position3DHex(0, 1, 0)
    g.add_prism(pos_port, "PORT", label="In")
    assert g[pos_port].is_port

    pos_dup_port = Position3DHex(2, 2, 0)
    with pytest.raises(
        TQECError, match="There is already a port with the same label In in the graph\\."
    ):
        g.add_prism(pos_dup_port, "PORT", label="In")


def test_prism_graph_add_pipe() -> None:
    """Test creating edges between two neighboring prisms with valid alignment validation."""
    g = PrismGraph()
    pos1 = Position3DHex(0, 0, 0)
    pos2 = Position3DHex(1, 1, 0)

    g.add_prism(pos1, "XX")
    g.add_prism(pos2, "XX")

    kind = PrismPipeKind(hor=BasisPrism.X, ver=BasisPrism.Z)
    g.add_pipe(pos1, pos2, kind)

    assert len(g.pipes) == 1
    pipe = g.pipes[0]
    assert isinstance(pipe, PrismPipe)
    assert pipe.kind == kind


def test_get_or_init_global_origin() -> None:
    """Test lexicon ordering extraction and the caching mechanism for global origins."""
    g = PrismGraph()

    pos_target = Position3DHex(2, 2, 0)
    pos_smallest = Position3DHex(0, 4, 0)
    pos_other = Position3DHex(1, 1, 0)

    g.add_prism(pos_target, "XX")
    g.add_prism(pos_smallest, "XX")
    g.add_prism(pos_other, "XX")

    d = 3
    macro_ref, micro_ref = g._get_or_init_global_origin(d)

    assert macro_ref == pos_smallest
    assert micro_ref == Position3DHex(2, 0, 0)

    cached_macro, cached_micro = g._get_or_init_global_origin(d)
    assert cached_macro == macro_ref
    assert cached_micro == micro_ref


def test_find_ver_hor_correlation_surface() -> None:
    """Test tqec CorrelationSurface translation back to layout mappings."""
    g = PrismGraph()
    pos_u = Position3DHex(0, 0, 0)
    pos_v = Position3DHex(1, 1, 0)

    g.add_prism(pos_u, "XX")
    g.add_prism(pos_v, "XX")

    # Setup spatial configuration: pipe.kind.ver = Z, pipe.kind.hor = X
    pipe_kind = PrismPipeKind(hor=BasisPrism.X, ver=BasisPrism.Z)
    g.add_pipe(pos_u, pos_v, pipe_kind)
    pipe = g.pipes[0]

    # Create our fake layout objects instead of using MagicMock
    fake_zx = FakeZXGraph(positions={10: pos_u, 20: pos_v})

    g.to_zx_graph = lambda *args, **kwargs: fake_zx

    fake_node_u = FakeNode(node_id=10, basis=Basis.X)
    fake_node_v = FakeNode(node_id=20, basis=Basis.X)
    fake_edge = FakeEdge(u_node=fake_node_u, v_node=fake_node_v)
    fake_surface = FakeCorrelationSurface(span_edges=[fake_edge])

    result = g.find_ver_hor_correlation_surface(fake_surface)

    assert pipe in result
    basis_res, orientation_res = result[pipe]
    assert basis_res == BasisPrism.X
    assert orientation_res == "ver"


def test_stabilizers_and_product() -> None:
    """Test everything by running an explicit example against a saved snapshot."""
    # Build up the active graph
    g = PrismGraph("randomlarge")

    prisms = [
        (Position3DHex(0, 0, 0), "NN", ""),
        (Position3DHex(1, 1, 0), "ZZ", ""),
        (Position3DHex(2, 0, 0), "ZZ", ""),
        (Position3DHex(3, 1, 0), "ZZ", ""),
        (Position3DHex(2, 2, 0), "ZZ", ""),
        (Position3DHex(1, 3, 0), "NN", ""),
        (Position3DHex(4, 0, 0), "NN", ""),
        (Position3DHex(0, 0, -1), "XN", ""),
        (Position3DHex(0, 0, 1), "NX", ""),
        (Position3DHex(4, 0, -1), "XN", ""),
        (Position3DHex(4, 0, 1), "NX", ""),
        (Position3DHex(1, 3, -1), "XN", ""),
        (Position3DHex(1, 3, 1), "NX", ""),
    ]

    for pos, kind, label in prisms:
        g.add_prism(pos, kind, label)

    pipe_kind = PrismPipeKind(hor=BasisPrism.Z, ver=BasisPrism.X)
    g.add_pipe(prisms[0][0], prisms[1][0], pipe_kind)
    g.add_pipe(prisms[1][0], prisms[2][0], pipe_kind)
    g.add_pipe(prisms[2][0], prisms[3][0], pipe_kind)
    g.add_pipe(prisms[3][0], prisms[4][0], pipe_kind)
    g.add_pipe(prisms[4][0], prisms[5][0], pipe_kind)
    g.add_pipe(prisms[3][0], prisms[6][0], pipe_kind)

    pipe_kind = PrismPipeKind(hor=BasisPrism.N, ver=BasisPrism.N)
    g.add_pipe(prisms[0][0], prisms[7][0], pipe_kind)
    g.add_pipe(prisms[0][0], prisms[8][0], pipe_kind)
    g.add_pipe(prisms[6][0], prisms[9][0], pipe_kind)
    g.add_pipe(prisms[6][0], prisms[10][0], pipe_kind)
    g.add_pipe(prisms[5][0], prisms[11][0], pipe_kind)
    g.add_pipe(prisms[5][0], prisms[12][0], pipe_kind)

    z = 0
    d = 3

    (
        stabs_x,
        stabs_z,
        _,
        dct_single_type_stabs,
        dct_patch_stabilizers,
    ) = g.stabilizers_timeslice(z, d)
    star_ops_x, star_ops_z = g.star_operator_timeslice(z, d)

    result = g.stabilizer_product_timeslice(
        z, d, dct_single_type_stabs, dct_patch_stabilizers, testing=True
    )

    pkl_path = os.path.join(os.path.dirname(__file__), "reference_stabilizer_data.pkl")

    if not os.path.exists(pkl_path):
        pytest.fail(f"Reference file snapshot not found at: {pkl_path}.")

    with open(pkl_path, "rb") as f:
        ref = pickle.load(f)

    assert stabs_x == ref["stabs_x"]
    assert stabs_z == ref["stabs_z"]

    assert star_ops_x == ref["star_ops_x"]
    assert star_ops_z == ref["star_ops_z"]

    assert dct_single_type_stabs == ref["dct_single_type_stabs"]
    assert dct_patch_stabilizers == ref["dct_patch_stabilizers"]

    assert result.assignment == ref["result"].assignment
    assert result.paths_x == ref["result"].paths_x
    assert result.paths_z == ref["result"].paths_z
    assert result.stars_x == ref["result"].stars_x
    assert result.stars_z == ref["result"].stars_z
    assert result.stabilizer_products_x == ref["result"].stabilizer_products_x
    assert result.stabilizer_products_z == ref["result"].stabilizer_products_z
