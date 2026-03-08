"""ZX graph with 3D positions on a hex lattice."""

from __future__ import annotations

from collections.abc import Mapping
from fractions import Fraction
from typing import TYPE_CHECKING

from pyzx.graph.graph_s import GraphS
from pyzx.utils import EdgeType, VertexType

from tqec.computation.pipe_prism import PrismPipe
from tqec.computation.prism import Position3DHex, Prism
from tqec.computation.prism_graph import PrismGraph
from tqec.interop.pyzx.utils import prism_kind_to_zx
from tqec.utils.exceptions import TQECError

if TYPE_CHECKING:
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.axes3d import Axes3D

class PositionedHexZX:
    def __init__(self, g: GraphS, positions: Mapping[int, Position3DHex]):
        """Represent a ZX graph with 3D positions on a hex lattice."""
        self.check_preconditions(g, positions)

        self._g = g
        self._positions: dict[int, Position3DHex] = dict(positions)

    @staticmethod
    def check_preconditions(g: GraphS, positions: Mapping[int, Position3DHex]) -> None:
        pass

    @property
    def g(self) -> GraphS:
        """Return the internal ZX graph."""
        return self._g

    @property
    def positions(self) -> dict[int, Position3DHex]:
        """Return the 3D Hex positions of the vertices."""
        return self._positions

    @staticmethod
    def outgoing_pipes_from_prism(prism: Prism, prism_graph: PrismGraph) -> list[PrismPipe]:
        """Return the pipes connecting the given Prism."""
        return [
            data[PrismGraph._EDGE_DATA_KEY]
            for _, _, data in prism_graph._graph.edges(prism.position, data=True)
        ]

    @staticmethod
    def from_prism_block_graph(prism_graph: PrismGraph) -> PositionedHexZX:
        """Find the zx graph according to a prsim graph."""
        v2p: dict[int, Position3DHex] = {}
        p2v: dict[Position3DHex, int] = {}
        g = GraphS()

        for prism in prism_graph.prisms:#sorted(prism_graph.prisms, key=lambda c: c.position):
            neighbor_pipes = PositionedHexZX.outgoing_pipes_from_prism(prism, prism_graph)
            vt, phase = prism_kind_to_zx(prism.kind, neighbor_pipes)
            v = g.add_vertex(vt, phase=phase)
            v2p[v] = prism.position
            p2v[prism.position] = v

        for edge in prism_graph.pipes:
            et = EdgeType.HADAMARD if edge.kind.has_hadamard else EdgeType.SIMPLE
            g.add_edge((p2v[edge.u.position], p2v[edge.v.position]), et)

        return PositionedHexZX(g, v2p)

    def draw(
        self,
        *,
        figsize: tuple[float, float] = (5, 6),
        title: str | None = None,
        node_size: int = 400,
        hadamard_size: int = 200,
        edge_width: int = 1,
    ) -> tuple[Figure, Axes3D]:  # pragma: no cover
        """Plot the :py:class:`~tqec.interop.pyzx.positioned.PositionedZX` using matplotlib.

        Args:
            graph: The ZX graph to plot.
            figsize: The figure size. Default is ``(5, 6)``.
            title: The title of the plot. Default to the name of the graph.
            node_size: The size of the node in the plot. Default is ``400``.
            hadamard_size: The size of the Hadamard square in the plot. Default
                is ``200``.
            edge_width: The width of the edge in the plot. Default is ``1``.

        Returns:
            A tuple of the figure and the axes.

        """
        # Needs to be imported here to avoid pulling pyzx when importing this module.
        from tqec.interop.pyzx.plot import plot_positioned_zx_graph  # noqa: PLC0415

        return plot_positioned_zx_graph(
            self,
            figsize=figsize,
            title=title,
            node_size=node_size,
            hadamard_size=hadamard_size,
            edge_width=edge_width,
        )