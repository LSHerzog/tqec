"""Prism graph representation of a logical computation with color code."""

from __future__ import annotations

from networkx import Graph, is_connected
from tqec.computation.prism import Port, Position3DHex, PrismKind, prism_kind_from_string, Prism
from tqec.computation.pipe_prism import PrismPipeKind, PrismPipe
from typing import TYPE_CHECKING, Any, cast
from tqec.computation.correlation import find_correlation_surfaces

from tqec.utils.exceptions import TQECError

if TYPE_CHECKING:
    from tqec.interop.pyzx.positioned_prism import PositionedHexZX
    from tqec.computation.correlation import CorrelationSurface


class PrismGraph:
    _NODE_DATA_KEY: str = "tqec_node_data"
    _EDGE_DATA_KEY: str = "tqec_edge_data"

    def __init__(self, name: str = "") -> None:
        """Prism Graph Rep of logical computation."""
        self._name = name
        self._graph: Graph[Position3DHex] = Graph()
        self._ports: dict[str, Position3DHex] = {}

    @property
    def prisms(self) -> list[Prism]:
        """List of prisms (nodes) in the graph."""
        return [data[self._NODE_DATA_KEY] for _, data in self._graph.nodes(data=True)]

    @property
    def pipes(self) -> list[PrismPipe]:
        """The list of pipes (edges) in the graph."""
        return [data[self._EDGE_DATA_KEY] for _, _, data in self._graph.edges(data=True)]


    def add_prism(self, position: Position3DHex, kind: PrismKind | str, label: str = "") -> Position3DHex:
        """Add Prism to the graph."""
        #if position in self:
        #    raise TQECError(f"Cube already exists at position {position}.")
        if isinstance(kind, str):
            kind = prism_kind_from_string(kind)
        if kind == Port() and label in self._ports:
            raise TQECError(f"There is already a port with the same label {label} in the graph.")

        self._graph.add_node(position, **{self._NODE_DATA_KEY: Prism(position, kind, label)})
        if kind == Port():
            self._ports[label] = position
        return position

    def add_pipe(self, pos1: Position3DHex, pos2: Position3DHex, kind: PrismPipeKind):
        """Add a pipe."""
        u, v = self[pos1], self[pos2] #uses getitem to retrieve u and v prisms
        pipe = PrismPipe.from_prisms(u, v, kind) #tests some stuff
        self._graph.add_edge(pos1, pos2, **{self._EDGE_DATA_KEY: pipe})

        #!todo also check that directly neighboring pipes must have the same color (e.g. one middle
        #!todo prism with hor/ver colors swapped in the left and right pipe is not allowed.)

    def __getitem__(self, position: Position3DHex) -> Prism:
        return cast(Prism, self._graph.nodes[position][self._NODE_DATA_KEY])

    def to_zx_graph(self) -> PositionedHexZX:
        """Convert the block graph to a positioned PyZX graph on a hex lattice.

        Returns:
            A :py:class:`~tqec.interop.pyzx.positioned_prism.PositionedHexZX` object
            converted from the block graph.

        """
        # Needs to be imported here to avoid pulling pyzx when importing this module.
        from tqec.interop.pyzx.positioned_prism import PositionedHexZX  # noqa: PLC0415

        return PositionedHexZX.from_prism_block_graph(self)

    def find_correlation_surfaces(self) -> list[CorrelationSurface]:
        """Find the correlation surfaces in the block graph.

        Returns:
            The list of correlation surfaces.

        """
        return find_correlation_surfaces(self.to_zx_graph().g)

    def view_as_html(self):
        """Plot 3d Plot."""
        #!TODO generalize this, also with removing side walls etc.
        import io  # noqa: I001, PLC0415

        from tqec.interop.collada.html_viewer import display_collada_model  # noqa: PLC0415
        from tqec.interop.collada.read_write_prism import write_prism_graph_to_dae_file  # noqa: PLC0415

        buf = io.BytesIO()
        write_prism_graph_to_dae_file(self, buf, spacing = 3.0)
        return display_collada_model(buf.getvalue())  # pass bytes directly
