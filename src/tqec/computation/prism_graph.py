"""Prism graph representation of a logical computation with color code."""

from __future__ import annotations

from networkx import Graph, is_connected
from tqec.computation.prism import Port, Position3DHex, PrismKind, prism_kind_from_string, Prism, ZXPrism
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


    def stabilizers_timeslice(self, z: int, d: int):
        """Build the stabilizers of a given time slice and given distance d."""
        #filter prism and horizontal pipes of some given time slice
        current_prisms = []
        for pos, attrs in self._graph.nodes(data=True):
            if pos.z == z:
                prism = attrs[self._NODE_DATA_KEY]
                current_prisms.append(prism)

        current_pipes = []
        for pos1, pos2, attrs in self._graph.edges(data=True):
            if pos1.z == z and pos2.z == z:
                edge = attrs[self._EDGE_DATA_KEY]
                if edge.kind.is_spatial:
                    current_pipes.append(edge)

        #find microscopic centroid for 1st prism (the first prism it the "origin")
        if current_prisms[0].position.x % 2 == 0:
            #upwards
            centroid = Position3DHex(d-1, 0, z)
        else:
            #downwards
            centroid = Position3DHex(d-1, -d-1, z)
        centroids = [centroid]

        #find for each prism position the microscopic centroid position
        for prism in current_prisms[1:]:
            path = current_prisms[0].position.shortest_path_spatial(prism.position)
            current_centroid = centroid
            current_macro = current_prisms[0].position
            for macro_next in path[1:]:
                #yields next centroid from current centroid and current macro and el
                current_centroid = current_macro.macro_diff_to_micro_diff(d, current_centroid, macro_next)
                current_macro = macro_next #overwrite for neighbor consistentcy
            centroids.append(current_centroid)

        for el in centroids:
            print(el)

        left_corner_lst = [] #left_corner depends on x even or odd in macro
        for macro, micro_centroid in zip(current_prisms, centroids):
            left_corner = micro_centroid
            if macro.position.x % 2 == 0:
                for _ in range(d-1):
                    left_corner = left_corner.shift_standard_direction_minus2()
            else:
                for _ in range(d-1):
                    left_corner = left_corner.shift_standard_direction_minus1()
            left_corner_lst.append(left_corner)

        for el in left_corner_lst:
            print(el)

        #place the current_prisms on the microscopic lattice
        #! ONLY TEMP BECAUSE BOUNDARY REMOVAL DEPENDING ON PIPE
        stabilizers = []
        for prism, left_corner in zip(current_prisms, left_corner_lst):
            #is the prism end of a pipe?
            pipes_dirs = []
            for pipe in current_pipes:
                if prism in (pipe.u, pipe.v):
                    pipes_dirs.append(pipe.direction_connecting_bdry())  # noqa: PERF401
            pipes_dirs = [el for el in ["a", "b", "c"] if el not in pipes_dirs] #flip the elements
            if prism.position.x % 2 == 0:
                stabilizers += ZXPrism.patch_stabilizers(d, "upwards", left_corner, pipes_dirs)
            else:
                stabilizers += ZXPrism.patch_stabilizers(d, "downwards", left_corner, pipes_dirs)
        #connect the prisms according to current_pipes on the micro lattice

        return stabilizers