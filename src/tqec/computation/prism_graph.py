"""Prism graph representation of a logical computation with color code."""

from __future__ import annotations

from networkx import Graph, is_connected
from prism import Port, Position3DHex, PrismKind, prism_kind_from_string, Prism
from pipe_prism import PrismPipeKind, PrismPipe
from typing import TYPE_CHECKING, Any, cast

from tqec.utils.exceptions import TQECError


class PrismGraph:
    _NODE_DATA_KEY: str = "tqec_node_data"
    _EDGE_DATA_KEY: str = "tqec_edge_data"

    def __init__(self, name: str = "") -> None:
        """Prism Graph Rep of logical computation."""
        self._name = name
        self._graph: Graph[Position3DHex] = Graph()
        self._ports: dict[str, Position3DHex] = {}

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