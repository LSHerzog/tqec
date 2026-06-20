"""Prism graph representation of a logical computation with color code."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import networkx as nx
from networkx import Graph

from tqec.computation.correlation import find_correlation_surfaces
from tqec.computation.pipe_prism import PrismPipe, PrismPipeKind
from tqec.computation.prism import (
    BasisPrism,
    Port,
    Position3DHex,
    Prism,
    PrismKind,
    ZXPrism,
    prism_kind_from_string,
)
from tqec.utils.enums import Basis
from tqec.utils.exceptions import TQECError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from tqec.computation.correlation import CorrelationSurface
    from tqec.interop.pyzx.positioned_prism import PositionedHexZX


@dataclass
class StabilizerProductResult:
    assignment: dict
    paths_x: list
    paths_z: list
    stars_x: list
    stars_z: list
    stabilizer_products_x: list
    stabilizer_products_z: list


class PrismGraph:
    _NODE_DATA_KEY: str = "tqec_node_data"
    _EDGE_DATA_KEY: str = "tqec_edge_data"

    def __init__(self, name: str = "") -> None:
        """Prism Graph Rep of logical computation."""
        self._name = name
        self._graph: Graph[Position3DHex] = Graph()
        self._ports: dict[str, Position3DHex] = {}
        self.seed_star_op: dict[int, tuple] = {}
        self.prism_pipes_to_data_qubits_full = {}
        # per key=d, collect an origin of the whole diagram, both macro and micro corner.
        self._global_micro_origin: dict[int, tuple[Position3DHex, Position3DHex]] = {}

    @property
    def prisms(self) -> list[Prism]:
        """List of prisms (nodes) in the graph."""
        return [data[self._NODE_DATA_KEY] for _, data in self._graph.nodes(data=True)]

    @property
    def pipes(self) -> list[PrismPipe]:
        """The list of pipes (edges) in the graph."""
        return [data[self._EDGE_DATA_KEY] for _, _, data in self._graph.edges(data=True)]

    def add_prism(
        self, position: Position3DHex, kind: PrismKind | str, label: str = ""
    ) -> Position3DHex:
        """Add Prism to the graph."""
        # if position in self:
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
        u, v = self[pos1], self[pos2]  # uses getitem to retrieve u and v prisms
        pipe = PrismPipe.from_prisms(u, v, kind)  # tests some stuff
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

    def find_ver_hor_correlation_surface(self, correlation_surface: CorrelationSurface):
        """Turn a correlation surface native to tqec into representation with more info.

        Since a CorrelationSurface is just a list of edges with basis assignments,
        it is necessary (for SE extraction) to know more than that in relation to the prismgraph.
        Only correlation surfaces of horizontal pipes are relevant for the SE,
        thus for each horizontal edge, determine based on hor/ver colors of the pipe
        whether the relevant correlation surface is horizontal or vertical.
        """
        zx = self.to_zx_graph()
        horizontal_pipes_cs: dict[PrismPipe, tuple[BasisPrism, str]] = {}
        # per pipe we store a string "X" or "Z" and a string "hor" or "ver"
        # the first determines the type of the CS  and the second whether it is about
        # about a vertical cs = spread star operator
        # or about a horizontl cs = product of stabilizers
        for edge in correlation_surface.span:
            u_id = edge.u.id
            v_id = edge.v.id
            pos_u = zx._positions[u_id]  # Position3DHex
            pos_v = zx._positions[v_id]
            # only relevant if horizontal pipe
            if pos_u.z == pos_v.z:
                # find the edge in the prsimgraph with pos_u and pos_v
                for pipe in self.pipes:
                    if (pipe.u.position == pos_u and pipe.v.position == pos_v) or (
                        pipe.v.position == pos_u and pipe.u.position == pos_v
                    ):
                        vertical_cs_type = pipe.kind.ver
                        horizontal_cs_type = pipe.kind.hor
                        if edge.u.basis == Basis.Z and edge.v.basis == Basis.Z:
                            if vertical_cs_type == BasisPrism.Z:
                                horizontal_pipes_cs.update({pipe: (BasisPrism.Z, "hor")})
                            elif horizontal_cs_type == BasisPrism.Z:
                                horizontal_pipes_cs.update({pipe: (BasisPrism.Z, "ver")})
                        elif edge.u.basis == Basis.X and edge.v.basis == Basis.X:
                            if vertical_cs_type == BasisPrism.X:
                                horizontal_pipes_cs.update({pipe: (BasisPrism.X, "hor")})
                            elif horizontal_cs_type == BasisPrism.X:
                                horizontal_pipes_cs.update({pipe: (BasisPrism.X, "ver")})
                        else:
                            raise NotImplementedError("No mixed edges implemented yet.")
                        break
        return horizontal_pipes_cs

    def view_as_html(self):
        """Plot 3d Plot."""
        #!TODO generalize this, also with removing side walls etc.
        import io  # noqa: I001, PLC0415

        from tqec.interop.collada.html_viewer import display_collada_model  # noqa: PLC0415
        from tqec.interop.collada.read_write_prism import write_prism_graph_to_dae_file  # noqa: PLC0415

        buf = io.BytesIO()
        write_prism_graph_to_dae_file(self, buf, spacing=3.0)
        return display_collada_model(buf.getvalue())  # pass bytes directly

    def _get_or_init_global_origin(self, d: int) -> tuple[Position3DHex, Position3DHex]:
        """Return (macro_ref, micro_centroid_ref) for the whole diagram.

        The macro_ref is the lexicographically smallest (x, y) position among
        all prisms in the graph. Its microscopic centroid is fixed at the same
        formula currently used in corners_timeslice, and cached for reuse.
        """
        if d in self._global_micro_origin:
            return self._global_micro_origin[d]

        # Pick the canonical macro reference — smallest (x, y) in the whole graph.
        all_positions = [pos for pos, _ in self._graph.nodes(data=True)]
        macro_ref = min(all_positions, key=lambda p: (p.x, p.y))

        # Compute micro centroid
        # but at z=0 (z is stripped and re-added per layer later).
        if macro_ref.x % 2 == 0:
            micro_ref = Position3DHex(x=d - 1, y=0, z=0)
        else:
            micro_ref = Position3DHex(x=d - 1, y=-d - 1, z=0)

        self._global_micro_origin[d] = (macro_ref, micro_ref)
        return macro_ref, micro_ref

    def corners_timeslice(self, z: int, d: int, current_prisms):
        """Find the corners to build stabilizers from them for current timeslice."""
        macro_ref, micro_ref_z0 = self._get_or_init_global_origin(d)
        # Reattach the current z to the reference centroid.
        micro_ref = Position3DHex(micro_ref_z0.x, micro_ref_z0.y, z)
        macro_ref_at_z = Position3DHex(macro_ref.x, macro_ref.y, z)

        centroids = []
        for prism in current_prisms:
            target = Position3DHex(prism.position.x, prism.position.y, z)
            if target == macro_ref_at_z:
                centroids.append(micro_ref)
            else:
                # Walk from the global macro_ref to this prism's (x,y).
                path = macro_ref_at_z.shortest_path_spatial(target)
                current_centroid = micro_ref
                current_macro = macro_ref_at_z
                for macro_next in path[1:]:
                    current_centroid = current_macro.macro_diff_to_micro_diff(
                        d, current_centroid, macro_next
                    )
                    current_macro = macro_next
                centroids.append(current_centroid)

        left_corner_lst = []  # left_corner depends on x even or odd in macro
        for macro, micro_centroid in zip(current_prisms, centroids):
            left_corner = micro_centroid
            if macro.position.x % 2 == 0:
                for _ in range(d - 1):
                    left_corner = left_corner.shift_standard_direction_minus2()
            else:
                for _ in range(d - 1):
                    left_corner = left_corner.shift_standard_direction_minus1()
            left_corner_lst.append(left_corner)

        return left_corner_lst

    @staticmethod
    def split_into_connected_components(
        positions: list[Position3DHex],
    ) -> list[list[Position3DHex]]:
        """Split positions into sublists.

        Split positions into sublists where elements are
        connected via neighbour or next-nearest-neighbour.
        """
        # each position starts in its own component
        parent = {pos: pos for pos in positions}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]  # path compression
                x = parent[x]
            return x

        def union(x, y):
            parent[find(x)] = find(y)

        for i, pos1 in enumerate(positions):
            for pos2 in positions[i + 1 :]:
                if pos1.is_neighbour(pos2) or pos1.is_next_nearest_neighbour(pos2):
                    union(pos1, pos2)

        # group by root
        components: dict[Position3DHex, list[Position3DHex]] = {}
        for pos in positions:
            root = find(pos)
            components.setdefault(root, []).append(pos)

        return list(components.values())

    # assign X or Z via pipes: find any pipe connected to this prism and read its kind
    @staticmethod
    def _get_basis_from_pipes(prism_pos: Position3DHex, current_pipes: list[PrismPipe]) -> str:
        for pipe in current_pipes:
            if prism_pos in (pipe.u.position, pipe.v.position):
                if pipe.kind.hor == BasisPrism.X and pipe.kind.ver == BasisPrism.Z:
                    return "X"
                elif pipe.kind.hor == BasisPrism.Z and pipe.kind.ver == BasisPrism.X:
                    return "Z"
        return "XZ"  # fallback for isolated prisms with no pipes, for them it's both X and Z

    def _walk_star_op_to_prism(
        self,
        star_op: list[Position3DHex],
        from_prism: Prism,
        to_prism: Prism,
        z: int,
        d: int,
    ) -> list[Position3DHex]:
        """Walk a star operator from from_prism to to_prism via consecutive reflections.

        Both from_prism and to_prism must share the same z.
        Returns the star operator sitting on to_prism.

        #!DEDUPLICATE
        """
        path = from_prism.position.shortest_path_spatial(to_prism.position)
        path_prisms = [Prism(pos, ZXPrism(BasisPrism.N, BasisPrism.N), "") for pos in path]
        left_corner_lst = self.corners_timeslice(z, d, path_prisms)

        star_op_tmp = star_op.copy()
        for idx, (pos, corner) in enumerate(zip(path, left_corner_lst)):
            triangle_type = "upwards" if pos.x % 2 == 0 else "downwards"
            nodes_triangle_bdry = ZXPrism.patch_triangle_bdry(d, corner, triangle_type)
            if idx + 1 < len(path):
                pipe = PrismPipe(
                    path_prisms[idx],
                    path_prisms[idx + 1],
                    PrismPipeKind(hor=BasisPrism.N, ver=BasisPrism.N),
                )
                direction = pipe.direction_connecting_bdry()
                star_op_tmp = ZXPrism.reflect_star_operator(
                    star_op_tmp, direction, triangle_type, nodes_triangle_bdry
                )
        return star_op_tmp

    def star_operator_timeslice(self, z: int, d: int):
        """Generate the star operator for a time slice."""
        # current timeslice
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

        if d in self.seed_star_op:  # load an already generated seed
            cached_star_op, cached_prism = self.seed_star_op[d]
            # Remap micro-positions to current z
            star_op_init = [Position3DHex(p.x, p.y, z) for p in cached_star_op]
            cached_prism = self.seed_star_op[d][1]
            # update z to current timeslice
            init_prism = Prism(
                Position3DHex(cached_prism.position.x, cached_prism.position.y, z),
                cached_prism.kind,
                cached_prism.label,
            )

            # If the cached init_prism is not present in this timeslice, walk the
            # star operator from the cached (phantom) position to the first actual
            # prism in this layer so that init_prism and star_op_init are consistent
            # with what exists at layer z.
            current_macro_positions = {p.position for p in current_prisms}
            if init_prism.position not in current_macro_positions:
                actual_init = current_prisms[0]
                star_op_init = self._walk_star_op_to_prism(
                    star_op_init, init_prism, actual_init, z, d
                )
                init_prism = actual_init

        else:
            # generate initial star operator
            init_prism = current_prisms[0]
            if init_prism.position.x % 2 == 0:
                triangle_type = "upwards"
            else:
                triangle_type = "downwards"

            left_corner_lst = self.corners_timeslice(
                z, d, [current_prisms[0]]
            )  # only 1 left corner necessary at this point

            nodes_triangle_bdry = ZXPrism.patch_triangle_bdry(d, left_corner_lst[0], triangle_type)
            star_op_init = ZXPrism.star_operator_patch(triangle_type, nodes_triangle_bdry)
            self.seed_star_op[d] = (star_op_init.copy(), init_prism)

        star_op_final = star_op_init.copy()

        # need to map star op to prisms and hence pipes for X/Z assignment later
        prism_to_star_op: dict[Position3DHex, list[Position3DHex]] = {
            init_prism.position: star_op_init.copy()
        }

        # find path to any other patch with finding the macro path.
        for prism in [p for p in current_prisms if p != init_prism]:
            path = init_prism.position.shortest_path_spatial(prism.position)
            # translate to prisms to have no type mismatch, but random prism generation here
            path_prisms = [Prism(pos, ZXPrism(BasisPrism.N, BasisPrism.N), "") for pos in path]
            left_corner_lst_tmp = self.corners_timeslice(z, d, path_prisms)
            #!todo deduplicate this code with _reflect_to
            # go through each macro path and reflect consecutively until destination patch reached
            # this yields some repetitive computations but not too bad i think.
            star_op_tmp = star_op_init.copy()
            for idx, (pos, corner) in enumerate(zip(path, left_corner_lst_tmp)):
                if pos.x % 2 == 0:
                    triangle_type = "upwards"
                else:
                    triangle_type = "downwards"
                nodes_triangle_bdry = ZXPrism.patch_triangle_bdry(d, corner, triangle_type)
                # find direction by creating dummy pipe
                if idx + 1 < len(path):
                    pipe = PrismPipe(
                        path_prisms[idx],
                        path_prisms[idx + 1],
                        PrismPipeKind(hor=BasisPrism.N, ver=BasisPrism.N),
                    )
                    direction = pipe.direction_connecting_bdry()
                    star_op_tmp = ZXPrism.reflect_star_operator(
                        star_op_tmp, direction, triangle_type, nodes_triangle_bdry
                    )
                else:
                    # if last element reached, we have our star op and add
                    star_op_final += star_op_tmp.copy()
                    prism_to_star_op[prism.position] = star_op_tmp.copy()

        # depending on the pipe ver/hor, decide whether the star operator is an x or z logical
        partitioned_star_ops = PrismGraph.split_into_connected_components(star_op_final)
        # deduplicate
        partitioned_star_ops = [
            list(dict.fromkeys(component)) for component in partitioned_star_ops
        ]
        star_ops_x = []
        star_ops_z = []
        for component in partitioned_star_ops:
            component_set = set(component)
            for prism_pos, star_op in prism_to_star_op.items():
                if component_set & set(star_op):
                    basis = self._get_basis_from_pipes(prism_pos, current_pipes)
                    break
            if basis == "X":
                star_ops_x.append(component)
            elif basis == "Z":
                star_ops_z.append(component)
            elif basis == "XZ":
                star_ops_x.append(component)
                star_ops_z.append(component)

        return star_ops_x, star_ops_z

    def stabilizers_timeslice(
        self, z: int, d: int
    ) -> tuple[
        list[list[Position3DHex]],
        list[list[Position3DHex]],
        list[list[Position3DHex]],
        dict[PrismPipe, list[list[Position3DHex]]],
        dict[Prism, list[list[Position3DHex]]],
    ]:
        """Build the stabilizers of a given time slice and given distance d."""
        # filter prism and horizontal pipes of some given time slice
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

        left_corner_lst = self.corners_timeslice(z, d, current_prisms)

        # initialize a dictionary that maps the macro position of a prism or
        # a pipe to the corresponding data qubts. Needed later for SE circuits
        patch_pipe_to_data_qubits_dct = {}

        # place the current_prisms on the microscopic lattice
        #!TODO make this more efficient and avoid repeated calls
        # of patch_stabilizers, cache the objects and translate them.
        stabilizers = []
        # collect all nodes_triangle_bdry for each prism, in same order as current_prisms
        prism_bdries = {}
        prism_bdries_filtered = []
        dct_patch_stabilizers: dict[Prism, list[list[Position3DHex]]] = {}
        for prism, left_corner in zip(current_prisms, left_corner_lst):
            # is the prism end of a pipe?
            pipes_dirs = []
            for pipe in current_pipes:
                if prism in (pipe.u, pipe.v):
                    pipes_dirs.append(pipe.direction_connecting_bdry())  # noqa: PERF401
            pipes_dirs_opp = [
                el for el in ["a", "b", "c"] if el not in pipes_dirs
            ]  # flip the elements
            if prism.position.x % 2 == 0:
                stabs, nodes_triangle_bdry = ZXPrism.patch_stabilizers(
                    d, "upwards", left_corner, pipes_dirs_opp
                )
            else:
                stabs, nodes_triangle_bdry = ZXPrism.patch_stabilizers(
                    d, "downwards", left_corner, pipes_dirs_opp
                )
            # add to patch_pipe_to_data_qubits_dct
            data_qubits = list(set([p for stab in stabs for p in stab]))
            patch_pipe_to_data_qubits_dct.update({prism: data_qubits})

            stabilizers += stabs
            dct_patch_stabilizers.update({prism: stabs})
            # prism_bdries.append(nodes_triangle_bdry)
            prism_bdries.update({prism.position: nodes_triangle_bdry})
            prism_bdries_filtered.append({k: nodes_triangle_bdry[k] for k in pipes_dirs})

        # collect weight-2 stabilizers at connecting bdries.
        all_weight_2_stabs = ZXPrism.find_pairs_with_two_overlaps(stabilizers)

        # generate the weight-3,-5,-6 stabilizers per pipe and store info about which pipe
        dct_single_type_stabs: dict[PrismPipe, list[list[Position3DHex]]] = {}
        #!TODO also sort the weight 2 stabs here into correct lists.

        for pipe in current_pipes:
            stabs_list = []
            bdry_pair_dir = pipe.direction_connecting_bdry()
            bdry1 = prism_bdries[pipe.u.position][bdry_pair_dir]
            bdry2 = prism_bdries[pipe.v.position][bdry_pair_dir]
            # the bdries are built such that they are ordered correctly and can be paired up
            stab_temp = []
            for idx in range(len(bdry1)):
                pos1 = bdry1[idx]
                pos2 = bdry2[idx]
                if pos1 not in stab_temp and pos2 not in stab_temp:
                    stab_temp.append(pos1)
                    stab_temp.append(pos2)
                # overlap with any weight 2
                pos1_neighbor_weight2 = [
                    sublist
                    for sublist in all_weight_2_stabs
                    if any(pos1.is_neighbour(pos) for pos in sublist)
                ]
                pos2_neighbor_weight2 = [
                    sublist
                    for sublist in all_weight_2_stabs
                    if any(pos2.is_neighbour(pos) for pos in sublist)
                ]
                pos1neigh, pos2neigh = None, None
                if pos1_neighbor_weight2 and pos2_neighbor_weight2:
                    assert len(pos1_neighbor_weight2) == 1
                    for el in pos1_neighbor_weight2[0]:
                        if pos1.is_neighbour(el):
                            pos1neigh = el
                            break

                    assert len(pos2_neighbor_weight2) == 1
                    for el in pos1_neighbor_weight2[0]:
                        if pos1.is_neighbour(el):
                            pos2neigh = el
                            break
                flag = False
                if idx + 1 <= len(bdry1) - 1 and idx + 1 <= len(bdry2) - 1:
                    if pos1.is_neighbour(bdry1[idx + 1]) and pos2.is_neighbour(bdry2[idx + 1]):
                        if bdry1[idx + 1] not in stab_temp and bdry2[idx + 1] not in stab_temp:
                            stab_temp.append(bdry1[idx + 1])
                            stab_temp.append(bdry2[idx + 1])
                        flag = True
                else:
                    # last element of the pairs -> add the final stabilizer
                    stab_temp = ZXPrism.order_stabilizer(stab_temp)
                    stabs_list.append(stab_temp.copy())
                    break
                # if pos1_neighbor_weight2 and pos2_neighbor_weight2:
                if pos1neigh and pos2neigh:
                    if pos1neigh == pos2neigh:
                        if pos1neigh not in stab_temp:
                            stab_temp.append(pos1neigh)

                        if pos1_neighbor_weight2[0] not in stabs_list:
                            stabs_list.append(
                                pos1_neighbor_weight2[0]
                            )  #!TODO ADD WEGIHT 2 STABILIZERS TOO

                        if flag is False:
                            # whenever wight 2 is touched, close stabilizer and start new one.
                            stab_temp = ZXPrism.order_stabilizer(stab_temp)
                            stabs_list.append(stab_temp.copy())
                            stab_temp = []
            dct_single_type_stabs.update({pipe: stabs_list.copy()})
            stabs_list_2 = [stab for stab in stabs_list if len(stab) == 2]
            data_qubits = list(set([p for stab in stabs_list_2 for p in stab]))
            patch_pipe_to_data_qubits_dct.update({pipe: data_qubits})

        # remove pipe data qubits from the corresponding prism entries
        # (the prisms should not contain the pipe qubits at the intersection)
        for key in patch_pipe_to_data_qubits_dct.keys():  # noqa: PLC0206
            if isinstance(key, PrismPipe):
                pipe_qubits = set(patch_pipe_to_data_qubits_dct[key])
                for prism_key in (key.u, key.v):
                    if prism_key in patch_pipe_to_data_qubits_dct:
                        patch_pipe_to_data_qubits_dct[prism_key] = [
                            q
                            for q in patch_pipe_to_data_qubits_dct[prism_key]
                            if q not in pipe_qubits
                        ]

        # connect the prisms according to current_pipes on the micro lattice -> distinction x, z
        stabilizers_x = stabilizers.copy()
        stabilizers_z = stabilizers.copy()
        # in a connected object, all pipes must be of same kind, but you may have multiple not
        # not directly connected parts in a timeslice, hence go through each pipe separately
        for pipe in current_pipes:
            hor = pipe.kind.hor
            ver = pipe.kind.ver
            bdry_pair_dir = pipe.direction_connecting_bdry()
            if hor == BasisPrism.X and ver == BasisPrism.Z:
                # hor=X means that init/meas in X basis, thus single type stabilizers at bdry are Z
                stabilizers_z += dct_single_type_stabs[pipe]
                pass
            elif hor == BasisPrism.Z and ver == BasisPrism.X:
                # hor=Z means that init/meas in Z basis, thus single type stabilizers at bdry are X
                stabilizers_x += dct_single_type_stabs[pipe]
                pass
            elif BasisPrism.N in (hor, ver):
                raise TQECError("Horizontal pipes should not be N")
            else:
                raise TQECError("Horizontal pipes have wrong colors for ver,hor.")

        self.prism_pipes_to_data_qubits_full.update({z: patch_pipe_to_data_qubits_dct})
        return (
            stabilizers_x,
            stabilizers_z,
            all_weight_2_stabs,
            dct_single_type_stabs,
            dct_patch_stabilizers,
        )

    @staticmethod
    def find_origin_vertex_stab(stabilizer: list[Position3DHex]) -> Position3DHex:
        r"""Define an origin vertex for each weight-6 stabilizer.

          o
        /   \
        x    o
        |    |
        o    o
        \    /
           o
        the x defines the origin vertex we define. the vertical axis is direction C.
        """
        if len(stabilizer) != 6:
            raise ValueError("`find_origin_vertex_stab` only works for weight 6 stabilizers.")
        for idx, vertex in enumerate(stabilizer):
            neigh1 = stabilizer[(idx - 1) % len(stabilizer)]
            neigh2 = stabilizer[(idx + 1) % len(stabilizer)]
            if (
                vertex.x - neigh1.x == -1
                and vertex.y - neigh1.y == -1
                and vertex.x - neigh2.x == -1
                and vertex.y - neigh2.y == 1
            ):
                return vertex
            elif (
                vertex.x - neigh2.x == -1
                and vertex.y - neigh2.y == -1
                and vertex.x - neigh1.x == -1
                and vertex.y - neigh1.y == 1
            ):
                return vertex
        raise TQECError("`origin vertex` could not be found in given plaquette.")

    @staticmethod
    def filter_isolated_patch_stabilizers(
        stabilizers: list[list[Position3DHex]],
    ) -> list[list[Position3DHex]]:
        """Remove weight-4 stabilizers.

        Remove weight-4 stabilizers that form an isolated group
        with no overlap with any other stabilizer.
        """
        weight4 = [stab for stab in stabilizers if len(stab) == 4]
        others = [stab for stab in stabilizers if len(stab) != 4]

        isolated = []
        for stab in weight4:
            stab_set = set(stab)
            touches_non_weight4 = any(stab_set & set(other) for other in others)
            if not touches_non_weight4:
                isolated.append(stab)
        return [stab for stab in stabilizers if stab not in isolated]

    @staticmethod
    def find_three_coloring_stabilizers(stabilizers: list[list[Position3DHex]]) -> dict:
        """Find an assignment of rgb colors to the stabilizers.

        Exclude weight-2 stabilizers for this consideration.
        Uses the fact that origin vertices of weight-6 stabilizers form a
        hexagonal lattice, which is 3-colorable by coordinate parity.
        """
        # filter stabilizers that constitute a d=3 single patch of three stabilizers of each kind.
        # they destroy the coloring and are not needed for the
        # construction of horizontal correlatino surfaces
        stabilizers = PrismGraph.filter_isolated_patch_stabilizers(stabilizers)

        weight6 = [stab for stab in stabilizers if len(stab) == 6]
        if not weight6:
            raise TQECError("No weight-6 stabilizer found to seed the 3-coloring.")
        others = [stab for stab in stabilizers if len(stab) != 6 and len(stab) != 2]

        COLORS = ["red", "green", "blue"]  # noqa: N806
        assignment = {}

        # Get the origin vertex of the seed stabilizer
        seed_stab = weight6[0]
        seed_origin = PrismGraph.find_origin_vertex_stab(seed_stab)

        for stab in weight6:
            origin = PrismGraph.find_origin_vertex_stab(stab)
            dx = origin.x - seed_origin.x
            dy = origin.y - seed_origin.y
            color_index = ((dx - dy) // 2) % 3
            assignment[tuple(stab)] = COLORS[color_index]

        # remaining stabilizers: weight 3 and weight 5 and weight 4
        # check what the neighboring weight-6 colors are and take the non appearing color.
        unassigned = list(others)
        while unassigned:
            still_unassigned = []
            for other_stab in unassigned:
                other_vertices = set(other_stab)
                neighboring_colors = set()
                for stab in weight6:
                    if set(stab) & other_vertices:
                        neighboring_colors.add(assignment[tuple(stab)])
                for stab in others:
                    key = tuple(stab)
                    if key in assignment and set(stab) & other_vertices:
                        neighboring_colors.add(assignment[key])
                remaining = [c for c in COLORS if c not in neighboring_colors]
                if len(remaining) == 1:
                    assignment[tuple(other_stab)] = remaining[0]
                else:
                    still_unassigned.append(other_stab)
            if len(still_unassigned) == len(unassigned):
                raise TQECError(
                    "Could not resolve coloring — stuck with ambiguous boundary stabilizers."
                )
            unassigned = still_unassigned

        return assignment

    def find_all_linear_paths_timeslice(self, z: int) -> list[list[Position3DHex]]:
        """Find all simple linear paths through the prism graph restricted to a fixed z slice."""
        # Restrict to nodes in the given z-slice
        nodes_in_slice = [n for n in self._graph.nodes() if n.z == z]
        subgraph = self._graph.subgraph(nodes_in_slice)

        # Find endpoints in the subgraph
        endpoints = [n for n, deg in subgraph.degree() if deg == 1]

        # If no endpoints exist (e.g. pure cycle), fall back to all nodes in slice
        if not endpoints:
            endpoints = list(subgraph.nodes())

        all_paths: list[list[Position3DHex]] = []

        for source in endpoints:
            for target in endpoints:
                if source >= target:
                    continue
                for node_path in nx.all_simple_paths(subgraph, source, target):
                    all_paths.append(node_path)
                    break  # only one path needed

        return all_paths

    @staticmethod
    def find_boundary_stabilizers(
        dct_patch_stabilizers: dict[Prism, list[list[Position3DHex]]],
        pos_a: Position3DHex,
        pos_b: Position3DHex,
        dct_single_type_stabs: dict[PrismPipe, list[list[Position3DHex]]],
    ) -> list[list[Position3DHex]]:
        """Find stabilizers at the boundary between pos_a and pos_b.

        Restricts to only the pipe connecting pos_a and pos_b directly.

        Args:
            dct_patch_stabilizers: mapping from prism to its list of stabilizers.
            pos_a: position of the first prism.
            pos_b: position of the second prism.
            dct_single_type_stabs: mapping from pipe to its boundary stabilizers.

        Returns:
            The stabilizers from the pipe between pos_a and pos_b that share
            at least one vertex with pos_a's patch.

        """
        stabs_a = next(
            (stabs for prism, stabs in dct_patch_stabilizers.items() if prism.position == pos_a),
            None,
        )
        if stabs_a is None:
            raise TQECError(f"No patch found for position {pos_a}.")

        vertices_a = set(v for stab in stabs_a for v in stab)

        # restrict to only the pipe directly connecting pos_a and pos_b
        connecting_pipe_stabs = [
            stab
            for pipe, stabs in dct_single_type_stabs.items()
            if {pipe.u.position, pipe.v.position} == {pos_a, pos_b}
            for stab in stabs
            if set(stab) & vertices_a
        ]
        return connecting_pipe_stabs

    @staticmethod
    def count_stabilizer_appearances(
        position: Position3DHex,
        stabilizers: list[list[Position3DHex]],
    ) -> int:
        """Count how many stabilizers touch the given position."""
        return sum(1 for stab in stabilizers if position in stab)

    @staticmethod
    def find_neighboring_bdry_stabilizer(
        init_stabilizer: list[Position3DHex],
        stabilizers: list[list[Position3DHex]],
        single_type_stabs: list[list[Position3DHex]],
        no_filter: bool = False,
    ) -> list[list[Position3DHex]]:
        """Find stabilizers at the boundary of the patch.

        Boundary does not mean the connecting STDW but the real boundary.
        A stabilizer is considered a boundary stabilizer if it touches init_stabilizer
        and has at least one vertex that appears in at most 2 stabilizers in the list.
        Stabilizers are excluded if ALL their low-appearance vertices (<=2) are
        exclusively part of single_type_stabs.
        """
        init_set = set(init_stabilizer)

        # find neighbors: stabilizers that share at least one vertex with init_stabilizer
        neighbors = [
            stab
            for stab in stabilizers
            if set(stab) != set(init_stabilizer) and set(stab) & init_set
        ]
        if no_filter:
            return neighbors

        single_type_verts = set(v for stab in single_type_stabs for v in stab)

        result = []
        for stab in neighbors:
            # vertices of this stabilizer that appear in at most 2 stabilizers
            low_appearance_verts = [
                v for v in stab if PrismGraph.count_stabilizer_appearances(v, stabilizers) <= 2
            ]
            if not low_appearance_verts:
                continue
            # exclude only if ALL low-appearance vertices are single_type verts
            if not all(v in single_type_verts for v in low_appearance_verts):
                result.append(stab)

        return result

    def _reflect_to(
        self, target_pos: Position3DHex, init_prism: Prism, star_op_init, z, d
    ) -> list[Position3DHex]:
        macro_path = init_prism.position.shortest_path_spatial(target_pos)
        path_prisms = [Prism(pos, ZXPrism(BasisPrism.N, BasisPrism.N), "") for pos in macro_path]
        left_corner_lst = self.corners_timeslice(z, d, path_prisms)
        star_op_tmp = star_op_init.copy()
        for idx, (pos, corner) in enumerate(zip(macro_path, left_corner_lst)):
            triangle_type = "upwards" if pos.x % 2 == 0 else "downwards"
            nodes_triangle_bdry = ZXPrism.patch_triangle_bdry(d, corner, triangle_type)
            if idx + 1 < len(macro_path):
                dummy_pipe = PrismPipe(
                    path_prisms[idx],
                    path_prisms[idx + 1],
                    PrismPipeKind(hor=BasisPrism.N, ver=BasisPrism.N),
                )
                direction = dummy_pipe.direction_connecting_bdry()
                star_op_tmp = ZXPrism.reflect_star_operator(
                    star_op_tmp, direction, triangle_type, nodes_triangle_bdry
                )
        return star_op_tmp

    def _test_stabilizer_product_timeslice(self, stabilizer_product, start_star, end_star):
        """Test whether a stabilizer product is valid.

        i.e. test whether it is trivial everywhere despite at the verices of the star_operators.
        """
        flattened = list(set([v for stab in stabilizer_product for v in stab]))
        for node in flattened:
            touches = self.count_stabilizer_appearances(node, stabilizer_product)
            if node in start_star + end_star:
                if touches % 2 == 0:
                    raise TQECError(
                        "The construction of the stabilizer product is wrong //"
                        f" (even number of stabilizers at logical op position {node} //"
                        f"has {touches} touches)."
                    )
            else:  # noqa: PLR5501
                if touches % 2 != 0:
                    raise TQECError(
                        "The construction of the stabilizer product is wrong //"
                        f" (odd number of stabilizers at trivial position {node} //"
                        f"has {touches} touches)."
                    )
        # logger.info("Stabilizer product test passed.")
        print("stabilizer tests passed")

    @staticmethod
    def _helper_bdry_start_end(single_type_stabs, star, patch_stabs):
        bdry_bdry = [stab for stab in single_type_stabs if len(stab) == 3 or len(stab) == 5]
        assert len(bdry_bdry) == 2, f"Internal error. {bdry_bdry}"
        d = len(single_type_stabs)
        bdry_1 = []
        temp_neigh = [bdry_bdry[0]]
        seen = [bdry_bdry[0]]
        while not set([v for stab in temp_neigh for v in stab]) & set(star):
            temp_neigh = PrismGraph.find_neighboring_bdry_stabilizer(
                temp_neigh[0], patch_stabs, single_type_stabs
            )
            temp_neigh = [stab for stab in temp_neigh if not any(set(stab) == set(s) for s in seen)]
            seen += temp_neigh
            bdry_1 += temp_neigh

        bdry_2 = []
        temp_neigh = [bdry_bdry[1]]
        seen = [bdry_bdry[1]]
        if d == 3:
            # the structure for d=3 is very peculiar and
            # i think very sensible to changes in the code
            temp_neigh = PrismGraph.find_neighboring_bdry_stabilizer(
                temp_neigh[0], patch_stabs, single_type_stabs
            )
            bdry_2 += [
                neigh for neigh in temp_neigh if set(neigh) not in [set(el) for el in bdry_1]
            ]
        else:
            while not set([v for stab in temp_neigh for v in stab]) & set(star):
                temp_neigh = PrismGraph.find_neighboring_bdry_stabilizer(
                    temp_neigh[0], patch_stabs, single_type_stabs
                )
                temp_neigh = [
                    stab for stab in temp_neigh if not any(set(stab) == set(s) for s in seen)
                ]
                seen += temp_neigh
                bdry_2 += temp_neigh

        return bdry_1, bdry_2

    @staticmethod
    def _helper_split_patch_in_thirds(star, patch_stabs):
        """Find regions of the patch determined by star.

        The regions are not perfect, but next helper can handle this
        """
        # flatten patch stabs into unique nodes
        flattened_patch = set([v for stab in patch_stabs for v in stab])

        remaining = flattened_patch - set(star)
        regions = []

        def flood(node: Position3DHex, component: set[Position3DHex]) -> None:
            component.add(node)
            remaining.discard(node)
            for nb in node.neighbors_spatial():
                if nb in remaining:
                    flood(nb, component)

        while remaining:
            component: set[Position3DHex] = set()
            flood(next(iter(remaining)), component)
            regions.append(component)

        # include also the neighboring star nodes
        for star_node in star:
            for region in regions:
                if any(nb in region for nb in star_node.neighbors_spatial()):
                    region.add(star_node)
        return regions

    @staticmethod
    def _get_color(stab, assignment):
        stab_set = set(stab)
        return next(color for key, color in assignment.items() if set(key) == stab_set)

    @staticmethod
    def _helper_fill_third(bdry, patch_stabs, regions, assignment):
        # determine the region that contains nodes from bdry[0] which is just a random choice
        for region in regions:
            if any(node in bdry[0] for node in region):
                region_current = region
                break

        # check which stabilizers are in the region by checking that
        # 1. a weight 6 stabilizer needs at least 5 nodes overlap with the region
        # 2. a weight 4 stabilizer needs at least 3 nodes overlap with the region
        # these special rules apply because the regions of _helper_split_patch_in_thirds
        # are  not perfect
        region_stabs = []
        for stab in patch_stabs:
            overlap = sum(1 for node in stab if node in region_current)
            if len(stab) == 4:
                if overlap >= 3:
                    region_stabs.append(stab)
            elif len(stab) == 6:
                if overlap >= 5:
                    region_stabs.append(stab)
            else:
                raise TQECError("internal error")

        # filter the color
        d = int((8 * len(patch_stabs) / 3 + 1) ** 0.5)
        if d == 3:
            if len(bdry) == 1:
                colors = [PrismGraph._get_color(bdry[0], assignment)]
                stabilizer_product_temp = [
                    stab for stab in region_stabs if assignment[tuple(stab)] in colors
                ]
            else:
                raise TQECError("d=3 edge case error.")
        else:
            colors = [
                PrismGraph._get_color(bdry[0], assignment),
                PrismGraph._get_color(bdry[1], assignment),
            ]
            stabilizer_product_temp = [
                stab for stab in region_stabs if assignment[tuple(stab)] in colors
            ]

        return stabilizer_product_temp

    def stabilizer_product_timeslice(
        self, z: int, d: int, dct_single_type_stabs, dct_patch_stabilizers, testing: bool = True
    ):
        """Construct the logical operator corresponding to a horizonatl correlation surface.

        this means that we look for a stabilizer product that transports a logical
        from one to another place spatially.

        This requires the stabilizers from `stabilizers_timeslice` as input.

        There are always different possibilities to go through a pipe diagram.
        This method generates all possible stabilizer products.
        """
        all_paths = self.find_all_linear_paths_timeslice(z=z)
        all_paths_stabilizer_product = []
        all_paths_stars = []

        # ----------------setup 3-coloring and seed star operator ------------------

        # 3coloring of full stabilizers whatever the z or x kind, this is specified later
        assignment = self.find_three_coloring_stabilizers(
            [stab for stabs in dct_single_type_stabs.values() for stab in stabs]
            + [stab for stabs in dct_patch_stabilizers.values() for stab in stabs]
        )

        # load a cached star operator or create a new seed star op and cache it
        if d in self.seed_star_op:  # load an already generated seed
            cached_star_op, cached_prism = self.seed_star_op[d]
            # Remap micro-positions to current z
            star_op_init = [Position3DHex(p.x, p.y, z) for p in cached_star_op]
            cached_prism = self.seed_star_op[d][1]
            # update z to current timeslice
            init_prism = Prism(
                Position3DHex(cached_prism.position.x, cached_prism.position.y, z),
                cached_prism.kind,
                cached_prism.label,
            )

        else:
            # generate initial star operator
            init_prism_pos = all_paths[0][0]  # choose some element
            if init_prism_pos.x % 2 == 0:
                triangle_type = "upwards"
            else:
                triangle_type = "downwards"

            init_prism = Prism(
                init_prism_pos,
                "NN",  # just some choice
                "",
            )
            # only 1 left corner necessary at this point
            left_corner_lst = self.corners_timeslice(z, d, [init_prism])

            nodes_triangle_bdry = ZXPrism.patch_triangle_bdry(d, left_corner_lst[0], triangle_type)
            star_op_init = ZXPrism.star_operator_patch(triangle_type, nodes_triangle_bdry)
            self.seed_star_op[d] = (star_op_init.copy(), init_prism)

        # --------------go through all paths and collect the stabilizer products-----------------

        for path in all_paths:
            stabilizer_product = []
            if len(path) == 2:
                start_point = path[0]
                end_point = path[-1]
                boundary = self.find_boundary_stabilizers(
                    dct_patch_stabilizers, start_point, end_point, dct_single_type_stabs
                )
                stabilizer_product += [el for el in boundary if len(el) != 2]
            else:
                for idx, prism_pos in enumerate(path[1:-1], start=1):
                    # find boundary stabilizers between previous and current prism
                    boundary_left = self.find_boundary_stabilizers(
                        dct_patch_stabilizers, prism_pos, path[idx - 1], dct_single_type_stabs
                    )
                    boundary_right = self.find_boundary_stabilizers(
                        dct_patch_stabilizers, prism_pos, path[idx + 1], dct_single_type_stabs
                    )

                    stabilizer_product += [
                        el for el in boundary_left if len(el) != 2
                    ]  # add the boundary operators too
                    if idx == len(path) - 2:
                        stabilizer_product += [el for el in boundary_right if len(el) != 2]

                    # determine color of left boundary
                    color_left = None
                    for stab in boundary_left:
                        if len(stab) != 2:
                            stab_set = set(stab)
                            for key, color in assignment.items():
                                if set(key) == stab_set:
                                    color_left = color
                                    break
                        if color_left is not None:
                            break

                    # determine color of right boundary
                    color_right = None
                    for stab in boundary_right:
                        if len(stab) != 2:
                            stab_set = set(stab)
                            for key, color in assignment.items():
                                if set(key) == stab_set:
                                    color_right = color
                                    break
                        if color_right is not None:
                            break

                    # add all stabilizers of this prism with matching colors
                    prism_stabilizers = next(
                        stabs
                        for prism, stabs in dct_patch_stabilizers.items()
                        if prism.position == prism_pos
                    )
                    for stab in prism_stabilizers:
                        stab_set = set(stab)
                        for key, color in assignment.items():
                            if set(key) == stab_set:
                                if color in (color_left, color_right):
                                    stabilizer_product.append(stab)
                                break

                    # determine which weight-2 operators are needed at the interfaces of two patches
                    if idx != len(path) - 1 and idx != 1:  # not at the bdry of the path
                        weight_2_stabs = [el for el in boundary_left if len(el) == 2]
                        # go through the weight 2 stabilizers, at each data qubit,
                        # at each data qubit, check how many data qubits are touched
                        # by a stabilizer in the product
                        # add the weight 2 stabilizer to the product if odd number
                        # of stabilizer touch the qubit.
                        for stab in weight_2_stabs:
                            bool_lst = []  # bool for each data qubit, True if odd, False if even
                            for qubit in stab:
                                touches = self.count_stabilizer_appearances(
                                    qubit, stabilizer_product
                                )
                                if touches % 2 == 0:
                                    bool_lst.append(False)
                                else:
                                    bool_lst.append(True)
                            if any(bool_lst) and not all(bool_lst):
                                raise TQECError("Expected all True or all False, got a mix.")
                            if all(bool_lst):
                                stabilizer_product.append(stab)

            # ----------------star ops---------------------
            start_point = path[0]
            end_point = path[-1]
            start_star = self._reflect_to(start_point, init_prism, star_op_init, z, d)
            end_star = self._reflect_to(end_point, init_prism, star_op_init, z, d)

            # ---------------------------------------
            # find color-pair of the two thirds connected to the bulk patches.
            # for patch at start_point find the color of the weight-4 stabilizers

            # Get patch stabilizers and single-type stabs for start_point
            start_patch_stabs = next(
                stabs
                for prism, stabs in dct_patch_stabilizers.items()
                if prism.position == start_point
            )
            start_single_type_stabs = [
                stab
                for pipe, stabs in dct_single_type_stabs.items()
                if pipe.u.position == start_point or pipe.v.position == start_point  # noqa: PLR1714
                for stab in stabs
            ]

            # end
            end_patch_stabs = next(
                stabs
                for prism, stabs in dct_patch_stabilizers.items()
                if prism.position == end_point
            )
            end_single_type_stabs = [
                stab
                for pipe, stabs in dct_single_type_stabs.items()
                if pipe.u.position == end_point or pipe.v.position == end_point  # noqa: PLR1714
                for stab in stabs
            ]

            bdry_1_from_start, bdry_2_from_start = self._helper_bdry_start_end(
                start_single_type_stabs, start_star, start_patch_stabs
            )
            stabilizer_product += bdry_1_from_start
            stabilizer_product += bdry_2_from_start
            bdry_1_from_end, bdry_2_from_end = self._helper_bdry_start_end(
                end_single_type_stabs, end_star, end_patch_stabs
            )
            stabilizer_product += bdry_1_from_end
            stabilizer_product += bdry_2_from_end

            regions_start = self._helper_split_patch_in_thirds(start_star, start_patch_stabs)
            regions_end = self._helper_split_patch_in_thirds(end_star, end_patch_stabs)

            # from bdry_1_from_start to middle
            stabilizer_product += self._helper_fill_third(
                bdry_1_from_start, start_patch_stabs, regions_start, assignment
            )
            # from bdry_2_from_start to middle
            stabilizer_product += self._helper_fill_third(
                bdry_2_from_start, start_patch_stabs, regions_start, assignment
            )
            # from bdry_1_from_end to middle
            stabilizer_product += self._helper_fill_third(
                bdry_1_from_end, end_patch_stabs, regions_end, assignment
            )
            # from bdry_2_from_end to middle
            stabilizer_product += self._helper_fill_third(
                bdry_2_from_end, end_patch_stabs, regions_end, assignment
            )

            all_paths_stars.append([start_star, end_star])

            # remove duplicate stabilizers
            stabilizer_product = [
                stab
                for i, stab in enumerate(stabilizer_product)
                if not any(set(stab) == set(s) for s in stabilizer_product[:i])
            ]

            # add weight-2 stabilizers according to the filling of the end/start patch
            boundary_start = self.find_boundary_stabilizers(
                dct_patch_stabilizers, start_point, path[1], dct_single_type_stabs
            )
            boundary_end = self.find_boundary_stabilizers(
                dct_patch_stabilizers, end_point, path[-2], dct_single_type_stabs
            )
            boundary_start_two = [el for el in boundary_start if len(el) == 2]
            boundary_end_two = [el for el in boundary_end if len(el) == 2]

            for stab_weight2 in boundary_start_two + boundary_end_two:
                bool_lst = []
                for node in stab_weight2:
                    touches = self.count_stabilizer_appearances(node, stabilizer_product)
                    bool_lst.append(touches % 2 != 0)
                if any(bool_lst) and not all(bool_lst):
                    raise TQECError(
                        "Expected all True or all False for boundary weight-2 stabilizer."
                    )
                if all(bool_lst):
                    stabilizer_product.append(stab_weight2)

            # remaining weight-2 stabilizers at bdries
            # check those weight-6 stabilizers that have qubits that are not touched
            # these are placed along STDWs not part of the current path
            # to make them trivial, add the respective weight-2 stabilizer
            star_set_full = set(start_star) | set(end_star)
            weight6_in_product = [
                stab
                for stab in stabilizer_product
                if len(stab) == 6 and not (set(stab) & star_set_full)
            ]
            for stab in weight6_in_product:
                once_touched = [
                    qubit
                    for qubit in stab
                    if self.count_stabilizer_appearances(qubit, stabilizer_product) == 1
                    and qubit not in (start_star, end_star)
                ]
                if len(once_touched) == 2 and once_touched[0].is_neighbour(once_touched[1]):
                    stabilizer_product.append(once_touched)

            all_paths_stabilizer_product.append(stabilizer_product.copy())

            if testing:
                self._test_stabilizer_product_timeslice(stabilizer_product, start_star, end_star)

        # ---------determine whether the CS is X or Z type.---------
        all_paths_stabilizer_product_x = []
        all_paths_stabilizer_product_z = []
        all_paths_x = []
        all_paths_z = []
        stars_z = []
        stars_x = []

        current_pipes = []
        for pos1, pos2, attrs in self._graph.edges(data=True):
            if pos1.z == z and pos2.z == z:
                edge = attrs[self._EDGE_DATA_KEY]
                if edge.kind.is_spatial:
                    current_pipes.append(edge)
        # just take the first prism pos, it is guaranteed by construction
        # of the prism graph that it has to be consistent
        for path, stab_prod, star in zip(all_paths, all_paths_stabilizer_product, all_paths_stars):
            basis = self._get_basis_from_pipes(path[0], current_pipes)
            if (
                basis == "Z"
            ):  # turned around bc get basis is for vertical CS and here we have horizontal CS
                all_paths_stabilizer_product_x.append(stab_prod)
                all_paths_x.append(path)
                stars_x.append(star)
            elif basis == "X":
                all_paths_stabilizer_product_z.append(stab_prod)
                all_paths_z.append(path)
                stars_z.append(star)
            else:
                raise TQECError("Internal error.")

        # assignment is order dependent, so please return the specific
        # assignment which is in general not unique
        return StabilizerProductResult(
            assignment=assignment,
            paths_x=all_paths_x,
            paths_z=all_paths_z,
            stars_x=stars_x,
            stars_z=stars_z,
            stabilizer_products_x=all_paths_stabilizer_product_x,
            stabilizer_products_z=all_paths_stabilizer_product_z,
        )
