"""Prism graph representation of a logical computation with color code."""

from __future__ import annotations

from networkx import Graph, is_connected
from tqec.computation.prism import Port, Position3DHex, PrismKind, BasisPrism, prism_kind_from_string, Prism, ZXPrism
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

    def corners_timeslice(self, z: int, d: int, current_prisms):
        """Find the corners to build stabilizers from them for current timeslice."""
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

        return left_corner_lst

    @staticmethod
    def split_into_connected_components(
        positions: list[Position3DHex],
    ) -> list[list[Position3DHex]]:
        """Split positions into sublists where elements are connected via neighbour or next-nearest-neighbour."""
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
            for pos2 in positions[i+1:]:
                if pos1.is_neighbour(pos2) or pos1.is_next_nearest_neighbour(pos2):
                    union(pos1, pos2)

        # group by root
        components: dict[Position3DHex, list[Position3DHex]] = {}
        for pos in positions:
            root = find(pos)
            components.setdefault(root, []).append(pos)

        return list(components.values())

    def star_operator_timeslice(self, z:int, d:int):
        """Generate the star operator for a time slice."""
        #current timeslice
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

        left_corner_lst = self.corners_timeslice(z, d, [current_prisms[0]]) #only 1 left corner necessary at this point

        #generate initial star operator
        init_prism = current_prisms[0]
        if init_prism.position.x % 2 == 0:
            triangle_type = "upwards"
        else:
            triangle_type = "downwards"

        nodes_triangle_bdry = ZXPrism.patch_triangle_bdry(d, left_corner_lst[0], triangle_type)
        star_op_init = ZXPrism.star_operator_patch(triangle_type, nodes_triangle_bdry)
        star_op_final = star_op_init.copy()

        #need to map star op to prisms and hence pipes for X/Z assignment later
        prism_to_star_op: dict[Position3DHex, list[Position3DHex]] = {
                init_prism.position: star_op_init.copy()
            }

        #find path to any other patch with finding the macro path.
        for prism in current_prisms[1:]:
            path = init_prism.position.shortest_path_spatial(prism.position)
            #translate to prisms to have no type mismatch, but random prism generation here
            path_prisms = [Prism(pos, ZXPrism(BasisPrism.N, BasisPrism.N), "") for pos in path]
            left_corner_lst_tmp = self.corners_timeslice(z, d, path_prisms)

            #go through each macro path and reflect consecutively until destination patch reached
            #this yields some repetetive computations but not too bad i think.
            star_op_tmp = star_op_init.copy()
            for idx, (pos, corner) in enumerate(zip(path, left_corner_lst_tmp)):
                if pos.x % 2 == 0:
                    triangle_type = "upwards"
                else:
                    triangle_type = "downwards"
                nodes_triangle_bdry = ZXPrism.patch_triangle_bdry(d, corner, triangle_type)
                #find direction by creating dummy pipe
                if idx+1 < len(path):
                    pipe = PrismPipe(path_prisms[idx], path_prisms[idx+1], PrismPipeKind(hor = BasisPrism.N, ver = BasisPrism.N))
                    direction = pipe.direction_connecting_bdry()
                    star_op_tmp = ZXPrism.reflect_star_operator(star_op_tmp, direction, triangle_type, nodes_triangle_bdry)
                else:
                    #if last element reached, we have our star op and add
                    star_op_final += star_op_tmp.copy()
                    prism_to_star_op[prism.position] = star_op_tmp.copy()

        # assign X or Z via pipes: find any pipe connected to this prism and read its kind
        def get_basis_from_pipes(prism_pos: Position3DHex) -> str:
            for pipe in current_pipes:
                if prism_pos in (pipe.u.position, pipe.v.position):
                    if pipe.kind.hor == BasisPrism.X and pipe.kind.ver == BasisPrism.Z:
                        return "X"
                    elif pipe.kind.hor == BasisPrism.Z and pipe.kind.ver == BasisPrism.X:
                        return "Z"
            return "X"  # fallback for isolated prisms with no pipes


        #depending on the pipe ver/hor, decide whether the star operator is an x or z logical
        partitioned_star_ops = PrismGraph.split_into_connected_components(star_op_final)
        star_ops_x = []
        star_ops_z = []
        for component in partitioned_star_ops:
            component_set = set(component)
            basis = "X"  # fallback
            for prism_pos, star_op in prism_to_star_op.items():
                if component_set & set(star_op):
                    basis = get_basis_from_pipes(prism_pos)
                    break
            if basis == "X":
                star_ops_x.append(component)
            else:
                star_ops_z.append(component)

        return star_ops_x, star_ops_z

    def stabilizers_timeslice(self, z: int, d: int) -> tuple[list[list[Position3DHex]], list[list[Position3DHex]]]:
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

        left_corner_lst = self.corners_timeslice(z, d, current_prisms)

        #place the current_prisms on the microscopic lattice
        #!TODO make this more efficient and avoid repeated calls of patch_stabilizers, cache the objects and translate them.
        stabilizers = []
        prism_bdries = {} #collect all nodes_triangle_bdry for each prism, in same order as current_prisms
        prism_bdries_filtered = []
        for prism, left_corner in zip(current_prisms, left_corner_lst):
            #is the prism end of a pipe?
            pipes_dirs = []
            for pipe in current_pipes:
                if prism in (pipe.u, pipe.v):
                    pipes_dirs.append(pipe.direction_connecting_bdry())  # noqa: PERF401
            pipes_dirs_opp = [el for el in ["a", "b", "c"] if el not in pipes_dirs] #flip the elements
            if prism.position.x % 2 == 0:
                stabs, nodes_triangle_bdry = ZXPrism.patch_stabilizers(d, "upwards", left_corner, pipes_dirs_opp)
            else:
                stabs, nodes_triangle_bdry = ZXPrism.patch_stabilizers(d, "downwards", left_corner, pipes_dirs_opp)
            stabilizers += stabs
            #prism_bdries.append(nodes_triangle_bdry)
            prism_bdries.update({prism.position: nodes_triangle_bdry})
            prism_bdries_filtered.append({k: nodes_triangle_bdry[k] for k in pipes_dirs})

        #collect weight-2 stabilizers at connecting bdries.
        all_weight_2_stabs = ZXPrism.find_pairs_with_two_overlaps(stabilizers)

        #generate the weight-3,-5,-6 stabilizers per pipe and store info about which pipe
        dct_single_type_stabs = {}

        #!TODO also sort the weight 2 stabs here into correct lists.

        for pipe in current_pipes:
            stabs_list = []
            bdry_pair_dir = pipe.direction_connecting_bdry()
            bdry1 = prism_bdries[pipe.u.position][bdry_pair_dir]
            bdry2 = prism_bdries[pipe.v.position][bdry_pair_dir]
            #the bdries are built such that they are ordered correctly and can be paired up
            stab_temp = []
            for idx in range(len(bdry1)):
                pos1 = bdry1[idx]
                pos2 = bdry2[idx]
                if pos1 not in stab_temp and pos2 not in stab_temp:
                    stab_temp.append(pos1)
                    stab_temp.append(pos2)
                #overlap with any weight 2
                pos1_neighbor_weight2 = [sublist for sublist in all_weight_2_stabs if any(pos1.is_neighbour(pos) for pos in sublist)]
                pos2_neighbor_weight2 = [sublist for sublist in all_weight_2_stabs if any(pos2.is_neighbour(pos) for pos in sublist)]
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
                if idx+1 <= len(bdry1)-1 and idx+1 <= len(bdry2)-1:
                    if pos1.is_neighbour(bdry1[idx+1]) and pos2.is_neighbour(bdry2[idx+1]):
                        if bdry1[idx+1] not in stab_temp and bdry2[idx+1] not in stab_temp:
                            stab_temp.append(bdry1[idx+1])
                            stab_temp.append(bdry2[idx+1])
                        flag = True
                else:
                    #last element of the pairs -> add the final stabilizer
                    stab_temp = ZXPrism.order_stabilizer(stab_temp)
                    stabs_list.append(stab_temp.copy())
                    break
                #if pos1_neighbor_weight2 and pos2_neighbor_weight2:
                if pos1neigh and pos2neigh:
                    if pos1neigh == pos2neigh:
                        if pos1neigh not in stab_temp:
                            stab_temp.append(pos1neigh)

                        if pos1_neighbor_weight2[0] not in stabs_list:
                            stabs_list.append(pos1_neighbor_weight2[0]) #!TODO ADD WEGIHT 2 STABILIZERS TOO

                        if flag is False:
                            #whenever wight 2 is touched, close stabilizer and start new one.
                            stab_temp = ZXPrism.order_stabilizer(stab_temp)
                            stabs_list.append(stab_temp.copy())
                            stab_temp = []
            dct_single_type_stabs.update({pipe: stabs_list.copy()})



        #connect the prisms according to current_pipes on the micro lattice -> distinction x, z
        stabilizers_x = stabilizers.copy()
        stabilizers_z = stabilizers.copy()
        #in a connected object, all pipes must be of same kind, but you may have multiple not
        #not directly connected parts in a timeslice, hence go through each pipe separately
        for pipe in current_pipes:
            hor = pipe.kind.hor
            ver = pipe.kind.ver
            bdry_pair_dir = pipe.direction_connecting_bdry()
            if hor == BasisPrism.X and ver == BasisPrism.Z:
                #hor=X means that init/meas in X basis, thus single type stabilizers at bdry are Z
                stabilizers_z += dct_single_type_stabs[pipe]
                pass
            elif hor == BasisPrism.Z and ver == BasisPrism.X:
                #hor=Z means that init/meas in Z basis, thus single type stabilizers at bdry are X
                stabilizers_x += dct_single_type_stabs[pipe]
                pass
            elif BasisPrism.N in (hor, ver):
                raise TQECError("Horizontal pipes should not be N")
            else:
                raise TQECError("Horizontal pipes have wrong colors for ver,hor.")

        return stabilizers_x, stabilizers_z, all_weight_2_stabs, dct_single_type_stabs