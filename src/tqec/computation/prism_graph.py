"""Prism graph representation of a logical computation with color code."""

from __future__ import annotations

from networkx import Graph, is_connected
import networkx as nx
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
        self.seed_star_op: dict[int, tuple] = {}

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


        if d in self.seed_star_op: #load an already generated seed
            cached_star_op, cached_prism = self.seed_star_op[d]
            # Remap micro-positions to current z
            star_op_init = [
                Position3DHex(p.x, p.y, z) for p in cached_star_op
            ]
            cached_prism = self.seed_star_op[d][1]
            # update z to current timeslice
            init_prism = Prism(
                Position3DHex(cached_prism.position.x, cached_prism.position.y, z),
                cached_prism.kind,
                cached_prism.label,
            )

        else:
            #generate initial star operator
            init_prism = current_prisms[0]
            if init_prism.position.x % 2 == 0:
                triangle_type = "upwards"
            else:
                triangle_type = "downwards"

            left_corner_lst = self.corners_timeslice(z, d, [current_prisms[0]]) #only 1 left corner necessary at this point

            nodes_triangle_bdry = ZXPrism.patch_triangle_bdry(d, left_corner_lst[0], triangle_type)
            star_op_init = ZXPrism.star_operator_patch(triangle_type, nodes_triangle_bdry)
            self.seed_star_op[d] = (star_op_init.copy(), init_prism)

        star_op_final = star_op_init.copy()

        #need to map star op to prisms and hence pipes for X/Z assignment later
        prism_to_star_op: dict[Position3DHex, list[Position3DHex]] = {
                init_prism.position: star_op_init.copy()
            }

        #find path to any other patch with finding the macro path.
        for prism in [p for p in current_prisms if p != init_prism]:
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
            return "XZ"  # fallback for isolated prisms with no pipes, for them it's both X and Z


        #depending on the pipe ver/hor, decide whether the star operator is an x or z logical
        partitioned_star_ops = PrismGraph.split_into_connected_components(star_op_final)
        star_ops_x = []
        star_ops_z = []
        for component in partitioned_star_ops:
            component_set = set(component)
            for prism_pos, star_op in prism_to_star_op.items():
                if component_set & set(star_op):
                    basis = get_basis_from_pipes(prism_pos)
                    break
            if basis == "X":
                star_ops_x.append(component)
            elif basis == "Z":
                star_ops_z.append(component)
            elif basis == "XZ":
                star_ops_x.append(component)
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
        dct_patch_stabilizers = {}
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
            dct_patch_stabilizers.update({prism: stabs})
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

        return stabilizers_x, stabilizers_z, all_weight_2_stabs, dct_single_type_stabs, dct_patch_stabilizers

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
        if len(stabilizer)!=6:
            raise ValueError("`find_origin_vertex_stab` only works for weight 6 stabilizers.")
        for idx, vertex in enumerate(stabilizer):
            neigh1 = stabilizer[(idx-1)%len(stabilizer)]
            neigh2 = stabilizer[(idx+1)%len(stabilizer)]
            if vertex.x - neigh1.x == -1 and vertex.y - neigh1.y == -1 and vertex.x - neigh2.x == -1 and vertex.y - neigh2.y == 1:
                return vertex
            elif vertex.x - neigh2.x == -1 and vertex.y - neigh2.y == -1 and vertex.x - neigh1.x == -1 and vertex.y - neigh1.y == 1:
                return vertex
        raise TQECError("`origin vertex` could not be found in given plaquette.")

    @staticmethod
    def find_three_coloring_stabilizers(stabilizers: list[list[Position3DHex]]) -> dict:
        """Find an assignment of rgb colors to the stabilizers.

        Exclude weight-2 stabilizers for this consideration.
        Uses the fact that origin vertices of weight-6 stabilizers form a 
        hexagonal lattice, which is 3-colorable by coordinate parity.
        """
        weight6 = [stab for stab in stabilizers if len(stab) == 6]
        if not weight6:
            raise TQECError("No weight-6 stabilizer found to seed the 3-coloring.")
        others = [stab for stab in stabilizers if len(stab) != 6 and len(stab)!= 2]

        COLORS = ["red", "green", "blue"]
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

        #remaining stabilizers: weight 3 and weight 5 and weight 4
        # check what the neighboring weight-6 colors are and take the non appearing color.

        #for other_stab in others:
        #    other_vertices = set(other_stab)
        #    neighboring_colors = set()
        #    for stab in weight6:
        #        if set(stab) & other_vertices:
        #            neighboring_colors.add(assignment[tuple(stab)])
        #    remaining = [c for c in COLORS if c not in neighboring_colors]
        #    assignment[tuple(other_stab)] = remaining[0]
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
                raise TQECError("Could not resolve coloring — stuck with ambiguous boundary stabilizers.")
            unassigned = still_unassigned

        return assignment

    def find_all_linear_paths_timeslice(self, z: int) -> list[list[PrismPipe]]:
        """Find all simple linear paths through the prism graph.

        This is needed to find all possible horizontal correlation surfaces of a slice.
        """
        # Find all nodes that are endpoints (degree 1) or isolated (degree 0)
        # Paths must start and end at such nodes
        endpoints = [n for n, deg in self._graph.degree() if deg == 1 and n.z == z]

        # If no endpoints exist (e.g. pure cycle), fall back to all nodes
        if not endpoints:
            endpoints = list(self._graph.nodes())

        all_paths: list[list[PrismPipe]] = []

        for source in endpoints:
            for target in endpoints:
                if source >= target:  # avoid duplicates and self-paths
                    continue
                for node_path in nx.all_simple_paths(self._graph, source, target):
                    all_paths.append(node_path)
                    break #only one node path (more will not be possible anyways)
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
        stabs_a = next((stabs for prism, stabs in dct_patch_stabilizers.items() if prism.position == pos_a), None)
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
        no_filter: bool = False
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
            stab for stab in stabilizers
            if stab is not init_stabilizer and set(stab) & init_set
        ]
        if no_filter:
            return neighbors

        single_type_verts = set(v for stab in single_type_stabs for v in stab)

        result = []
        for stab in neighbors:
            # vertices of this stabilizer that appear in at most 2 stabilizers
            low_appearance_verts = [
                v for v in stab
                if PrismGraph.count_stabilizer_appearances(v, stabilizers) <= 2
            ]
            if not low_appearance_verts:
                continue
            # exclude only if ALL low-appearance vertices are single_type verts
            if not all(v in single_type_verts for v in low_appearance_verts):
                result.append(stab)

        return result

    def stabilizer_product_timeslice(self, z: int, d: int, dct_single_type_stabs, dct_patch_stabilizers):
        """Construct the logical operator corresponding to a horizonatl correlation surface.

        this means that we look for a stabilizer product that transports a logical
        from one to another place spatially.

        This requires the stabilizers from `stabilizers_timeslice` as input.

        There are always different possibilities to go through a pipe diagram.
        This method generates all possible stabilizer products.
        """
        all_paths = self.find_all_linear_paths_timeslice(z = z)
        all_paths_stabilizer_product = []
        x_or_z = [] #list of strings for all_paths_stabilizer_product that determins whether X or Z stabilizer product depending on pipe type.

        #3coloring of full stabilizers whatever the z or x kind, this is specified later
        assignment = self.find_three_coloring_stabilizers(
            [stab for stabs in dct_single_type_stabs.values() for stab in stabs]
            + [stab for stabs in dct_patch_stabilizers.values() for stab in stabs]
        )

        def _get_color(stab):
            stab_set = set(stab)
            return next(color for key, color in assignment.items() if set(key) == stab_set)
        
        all_paths_stars = []


        #load a cached star operator or create a new seed star op and cache it
        if d in self.seed_star_op: #load an already generated seed
            cached_star_op, cached_prism = self.seed_star_op[d]
            # Remap micro-positions to current z
            star_op_init = [
                Position3DHex(p.x, p.y, z) for p in cached_star_op
            ]
            cached_prism = self.seed_star_op[d][1]
            # update z to current timeslice
            init_prism = Prism(
                Position3DHex(cached_prism.position.x, cached_prism.position.y, z),
                cached_prism.kind,
                cached_prism.label,
            )

        else:
            #generate initial star operator
            init_prism_pos = all_paths[0][0]#choose some element
            if init_prism_pos.x % 2 == 0:
                triangle_type = "upwards"
            else:
                triangle_type = "downwards"

            init_prism = Prism(
                init_prism_pos,
                "NN", #just some choice
                ""
            )
            left_corner_lst = self.corners_timeslice(z, d, [init_prism]) #only 1 left corner necessary at this point

            nodes_triangle_bdry = ZXPrism.patch_triangle_bdry(d, left_corner_lst[0], triangle_type)
            star_op_init = ZXPrism.star_operator_patch(triangle_type, nodes_triangle_bdry)
            self.seed_star_op[d] = (star_op_init.copy(), init_prism)

        for path in all_paths:
            stabilizer_product = []

            for idx, prism_pos in enumerate(path[1:-1], start=1):
                # find boundary stabilizers between previous and current prism
                boundary_left = self.find_boundary_stabilizers(
                    dct_patch_stabilizers, prism_pos, path[idx-1], dct_single_type_stabs)
                boundary_right = self.find_boundary_stabilizers(
                    dct_patch_stabilizers, prism_pos, path[idx+1], dct_single_type_stabs)

                stabilizer_product += [el for el in boundary_left if len(el)!=2] #add the boundary operators too
                if idx == len(path)-2:
                    stabilizer_product += [el for el in boundary_right if len(el)!=2]

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

                if idx == 1:
                    color_stdw_start = color_left
                    print("color start", color_stdw_start)
                elif idx == len(path):
                    color_stdw_end = color_right
                    print("color end", color_stdw_end)

                # add all stabilizers of this prism with matching colors
                prism_stabilizers = next(
                    stabs for prism, stabs in dct_patch_stabilizers.items()
                    if prism.position == prism_pos
                )
                for stab in prism_stabilizers:
                    stab_set = set(stab)
                    for key, color in assignment.items():
                        if set(key) == stab_set:
                            if color in (color_left, color_right):
                                stabilizer_product.append(stab)
                            break

                #determine which weight-2 operators are needed
                if idx != len(path)-1 and idx!=1: #not at the bdry of the path
                    weight_2_stabs = [el for el in boundary_left if len(el)==2]
                    #go through the weight 2 stabilizers, at each data qubit,
                    #at each data qubit, check how many data qubits are touched by a stabilizer in the product
                    #add the weight 2 stabilizer to the product if odd number of stabilizer touch the qubit.
                    for stab in weight_2_stabs:
                        bool_lst = [] #bool for each data qubit, True if odd, False if even
                        for qubit in stab:
                            touches = self.count_stabilizer_appearances(qubit, stabilizer_product)
                            if touches%2==0:
                                bool_lst.append(False)
                            else:
                                bool_lst.append(True)
                        if any(bool_lst) and not all(bool_lst):
                            raise TQECError("Expected all True or all False, got a mix.")
                        if all(bool_lst):
                            stabilizer_product.append(stab)

            #----------------star ops---------------------
            def _reflect_to(target_pos: Position3DHex) -> list[Position3DHex]:
                #! use this method maybe also in the star slice generation. currently duplicate code
                macro_path = init_prism.position.shortest_path_spatial(target_pos)
                path_prisms = [Prism(pos, ZXPrism(BasisPrism.N, BasisPrism.N), "") for pos in macro_path]
                left_corner_lst = self.corners_timeslice(z, d, path_prisms)
                star_op_tmp = star_op_init.copy()
                for idx, (pos, corner) in enumerate(zip(macro_path, left_corner_lst)):
                    triangle_type = "upwards" if pos.x % 2 == 0 else "downwards"
                    nodes_triangle_bdry = ZXPrism.patch_triangle_bdry(d, corner, triangle_type)
                    if idx + 1 < len(macro_path):
                        dummy_pipe = PrismPipe(
                            path_prisms[idx], path_prisms[idx + 1],
                            PrismPipeKind(hor=BasisPrism.N, ver=BasisPrism.N)
                        )
                        direction = dummy_pipe.direction_connecting_bdry()
                        star_op_tmp = ZXPrism.reflect_star_operator(star_op_tmp, direction, triangle_type, nodes_triangle_bdry)
                return star_op_tmp

            start_point = path[0]
            end_point = path[-1]
            start_star = _reflect_to(start_point)
            end_star   = _reflect_to(end_point)

            #---------------------------------------
            #find color-pair of the two thirds connected to the bulk patches.
            #for patch at start_point find the color of the weight-4 stabilizers

            # Get patch stabilizers and single-type stabs for start_point
            start_patch_stabs = next(
                stabs for prism, stabs in dct_patch_stabilizers.items()
                if prism.position == start_point
            )
            start_single_type_stabs = [
                stab
                for pipe, stabs in dct_single_type_stabs.items()
                if pipe.u.position == start_point or pipe.v.position == start_point
                for stab in stabs
            ]

            #end
            end_patch_stabs = next(
                stabs for prism, stabs in dct_patch_stabilizers.items()
                if prism.position == end_point
            )
            end_single_type_stabs = [
                stab
                for pipe, stabs in dct_single_type_stabs.items()
                if pipe.u.position == end_point or pipe.v.position == end_point
                for stab in stabs
            ]

            def _helper_bdry_start_end(single_type_stabs, star, patch_stabs):
                bdry_bdry = [stab for stab in single_type_stabs if len(stab) == 3 or len(stab) == 5]
                assert len(bdry_bdry) == 2, f"Internal error. {bdry_bdry}"

                bdry_1 = []
                temp_neigh = [bdry_bdry[0]]
                seen = [bdry_bdry[0]]
                while not set([v for stab in temp_neigh for v in stab]) & set(star):
                    temp_neigh = self.find_neighboring_bdry_stabilizer(temp_neigh[0], patch_stabs, single_type_stabs)
                    temp_neigh = [stab for stab in temp_neigh if not any(set(stab) == set(s) for s in seen)]
                    seen += temp_neigh
                    bdry_1 += temp_neigh

                bdry_2 = []
                temp_neigh = [bdry_bdry[1]]
                seen = [bdry_bdry[1]]
                while not set([v for stab in temp_neigh for v in stab]) & set(star):
                    temp_neigh = self.find_neighboring_bdry_stabilizer(temp_neigh[0], patch_stabs, single_type_stabs)
                    temp_neigh = [stab for stab in temp_neigh if not any(set(stab) == set(s) for s in seen)]
                    seen += temp_neigh
                    bdry_2 += temp_neigh

                return bdry_1, bdry_2

            bdry_1_from_start, bdry_2_from_start = _helper_bdry_start_end(start_single_type_stabs, start_star, start_patch_stabs)
            stabilizer_product += bdry_1_from_start
            stabilizer_product += bdry_2_from_start
            bdry_1_from_end, bdry_2_from_end = _helper_bdry_start_end(end_single_type_stabs, end_star, end_patch_stabs)
            stabilizer_product += bdry_1_from_end
            stabilizer_product += bdry_2_from_end

            def _helper_walk_bdry_to_middle_star(bdry, star, patch_stabs, single_type_stabs):
                stabilizer_product_temp = []
                seen = bdry
                colors = [_get_color(bdry[0]), _get_color(bdry[1])]
                temp_neigh = bdry
                star_set = set(star)
                seen_star_vertices = set(v for stab in bdry for v in stab if v in star_set)
                flag = True
                while flag:
                    temp_neigh_second = []
                    for el in temp_neigh:
                        temp_neigh_second += self.find_neighboring_bdry_stabilizer(el, patch_stabs, single_type_stabs, True)
                    temp_neigh = [stab for stab in temp_neigh_second if not any(set(stab) == set(s) for s in seen)]
                    seen += temp_neigh
                    temp_neigh = [stab for stab in temp_neigh if assignment[tuple(stab)] in colors]  # filter color

                    # filter out stabilizers that touch already-seen star vertices
                    temp_neigh = [
                        stab for stab in temp_neigh
                        if not (set(stab) & star_set & seen_star_vertices)
                    ]

                    if not temp_neigh:
                        flag = False
                    else:
                        # update seen star vertices before adding
                        seen_star_vertices |= set(v for stab in temp_neigh for v in stab if v in star_set)
                        stabilizer_product_temp += temp_neigh
                return stabilizer_product_temp

            #from bdry_1_from_start to middle
            stabilizer_product += _helper_walk_bdry_to_middle_star(bdry_1_from_start, start_star, start_patch_stabs, start_single_type_stabs)
            #from bdry_2_from_start to middle
            stabilizer_product += _helper_walk_bdry_to_middle_star(bdry_2_from_start, start_star, start_patch_stabs, start_single_type_stabs)
            #from bdry_1_from_end to middle
            stabilizer_product += _helper_walk_bdry_to_middle_star(bdry_1_from_end, end_star, end_patch_stabs, end_single_type_stabs)
            #from bdry_2_from_end to middle
            stabilizer_product += _helper_walk_bdry_to_middle_star(bdry_2_from_end, end_star, end_patch_stabs, end_single_type_stabs)

            #fill missing
            #most likely this is not yet correct, thus one has to add further stabilizers along the star arm pointing into the stdw
            def _fill_missing_star(star, patch_stabs):
                product_temp = []
                while True:
                    evenly_touched = [
                        v for v in star
                        if self.count_stabilizer_appearances(v, stabilizer_product + product_temp) % 2 == 0
                    ]
                    if not evenly_touched:
                        break
                    evenly_touched_set = set(evenly_touched)
                    newly_added = []
                    covered_this_round = set()
                    for stab in patch_stabs:
                        touching = set(stab) & evenly_touched_set
                        # only add if it touches evenly-touched vertices not yet covered this round
                        if len(touching - covered_this_round) >= 2:
                            newly_added.append(stab)
                            covered_this_round |= touching
                    if not newly_added:
                        break
                    product_temp += newly_added
                return product_temp

            #!TODO debug this!!!!!!!!!!!!

            stabilizer_product += _fill_missing_star(start_star, start_patch_stabs)
            stabilizer_product += _fill_missing_star(end_star, end_patch_stabs)

            all_paths_stars.append([start_star, end_star])

            #!TODO add weight-2 stabilizer at start/end stdw based on the 

            #------------------remaining weight-2 stabilizers---------------------
            #check those weight-6 stabilizers that have qubits that are not touched
            #these are placed along STDWs not part of the current path
            #to make them trivial, add the respective weight-2 stabilizer
            weight6_in_product = [stab for stab in stabilizer_product if len(stab) == 6]
            for stab in weight6_in_product:
                once_touched = [
                    qubit for qubit in stab
                    if self.count_stabilizer_appearances(qubit, stabilizer_product) == 1
                    and qubit not in (start_star, end_star)
                ]
                if len(once_touched) == 2 and once_touched[0].is_neighbour(once_touched[1]):
                    stabilizer_product.append(once_touched)
            all_paths_stabilizer_product.append(stabilizer_product.copy())

        #!TODO
        #!add test that every node in the stars has to be touched odd times + all others even times

        return assignment, all_paths, all_paths_stabilizer_product, all_paths_stars #assignment is order dependent, so please return the specific assignment which is in general not unique
