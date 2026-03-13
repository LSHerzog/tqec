"""Defines the :py:class:`~tqec.computation.prism.Prism` class."""

from __future__ import annotations

from dataclasses import astuple, dataclass
from enum import Enum
from typing import Any
import numpy as np
from collections import Counter
from itertools import combinations

from tqec.utils.exceptions import TQECError

@dataclass(frozen=True, order=True)
class Vec3DHex:
    x: int
    y: int
    z: int

class Position3DHex(Vec3DHex):
    """A 3D position for 3d space based on hex lattice."""

    x: int
    y: int
    z: int

    def distance_spatial(self, other: Position3DHex) -> int:
        """Determine distance between two positions."""
        if self.y % 2 == 0:
            xy_self = -self.x-self.y
        else:
            xy_self = -self.x-self.y +1

        if other.y % 2 == 0:
            xy_other = -other.x - other.y
        else:
            xy_other = -other.x - other.y +1
        return max([abs(self.x-other.x), abs(self.y-other.y), abs(xy_self -xy_other)])

    def neighbors_spatial(self) -> list[Position3DHex]:
        """Find all spatial neighbors of the point."""
        if self.y % 2 == 0:
            return [Position3DHex(self.x-1, self.y+1, self.z),
                    Position3DHex(self.x+1, self.y+1, self.z),
                    Position3DHex(self.x+1, self.y-1, self.z)]
        else:
            return [Position3DHex(self.x-1, self.y-1, self.z),
                    Position3DHex(self.x-1, self.y+1, self.z),
                    Position3DHex(self.x+1, self.y-1, self.z)]

    def is_next_nearest_neighbour(self, other: Position3DHex) -> bool:
        """Check whether two points are next-nearest neighbours in the triangular lattice."""
        return bool(set(self.neighbors_spatial()) & set(other.neighbors_spatial()))

    def shortest_path_spatial(self, other: Position3DHex) -> list[Position3DHex]:
        """Find shortest Path between two points."""
        path = [self]
        current = self
        while (current.x, current.y) != (other.x, other.y):
            neighbors = current.neighbors_spatial()
            best = min(neighbors, key=lambda n: n.distance_spatial(other))
            path.append(best)
            current = best
        return path

    def macro_diff_to_micro_diff(self, d:int, centroid_current, macro_next) -> Position3DHex:
        """Translate Macroscopic positions to microscopic poistitions which are d dependent.

        The self is the position of the current patch,
        centroid_current is the microscopic position of this patch's centroid.
        This function takes another macroscopic neighbor macro_next and finds its centroid

        If your input does not match up, this method yields nonsense.
        """
        if not self.is_neighbour(macro_next):
            raise TQECError("Microscopic cnetroids can only be found between neighboring patches.")

        if self.x % 2 == 0:
            if self.x - macro_next.x == -1 and self.y - macro_next.y == -1:
                for _ in range(d+1):
                    centroid_current = centroid_current.shift_standard_direction_plus2()
            elif self.x - macro_next.x == -1 and self.y - macro_next.y == +1:
                for _ in range(d+1):
                    centroid_current = centroid_current.shift_standard_direction_minus1()
            elif self.x - macro_next.x == +1 and self.y - macro_next.y == -1:
                for _ in range(d+1):
                    centroid_current = centroid_current.shift_standard_direction_minus3()
            else:
                raise TQECError("something off")
        else:  # noqa: PLR5501
            if self.x - macro_next.x == -1 and self.y - macro_next.y == 1:
                for _ in range(d+1):
                    centroid_current = centroid_current.shift_standard_direction_plus3()
            elif self.x - macro_next.x == 1 and self.y - macro_next.y == -1:
                for _ in range(d+1):
                    centroid_current = centroid_current.shift_standard_direction_plus1()
            elif self.x - macro_next.x == 1 and self.y - macro_next.y == 1:
                for _ in range(d+1):
                    centroid_current = centroid_current.shift_standard_direction_minus2()
            else:
                raise TQECError("something off")

        return centroid_current

    def is_neighbour(self, other: Position3DHex) -> bool:
        """Check whether two Prisms are neighbors.

        https://github.com/mhwombat/grid/wiki/Implementation%3A-Triangular-tiles
        """
        #temporal neighbors
        _is_neighbor = False
        if self.x == other.x and self.y == other.y and abs(self.z - other.z) == 1:
            _is_neighbor = True
        elif self.y%2==0 and self.z == other.z:
            if other.x-self.x==-1 and other.y-self.y==1:
                _is_neighbor = True
            elif other.x-self.x==1 and other.y-self.y==1:
                _is_neighbor = True
            elif other.x-self.x==1 and other.y-self.y==-1:
                _is_neighbor = True
        elif self.y%2!=0 and self.z == other.z:
            if other.x-self.x==-1 and other.y-self.y==-1:
                _is_neighbor = True
            elif other.x-self.x==-1 and other.y-self.y==1:
                _is_neighbor = True
            elif other.x-self.x==1 and other.y-self.y==-1:
                _is_neighbor = True
        return _is_neighbor

    def to_euclidean(
        self,
        scale: float = 1.0,
        z_spacing: float = 1.0,
    ) -> tuple[float, float, float]:
        """Translate hex/triangular coord to Euclidean coords.

        xy scaling matches read_write_prism.py: the centroid is scaled
        uniformly, so centroid-to-centroid distances expand by `scale`
        in every direction.

        scale     = 1 + spacing        (xy, from write_prism_graph_to_dae_file)
        z_spacing = 1 + spacing/√3     (z,  from write_prism_graph_to_dae_file)
        """
        import math
        h = math.sqrt(3) / 2
        pointing_up = (self.x % 2 == 0)
        cy_offset = h / 3 if pointing_up else 2 * h / 3

        # cell origin (bottom-left corner of triangle template)
        ox = self.x / 2.0 + (self.y // 2) * 0.5
        oy = (self.y // 2) * h

        # centroid scaled uniformly (matches _scaled_cell_origin in read_write_prism.py)
        cx = (ox + 0.5) * scale
        cy = (oy + cy_offset) * scale

        return cx, cy, self.z * z_spacing

    def shift_standard_direction_plus1(self) -> Position3DHex:
        """Shift along direction +1.

        This is a standard direction in the triangular lattice.
        """
        if self.x%2==0:
            x = self.x+1
        else:
            x = self.x-1
        y = self.y+1
        tup = [x, y, self.z]
        return Position3DHex(*tup.copy())

    def shift_standard_direction_minus1(self) -> Position3DHex:
        """Shift along direction -1.

        This is a standard direction in the triangular lattice.
        """
        if self.x%2==0:
            x = self.x+1
        else:
            x = self.x-1
        y = self.y-1
        tup = [x, y, self.z]
        return Position3DHex(*tup.copy())

    def shift_standard_direction_plus2(self) -> Position3DHex:
        """Shift along direction +2.

        This is a standard direction in the triangular lattice.
        """
        if self.x%2==0:
            y = self.y+1
        else:
            y = self.y-1
        x = self.x+1
        tup = [x, y, self.z]
        return Position3DHex(*tup.copy())

    def shift_standard_direction_minus2(self) -> Position3DHex:
        """Shift along direction -2.

        This is a standard direction in the triangular lattice.
        """
        if self.x%2==0:
            y = self.y+1
        else:
            y = self.y-1
        x = self.x-1
        return Position3DHex(x,y,self.z)

    def shift_standard_direction_plus3(self) -> Position3DHex:
        """Shift along direction +3.

        This is a standard direction in the triangular lattice.
        """
        x = self.x+1
        y = self.y-1
        tup = [x, y, self.z]
        return Position3DHex(*tup.copy())

    def shift_standard_direction_minus3(self) -> Position3DHex:
        """Shift along direction -3.

        This is a standard direction in the triangular lattice.
        """
        x = self.x-1
        y = self.y+1
        tup = [x, y, self.z]
        return Position3DHex(*tup.copy())


    def shift_triangle_direction_a(self) -> Position3DHex:
        """Shift along direction a for generating triangles.

        Important: This method should only be used to create triangle boundaries
        for microscopic positions to find stabilizers.
        This is NOT for macroscopic prism positinos.
        """
        #assert self.z == 0, "for microscopic positions, z=0 is necessary."
        x = self.x+1
        y = self.y+1
        return Position3DHex(x,y,self.z)

    def shift_triangle_direction_b(self) -> Position3DHex:
        """Shift along direction b for generating triangles.

        Important: This method should only be used to create triangle boundaries
        for microscopic positions to find stabilizers.
        This is NOT for macroscopic prism positinos.
        """
        #assert self.z == 0, "for microscopic positions, z=0 is necessary."
        if self.x%2==0:
            x = self.x + 3
            y = self.y - 1
        else:
            x = self.x + 1
            y = self.y -1
        return Position3DHex(x,y,self.z)

    def shift_triangle_direction_c(self) -> Position3DHex:
        """Shift along direction c for generating triangles.

        Important: This method should only be used to create triangle boundaries
        for microscopic positions to find stabilizers.
        This is NOT for macroscopic prism positinos.
        """
        #assert self.z == 0, "for microscopic positions, z=0 is necessary."
        if self.x%2==0:
            x = self.x - 1
            y = self.y + 3
        else:
            x = self.x - 1
            y = self.y + 1
        return Position3DHex(x,y,self.z)

class BasisPrism(Enum):
    X = "X"
    Z = "Z"
    N = "N" #if no basis fixed

    def __str__(self) -> str:
        return self.value

@dataclass(frozen=True)
class ZXPrism:
    """The ZX triangular prism.

    Attributes:
        prep: Basis of the prep (bottom) face
        meas: Basis of the meas (top) face

    """

    prep: BasisPrism
    meas: BasisPrism

    def as_tuple(self) -> tuple[BasisPrism, BasisPrism]:
        """Return a tuple of ``(self.prep, self.meas)``.

        Returns:
            A tuple of ``(self.prep, self.meas)``.

        """
        return (self.prep, self.meas)

    def __str__(self) -> str:
        return f"{self.prep}{self.meas}"

    @staticmethod
    def all_kinds() -> list[ZXPrism]:
        """Return all the allowed ``ZXPrism`` instances.

        Returns:
            The list of all the allowed ``ZXPrism`` instances.

        """
        return [ZXPrism.from_str(s) for s in ["XX", "ZZ", "XN", "ZN", "NX", "NZ"]]

    @staticmethod
    def from_str(string: str) -> ZXPrism:
        """Create a cube kind from the string representation.

        Args:
            string: A 2-character string consisting of ``'X'`` or ``'Z'``, representing
                the prep and meas colors

        Returns:
            The :py:class:`~tqec.computation.cube.ZXCube` instance constructed from
            the string representation.

        """
        return ZXPrism(*map(BasisPrism, string.upper()))

    @staticmethod
    def patch_triangle_bdry(d: int, left_corner: Position3DHex, triangle_type: str) -> dict[str, list[Position3DHex]]:
        r"""Find the microscopic positions of boundary vertices of a triangle.

        Assumes that left_corner has x even and y even.
        We choose this convention throughout this code.

        left_corner differs for the two triangle_types. `x` denotes the position of `left_corner`
        if roughly mapped to euclidian coords:
        triangle_type "upwards":
            /|
          /  |
        x    |
          \  |
            \|
        triangle type "downwards:
        | \
        |   \
        |    /
        |   /
        | /
        x
        """
        if triangle_type not in {"upwards", "downwards"}:
            raise TQECError("Incorrect microscopic triangle alignment chosen.")
        if triangle_type == "upwards":
            if left_corner.x % 2 == 0:
                #direction a
                dira = [left_corner]
                for i in range(1,d):
                    dira.append(dira[i-1].shift_triangle_direction_a())
                #direction b
                dirb = [left_corner]
                for i in range(1,d):
                    dirb.append(dirb[i-1].shift_triangle_direction_b())
                #direction c
                dirc = [dirb[-1]]
                for i in range(1,d):
                    dirc.append(dirc[i-1].shift_triangle_direction_c())
            else:
                raise TQECError("`left_corner` requires x to be even. This is the convention here")
        elif triangle_type == "downwards":
            if left_corner.x % 2 == 0:
                dira = [left_corner]
                for i in range(1,d):
                    dira.append(dira[i-1].shift_triangle_direction_a())
                dirc = [left_corner]
                for i in range(1,d):
                    dirc.append(dirc[-1].shift_triangle_direction_c())
                dirb = [dirc[-1]]
                for i in range(1,d):
                    dirb.append(dirb[-1].shift_triangle_direction_b())
            else:
                raise TQECError("`left_corner` requires x to be even. This is the convention here")

        return {"a": dira[::-1], "b": dirb, "c": dirc} #dira turned around for consistency in reduce_weight_six_to_four

    @staticmethod
    def patch_adjacent_bulk_stabilizers(init_point: Position3DHex) -> list[list[Position3DHex]]:
        """Find the stabilizer 3 plaquettes corresponding to the init_point."""
        if init_point.x % 2 == 0:
            #note that this choice is not unique as some directions yield same results for this case
            plaq1 = [init_point]
            plaq1.append(plaq1[-1].shift_standard_direction_plus2())
            plaq1.append(plaq1[-1].shift_standard_direction_plus2())
            plaq1.append(plaq1[-1].shift_standard_direction_plus3())
            plaq1.append(plaq1[-1].shift_standard_direction_minus2())
            plaq1.append(plaq1[-1].shift_standard_direction_minus2())
            plaq2 = [init_point]
            plaq2.append(plaq2[-1].shift_standard_direction_plus1())
            plaq2.append(plaq2[-1].shift_standard_direction_plus1())
            plaq2.append(plaq2[-1].shift_standard_direction_minus2())
            plaq2.append(plaq2[-1].shift_standard_direction_minus2())
            plaq2.append(plaq2[-1].shift_standard_direction_plus3())
            plaq3 = [init_point]
            plaq3.append(plaq3[-1].shift_standard_direction_minus2())
            plaq3.append(plaq3[-1].shift_standard_direction_minus2())
            plaq3.append(plaq3[-1].shift_standard_direction_plus3())
            plaq3.append(plaq3[-1].shift_standard_direction_plus3())
            plaq3.append(plaq3[-1].shift_standard_direction_plus2())
        else:
            plaq1 = [init_point]
            plaq1.append(plaq1[-1].shift_standard_direction_plus3())
            plaq1.append(plaq1[-1].shift_standard_direction_plus3())
            plaq1.append(plaq1[-1].shift_standard_direction_minus2())
            plaq1.append(plaq1[-1].shift_standard_direction_minus2())
            plaq1.append(plaq1[-1].shift_standard_direction_plus1())
            plaq2 = [init_point]
            plaq2.append(plaq2[-1].shift_standard_direction_plus1())
            plaq2.append(plaq2[-1].shift_standard_direction_plus1())
            plaq2.append(plaq2[-1].shift_standard_direction_plus3())
            plaq2.append(plaq2[-1].shift_standard_direction_plus3())
            plaq2.append(plaq2[-1].shift_standard_direction_minus2())
            plaq3 = [init_point]
            plaq3.append(plaq3[-1].shift_standard_direction_minus3())
            plaq3.append(plaq3[-1].shift_standard_direction_minus3())
            plaq3.append(plaq3[-1].shift_standard_direction_minus1())
            plaq3.append(plaq3[-1].shift_standard_direction_minus1())
            plaq3.append(plaq3[-1].shift_standard_direction_plus3())

        return [plaq1, plaq2, plaq3]

    @staticmethod
    def check_within_bdrys(pos: Position3DHex, bdry_dct: dict[str, list[Position3DHex]], d: int) -> bool:
        """Check whether a given pos is contained in the bdry.

        This is greedily done by checking each direction d steps and
        whether 2 boundary elements are encountered.
        """
        within_bdries = False
        bdry = []
        for lst in bdry_dct.values():#flattened bdries
            bdry += lst

        #direction 1
        tmpp = [pos]
        for _ in range(d-1):
            tmpp.append(tmpp[-1].shift_standard_direction_plus1())
        tmpm = [pos]
        for _ in range(d-1):
            tmpm.append(tmpm[-1].shift_standard_direction_minus1())

        if any(p in bdry for p in tmpp) and any(p in bdry for p in tmpm):
            within_bdries = True
            return within_bdries

        #direction 2
        tmpp = [pos]
        for _ in range(d-1):
            tmpp.append(tmpp[-1].shift_standard_direction_plus2())
        tmpm = [pos]
        for _ in range(d-1):
            tmpm.append(tmpm[-1].shift_standard_direction_minus2())

        if any(p in bdry for p in tmpp) and any(p in bdry for p in tmpm):
            within_bdries = True
            return within_bdries

        #direction 3
        tmpp = [pos]
        for _ in range(d-1):
            tmpp.append(tmpp[-1].shift_standard_direction_plus3())
        tmpm = [pos]
        for _ in range(d-1):
            tmpm.append(tmpm[-1].shift_standard_direction_minus3())

        if any(p in bdry for p in tmpp) and any(p in bdry for p in tmpm):
            within_bdries = True
            return within_bdries

        return within_bdries

    @staticmethod
    def remove_duplicate_stabilizers(stabilizers: list[list[Position3DHex]]) -> list[list[Position3DHex]]:
        """Remove duplicate stabilizers, keeping one copy of each."""
        seen = set()
        result = []
        for stab in stabilizers:
            key = frozenset(stab)
            if key not in seen:
                seen.add(key)
                result.append(stab)
        return result

    @staticmethod
    def remove_low_overlap_stabilizers(stabilizers: list[list[Position3DHex]], bdry_dct: dict[str, list[Position3DHex]], d: int) -> list[list[Position3DHex]]:
        """Remove Stabilizers that have <=2 nodes within the patch."""
        stabilizers_reduced = []
        for stab in stabilizers:
            bool_lst = [ZXPrism.check_within_bdrys(node, bdry_dct, d) for node in stab]
            num_inside = np.sum([1 for el in bool_lst if el])
            if num_inside > 2:
                stabilizers_reduced.append(stab)
        return stabilizers_reduced

    @staticmethod
    def reduce_weight_six_to_four(stabilizers: list[list[Position3DHex]], reduce_bdry: list[str], nodes_triangle_bdry: dict[str, list[Position3DHex]], d: int) -> list[list[Position3DHex]]:
        """Reduce stabilizers along one or more given boundaries.

        weight-6 stabilizers are reduced to weight-4 at the boundary.
        """
        if any([el not in {"a", "b", "c"} for el in reduce_bdry]):
            raise TQECError("`reduce_bdry` should have elements `a`, `b` and/or `c`.")

        for el in reduce_bdry:
            for node in nodes_triangle_bdry[el][:-1]: #depends on the order of the boundary node lists
                for i, stab in enumerate(stabilizers):
                    bool_lst = [ZXPrism.check_within_bdrys(node, nodes_triangle_bdry, d) for node in stab]
                    num_inside = np.sum([1 for el in bool_lst if el])
                    if node.x %2 == 0 and node in stab and num_inside <= 4:
                        stab_reduced = [el for j, el in enumerate(stab) if bool_lst[j]]
                        stabilizers[i] = stab_reduced

        return stabilizers

    @staticmethod
    def patch_stabilizers(d: int, triangle_type: str, left_corner: Position3DHex, reduce_bdry: list[str]) -> tuple[list[list[Position3DHex]]]:
        """Create patch stabilizers for a single patch.

        Since the stabilizers are self-dual if no merge is performed, this returns a single list
        of stabilizers representing both X and Z stabilizers.

        Note that Position3DHex has z=0 all the time, because we are considering microscopic
        positions here in 2D.

        `reduce_bdry` decides along which boundary the weight-6 stabilizers `outside` the triangle
        are reduced to weight-4 stabilizers.
        """
        nodes_triangle_bdry = ZXPrism.patch_triangle_bdry(d, left_corner, triangle_type)

        stabilizers =  []
        #add plaquettes along boundary
        for bdry in nodes_triangle_bdry.values():
            for node in bdry:
                if node.x % 2 == 0:
                    stabilizers += ZXPrism.patch_adjacent_bulk_stabilizers(node)

        appear_once_within = [left_corner]#just initialize somehow to start while loop
        while len(appear_once_within) != 0:
            #flatten the stabilizers
            flattened_stabs = [p for sublist in stabilizers for p in sublist]

            # filter nodes that are within the triangle AND appear only once in the stabilizers
            appear_once = [p for p in flattened_stabs if Counter(flattened_stabs)[p] == 1]
            appear_once_within = [p for p in appear_once if ZXPrism.check_within_bdrys(p, nodes_triangle_bdry, d)]
            if len(appear_once_within) == 0:
                break

            #fill up sprialing to the center of the triangle
            for node in appear_once_within:
                stabilizers += ZXPrism.patch_adjacent_bulk_stabilizers(node)

            stabilizers = ZXPrism.remove_duplicate_stabilizers(stabilizers)

        stabilizers = ZXPrism.remove_low_overlap_stabilizers(stabilizers, nodes_triangle_bdry, d)
        stabilizers = ZXPrism.reduce_weight_six_to_four(stabilizers, reduce_bdry, nodes_triangle_bdry, d)

        if len(stabilizers) !=  (3*(d**2-1))/8:
            raise TQECError("Internal issue with patch stabilizer construction.")

        return stabilizers, nodes_triangle_bdry

    @staticmethod
    def order_stabilizer(stabilizer: list[Position3DHex]):
        """Order a stabilizer such that adjacent data qubit positions appear subsequently."""
        if len(stabilizer) <= 2:
            return stabilizer

        # Start from an endpoint (only one neighbour in the set), or fall back to first element
        start = next(
            (pos for pos in stabilizer if sum(1 for other in stabilizer if pos.is_neighbour(other)) == 1),
            stabilizer[0]
        )

        ordered = [start]
        while len(ordered) < len(stabilizer):
            next_pos = next(
                (pos for pos in stabilizer if pos not in ordered and ordered[-1].is_neighbour(pos)),
                None
            )
            if next_pos is None:
                break
            ordered.append(next_pos)

        return ordered


    @staticmethod
    def find_pairs_with_two_overlaps(stabilizers: list[list[Position3DHex]]) -> list[list[Position3DHex]]:
        """From a list of stabilizers, find the weight-2 overlaps of connected weight-6 operators.

        These form the weight-2 stabilizers between patches in the STDW.

        If not the perfectly right inputs are given here, the function yields nonsense.
        """
        six_element_lists = [sublist for sublist in stabilizers if len(sublist) == 6]

        #Find weight-2 overlaps
        result: list[list[Position3DHex]] = []
        for list_a, list_b in combinations(six_element_lists, 2):
            overlapping = [pos for pos in list_a if pos in list_b]
            if len(overlapping) == 2:
                counts = [sum(1 for sublist in stabilizers if el in sublist) for el in overlapping]
                if counts[0] == 2 and counts[1]==2:
                    result.append(overlapping)

        return result

    @staticmethod
    def star_operator_patch(triangle_type: str, nodes_triangle_bdry: dict):
        """Search for a star operator in a given patch.

        for triangle_type upwards:
        Start at bdry of type "a" -> go d steps in "+3" direction
        Start at bdry of type "b" -> go d steps in "+1" direction
        Start at bdry of type "c" -> go d steps in "-2" direction

        for triangle_type downwards:
        Start at bdry of type "a" -> go d steps in "-3" direction
        Start at bdry of type "b" -> go d steps in "-1" direction
        Start at bdry of type "c" -> go d steps in "+2" direction
        """
        if triangle_type not in {"upwards", "downwards"}:
            raise TQECError("Incorrect microscopic triangle alignment chosen.")

        d = len(nodes_triangle_bdry["a"])

        def walk(start: Position3DHex, bdry: str) -> list[Position3DHex]:
            if triangle_type == "upwards":
                shift_fn = {
                    "a": lambda p: p.shift_standard_direction_plus3(),
                    "b": lambda p: p.shift_standard_direction_plus1(),
                    "c": lambda p: p.shift_standard_direction_minus2(),
                }[bdry]
            elif triangle_type == "downwards":
                shift_fn = {
                    "a": lambda p: p.shift_standard_direction_minus3(),
                    "b": lambda p: p.shift_standard_direction_minus1(),
                    "c": lambda p: p.shift_standard_direction_plus2(),
                }[bdry]
            temp_nodes = [start]
            for i in range(d):
                if i % 2 == 0:
                    temp_nodes.append(shift_fn(temp_nodes[-1]))
                else:
                    temp_nodes.append(shift_fn(shift_fn(temp_nodes[-1])))
            return temp_nodes

        # try all combinations of starting indices across the three boundaries
        def middle_out(n: int) -> list[int]:
            """Yield indices starting from the middle, alternating left and right."""
            mid = n // 2
            indices = [mid]
            for offset in range(1, n):
                if mid - offset >= 1:
                    indices.append(mid - offset)
                if mid + offset <= n - 2:
                    indices.append(mid + offset)
            return indices

        for idx_a in middle_out(len(nodes_triangle_bdry["a"])):
            for idx_b in middle_out(len(nodes_triangle_bdry["b"])):
                for idx_c in middle_out(len(nodes_triangle_bdry["c"])):
                    walk_a = walk(nodes_triangle_bdry["a"][idx_a], "a")
                    walk_b = walk(nodes_triangle_bdry["b"][idx_b], "b")
                    walk_c = walk(nodes_triangle_bdry["c"][idx_c], "c")

                    intersection = set(walk_a) & set(walk_b) & set(walk_c)
                    if intersection:
                        intersection_point = next(iter(intersection))
                        star_op = [intersection_point]
                        arms = [[], [], []] #a,b,c
                        for k, walk_nodes in enumerate([walk_a, walk_b, walk_c]):
                            for el in walk_nodes:
                                if el == intersection_point:
                                    break
                                star_op.append(el)
                                arms[k].append(el)

                        check_ok_middle = len(arms[0]) % 2 == 0 and len(arms[1]) % 2 == 0 and len(arms[2]) % 2 == 0 #even if middle point not included
                        thres = 0
                        check_ok_len = len(arms[0]) > thres and len(arms[1]) > thres and len(arms[2]) > thres 
                        if check_ok_middle and check_ok_len:
                            return star_op

        raise TQECError("No intersection point found for star operator.")

    @staticmethod
    def reflect_node(node: Position3DHex, direction: str, triangle_type: str, nodes_triangle_bdry: dict):
        """Reflect a node along a given axis."""
        if direction not in ("a", "b", "c") or triangle_type not in ("upwards", "downwards"):
            raise TQECError("incorrect string input(s).")

        d = len(nodes_triangle_bdry["a"]) #all have the same length thsu just choose a randomly.

        reflect_shift = {
            ("upwards",   "a"): "shift_standard_direction_minus3",
            ("upwards",   "b"): "shift_standard_direction_minus1",
            ("upwards",   "c"): "shift_standard_direction_plus2",
            ("downwards", "a"): "shift_standard_direction_plus3",
            ("downwards", "b"): "shift_standard_direction_plus1",
            ("downwards", "c"): "shift_standard_direction_minus2",
        }

        def get_reflection_shift(el: Position3DHex, direction: str, triangle_type: str) -> Position3DHex:
            method_name = reflect_shift[(triangle_type, direction)]
            return getattr(el, method_name)()

        #find the nodes on the reflection axis
        nodes_reflection_axis = [
            get_reflection_shift(el, direction, triangle_type)
            for el in nodes_triangle_bdry[direction]
        ]

        #go from node to corresponding element in reflection axis and count steps
        #go the same number of steps in same direction -> result: reflected node
        current = node
        for steps in range(1, d + 1):
            current = get_reflection_shift(current, direction, triangle_type)
            if current in nodes_reflection_axis:
                break

        #go "steps" many steps further in that direction
        for _ in range(steps):
            current = get_reflection_shift(current, direction, triangle_type)

        return current

    @staticmethod
    def reflect_star_operator(star_op: list[Position3DHex], direction: str, triangle_type: str, nodes_triangle_bdry: dict):
        """Reflect the star operator along the given direction.

        Note that triangle_type and nodes_triangle_bdry must fit to the current star op, otherwise the result wirll be nonsense.
        """
        star_op_reflected = [
            ZXPrism.reflect_node(el, direction, triangle_type, nodes_triangle_bdry)
            for el in star_op
        ]
        return star_op_reflected

class Port:
    """Prism kind representing the open ports in the block graph.

    The open ports correspond to the input/output of the computation represented by the block graph.
    They will have no effect on the functionality of the logical computation itself and should be
    invisible when visualizing the computation model.

    """

    def __str__(self) -> str:
        return "PORT"

    def __hash__(self) -> int:
        return hash(Port)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Port)


PrismKind = ZXPrism | Port

def prism_kind_from_string(s: str) -> PrismKind:
    """Create a cube kind from the string representation."""
    match s.upper():
        case "PORT" | "P":
            return Port()
        case _:
            return ZXPrism.from_str(s)

@dataclass(frozen=True)
class Prism:
    """creates triangular prisms representing a patch + time dimension.

    More general than ZXPrism.
    """

    position: Position3DHex
    kind: PrismKind
    label: str = ""

    def __str__(self) -> str:
        return f"{self.kind}{self.position}"

    def to_dict(self) -> dict[str, Any]:
        """Return the dictionary representation of the prism."""
        return {
            "position": self.position.as_tuple(),
            "kind": str(self.kind),
            "label": self.label,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> Prism:
        """Create a prsim from the dictionary representation.

        Args:
            data: The dictionary representation of the cube.

        Returns:
            The :py:class:`~tqec.computation.cube.Cube` instance created from the
            dictionary representation.

        """
        return Prism(
            position=Position3DHex(*data["position"]),
            kind=prism_kind_from_string(data["kind"]),
            label=data["label"],
        )

    @property
    def is_zx_prism(self) -> bool:
        """Check whether prism is ZXPrism."""
        return isinstance(self.kind, ZXPrism)

    @property
    def is_port(self) -> bool:
        """Check whether the prism is a port."""
        return isinstance(self.kind, Port)
