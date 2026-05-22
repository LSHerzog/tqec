import re
from collections import defaultdict
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sinter
import stim
import tesseract_decoder

from tqec.computation.correlation import CorrelationSurface
from tqec.computation.pipe_prism import PrismPipe, PrismPipeKind
from tqec.computation.prism import BasisPrism, Port, Position3DHex, Prism, ZXPrism
from tqec.computation.prism_graph import PrismGraph, StabilizerProductResult
from tqec.utils.exceptions import TQECError


@dataclass
class PositionEntry:
    """Represent multiple possible coordinate representations."""

    hex: Position3DHex | None
    rect: tuple[int, int]
    label: int | None

@dataclass
class StabilizerInfo:
    data_qubits: list[PositionEntry]
    top: list[PositionEntry]
    bottom: list[PositionEntry]
    sides: list[PositionEntry]
    ancilla: list[PositionEntry]
    stab_type: str

@dataclass
class PrismData:
    positions: list[PositionEntry] = field(default_factory=list)
    stabilizers: list[StabilizerInfo] | None = field(default_factory=list)

@dataclass
class MeasRecInfo:
    meas_type: str #ancilla_mz, ancilla_mx, data_mz, data_mx
    pipe_prism: PrismPipe | Prism
    stabilizer: StabilizerInfo | None
    abs_rec: int
    z_value: int
    round: int
    label: int #qubit label on which the M is done
    tick : int
@dataclass(frozen=True)
class ZPM:
    """Represent a single ``ZPM`` string.

    ## Format specification
    The ``ZPM`` string is a standard format used to unambiguously describe
    the state of a single qubit within a layer of a prism graph. It is a
    3+-character string. See each attribute docstring for more details on the
    possible values for each character.

    ## Example
    The following ::
        12XZ
        zzpm

    represents a qubit at layer ``12``, prepared in the ``X`` basis and
    measured in the ``Z`` basis.

    Attributes:
        z: layer index of the prism graph. Should be a non-negative integer.
        p: prepare basis of the qubit at the beginning of the layer
            (``X``, ``Z``, or ``N`` if no basis is fixed).
        m: measurement basis of the qubit at the end of the layer
            (``X``, ``Z``, or ``N`` if no basis is fixed).
    """

    z: int
    p: BasisPrism
    m: BasisPrism

    @classmethod
    def from_string(cls, zpm_string: str) -> "ZPM":
        """Initialize the ZPM object from a string.

        The string must consist of one or more digits followed by exactly
        two uppercase letters representing the prepare and measure bases.

        Raises:
            ValueError: if an invalid ``zpm_string`` is provided.

        """
        match = re.fullmatch(r"(\d+)([A-Z])([A-Z])", zpm_string)
        if match is None:
            raise ValueError(
                f"Invalid zpm string '{zpm_string}': expected format '{{z}}{{p}}{{m}}' "
                "where z is one or more digits and p, m are uppercase letters."
            )
        z_str, p_str, m_str = match.groups()

        if p_str not in BasisPrism._value2member_map_:
            raise ValueError(
                f"Invalid character '{p_str}' for the P field: "
                f"must be one of {list(BasisPrism._value2member_map_)}."
            )
        if m_str not in BasisPrism._value2member_map_:
            raise ValueError(
                f"Invalid character '{m_str}' for the M field: "
                f"must be one of {list(BasisPrism._value2member_map_)}."
            )

        return cls(z=int(z_str), p=BasisPrism(p_str), m=BasisPrism(m_str))

    def __str__(self) -> str:
        return f"{self.z}{self.p}{self.m}"


class SyndromeExtractionStimCC:
    def __init__(self, prism_graph: PrismGraph, d: int):
        """Construct syndrome extraction circuits for a prism_graph."""
        self.prism_graph = prism_graph
        self.d = d
        self.z_values = sorted({pos.z for pos in self.prism_graph._graph.nodes})

    def retrieve_stabilizers_operators(self):
        """Retrieve all stabilizers and logical ooperators for all z in the prism_graph."""
        result_dct = {z : dict() for z in self.z_values} #each of those dicts will contain stabilizers, operator products and star operators
        for z in self.z_values:
            stabs_x, stabs_z, _, dct_single_type_stabs, dct_patch_stabilizers = self.prism_graph.stabilizers_timeslice(z, self.d)
            result_dct[z].update({"stabs": [stabs_x, stabs_z]})
            #if self.d > 5: #not possible for d=3, d=5
            star_ops_x, star_ops_z = self.prism_graph.star_operator_timeslice(z, self.d)
            try:
                result = self.prism_graph.stabilizer_product_timeslice(z, self.d, dct_single_type_stabs, dct_patch_stabilizers, testing = True)
            except TQECError as e:
                #for d=3 the 3 coloring cannot be created for a single patch as it does not contain weight 6 stabilizers
                #but we do not care and skip this because for a single patch there's no stabilizer product anyway
                allowed_messages = [
                    "No weight-6 stabilizer found to seed the 3-coloring.",
                    "Could not resolve coloring — stuck with ambiguous boundary stabilizers."
                ]
                if not any(msg in str(e) for msg in allowed_messages):
                    raise TQECError("There is some issue with the stabilizer product construction.")
                else:
                    result = StabilizerProductResult(
                        assignment={},
                        paths_x=[],
                        paths_z=[],
                        stars_x=[],
                        stars_z=[],
                        stabilizer_products_x=[],
                        stabilizer_products_z=[],
                    )

            result_dct[z].update({"star": [star_ops_x, star_ops_z]})
            result_dct[z].update({"product": result})
            #result_dct[z].update({"assignment": self.prism_graph.prism_pipes_to_data_qubits_full})
        self.result_dct = result_dct

    def create_mapping(self):
        """Create a mapping of each data_qubit position to an int."""
        #positions = set()
        #for z in self.z_values:
        #    [stabs_x, stabs_z] = self.result_dct[z]["stabs"]
        #    stabs = stabs_x + stabs_z #double stabs removed
        #    positions_temp = {item for sublist in stabs for item in sublist}
        #    #set all z values to 0, because we want the mapping agnostically of z
        #    positions_temp = {Position3DHex(x = pos.x, y = pos.y, z = 0) for pos in positions_temp}
        #    positions.update(positions_temp)
        #self.mapping = {value: key for key,value in enumerate(positions)}
        positions_ordered = []
        seen = set()

        # collect all prisms first, then all pipes — consistent with any z layer
        # use z=0 projected positions
        for z in self.z_values:
            prism_pipes_data = self.prism_graph.prism_pipes_to_data_qubits_full[z]

            # first pass: prisms only
            for pipe_prism, data_qubits in prism_pipes_data.items():
                if isinstance(pipe_prism, Prism):
                    for pos in data_qubits:
                        flat = Position3DHex(x=pos.x, y=pos.y, z=0)
                        if flat not in seen:
                            positions_ordered.append(flat)
                            seen.add(flat)

            # second pass: pipes only
            for pipe_prism, data_qubits in prism_pipes_data.items():
                if isinstance(pipe_prism, PrismPipe):
                    for pos in data_qubits:
                        flat = Position3DHex(x=pos.x, y=pos.y, z=0)
                        if flat not in seen:
                            positions_ordered.append(flat)
                            seen.add(flat)

        self.mapping = {pos: idx for idx, pos in enumerate(positions_ordered)}

    def check_matrix(self, z:int, stab_type: str):
        """Create check matrix given some stabilizers and a mapping."""
        mapping = self.mapping
        if stab_type == "Z":
            stabs = self.result_dct[z]["stabs"][1]
        elif stab_type == "X":
            stabs = self.result_dct[z]["stabs"][0]
        else:
            raise ValueError("Wrong stab_type input. must be X or Z string.")
        cols = len(mapping)
        rows = len(stabs)
        arr = np.zeros((rows, cols), dtype = int)
        for k,stab in enumerate(stabs):
            stab_ad = [Position3DHex(x=pos.x, y=pos.y, z=0) for pos in stab]#set z=0 for all positions in stab
            stab_int = [mapping[pos] for pos in stab_ad]
            for label in stab_int:
                arr[k, label] = 1
        return arr

    @staticmethod
    def tanner_graph(arr : np.array):
        """Create the tanner graph of given check matrix arr."""
        rows, cols = arr.shape
        tg = nx.Graph()
        # qubit nodes
        tg.add_nodes_from([f"q{i}" for i in range(cols)])
        # check nodes
        tg.add_nodes_from([f"c{i}" for i in range(rows)])
        for i in range(rows):
            for j in range(cols):
                if arr[i, j] == 1:
                    tg.add_edge(f"c{i}", f"q{j}")
        return tg

    @staticmethod
    def tanner_coloring(tg: nx.Graph):
        """Retrieve edge coloring of given Tanner graph."""
        line_graph = nx.line_graph(tg)
        edge_coloring = nx.coloring.greedy_color(line_graph, strategy="largest_first")
        return edge_coloring

    @staticmethod
    def regroup_by_color(edge_coloring: dict) -> dict:
        """Regroup edge coloring dict from {edge: color} to {color: list[edges]}."""
        grouped = defaultdict(list)
        for edge, color in edge_coloring.items():
            grouped[color].append(edge)
        return dict(grouped)

    @staticmethod
    def canonical(stab):
        """Sort positions to make order irrelevant."""
        return tuple(sorted(stab, key=lambda p: (p.x, p.y, p.z)))

    def reorder_stabilizers(self, z: int):
        """Reorder stabilizers and classify them as X, Z, or XZ."""
        stabs_x, stabs_z = self.result_dct[z]["stabs"]

        set_x = {self.canonical(stab) for stab in stabs_x}
        set_z = {self.canonical(stab) for stab in stabs_z}

        dct_stabilizers = {}
        all_stabs = set_x | set_z
        for stab in all_stabs:
            if stab in set_x and stab in set_z:
                dct_stabilizers[stab] = "XZ"
            elif stab in set_x:
                dct_stabilizers[stab] = "X"
            else:
                dct_stabilizers[stab] = "Z"
        return dct_stabilizers

    def reorder_all_stabilizers(self):
        """Apply reordering for all z."""
        dct_stabilizers_all = {}
        for z in self.z_values:
            dct_stabilizers = self.reorder_stabilizers(z)
            dct_stabilizers_all.update({z : dct_stabilizers})
        self.dct_stabilizers_all = dct_stabilizers_all

    def create_mapping_ancillas(self):
        #!TODO CORRECT THIS
        #ordered version of ancilla mapping, i.e. labels consecutive labels belong to a patch
        #single ancilla per plaquette, even for the weight-2 stabilizers
        positions_ordered = []
        seen = set()
        max_label = max(list(self.mapping.values()))
        print("max_label", max_label)

        weight_6_stabilizers = []
        weight_4_stabilizers = []

        def is_stab_new(stab, seen):
            """
            Returns True if stab is genuinely new, i.e. not already represented in seen.
            A stab is considered already seen if:
            - it is directly in seen, OR
            - it is a weight-4 stab and a weight-6 stab in seen contains it (overlap == 4), OR
            - it is a weight-6 stab and a weight-4 stab in seen is contained in it (overlap == 4)
            """
            if stab in seen:
                return False
            stab_set = set(stab)
            for seen_stab in seen:
                if len(stab_set & set(seen_stab)) == 4:
                    return False
            return True

        #collect prisms first, then pipes
        for z in self.z_values:
            prism_pipes_data = self.prism_graph.prism_pipes_to_data_qubits_full[z]
            stabilizers = self.dct_stabilizers_all[z]#this contains also info about type of stabilizer!
            stabilizers = {
                tuple([Position3DHex(pos.x, pos.y, 0) for pos in stab]) : val
                for stab, val in stabilizers.items()}
            print("stabilizers", stabilizers)
            #handle single type stabilizers ('X', 'Z') as pipe stabilizers and double type ('XZ') stabilizers as patch stabilizers
            #prisms only
            for pipe_prism, data_qubits in prism_pipes_data.items():
                data_qubits_temp = [Position3DHex(x=pos.x, y=pos.y, z=0) for pos in data_qubits]
                if isinstance(pipe_prism, Prism):
                    #filter only double type stabilizers
                    stabilizers_temp = [stab for stab, stab_type in stabilizers.items() if stab_type == "XZ"]
                if isinstance(pipe_prism, PrismPipe):
                    #filter only single type stabilizers
                    stabilizers_temp = [stab for stab, stab_type in stabilizers.items() if stab_type == "X" or stab_type == "Z"]
                print("stabilizers_temp")
                for stab in stabilizers_temp:
                    if set(stab) & set(data_qubits_temp) != 0:
                        #!TODO make stab not in seen more complex. not only set comparison, but actually a weight-4 and weight-6 stabilizer can be equivalent
                        if is_stab_new(stab, seen):
                            seen.add(stab)
                            positions_ordered.append(stab)
                            if len(stab) == 4:
                                weight_4_stabilizers.append(stab)
                            elif len(stab) == 6:
                                weight_6_stabilizers.append(stab)

        mapping_ancillas = {pos: idx + max_label for idx, pos in enumerate(positions_ordered)}

        # for each weight-6 stabilizer, check if a weight-4 stabilizer is fully contained in it
        # if so, assign the same label; otherwise assign a new label
        # vice versa: for each weight-4 stabilizer, check if it is fully contained in a weight-6 stabilizer
        mapping_subset = {}
        for w6_stab in weight_6_stabilizers:
            w6_set = set(w6_stab)
            parent_label = None
            for w4_stab in weight_4_stabilizers:
                if len(set(w4_stab) & w6_set) == 4:
                    parent_label = mapping_ancillas[w4_stab]
                    break
            if parent_label is not None:
                mapping_subset[w6_stab] = parent_label
            else:
                mapping_subset[w6_stab] = mapping_ancillas[w6_stab]

        for w4_stab in weight_4_stabilizers:
            w4_set = set(w4_stab)
            for w6_stab in weight_6_stabilizers:
                if len(w4_set & set(w6_stab)) == 4:
                    mapping_subset[w4_stab] = mapping_subset[w6_stab]
                    break

        mapping_ancillas.update(mapping_subset)
        self.mapping_ancillas = mapping_ancillas

    """
    def create_mapping_ancillas(self):
        #Construct a mapping of ancilla labels (> mapping labels) with their stabilizer affiliations.
        #!FIRST SIMPLE VERSION WHERE ALL STABS HAVE OWN ANCILLA
        #! adapt integer assignment if less ancillas will be assumed later
        max_label = max(list(self.mapping.values()))
        union = set()
        for dct in self.dct_stabilizers_all.values():
            for stab in dct.keys():
                # project to z = 0
                stab_flat = tuple(
                    Position3DHex(x=p.x, y=p.y, z=0)
                    for p in stab
                )
                union.add(stab_flat)
        #if weight-4 and weight-6 stabilizers have max overlap, assign same ancilla
        mapping_ancillas = {}
        stab_sets = {stab: set(stab) for stab in union}
        weight6 = [stab for stab in union if len(stab) == 6]
        weight4 = [stab for stab in union if len(stab) == 4]
        mapping_ancillas = {}
        current_label = max_label + 1
        for stab in weight6:
            mapping_ancillas[stab] = current_label
            current_label += 1
        for stab4 in weight4:
            set4 = stab_sets[stab4]
            assigned = False
            for stab6 in weight6:
                if set4.issubset(stab_sets[stab6]):
                    mapping_ancillas[stab4] = mapping_ancillas[stab6]
                    assigned = True
                    break
            if not assigned:
                mapping_ancillas[stab4] = current_label
                current_label += 1
        others = [stab for stab in union if len(stab) not in (4, 6)]
        for stab in others:
            mapping_ancillas[stab] = current_label
            current_label += 1
        self.mapping_ancillas = mapping_ancillas
        """



    #! per timeslice define the init/meas basis per qubit or if none of those actions is performed
    #! based on pipe color or prism color
    def meas_prep_data_qubits(self):
        """Find meas/prep instructions for each data qubit for the whole pipe diagram."""
        #retrieve assignment of data qubits per patch / prism
        prism_pipes_to_data_qubits_full = self.prism_graph.prism_pipes_to_data_qubits_full
        #collect ZPM per prism or pipe
        prism_pipes_to_ZPM = dict()
        for z, prism_pipe_dct in prism_pipes_to_data_qubits_full.items():
            temp_dct = {}
            for pipe_prism in prism_pipe_dct.keys():
                if isinstance(pipe_prism, PrismPipe):
                    if pipe_prism.kind.hor == BasisPrism.X:
                        zpm = ZPM(z=z, p=BasisPrism.X, m=BasisPrism.X)
                    elif pipe_prism.kind.hor == BasisPrism.Z:
                        zpm = ZPM(z=z, p=BasisPrism.Z, m=BasisPrism.Z)
                    elif pipe_prism.kind.hor == BasisPrism.N:
                        zpm = ZPM(z=z, p=BasisPrism.N, m=BasisPrism.N)
                elif isinstance(pipe_prism, Prism):
                    if isinstance(pipe_prism.kind, ZXPrism):
                        zpm = ZPM(z=z, p=pipe_prism.kind.prep, m=pipe_prism.kind.meas)
                    elif isinstance(pipe_prism.kind, Port):
                        raise ValueError("For a stim circuit, no open ports are allowed.")
                else:
                    raise ValueError("internal issue.")
                temp_dct.update({pipe_prism : zpm})
            prism_pipes_to_ZPM.update({z: temp_dct})
        #note that prism_pipes_to_data_qubits_full and prsim_pipes_to_ZPM should be used together in create_stim_circuit
        self.prism_pipes_to_ZPM = prism_pipes_to_ZPM

    def assign_qubit_coords(self, scale: int = 20):
        """Create crumble positions."""
        self.qubit_coords = {}

        # --- data qubits ---
        for pos, q in self.mapping.items():
            cx, cy, _ = pos.to_euclidean(scale=1.0)

            ix = int(round(cx * scale))
            iy = int(round(cy * scale))

            self.qubit_coords[q] = (ix, iy)

        # --- ancillas ---
        for stab, anc in self.mapping_ancillas.items():
            coords = [
                Position3DHex(p.x, p.y, 0).to_euclidean(scale=1.0)[:2]
                for p in stab
            ]

            cx = sum(c[0] for c in coords) / len(coords)
            cy = sum(c[1] for c in coords) / len(coords)

            ix = int(round(cx * scale))
            iy = int(round(cy * scale))

            self.qubit_coords[anc] = (ix, iy)

    def add_qubit_coords_to_circuit(self, circuit: stim.Circuit) -> stim.Circuit:
        """Prepend QUBIT_COORDS to the circuit."""
        coord_circuit = stim.Circuit()

        for q, (x, y) in sorted(self.qubit_coords.items()):
            coord_circuit.append("QUBIT_COORDS", [q], [x, y])

        coord_circuit += circuit
        return coord_circuit

    @staticmethod
    def get_zpm_for_stab(
        stab,
        prism_pipes_data_temp,
        prism_pipes_zpm_temp,
        stabs_x,
        stabs_z,
    ):
        """Assign ZPM value from given stab."""
        stab_set = set(stab)

        #normalize to z=0
        stabs_x = [
            [Position3DHex(el.x, el.y, 0) for el in stab]
            for stab in stabs_x
        ]
        stabs_z = [
            [Position3DHex(el.x, el.y, 0) for el in stab]
            for stab in stabs_z
        ]

        in_x = any(stab_set == set(s) for s in stabs_x)
        in_z = any(stab_set == set(s) for s in stabs_z)

        # normalize pipe vs prism search space
        if in_x and in_z:
            allowed_types = (Prism,)
        else:
            allowed_types = (PrismPipe,)

        print("in_z", in_z)
        print("in_x", in_x)
        print("allowed_types", allowed_types)

        for pipe_prism, data_qubits in prism_pipes_data_temp.items():

            if not isinstance(pipe_prism, allowed_types):
                continue

            dq_set = {
                Position3DHex(q.x, q.y, 0)
                for q in data_qubits
            }

            if stab_set & dq_set:
                return prism_pipes_zpm_temp[pipe_prism]

        return None

    def stabilizer_changed_shape(
        self,
        stab: tuple,
        ancilla_label: int,
        mapping_ancillas_filtered_previous: dict,
    ) -> bool:
        """Check if a stabilizer changed shape compared to the previous z-layer.

        A stabilizer 'changes shape' when its ancilla label is shared with
        another stabilizer (i.e. a weight-4 stab is a subset of a weight-6 stab
        and they share an ancilla), AND the stab that was active under that
        ancilla label in the previous layer has a different support than the
        current one.

        Args:
            stab: the canonical stabilizer tuple (z=0 projected) of the current layer.
            ancilla_label: the integer ancilla label for this stabilizer.
            mapping_ancillas_filtered_previous: the mapping_ancillas_filtered
                dict from the previous z-layer.

        Returns:
            True if the stabilizer support changed shape, False otherwise.
        """
        # Find which stabilizer(s) in the previous layer shared this ancilla label
        prev_stabs_with_same_ancilla = [
            prev_stab
            for prev_stab, prev_label in mapping_ancillas_filtered_previous.items()
            if prev_label == ancilla_label
        ]

        if not prev_stabs_with_same_ancilla:
            # Ancilla didn't exist in previous layer at all
            return True

        # There should be at most one (by construction), but be safe
        prev_stab = prev_stabs_with_same_ancilla[0]

        # Compare support: same set of qubits = same shape
        return set(stab) != set(prev_stab)

    def find_extra_qubits(self, stab: tuple) -> tuple[list, set] | tuple[None, None]:
        """Find the extra qubits in the weight-6 parent stabilizer.

        Given a weight-4 stabilizer, finds the weight-6 stabilizer that contains
        it as a subset, and returns the 2 extra qubit labels and positions.

        Returns:
            (extra_qubit_labels, extra_qubits) if a weight-6 parent is found,
            (None, None) otherwise.
        """
        stab_set = set(stab)
        weight6_parent = None
        for candidate in self.mapping_ancillas:
            if len(candidate) == 6 and stab_set.issubset(set(candidate)):
                weight6_parent = candidate
                break
        if weight6_parent is None:
            return None, None
        extra_qubits = set(weight6_parent) - stab_set
        extra_qubit_labels = [self.mapping[q] for q in extra_qubits]
        return extra_qubit_labels, extra_qubits

    def find_weight2_subset_same_layer(
        self,
        stab_w6: tuple,
        mapping_ancillas_filtered: dict,
    ):
        """
        Given a weight-6 stabilizer, find a weight-2 stabilizer that is a subset
        of it and present in the SAME layer mapping.

        Returns:
            (weight2_stab, ancilla_label) or (None, None)
        """
        if len(stab_w6) != 6:
            return None, None

        stab6_set = set(stab_w6)

        for stab, anc in mapping_ancillas_filtered.items():
            if len(stab) == 2 and set(stab).issubset(stab6_set):
                return stab, anc

        return None, None

    def create_stim_circuit_naive(
            self,
            rounds,
            p_init: float,
            p_meas: float,
            p_idle: float,
            p_gate2: float,
        ) -> stim.Circuit:
        """Create a stim circuit of the object."""
        #! currently no horizontal correlation surfaces possible yet!
        #! summarize input from correlationsurface directly!

        #! logical_operator must be an operator on a full patch! otherwise not all detectors created.

        circuit = stim.Circuit()

        horizontal_cs_x_list = []
        horizontal_cs_z_list = []
        vertical_cs_x_list = []
        vertical_cs_z_list = []

        for s, z in enumerate(self.z_values):
            print(f"building circuit for z={z}...")
            #use both prism_pipes_to_ZPM and prism_pipes_to_data_qubits_full
            prism_pipes_zpm_temp = self.prism_pipes_to_ZPM[z]
            prism_pipes_data_temp = self.prism_graph.prism_pipes_to_data_qubits_full[z]

            print("prism_pipes_zpm_temp", prism_pipes_zpm_temp)
            print("prism_pipes_data_temp", prism_pipes_data_temp)

            if s != 0:
                meas_per_round_previous = meas_per_round #store the number of meas per roudn from previous 
            else:
                meas_data_qubits = 0 #initialize to 0 #number of data qubits that were measured zpm.m

            #initialize
            for prism_pipe in prism_pipes_zpm_temp.keys():
                zpm = prism_pipes_zpm_temp[prism_pipe]
                data_qubits = prism_pipes_data_temp[prism_pipe]
                mapped_data_qubits = [self.mapping[Position3DHex(x=el.x, y=el.y, z=0)] for el in data_qubits]
                if zpm.p == BasisPrism.Z:
                    print(f"initialization Z, {prism_pipe}")
                    circuit.append("R", mapped_data_qubits)
                    circuit.append("X_ERROR", mapped_data_qubits, p_init)
                elif zpm.p == BasisPrism.X:
                    print(f"initialization X, {prism_pipe}")
                    circuit.append("RX", mapped_data_qubits)
                    circuit.append("Z_ERROR", mapped_data_qubits, p_init)
                else:
                    print(f"no initialization for {prism_pipe}")
            #circuit.append("TICK", [])

            #retrieve X and Z tanner graph + coloring for current time step
            arr = self.check_matrix(z, "X")
            tg = self.tanner_graph(arr)
            edge_coloring_x = self.tanner_coloring(tg)
            edge_coloring_x = self.regroup_by_color(edge_coloring_x)
            arr = self.check_matrix(z, "Z")
            tg = self.tanner_graph(arr)
            edge_coloring_z = self.tanner_coloring(tg)
            edge_coloring_z = self.regroup_by_color(edge_coloring_z)

            stabs_x, stabs_z = self.result_dct[z]["stabs"]
            print("len stabs_x", len(stabs_x))
            print("len stabs_z", len(stabs_z))

            #extract stabilizer product (=horizontal correlation surface) if there is one or more of them in current z
            #if self.d>5:
            result = self.result_dct[z]["product"]
            star_ops_x, star_ops_z = self.result_dct[z]["star"]
            #else:
            #    print("No horizontal CS available for d<=5. your circuit will be wrong.")
            #    star_ops_x, star_ops_z = None, None
            #    result = None


            #!This must be adapted for larger structures where the `active` ancillas may differ per z layer
            #filter currently relevant mapping_ancillas
            mapping_ancillas_filtered = dict()
            for stab in stabs_x + stabs_z:
                stab_tup = tuple(self.canonical(stab))
                stab_tup = [Position3DHex(x=el.x, y=el.y, z=0) for el in stab_tup]
                stab_tup = tuple(stab_tup)
                mapping_ancillas_filtered.update({stab_tup: self.mapping_ancillas[stab_tup]})

            n_ancillas = len(list(mapping_ancillas_filtered))
            meas_per_round = 2 * n_ancillas  # MX block + M block per round

            for r in range(rounds):
                #! make this a separate method such that you can interchange the syndrome extraction schemes
                mapped_ancilla_qubits = mapping_ancillas_filtered.values()
                #--------X syndrome extraction----------
                #initialize ancillas in X
                circuit.append("RX", mapped_ancilla_qubits)
                circuit.append("Z_ERROR", mapped_ancilla_qubits, p_init)
                circuit.append("TICK", [])

                #collect the measurement indices for a horizontal correlation surface
                horizontal_cs_x = []#measurement labels
                stabilizer_products_x = None
                stabilizer_product_x_ancillas = None
                #if self.d>5:
                stabilizer_products_x = result.stabilizer_products_x
                if len(stabilizer_products_x)!=0:
                    stabilizer_products_x = stabilizer_products_x[0]#!HARDCODED TO FIRST PATH
                    stabilizer_products_x = [[Position3DHex(el.x, el.y, 0) for el in stabilizer] for stabilizer in stabilizer_products_x]
                    stabilizer_products_x = [self.canonical(stab) for stab in stabilizer_products_x]

                    stabilizer_product_x_ancillas = [] #collect the ancilla labels for those stabilizers
                    for stab in stabilizer_products_x:
                        corresponding_stab = self.canonical(stab)
                        corresponding_stab = [Position3DHex(x = el.x, y = el.y, z=0) for el in corresponding_stab]
                        ancilla_int = mapping_ancillas_filtered[tuple(corresponding_stab)]
                        stabilizer_product_x_ancillas.append(ancilla_int)

                print("stabilizer_product_x init", stabilizer_products_x)
                print("stabilizer_product_x ancillas", stabilizer_product_x_ancillas)

                active_qubits = set() #for idling noise
                for edges in edge_coloring_x.values():
                    #from labels "c+int" find the corresponding stabilizer and the ancilla label
                    #"q+int" describes data qubit label
                    #data qubit label = target, ancilla qubit label = control
                    for edge in edges:
                        q_int = int(edge[0][1:])
                        c_int = int(edge[1][1:])
                        corresponding_stab = self.canonical(stabs_x[c_int])
                        corresponding_stab = [Position3DHex(x = el.x, y = el.y, z=0) for el in corresponding_stab]
                        ancilla_int = mapping_ancillas_filtered[tuple(corresponding_stab)]
                        circuit.append("CNOT", [ancilla_int, q_int])
                        circuit.append("DEPOLARIZE2", [ancilla_int, q_int], p_gate2)
                        active_qubits.add(q_int)
                        active_qubits.add(ancilla_int)
                    #!ADAPT IF NOT ALL QUBITS ARE USED IN A z LAYER
                    #idle noise
                    all_qubits = set(self.mapping.values()) | set(self.mapping_ancillas.values())
                    idle_qubits = [q for q in all_qubits if q not in active_qubits]
                    if idle_qubits:
                        circuit.append("DEPOLARIZE1", idle_qubits, p_idle)
                    circuit.append("TICK", [])
                #measure all ancillas in X
                circuit.append("Z_ERROR", mapped_ancilla_qubits, p_meas)
                m_x_start = circuit.num_measurements
                circuit.append("MX", mapped_ancilla_qubits)
                circuit.append("TICK", [])

                if stabilizer_product_x_ancillas is not None:
                    for ancilla_int in stabilizer_product_x_ancillas:
                        pos_in_list = list(mapped_ancilla_qubits).index(ancilla_int)
                        abs_idx = m_x_start + pos_in_list
                        horizontal_cs_x.append(abs_idx)
                print("horizontal_cs_x", horizontal_cs_x)
                if r==0:
                    horizontal_cs_x_list.append(horizontal_cs_x)

                any_z_prep = any(zpm.p == BasisPrism.Z for zpm in prism_pipes_zpm_temp.values())
                any_x_prep = any(zpm.p == BasisPrism.X for zpm in prism_pipes_zpm_temp.values())

                #-------x stabilizer detectors---------
                #for anc_idx in range(n_ancillas):
                print("x stabilizers bulk (meas_data_qubits)", meas_data_qubits)
                for anc_idx, (stab, ancilla_int) in enumerate(mapping_ancillas_filtered.items()):
                    zpm_local = self.get_zpm_for_stab(stab, prism_pipes_data_temp, prism_pipes_zpm_temp, stabs_x, stabs_z)
                    rec_current = -(n_ancillas - anc_idx)# - meas_data_qubits #if nonzeor data qubits measured, this must be included in offset
                    # was the stabilizer shape changed from previous z layer to the current z layer? e.g. if a weight-4 stab becomes weight-6 in a split or vice versa
                    if s != 0:
                        stabilizer_change = self.stabilizer_changed_shape(stab, ancilla_int, mapping_ancillas_filtered_previous)
                        prism_pipes_zpm_temp_previous = self.prism_pipes_to_ZPM[self.z_values[s - 1]]
                        any_meas_prior = any(zpm.m != BasisPrism.N for zpm in prism_pipes_zpm_temp_previous.values())
                    else:
                        stabilizer_change = True #the elif below should not be triggered at z=0
                        any_meas_prior = False
                    print("stab x", stab)
                    print("ancilla int x", ancilla_int)
                    print("zpm local x", zpm_local)
                    print("stabilizer_change", stabilizer_change)
                    if r == 0:
                        if s!=0:
                            prev_keys = list(mapping_ancillas_filtered_previous.keys())
                            prev_values = list(mapping_ancillas_filtered_previous.values())
                        # first round: only deterministic if initialized in X basis
                        # if it's not the globally first round, then pay attention that bulk stabilizers still properly done
                        # i.e. if the current r==0 layer is actually not a new initialization of qubits.
                        # filter the ancillas belonging to a new initialized data qubits 
                        if zpm_local.p == BasisPrism.X: # and not any_z_prep:
                            circuit.append("DETECTOR", [stim.target_rec(rec_current)])
                            print("X, r=0, single detector", rec_current, "->", circuit.num_measurements+rec_current)
                        elif (zpm_local.p == BasisPrism.N and not any_z_prep and not any_meas_prior) or (
                            any_z_prep and zpm_local.p == BasisPrism.N and not stabilizer_change and not any_meas_prior
                            ) or (
                            not any_z_prep and zpm_local.p == BasisPrism.N and not stabilizer_change and any_meas_prior
                            ):
                            #rec_prev = rec_current - meas_per_round_previous - meas_data_qubits
                            if ancilla_int in prev_values:
                                #prev_anc_idx = prev_keys.index(stab)
                                prev_anc_idx = prev_values.index(ancilla_int)
                                #rec_prev = rec_current - meas_per_round_previous - meas_data_qubits - anc_idx_correction
                                n_ancillas_prev = len(prev_keys)
                                offset_previous = - (n_ancillas_prev - prev_anc_idx)
                                #rec_prev = rec_current - n_ancillas + offset_previous
                                rec_prev = - n_ancillas - n_ancillas_prev - meas_data_qubits + offset_previous
                                circuit.append("DETECTOR", [
                                    stim.target_rec(rec_current),
                                    stim.target_rec(rec_prev)
                                ])
                                print("X, r=0, double detector", rec_current, rec_prev, "->", circuit.num_measurements+rec_current, circuit.num_measurements+rec_prev)
                        elif (stabilizer_change and any_meas_prior and not any_z_prep_previous and zpm_local.p == BasisPrism.N):
                            #if the stabilizer changed shape and there was a prior meas in X basis
                            #then you need a detector that compares the current weight-4 stabilizer
                            #with the measurement outcomes of the previous layer that complete it
                            #to a weight-6 stabilizer
                            prev_meas_x = total_mapped_data_qubits_x_previous  # noqa: F821
                            extra_qubit_labels, extra_qubits = self.find_extra_qubits(stab)
                            if extra_qubit_labels is None:
                                continue
                            missing_targets = []
                            for q_label in extra_qubit_labels:
                                if q_label in prev_meas_x:
                                    pos_in_meas = prev_meas_x.index(q_label)
                                    offset = -(len(prev_meas_x) - pos_in_meas + n_ancillas)
                                    missing_targets.append(stim.target_rec(offset))
                            if len(missing_targets) == 2:
                                    circuit.append("DETECTOR", [stim.target_rec(rec_current)] + missing_targets)
                        elif (stabilizer_change and any_z_prep):
                            #within a merge, build detectors for weight-2 and weight-6 stabilizers where the weight-2 is a subset of weight-6
                            #i.e. if stabilizer changed and we have a horizontal z cs (aka any_x_prep)
                            #i.e. we add detectors within the same layer of syndrome measurements
                            #and furthermore the previous weight-4 corresponding measurement
                            weight2_stab, weight2_anc = self.find_weight2_subset_same_layer(
                                stab,
                                mapping_ancillas_filtered
                            )
                            if weight2_stab is None:
                                continue
                            weight2_idx = list(mapping_ancillas_filtered.values()).index(weight2_anc)
                            rec_weight2 = -(n_ancillas - weight2_idx)
                            #also previous layer's weight-4 stabilizer, i.e. same anc_idx
                            if ancilla_int not in prev_values:
                                continue
                            prev_anc_idx = prev_values.index(ancilla_int)
                            n_ancillas_prev = len(prev_values)
                            rec_previous = - n_ancillas - n_ancillas_prev - meas_data_qubits - (n_ancillas_prev - prev_anc_idx)
                            circuit.append("DETECTOR", [
                                stim.target_rec(rec_current),
                                stim.target_rec(rec_weight2),
                                stim.target_rec(rec_previous)
                            ])
                        elif (stabilizer_change and any_z_prep_previous):
                            #after a split we do the same as above only shifted:
                            #also add detectors that compare the same weight-4 Z stabilizer with its previous weight-2 stabilizer
                            #and the previous weight-6 stabilizer
                            #this requires a horizontal x cs in the previous layer
                            extra_qubit_labels, extra_qubits = self.find_extra_qubits(stab)
                            if extra_qubit_labels is None:
                                continue
                            weight2_prev_stab = None
                            for prev_stab in prev_keys:
                                if set(prev_stab) == extra_qubits:
                                    weight2_prev_stab = prev_stab
                                    break
                            if ancilla_int not in prev_values:
                                continue
                            if weight2_prev_stab is not None:
                                prev_anc_idx = prev_keys.index(weight2_prev_stab)
                                prev_w6_idx = prev_values.index(ancilla_int)
                                n_ancillas_prev = len(prev_keys)
                                offset_previous = - (len(prev_keys) - prev_anc_idx)
                                rec_prev = - n_ancillas_prev - n_ancillas - meas_data_qubits + offset_previous
                                offset_prev_w6 = -(n_ancillas_prev - prev_w6_idx)
                                rec_prev_w6 = - n_ancillas_prev - n_ancillas - meas_data_qubits + offset_prev_w6
                                circuit.append("DETECTOR", [
                                    stim.target_rec(rec_current),
                                    stim.target_rec(rec_prev),
                                    stim.target_rec(rec_prev_w6)
                                ])
                    else:
                        # middle rounds: always valid regardless of zpm.p
                        rec_prev = rec_current - meas_per_round
                        circuit.append("DETECTOR", [
                            stim.target_rec(rec_current),
                            stim.target_rec(rec_prev)
                        ])
                        print("X detector bulk:", rec_current, rec_prev,"->", circuit.num_measurements + rec_current, circuit.num_measurements+rec_prev)

                #--------Z syndrome extraction----------
                #initialize ancillas in Z
                circuit.append("R", mapped_ancilla_qubits)
                circuit.append("X_ERROR", mapped_ancilla_qubits, p_init)
                circuit.append("TICK", [])

                #collect the measurement indices for a horizontal correlation surface
                horizontal_cs_z = []
                stabilizer_products_z = None
                stabilizer_product_z_ancillas = None
                #if self.d>5:
                stabilizer_products_z = result.stabilizer_products_z
                if len(stabilizer_products_z)!=0:
                    stabilizer_products_z = stabilizer_products_z[0]#!HARD CODED TO FIRST PATH
                    stabilizer_products_z = [[Position3DHex(el.x, el.y, 0) for el in stabilizer] for stabilizer in stabilizer_products_z]
                    stabilizer_products_z = [self.canonical(stab) for stab in stabilizer_products_z]

                    stabilizer_product_z_ancillas = [] #collect the ancilla labels for those stabilizers
                    for stab in stabilizer_products_z:
                        corresponding_stab = self.canonical(stab)
                        corresponding_stab = [Position3DHex(x = el.x, y = el.y, z=0) for el in corresponding_stab]
                        ancilla_int = mapping_ancillas_filtered[tuple(corresponding_stab)]
                        stabilizer_product_z_ancillas.append(ancilla_int)


                print("stabilizer_product_z init", stabilizer_products_z)
                print("stabilizer_product_z ancillas", stabilizer_product_z_ancillas)

                active_qubits = set() #for idling noise
                for edges in edge_coloring_z.values():
                    #from labels "c+int" find the corresponding stabilizer and the ancilla label
                    #"q+int" describes data qubit label
                    #data qubit label = control, ancilla qubit label = target
                    for edge in edges:
                        q_int = int(edge[0][1:])
                        c_int = int(edge[1][1:])
                        corresponding_stab = self.canonical(stabs_z[c_int])
                        corresponding_stab = [Position3DHex(x = el.x, y = el.y, z=0) for el in corresponding_stab]
                        ancilla_int = mapping_ancillas_filtered[tuple(corresponding_stab)]
                        circuit.append("CNOT", [q_int, ancilla_int])
                        circuit.append("DEPOLARIZE2", [q_int, ancilla_int], p_gate2)
                        active_qubits.add(q_int)
                        active_qubits.add(ancilla_int)
                    #!ADAPT IF NOT ALL QUBITS ARE USED IN A z LAYER
                    #idle noise
                    all_qubits = set(self.mapping.values()) | set(self.mapping_ancillas.values())
                    idle_qubits = [q for q in all_qubits if q not in active_qubits]
                    if idle_qubits:
                        circuit.append("DEPOLARIZE1", idle_qubits, p_idle)
                    circuit.append("TICK", [])
                #measure all ancillas in Z
                circuit.append("X_ERROR", mapped_ancilla_qubits, p_meas)
                m_z_start = circuit.num_measurements
                circuit.append("M", mapped_ancilla_qubits)
                circuit.append("TICK", [])

                if stabilizer_product_z_ancillas is not None:
                    for ancilla_int in stabilizer_product_z_ancillas:
                        pos_in_list = list(mapped_ancilla_qubits).index(ancilla_int)
                        print("pos_in_list", pos_in_list)
                        print("m_z_start", m_z_start)
                        abs_idx = m_z_start + pos_in_list
                        print("abs_idx", abs_idx)
                        horizontal_cs_z.append(abs_idx)
                print("horizontal_cs_z", horizontal_cs_z)
                if r == 0:
                    horizontal_cs_z_list.append(horizontal_cs_z)

                #-------z stabilizer detectors---------
                #for anc_idx in range(n_ancillas):
                print("z stabilizers bulk (meas_data_qubits)", meas_data_qubits)
                for anc_idx, (stab, ancilla_int) in enumerate(mapping_ancillas_filtered.items()):
                    rec_current = -(n_ancillas - anc_idx)# - meas_data_qubits #if nonzeor data qubits measured, this must be included in offset
                    zpm_local = self.get_zpm_for_stab(stab, prism_pipes_data_temp, prism_pipes_zpm_temp, stabs_x, stabs_z)
                    print("stab", stab)
                    print("ancilla int", ancilla_int)
                    # was the stabilizer shape changed from previous z layer to the current z layer? e.g. if a weight-4 stab becomes weight-6 in a split or vice versa
                    if s != 0:
                        stabilizer_change = self.stabilizer_changed_shape(stab, ancilla_int, mapping_ancillas_filtered_previous)
                        prism_pipes_zpm_temp_previous = self.prism_pipes_to_ZPM[self.z_values[s - 1]]
                        any_meas_prior = any(zpm.m != BasisPrism.N for zpm in prism_pipes_zpm_temp_previous.values())
                    else:
                        stabilizer_change = True #the elif below should not be triggered at z=0
                        any_meas_prior = False
                    if r == 0:
                        if s!=0:
                            prev_keys = list(mapping_ancillas_filtered_previous.keys())
                            prev_values = list(mapping_ancillas_filtered_previous.values())
                        # first round: only deterministic if initialized in Z basis
                        if zpm_local.p == BasisPrism.Z:
                            circuit.append("DETECTOR", [stim.target_rec(rec_current)])
                            print("Z, r=0, single detector", rec_current, "->", circuit.num_measurements + rec_current)
                        elif (zpm_local.p == BasisPrism.N and not any_x_prep and not any_meas_prior) or (
                            any_x_prep and zpm_local.p == BasisPrism.N and not stabilizer_change and not any_meas_prior
                            ) or (
                            not any_x_prep and zpm_local.p == BasisPrism.N and not stabilizer_change and any_meas_prior  
                            ):
                            if ancilla_int in prev_values:
                                prev_anc_idx = prev_values.index(ancilla_int)
                                n_ancillas_prev = len(prev_keys)
                                offset_previous = - (n_ancillas_prev - prev_anc_idx)
                                print("offset previous", offset_previous)
                                rec_prev = - 2*n_ancillas - meas_data_qubits + offset_previous
                                circuit.append("DETECTOR", [
                                    stim.target_rec(rec_current),
                                    stim.target_rec(rec_prev)
                                ])
                                print("Z, r=0, double detector", rec_current, rec_prev, "->", circuit.num_measurements+rec_current, circuit.num_measurements+rec_prev)
                        elif (stabilizer_change and any_meas_prior and not any_x_prep_previous and zpm_local.p == BasisPrism.N):
                            #if the stabilizer changed shape and there was a prior meas in Z basis
                            #then you need a detector that compares the current weight-4 stabilizer
                            #with the measurement outcomes of the previous layer that complete it
                            #to a weight-6 stabilizer
                            prev_meas_z = total_mapped_data_qubits_z_previous  # noqa: F821
                            extra_qubit_labels, extra_qubits = self.find_extra_qubits(stab)
                            if extra_qubit_labels is None:
                                continue
                            missing_targets = []
                            for q_label in extra_qubit_labels:
                                if q_label in prev_meas_z:
                                    pos_in_meas = prev_meas_z.index(q_label)
                                    offset = -(len(prev_meas_z) - pos_in_meas + 2*n_ancillas) #difference to previous loop for X
                                    missing_targets.append(stim.target_rec(offset))
                            if len(missing_targets) == 2:
                                    circuit.append("DETECTOR", [stim.target_rec(rec_current)] + missing_targets)
                        elif (stabilizer_change and any_x_prep):
                            #within a merge, build detectors for weight-2 and weight-6 stabilizers where the weight-2 is a subset of weight-6
                            #i.e. if stabilizer changed and we have a horizontal z cs (aka any_x_prep)
                            #i.e. we add detectors within the same layer of syndrome measurements
                            #and furthermore the previous weight-4 corresponding measurement
                            weight2_stab, weight2_anc = self.find_weight2_subset_same_layer(
                                stab,
                                mapping_ancillas_filtered
                            )
                            if weight2_stab is None:
                                continue
                            weight2_idx = list(mapping_ancillas_filtered.values()).index(weight2_anc)
                            rec_weight2 = -(n_ancillas - weight2_idx)
                            #also previous layer's weight-4 stabilizer, i.e. same anc_idx
                            if ancilla_int not in prev_values:
                                continue
                            prev_anc_idx = prev_values.index(ancilla_int)
                            n_ancillas_prev = len(prev_values)
                            rec_previous = - 2 * n_ancillas - meas_data_qubits - (n_ancillas_prev - prev_anc_idx)
                            circuit.append("DETECTOR", [
                                stim.target_rec(rec_current),
                                stim.target_rec(rec_weight2),
                                stim.target_rec(rec_previous)
                            ])
                        elif (stabilizer_change and any_x_prep_previous):
                            #after a split we do the same as above only shifted:
                            #also add detectors that compare the same weight-4 X stabilizer with its previous weight-2 stabilizer
                            #and the previous weight-6 stabilizer
                            #this requires a horizontal z cs in the previous layer
                            extra_qubit_labels, extra_qubits = self.find_extra_qubits(stab)
                            print("extra qubits weight-2 stabilizer and stuff for horizontal z cs", extra_qubit_labels)
                            if extra_qubit_labels is None:
                                continue
                            weight2_prev_stab = None
                            for prev_stab in prev_keys:
                                if set(prev_stab) == extra_qubits:
                                    weight2_prev_stab = prev_stab
                                    break
                            if ancilla_int not in prev_values:
                                continue
                            if weight2_prev_stab is not None:
                                prev_anc_idx = prev_keys.index(weight2_prev_stab)
                                prev_w6_idx = prev_values.index(ancilla_int)
                                n_ancillas_prev = len(prev_keys)
                                offset_previous = - (len(prev_keys) - prev_anc_idx)
                                rec_prev = - 2*n_ancillas - meas_data_qubits + offset_previous
                                offset_prev_w6 = -(n_ancillas_prev - prev_w6_idx)
                                rec_prev_w6 = -2 * n_ancillas - meas_data_qubits + offset_prev_w6
                                circuit.append("DETECTOR", [
                                    stim.target_rec(rec_current),
                                    stim.target_rec(rec_prev),
                                    stim.target_rec(rec_prev_w6)
                                ])
                    else:
                        # middle rounds: always valid regardless of zpm.p
                        rec_prev = rec_current - meas_per_round
                        circuit.append("DETECTOR", [
                            stim.target_rec(rec_current),
                            stim.target_rec(rec_prev)
                        ])
                        print("Z detector bulk:", rec_current, rec_prev ,"->", circuit.num_measurements + rec_current, circuit.num_measurements+rec_prev)

            offset_start = circuit.num_measurements

            #check whether some zpm.m is Z or X and then measure accordingly.
            total_mapped_data_qubits_z = []
            total_mapped_data_qubits_x = []
            for prism_pipe in prism_pipes_zpm_temp.keys():
                zpm = prism_pipes_zpm_temp[prism_pipe]
                data_qubits = prism_pipes_data_temp[prism_pipe]
                mapped_data_qubits = [self.mapping[Position3DHex(x=el.x, y=el.y, z=0)] for el in data_qubits]
                if zpm.m == BasisPrism.Z:
                    total_current = circuit.num_measurements
                    print(f"Measure Z, {prism_pipe}")
                    circuit.append("X_ERROR", mapped_data_qubits, p_meas)
                    circuit.append("M", mapped_data_qubits)
                    meas_data_qubits += len(mapped_data_qubits)
                    total_mapped_data_qubits_z += mapped_data_qubits
                    #vertical Z correlation surface meas labels if there are any
                    if star_ops_z is not None:
                        measured_set = set(mapped_data_qubits)
                        star_ops_z_labels = []
                        for star_op in star_ops_z:
                            star_op_tr = [Position3DHex(x = pos.x, y = pos.y, z = 0) for pos in star_op]
                            lst = [self.mapping[pos] for pos in star_op_tr]
                            star_ops_z_labels.append(lst)
                        vertical_cs_z = [
                            total_current + mapped_data_qubits.index(qubit)
                            for star_op in star_ops_z_labels
                            for qubit in star_op
                            if qubit in measured_set
                        ]
                        vertical_cs_z_list.append(vertical_cs_z)

                elif zpm.m == BasisPrism.X:
                    print(f"Measure X, {prism_pipe}")
                    total_current = circuit.num_measurements
                    circuit.append("Z_ERROR", mapped_data_qubits, p_meas)
                    circuit.append("MX", mapped_data_qubits)
                    meas_data_qubits += len(mapped_data_qubits)
                    total_mapped_data_qubits_x += mapped_data_qubits
                    #vertical X correlation surface meas labels if there are any
                    if star_ops_x is not None:
                        measured_set = set(mapped_data_qubits)
                        star_ops_x_labels = []
                        for star_op in star_ops_x:
                            star_op_tr = [Position3DHex(x = pos.x, y = pos.y, z = 0) for pos in star_op]
                            lst = [self.mapping[pos] for pos in star_op_tr]
                            star_ops_x_labels.append(lst)
                        vertical_cs_x = [
                            total_current + mapped_data_qubits.index(qubit)
                            for star_op in star_ops_x_labels
                            for qubit in star_op
                            if qubit in measured_set
                        ]
                        vertical_cs_x_list.append(vertical_cs_x)
                        print("mapped data meas-------",mapped_data_qubits)
                        print("star ops x ----------",star_ops_x)
                        print("vertical_cs_x", vertical_cs_x)

                total = circuit.num_measurements
                #add OBS_INCLUDE only at the end of the diagram, i.e. at the largest available z
                if z == max(self.z_values):
                    print("ADD OBS_INCLUDE")
                    # build OBS_INCLUDE targets using measurement record offsets

                    #retrieve star operator from last zpm called in previous loop
                    current_hor_lst = None
                    current_star = None
                    if zpm.m == BasisPrism.Z:
                        if star_ops_z is not None:
                            current_star = star_ops_z[0] #!HARD CODED FOR NOW, IF MORE OPEN PIPES ADAPT THIS
                            current_hor_lst = horizontal_cs_z_list
                            current_ver_lst = vertical_cs_z_list
                    elif zpm.m == BasisPrism.X:
                        if star_ops_x is not None:
                            current_star = star_ops_x[0]
                            current_hor_lst = horizontal_cs_x_list
                            current_ver_lst = vertical_cs_x_list
                    if current_star is not None:
                        current_star = [Position3DHex(el.x,el.y,0) for el in current_star]

                        print("current star", current_star)
                        print("current_hor_lst", current_hor_lst)

                    print("mapped data qubits", mapped_data_qubits)
                    #translate current star into labels
                    current_star_labels = [self.mapping[pos] for pos in current_star]
                    obs_targets = []
                    for i, qubit in enumerate(mapped_data_qubits):
                        if current_star_labels is not None:
                            if qubit in current_star_labels:
                                offset = -(total - offset_start - i)
                                obs_targets.append(stim.target_rec(offset))
                                print("added star qubit to OBS", offset)
                        else: #use the whole patch as obs if no star op
                            offset = -(total - offset_start - i)
                            obs_targets.append(stim.target_rec(offset))

                    #add parity of corresponding horizontal cs:
                    if len(current_hor_lst) != 0 and not all(len(lst) == 0 for lst in current_hor_lst):#!hard coded to allow mem
                        #for horizontal_cs in current_hor_lst:
                        horizontal_cs = current_hor_lst[1]#!temp hard coded!!!!!
                        for el in horizontal_cs:
                            offset = -(total - el)
                            obs_targets.append(stim.target_rec(offset))
                            print("from horizontal CS added", offset)

                    #add parity from corresponding vertical cs:
                    if len(current_ver_lst)!=0:
                        print("current_ver_lst", current_ver_lst)
                        for lst in current_ver_lst[:-1]: #the last vertical star should not be included
                            for el in lst:
                                offset = -(total - el)
                                obs_targets.append(stim.target_rec(offset))
                                print("from vertical CS added", offset)

                    print("observable final", obs_targets)
                    circuit.append("OBSERVABLE_INCLUDE", obs_targets, 0)

                    #-------final round of detectors------
                    if zpm.m == BasisPrism.Z:
                        for anc_idx, (stab, ancilla_int) in enumerate(mapping_ancillas_filtered.items()):
                            # last M block ended at offset_start, data meas came after
                            # last Z ancilla meas for anc_idx:
                            # it is anc_idx-th measurement of the last M block
                            # M block ended at offset_start, so offset from total:
                            rec_last_anc = -(total - offset_start + n_ancillas - anc_idx)
                            # data qubit meas for this stabilizer's support
                            stab_data_qubits = [self.mapping[q] for q in stab]
                            data_targets = []
                            for q in stab_data_qubits:
                                if q in mapped_data_qubits:
                                    pos_in_obs = mapped_data_qubits.index(q)
                                    data_targets.append(stim.target_rec(-(total - offset_start - pos_in_obs)))
                            circuit.append("DETECTOR", [stim.target_rec(rec_last_anc)] + data_targets)
                            print("final Z detector:", rec_last_anc, data_targets)

                    elif zpm.m == BasisPrism.X:
                        for anc_idx, (stab, ancilla_int) in enumerate(mapping_ancillas_filtered.items()):
                            rec_last_anc = -(total - offset_start + 2*n_ancillas - anc_idx)
                            stab_data_qubits = [self.mapping[q] for q in stab]
                            data_targets = []
                            for q in stab_data_qubits:
                                if q in mapped_data_qubits:
                                    pos_in_obs = mapped_data_qubits.index(q)
                                    data_targets.append(stim.target_rec(-(total - offset_start - pos_in_obs)))
                            circuit.append("DETECTOR", [stim.target_rec(rec_last_anc)] + data_targets)
                            print("final X detector:", rec_last_anc, data_targets)

            #--------add data qubit measurement detectors not final measurement--------
            #add detectors for stabilizers fully included in the measurement
            if z != max(self.z_values):
                for stab in stabs_x:
                    stab_normalized = [Position3DHex(el.x, el.y, 0) for el in stab]
                    stab_int = [self.mapping[pos] for pos in stab_normalized]
                    if set(stab_int).issubset(set(total_mapped_data_qubits_x)):
                        stab_tup = tuple(self.canonical(stab_normalized))
                        ancilla_int = mapping_ancillas_filtered[stab_tup]
                        pos_ancilla = list(mapped_ancilla_qubits).index(ancilla_int)
                        lst_det = []
                        for q in stab_int:
                            pos_in_meas = total_mapped_data_qubits_x.index(q)
                            lst_det.append(stim.target_rec(-(len(total_mapped_data_qubits_x) - pos_in_meas)))
                        #add x ancilla previous
                        rec = -(len(total_mapped_data_qubits_x) + 2*n_ancillas - pos_ancilla) #2 ancillas bc x ancilla meas earlier
                        lst_det.append(stim.target_rec(rec))
                        circuit.append("DETECTOR", lst_det)
                for stab in stabs_z:
                    stab_normalized = [Position3DHex(el.x, el.y, 0) for el in stab]
                    stab_int = [self.mapping[pos] for pos in stab_normalized]
                    if set(stab_int).issubset(set(total_mapped_data_qubits_z)):
                        stab_tup = tuple(self.canonical(stab_normalized))
                        ancilla_int = mapping_ancillas_filtered[stab_tup]
                        pos_ancilla = list(mapped_ancilla_qubits).index(ancilla_int)
                        lst_det = []
                        for q in stab_int:
                            pos_in_meas = total_mapped_data_qubits_z.index(q)
                            lst_det.append(stim.target_rec(-(len(total_mapped_data_qubits_z) - pos_in_meas)))
                        #add z ancilla previous
                        #!why same rec as for stabs_x? n_ancillas = len(total_mapped_data_qubits) in current example. maybe exchange variables for more general versions?
                        rec = -(len(total_mapped_data_qubits_x) + 2*n_ancillas - pos_ancilla) #1 ancillas bc z ancilla meas later
                        lst_det.append(stim.target_rec(rec))
                        circuit.append("DETECTOR", lst_det)

            mapping_ancillas_filtered_previous = mapping_ancillas_filtered.copy()  # noqa: F841
            total_mapped_data_qubits_x_previous = total_mapped_data_qubits_x.copy()
            total_mapped_data_qubits_z_previous = total_mapped_data_qubits_z.copy()
            any_x_prep_previous, any_z_prep_previous = any_x_prep, any_z_prep

        self.assign_qubit_coords()
        circuit = self.add_qubit_coords_to_circuit(circuit)

        return circuit


    def run_all(self,
            rounds,
            p_init: float,
            p_meas: float,
            p_idle: float,
            p_gate2: float):
        """Run all methods."""
        self.retrieve_stabilizers_operators()
        self.create_mapping()
        print("self.mapping:")
        print(self.mapping)
        self.reorder_all_stabilizers()
        self.create_mapping_ancillas()
        print("self.mapping_ancillas")
        print(self.mapping_ancillas)
        self.meas_prep_data_qubits()

        circuit = self.create_stim_circuit_naive(
            rounds,
            p_init,
            p_meas,
            p_idle,
            p_gate2)
        return circuit

    #---------------new stuff for the quadratic mapping etc + superdense----------------------
    @staticmethod
    def is_stab_new(stab, seen):
        """Return True if stab is genuinely new, i.e. not already represented in seen.

        A stab is considered already seen if:
        - it is directly in seen, OR
        - it is a weight-4 stab and a weight-6 stab in seen contains it (overlap == 4), OR
        - it is a weight-6 stab and a weight-4 stab in seen is contained in it (overlap == 4)
        """
        if stab in seen:
            return False
        stab_set = set(stab)
        for seen_stab in seen:
            if len(stab_set & set(seen_stab)) == 4:
                return False
        return True

    @staticmethod
    def assign_stabilizer_info(stab) -> StabilizerInfo:
        """Assign the stabilizer info for a stab given in hex coordinates.

        This covers info about different weight-4 stabilizers and weight-6 stabilizers
        weight-2 stabilizers should not be covered by this.
        """
        #make sure that z=0
        stab = [Position3DHex(pos.x, pos.y, 0) for pos in stab]
        stab_rec = [pos.rectangular_map() for pos in stab]
        y_values = sorted({pos[1] for pos in stab_rec}) #available rec y values
        position_entries = [PositionEntry(hex = pos_hex, rect = pos_rec, label = None) for pos_hex, pos_rec in zip(stab, stab_rec)]
        if len(stab)==2: #minimal assignment for weight-2 stabilizers
            return StabilizerInfo(
                data_qubits = position_entries,
                top = None,
                bottom = None,
                sides = None,
                ancilla = None,
                stab_type = None
                )
        if len(y_values) == 3:
            #standard simple case
            top = sorted([pe for pe in position_entries if pe.rect[1] == y_values[2]], key=lambda pe: pe.rect[0])
            side = sorted([pe for pe in position_entries if pe.rect[1] == y_values[1]], key=lambda pe: pe.rect[0])
            bottom = sorted([pe for pe in position_entries if pe.rect[1] == y_values[0]], key=lambda pe: pe.rect[0])
            #assign ancillas
            if len(side) == 2:
                a = side[0].rect[0]
                b = side[-1].rect[0]
                ends = sorted([a,b])
                x_values = list(range(ends[0] + 1, ends[1]))
                y_rec = side[0].rect[1] #y is fix
                ancillas_rec = [(x, y_rec) for x in x_values]
                #ancillas = [PositionEntry(hex = None, rect = tuple(sorted(rec)), label = None) for rec in ancillas_rec]
                ancillas = [PositionEntry(hex=None, rect=rec, label=None) for rec in sorted(ancillas_rec)]
            elif len(side) == 1:
                #this is a sideways weight-4 stabilizer which is a bit deformed
                #there are 4 different shapes of that kind and they are covered by the following
                if len(bottom) == 2:
                    y_rec = side[0].rect[1]
                    x_values = [pe.rect[0] for pe in bottom]
                    ancillas_rec = [(x, y_rec) for x in x_values]
                    ancillas = [PositionEntry(hex=None, rect=rec, label=None) for rec in ancillas_rec]
                elif len(top) == 2:
                    y_rec = side[0].rect[1]
                    x_values = [pe.rect[0] for pe in top]
                    ancillas_rec = [(x, y_rec) for x in x_values]
                    ancillas = [PositionEntry(hex=None, rect=rec, label=None) for rec in ancillas_rec]
        elif len(y_values) == 2:
            row_a = sorted([pe for pe in position_entries if pe.rect[1] == y_values[1]], key=lambda pe: pe.rect[0])
            row_b = sorted([pe for pe in position_entries if pe.rect[1] == y_values[0]], key=lambda pe: pe.rect[0])
            # determine which row is the side by checking y-distance
            if y_values[1] - y_values[0] > 1:
                # gap between rows -> neither is adjacent, shouldn't happen
                raise ValueError("unexpected y gap")
            if len(stab) == 4:
                # check x-spread to determine sides: sides have larger x range
                spread_a = row_a[-1].rect[0] - row_a[0].rect[0] if len(row_a) > 1 else 0
                spread_b = row_b[-1].rect[0] - row_b[0].rect[0] if len(row_b) > 1 else 0
                if spread_a >= spread_b:
                    side = row_a
                    top = []
                    bottom = row_b
                else:
                    side = row_b
                    top = row_a
                    bottom = []
                # ancillas same as above
                if len(side) == 2: #weight-4 stabilizers
                    #weight-4 stabilizers
                    ends = sorted([side[0].rect[0], side[-1].rect[0]])
                    x_values = list(range(ends[0] + 1, ends[1]))
                    y_rec = side[0].rect[1]
                    ancillas_rec = [(x, y_rec) for x in x_values]
                    ancillas = [PositionEntry(hex=None, rect=rec, label=None) for rec in ancillas_rec]
            if len(stab) == 3:
                if len(row_a) == 2 and len(row_b) == 1:
                    side = row_b
                    top_bottom = row_a
                elif len(row_b) == 2 and len(row_a) == 1:
                    side = row_a
                    top_bottom = row_b
                y_rec = side[0].rect[1]
                x_values = [el.rect[0] for el in top_bottom]
                ancillas_rec = [(x, y_rec) for x in x_values]
                ancillas = [PositionEntry(hex=None, rect=rec, label=None) for rec in ancillas_rec]
                #determine top/bottom based on which y values are larger
                if top_bottom[0].rect[1] > y_rec:
                    top = top_bottom
                    bottom = []
                else:
                    top = []
                    bottom = top_bottom
        else:
            raise ValueError("wrong")

        return StabilizerInfo(
            data_qubits = position_entries,
            top = top,
            sides = side,
            bottom = bottom,
            ancilla = ancillas,
            stab_type = None
        )


    def hex_mapping_to_quadratic(self):
        """Map the hexagonal layout to a quadratic grid.

        This creates a mapping for both the data and ancilla qubits
        based on Position3DHex.rectangular_map() for the data qubits.
        """
        #first find stabilizerinfo for each stabilizer (no label assignment yet, no prism/pipe assignment yet)
        stabilizers_per_layer = {}
        for z in self.z_values:
            stabilizers = self.dct_stabilizers_all[z]
            lst_stab_info = []
            for stab, stab_type in stabilizers.items():
                stab_info = self.assign_stabilizer_info(stab)
                stab_info.stab_type = stab_type
                lst_stab_info.append(stab_info)
            stabilizers_per_layer.update({z: lst_stab_info})

        mapping_full: dict[int, dict[Prism | PrismPipe, PrismData]] = {}

        #find integer values for data qubits
        data_idx = 0
        seen: dict[Position3DHex, PositionEntry] = {}
        for z in self.z_values:
            dct_temp : dict[Prism | PrismPipe, PrismData] = {}

            prism_pipes_data = self.prism_graph.prism_pipes_to_data_qubits_full[z]
            # first pass: prisms only
            for pipe_prism, data_qubits in prism_pipes_data.items():
                #if not isinstance(pipe_prism, Prism):
                #    continue
                positions = []
                for pos in data_qubits:
                    flat = Position3DHex(x=pos.x, y=pos.y, z=0)
                    if flat not in seen:
                        position_entry = PositionEntry(hex = flat, rect = flat.rectangular_map(), label = data_idx)
                        positions.append(position_entry)
                        data_idx += 1
                        seen[flat] = position_entry
                    else:
                        positions.append(seen[flat])
                dct_temp.update({pipe_prism : PrismData(positions = positions.copy(), stabilizers = None)})
            # second pass: pipes only
            #for pipe_prism, data_qubits in prism_pipes_data.items():
            #    if not isinstance(pipe_prism, PrismPipe):
            #        continue
            #    positions = []
            #    for pos in data_qubits:
            #        flat = Position3DHex(x=pos.x, y=pos.y, z=0)
            #        if flat not in seen:
            #            position_entry = PositionEntry(hex = flat, rect = flat.rectangular_map(), label = data_idx)
            #            positions.append(position_entry)
            #            data_idx += 1
            #            seen[flat] = position_entry
            #        else:
            #            positions.append(seen[flat])
            #    dct_temp.update({pipe_prism : PrismData(positions = positions.copy(), stabilizers = None)})
            mapping_full.update({z: dct_temp})

        #adapt data_qubit labels in stab_info according to above labels
        rect_to_entry: dict[tuple, PositionEntry] = {pe.rect: pe for pe in seen.values()}
        seen_ancillas: dict[tuple, PositionEntry] = {}
        for z in self.z_values:
            stabilizers = stabilizers_per_layer[z]

            # initialize stabilizers list in each PrismData
            for prism_data in mapping_full[z].values():
                prism_data.stabilizers = []

            for stab_info in stabilizers:
                # assign labels to data qubits in stab_info from seen
                for pe in stab_info.data_qubits:
                    if pe.rect in rect_to_entry:
                        pe.label = rect_to_entry[pe.rect].label

                # assign ancilla labels
                if stab_info.ancilla:
                    for anc_pe in stab_info.ancilla:
                        if anc_pe.rect in seen_ancillas:
                            anc_pe.label = seen_ancillas[anc_pe.rect].label
                        else:
                            anc_pe.label = data_idx
                            data_idx += 1
                            seen_ancillas[anc_pe.rect] = anc_pe

                # find which prism/pipe this stabilizer belongs to by overlap
                stab_rects = {pe.rect for pe in stab_info.data_qubits}
                best_match = None
                best_overlap = 0
                for pipe_prism, prism_data in mapping_full[z].items():
                    prism_rects = {pe.rect for pe in prism_data.positions}
                    overlap = len(stab_rects & prism_rects)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match = pipe_prism
                if best_match is not None:
                    mapping_full[z][best_match].stabilizers.append(stab_info)

        self.mapping_full = mapping_full
        return mapping_full

    @staticmethod
    def append_tick_with_idle_noise(
        circuit,
        all_current_qubits,
        current_tick,
        p_idle,
        active_qubits: set,
    ):
        """Append idle noise and TICK to circuit."""
        all_qubits = {pe.label for pe in all_current_qubits}
        idle = all_qubits - active_qubits
        if idle:
            circuit.append("DEPOLARIZE1", list(idle), p_idle)
        circuit.append("TICK", [])
        return current_tick + 1

    @staticmethod
    def get_active_qubits_since_last_tick(circuit: stim.Circuit) -> set:
        """Get all qubit labels that had a gate since the last TICK."""
        active = set()
        for instruction in reversed(circuit):
            if instruction.name == "TICK":
                break
            for target in instruction.targets_copy():
                if target.is_qubit_target:
                    active.add(target.value)
        return active

    def stabilizer_changed_shape_bell(
        self,
        z: int,
        prism_pipe: Prism | PrismPipe,
        stab: StabilizerInfo,
    ) -> bool:
        """Check if a stabilizer changed its data qubit support compared to the previous z-layer.

        Finds the stabilizer in the previous layer that shares the same ancilla
        positions as the given stabilizer, then compares data qubit support.

        Args:
            z: current z layer index.
            prism_pipe: the prism or pipe this stabilizer belongs to.
            stab: the current StabilizerInfo to check.

        Returns:
            True if the data qubit support changed (or no matching previous
            stabilizer was found), False if the support is identical.
        """
        if z not in self.mapping_full or z - 1 not in self.mapping_full:
            raise ValueError("do not use this if this is the very first z layer.")

        prism_pipes_stabs_previous = self.mapping_full[z - 1]

        if isinstance(prism_pipe, Prism):
            prism_pipe_previous = Prism(
                position=Position3DHex(
                    prism_pipe.position.x,
                    prism_pipe.position.y,
                    prism_pipe.position.z - 1,
                ),
                kind=prism_pipe.kind,
            )
        elif isinstance(prism_pipe, PrismPipe):
            prism_pipe_previous = PrismPipe(
                u=Prism(
                    position=Position3DHex(
                        prism_pipe.u.position.x,
                        prism_pipe.u.position.y,
                        prism_pipe.u.position.z - 1,
                    ),
                    kind=prism_pipe.u.kind,
                ),
                v=Prism(
                    position=Position3DHex(
                        prism_pipe.v.position.x,
                        prism_pipe.v.position.y,
                        prism_pipe.v.position.z - 1,
                    ),
                    kind=prism_pipe.v.kind,
                ),
                kind=prism_pipe.kind,
            )
        #find the matching key as you cannot just take the current prism_pipe and adjust pos. this is not enough
        if isinstance(prism_pipe, Prism):
            matching_key = next(
                (k for k in prism_pipes_stabs_previous
                if isinstance(k, Prism) and k.position == prism_pipe_previous.position),
                None
            )
        elif isinstance(prism_pipe, PrismPipe):
            matching_key = next(
                (k for k in prism_pipes_stabs_previous
                if isinstance(k, PrismPipe)
                and k.u.position == prism_pipe_previous.u.position
                and k.v.position == prism_pipe_previous.v.position),
                None
            )
        stabs_previous = prism_pipes_stabs_previous[matching_key].stabilizers
        if not stabs_previous:
            raise ValueError("do not use this if this is the very first z layer.")

        # represent ancillas of current stab as a set of rect tuples for comparison
        if not stab.ancilla:
            raise ValueError("a stabilizer does not have ancillas.")
        current_ancilla_rects = {anc.rect for anc in stab.ancilla}
        print("current ancilla rects", current_ancilla_rects)

        # find the previous stabilizer that shares the same ancilla rect positions
        matching_prev_stab = None
        for prev_stab in stabs_previous:
            if not prev_stab.ancilla:
                continue
            prev_ancilla_rects = {anc.rect for anc in prev_stab.ancilla}
            if prev_ancilla_rects == current_ancilla_rects:
                matching_prev_stab = prev_stab
                print("match")
                break

        if matching_prev_stab is None:
            return True

        # compare data qubit support via rect tuples
        current_data_rects = {pe.rect for pe in stab.data_qubits}
        prev_data_rects = {pe.rect for pe in matching_prev_stab.data_qubits}
        print("current data rects", current_data_rects)
        print("prev data rects", prev_data_rects)

        return current_data_rects != prev_data_rects

    def pipe_matches(self, a: Prism | PrismPipe, b: Prism | PrismPipe) -> bool:
        """Compare two prisms/pipes ignoring z coordinate."""
        if type(a) is not type(b):
            return False
        if isinstance(a, Prism):
            return a.position.x == b.position.x and a.position.y == b.position.y
        return (
            a.u.position.x == b.u.position.x
            and a.u.position.y == b.u.position.y
            and a.v.position.x == b.v.position.x
            and a.v.position.y == b.v.position.y
        )

    def identify_correlation_surface_measurements(self, horizontal_pipes_cs, meas_rec_lst, num_meas):
        """Identify measurements that belong to a correlation surface."""
        #vertical correlation surface
        meas_rec_vertical = {} #measurement labels
        #horizontal correlation surface
        meas_rec_horizontal = {} #measurement labels

        for prism_pipe, (cs_type, cs_alignment) in horizontal_pipes_cs.items():
            u = prism_pipe.u
            v = prism_pipe.v
            if u.position.z != v.position.z:
                raise ValueError("Some error within pipe construction. Present horizontal pipe's u and v have different z values.")
            z_value = u.position.z
            if cs_alignment == "ver":
                #important: identify prisms that are neighbors of the given pipe.
                #because only the star operator on the neighboring prism is taken into account
                #measurement of data qubits in the interface are not taken into account.
                #the pipe itself does not host measurement outcomes of ver cs relevant for the OBS

                #search for data qubit measurements of either u or v prism
                #in meas rec lst, these data measurements are not containing a stabilizer
                star_ops_x, star_ops_z = self.result_dct[z_value]["star"]
                if cs_type == BasisPrism.Z:
                    star_ops_current = star_ops_z
                    meas_type = "MZ"
                elif cs_type == BasisPrism.X:
                    star_ops_current = star_ops_x
                    meas_type = "MX"
                else:
                    raise ValueError("Error during construction of correlation srufaces.")
                if len(star_ops_current)==0:
                    raise ValueError("Mismatch between existing star operators and requested vertical CS.")
                #find data qubit measurements on u and/or v in meas_rec_lst
                meas_label_abs_temp = []
                for meas_rec_info in meas_rec_lst:
                    if (
                        (meas_rec_info.pipe_prism in (u, v))
                        and meas_rec_info.stabilizer is None
                        and meas_rec_info.meas_type == meas_type
                    ):
                        meas_label_abs_temp.append((meas_rec_info.abs_rec, meas_rec_info.label))
                star_ops_labels = []
                for star_op in star_ops_current:
                    star_op_flat = [Position3DHex(p.x, p.y, 0) for p in star_op]
                    star_op_flat_set = set(star_op_flat)
                    labels = []
                    for prism_data in self.mapping_full[z_value].values():
                        for pe in prism_data.positions:
                            if pe.hex in star_op_flat_set:
                                labels.append(pe.label)
                    star_ops_labels.append(labels)
                #find intersection between star op and data qubit meas
                #compare tuple[1] with the star_ops_labels
                tmp = []
                for star_op_label in star_ops_labels:
                    for label in star_op_label:
                        for abs_rec, qubit_label in meas_label_abs_temp:
                            if qubit_label == label:
                                print("abs rec", abs_rec)
                                rec = abs_rec - num_meas -1
                                tmp.append(stim.target_rec(rec))
                meas_rec_vertical.update({(cs_type, prism_pipe): tmp})
            elif cs_alignment == "hor":
                result = self.result_dct[z_value]["product"]
                stabilizer_products_x = result.stabilizer_products_x
                stabilizer_products_z = result.stabilizer_products_z
                print("len stab prod x, z", len(stabilizer_products_x), len(stabilizer_products_z))
                print("horizontal corrleaiton surface")
                if cs_type == BasisPrism.Z:
                    stabilizer_product_current = stabilizer_products_z
                    meas_type = "MZ"
                elif cs_type == BasisPrism.X:
                    stabilizer_product_current = stabilizer_products_x
                    meas_type = "MX"
                else:
                    raise ValueError("incorrect cs type assignemnt.")
                print("CS TYPE", cs_type)
                tmp = []
                print("len stabilizers_product_currrent", stabilizer_product_current)
                for cs in stabilizer_product_current:
                    # check if any stabilizer in this cs overlaps with data qubits of u or v
                    uv_labels = set()
                    for meas_rec_info in meas_rec_lst:
                        if (
                            meas_rec_info.pipe_prism in (u, v)
                            and meas_rec_info.stabilizer is not None
                        ):
                            for pe in meas_rec_info.stabilizer.data_qubits:
                                uv_labels.add(pe.label)

                    cs_labels = set()
                    for stabilizer in cs:
                        stabilizer_flat = {Position3DHex(p.x, p.y, 0) for p in stabilizer}
                        for prism_data in self.mapping_full[z_value].values():
                            for pe in prism_data.positions:
                                if pe.hex in stabilizer_flat:
                                    cs_labels.add(pe.label)

                    if not uv_labels & cs_labels:
                        continue  # this cs doesn't belong to the current pipe
                    for stabilizer in cs:
                        print("stabilizer", stabilizer)
                        # normalize to z=0 and use a set for order-independent comparison
                        #stabilizer_flat = {Position3DHex(p.x, p.y, 0) for p in stabilizer}
                        stabilizer_flat = {Position3DHex(p.x, p.y, 0) for p in stabilizer}
                        z_value = stabilizer[0].z
                        match = next(
                            (m for m in meas_rec_lst
                            if m.meas_type == meas_type
                            and m.stabilizer is not None
                            and m.z_value == z_value
                            and {Position3DHex(pe.hex.x, pe.hex.y, 0) for pe in m.stabilizer.data_qubits if pe.hex is not None} == stabilizer_flat),
                            None
                        )
                        print("stabilizer new:", match.stabilizer.data_qubits)
                        if match is not None:
                            print("abs_rec", match.abs_rec-1)
                            print("match", match)
                            rec = match.abs_rec - num_meas - 1
                            tmp.append(stim.target_rec(rec))

                    meas_rec_horizontal.update({(cs_type, prism_pipe): tmp})
                print("tmp", tmp)
            else:
                raise ValueError("The alignment of CS must be ver or hor.")
        return meas_rec_vertical, meas_rec_horizontal

    @staticmethod
    def data_meas_rec_lst_pipe_prism(meas_rec_lst, z, rounds):
        """Collect the pipes of z-1 where data qubit measurements were performed."""
        meas_rec_lst_data = [
            m for m in meas_rec_lst
            if m.z_value == z-1 and m.stabilizer is None and m.round == rounds-1
            ]
        return meas_rec_lst_data

    def add_triple_detector_changed_shape_split_opposite(
        self,
        stab,
        meas_rec_lst,
        circuit,
        z,
        r,
        rounds,
        meas_rec_lst_data
        ):
        """Add `triple` detector after split with the opposite basis as the data qubit meas.

        weight-4 stabilizer of current z layer, with previous weight-2 and weight-6.
        """
        print("=========triple detector opposite==========")
        prism_pipes_zpm_previous = self.prism_pipes_to_ZPM[z-1]
        prism_pipes_stabs_previous = self.mapping_full[z-1]

        rec_lst = []
        #determine which elements of data_meas_pipe_prism are neighbors of current stab
        #for this we first need to find the previous weight-6 labels
        flag = False
        labels_weight6=None
        stab_set_label = {el.label for el in stab.data_qubits}
        for prism_pipe in prism_pipes_zpm_previous.keys():
            stabs = prism_pipes_stabs_previous[prism_pipe].stabilizers
            for stab_temp in stabs:
                stab_temp_labels = {pe.label for pe in stab_temp.data_qubits}
                if len(stab_temp_labels & stab_set_label) == 4:
                    labels_weight6 = stab_temp_labels
                    flag = True
                    break
            if flag:
                break
        neighbor_meas = [m for m in meas_rec_lst_data if m.label in labels_weight6]
        if len(neighbor_meas)==0:
            return

        #what is the meas_type of these? take first element and take opposite
        meas_types = [m.meas_type for m in neighbor_meas]
        if not all([meas_types[0] == meas_type for meas_type in meas_types]):
            raise ValueError("There are mixed measurement types on neighboring data qubits of previous z layer.")
        if meas_types[0] == "MX":
            meas_type = "MZ"
        elif meas_types[0] == "MZ":
            meas_type = "MX"

        #add rec for current stab:
        rec = next(
            m for m in meas_rec_lst
            if m.meas_type == meas_type
            and m.z_value == z
            and m.stabilizer == stab
            and m.round == r
        ).abs_rec - 1 - circuit.num_measurements
        rec_lst.append(rec)
        print("rec_lst", rec_lst)

        #add the previous weight-6 stabilizer
        flag = False
        for prism_pipe in prism_pipes_zpm_previous.keys():
            stabs = prism_pipes_stabs_previous[prism_pipe].stabilizers
            for stab_temp in stabs:
                stab_temp_labels = {pe.label for pe in stab_temp.data_qubits}
                if len(stab_temp_labels & stab_set_label) == 4:
                    rec = next(
                        m for m in meas_rec_lst
                        if m.meas_type == meas_type
                        and m.z_value == z-1
                        and m.stabilizer == stab_temp
                        and m.round == rounds-1
                    ).abs_rec - 1 - circuit.num_measurements
                    rec_lst.append(rec)
                    flag = True
                    break
            if flag:
                break
        print("rec_lst", rec_lst)

        #add the weight-2 stabilizer
        flag = False
        for prism_pipe in prism_pipes_zpm_previous.keys():
            stabs = prism_pipes_stabs_previous[prism_pipe].stabilizers
            for stab_temp in stabs:
                stab_temp_labels = {pe.label for pe in stab_temp.data_qubits}
                if len(stab_temp_labels & labels_weight6) == 2 and len(stab_temp_labels)==2:
                    rec = next(
                        m for m in meas_rec_lst
                        if m.meas_type == meas_type
                        and m.z_value == z-1
                        and m.stabilizer == stab_temp
                        and m.round == rounds-1
                    ).abs_rec - 1 - circuit.num_measurements
                    rec_lst.append(rec)
                    flag = True
                    break
            if flag:
                break
        print("rec_lst", rec_lst)

        circuit.append("DETECTOR", [stim.target_rec(rec) for rec in rec_lst])

    def add_triple_detector_changed_shape_split_same(
        self,
        stab,
        meas_rec_lst,
        circuit,
        z,
        r,
        rounds,
        meas_rec_lst_data
        ):
        """Add `triple` detector after split with the same basis as the data qubit meas.

        Strictly speaking, this is not a `triple` detector because the data qubit measurements are
        not summarized in a stabilizer. Thus there will be more recs in the detector, overall
        four: 1 current weight-4 stabilizer, 1 previous weight-6 stabilizer,
        2 data qubit meas previous.
        """
        prism_pipes_zpm_previous = self.prism_pipes_to_ZPM[z-1]
        prism_pipes_stabs_previous = self.mapping_full[z-1]

        rec_lst = []
        #determine which elements of data_meas_pipe_prism are neighbors of current stab
        #for this we first need to find the previous weight-6 labels
        flag = False
        labels_weight6=None
        stab_set_label = {el.label for el in stab.data_qubits}
        for prism_pipe in prism_pipes_zpm_previous.keys():
            stabs = prism_pipes_stabs_previous[prism_pipe].stabilizers
            for stab_temp in stabs:
                stab_temp_labels = {pe.label for pe in stab_temp.data_qubits}
                if len(stab_temp_labels & stab_set_label) == 4:
                    labels_weight6 = stab_temp_labels
                    flag = True
                    break
            if flag:
                break
        neighbor_meas = [m for m in meas_rec_lst_data if m.label in labels_weight6]
        if len(neighbor_meas)==0:
            return

        #what is the meas_type of these? take first element
        meas_types = [m.meas_type for m in neighbor_meas]
        if not all([meas_types[0] == meas_type for meas_type in meas_types]):
            raise ValueError("There are mixed measurement types on neighboring data qubits of previous z layer.")
        meas_type = meas_types[0]

        #add rec for current stab:
        rec = next(
            m for m in meas_rec_lst
            if m.meas_type == meas_type
            and m.z_value == z
            and m.stabilizer == stab
            and m.round == r
        ).abs_rec - 1 - circuit.num_measurements
        rec_lst.append(rec)

        #add the previous weight-6 stabilizer
        flag = False
        for prism_pipe in prism_pipes_zpm_previous.keys():
            stabs = prism_pipes_stabs_previous[prism_pipe].stabilizers
            for stab_temp in stabs:
                stab_temp_labels = {pe.label for pe in stab_temp.data_qubits}
                if len(stab_temp_labels & stab_set_label) == 4:
                    rec = next(
                        m for m in meas_rec_lst
                        if m.meas_type == meas_type
                        and m.z_value == z-1
                        and m.stabilizer == stab_temp
                        and m.round == rounds-1
                    ).abs_rec - 1 - circuit.num_measurements
                    rec_lst.append(rec)
                    flag = True
                    break
            if flag:
                break

        #add the two data qubit measurements.
        for neighbor in neighbor_meas:
            rec = neighbor.abs_rec - 1 - circuit.num_measurements
            rec_lst.append(rec)

        circuit.append("DETECTOR", [stim.target_rec(rec) for rec in rec_lst])

    def add_double_detector_changed_shape_merge_same(
        self,
        stab,
        meas_rec_lst,
        circuit,
        z,
        r,
        rounds,
        ):
        """Add a double detector in the initialization basis.

        This compares the current weight-6 stabilizer with the previous weight.4 stabilizer
        it compares the stabilizers of the basis in which zpm.p is done.
        these are naturally okay thus only two stabilizers needed
        """
        print("---------double detector same---------")
        stab_labels = {pe.label for pe in stab.data_qubits}
        #find the weight-4 stabilizer of the previous layer
        prism_pipes_zpm_previous = self.prism_pipes_to_ZPM[z-1]
        prism_pipes_stabs_previous = self.mapping_full[z-1]
        stab_weight4 = None
        flag=False
        for prism_pipe in prism_pipes_zpm_previous.keys():
            stabs = prism_pipes_stabs_previous[prism_pipe].stabilizers
            for stab_temp in stabs:
                stab_temp_labels = {pe.label for pe in stab_temp.data_qubits}
                if len(stab_temp_labels & stab_labels)==4 and len(stab_temp_labels)==4: #overlap 4 guarantees correct stab
                    stab_weight4 = stab_temp
                    flag = True
                    break
            if flag:
                break
        if stab_weight4 is None:
            return

        #find meas type based on initialization of adjacent pipe.
        #both stabilizers here are strictly speaking part of the prism, not the pipe
        #thus no knowledge about pipe initialization
        prism_pipes_zpm_temp = self.prism_pipes_to_ZPM[z]
        prism_pipes_stabs = self.mapping_full[z]
        flag = False
        for prism_pipe in prism_pipes_zpm_temp.keys():
            stabs = prism_pipes_stabs[prism_pipe].stabilizers
            for stab_temp in stabs:
                stab_temp_labels = {pe.label for pe in stab_temp.data_qubits}
                zpm = prism_pipes_zpm_temp[prism_pipe]
                #take the weight-2 stabilizer that has overlap with current weight-6
                #this is guaranteed in the adjacent pipe.
                if len(stab_labels & stab_temp_labels)==2 and len(stab_temp_labels)==2:
                    #this is the zpm we want
                    if zpm.p == BasisPrism.X:
                        meas_type = "MX"
                    elif zpm.p == BasisPrism.Z:
                        meas_type = "MZ"
                    flag = True
                    break
            if flag:
                break

        print("stab weight 4", [el.label for el in stab_weight4.data_qubits])
        print("meas type", meas_type)
        rec_lst = []

        #find the correct entry for previous weight-4 stabilizer
        rec = next(
            m for m in meas_rec_lst
            if m.meas_type == meas_type
            and m.z_value == z-1
            and m.stabilizer == stab_weight4
            and m.round == rounds-1
        ).abs_rec - 1 - circuit.num_measurements
        rec_lst.append(rec)
        print("rec_lst", rec_lst)

        #find the correct entry for current weight-6 stabilizer
        rec = next(
            m for m in meas_rec_lst
            if m.meas_type == meas_type
            and m.z_value == z
            and m.stabilizer == stab
            and m.round == r
        ).abs_rec - 1 - circuit.num_measurements
        rec_lst.append(rec)
        print("rec_lst", rec_lst)

        circuit.append("DETECTOR", [stim.target_rec(rec) for rec in rec_lst]) #inplace replace


    def add_triple_detector_changed_shape_merge_opposite(
        self,
        stab_weight6: StabilizerInfo,
        meas_rec_lst: list,
        circuit: stim.Circuit,
        z: int,
        r: int,
        rounds: int
        ):
        """Add a triple detector at the r=0 during a merge.

        this means that we add a detector that compares weight-4 of previous z layer,
        weight-6 and weight-2 of the current layer. the weight-4 and weight-2 are subsets
        of the weight-6 stabilizer
        """
        #usually, weight-4 stabilizer is given as input
        #find weight-6 stabilizer and weight-2 stabilizer
        stab_labels = {pe.label for pe in stab_weight6.data_qubits}
        print("stab_labels", stab_labels)

        if len(stab_labels) != 6:
            return

        stab_weight4 = None
        stab_weight2 = None
        meas_type = None

        #search for weight-2 stabilizer (requires already found weight-6)
        #determine the meas_type we are looking for. the weight-2 stab is in the pipe,
        # not in the prism, so this must be initailized
        flag = False
        prism_pipes_zpm_temp = self.prism_pipes_to_ZPM[z]
        prism_pipes_stabs = self.mapping_full[z]
        for prism_pipe in prism_pipes_zpm_temp.keys():
            stabs = prism_pipes_stabs[prism_pipe].stabilizers
            zpm = prism_pipes_zpm_temp[prism_pipe]
            for stab in stabs:
                stab_temp_labels = {pe.label for pe in stab.data_qubits}
                if stab_temp_labels & stab_labels and len(stab.data_qubits)==2:
                    stab_weight2 = stab
                    #meas type is opposite of init
                    if zpm.p == BasisPrism.Z:
                        meas_type = "MX"
                    elif zpm.p == BasisPrism.X:
                        meas_type = "MZ"
                    flag = True
                    break
            if flag:
                break

        #find weight-4 stabilizer of previous z layer
        prism_pipes_zpm_previous = self.prism_pipes_to_ZPM[z-1]
        prism_pipes_stabs_previous = self.mapping_full[z-1]
        flag = False
        for prism_pipe in prism_pipes_zpm_previous.keys():
            stabs = prism_pipes_stabs_previous[prism_pipe].stabilizers
            for stab in stabs:
                stab_temp_labels = {pe.label for pe in stab.data_qubits}
                if stab_labels == stab_temp_labels | set(pe.label for pe in stab_weight2.data_qubits):
                    stab_weight4 = stab
                    flag = True
                    break
            if flag:
                break
        if stab_weight4 is None: #this is a weight-6 stabilizer fully in the middle of the interface
            return
        rec_lst = []
        #find the rec labels for the triple detector.
        #weight4 from previous layer
        rec = next(
            m for m in meas_rec_lst
            if m.meas_type == meas_type
            and m.z_value == z-1
            and m.stabilizer == stab_weight4
            and m.round == rounds-1
        ).abs_rec - 1 - circuit.num_measurements
        rec_lst.append(rec)

        for stab_temp in [stab_weight2, stab_weight6]:
            rec = next(
                m for m in meas_rec_lst
                if m.meas_type == meas_type
                and m.z_value == z
                and m.stabilizer == stab_temp
                and m.round == r
            ).abs_rec - 1 - circuit.num_measurements
            rec_lst.append(rec)

        circuit.append("DETECTOR", [stim.target_rec(rec) for rec in rec_lst]) #inplace replace

    def add_trivial_detector_changed_shape(
        self,
        stab: StabilizerInfo,
        prism_pipe: Prism | PrismPipe,
        prism_pipes_stabs: dict,
        prism_pipes_zpm_temp: dict,
        meas_rec_lst: list,
        circuit: stim.Circuit,
        z: int,
        r: int,
    ):
        """Add a trivial detector for an XZ stabilizer that changed shape at r=0.

        When a stabilizer changed shape and the initialization basis matches
        one of the syndrome bases, that syndrome measurement is deterministic.
        Only performed if the very first z layer also initialized in the same basis. #!TODO is this the right condition?
        """
        first_z = self.z_values[0]
        first_zpm = self.prism_pipes_to_ZPM[first_z].get(prism_pipe)
        if first_zpm is None:
            # prism_pipe didn't exist at the first z layer, find by position match
            first_zpm = next(
                (zpm for pp, zpm in self.prism_pipes_to_ZPM[first_z].items()
                if self.pipe_matches(pp, prism_pipe)),
                None
            )
        if first_zpm is None:
            return

        stab_labels = {pe.label for pe in stab.data_qubits}
        for candidate_pipe, candidate_prism_data in prism_pipes_stabs.items():
            candidate_labels = {pe.label for pe in candidate_prism_data.positions}
            if stab_labels & candidate_labels:
                candidate_zpm = prism_pipes_zpm_temp[candidate_pipe]
                if stab.stab_type == "XZ":
                    if candidate_zpm.p == BasisPrism.X and first_zpm.p == BasisPrism.X:
                        rec = next(
                            m for m in meas_rec_lst
                            if m.meas_type == "MX"
                            and m.pipe_prism == prism_pipe
                            and m.z_value == z
                            and m.stabilizer == stab
                            and m.round == r
                        ).abs_rec - 1 - circuit.num_measurements
                        circuit.append("DETECTOR", [stim.target_rec(rec)])
                    elif candidate_zpm.p == BasisPrism.Z and first_zpm.p == BasisPrism.Z:
                        rec = next(
                            m for m in meas_rec_lst
                            if m.meas_type == "MZ"
                            and m.pipe_prism == prism_pipe
                            and m.z_value == z
                            and m.stabilizer == stab
                            and m.round == r
                        ).abs_rec - 1 - circuit.num_measurements
                        circuit.append("DETECTOR", [stim.target_rec(rec)])
    @staticmethod
    def create_data_measurement_detectors(prism_pipe, stabs, zpm, meas_rec_lst, circuit, z, r):
        """Create detectors based on data qubit measurements.

        i.e. for some stabilizer, take the data qubit measurements of the respective qubits
        and compare to the previous layer's stabilizer ancilla measurements.

        this is for a current pipe/prism.

        #! ATTENTION! in the future, if there may be multiple data qubit measurements
        #! on different pipes/prisms in the same z layer, one has to specifiy pipe_prism
        #! explicitly. otherwise the wrong stabilizer measurements may be assigned to the detector
        """
        if zpm.m == BasisPrism.N:
            return circuit #do nothing if no measurement!
        #only go through the stabilizers that are relevant for zpm.m
        stabs_filtered = []
        for stab in stabs:
            if zpm.m == BasisPrism.Z:
                if stab.stab_type in {"XZ", "Z"}:
                    stabs_filtered.append(stab)
            elif zpm.m == BasisPrism.X:
                if stab.stab_type in {"XZ", "X"}:
                    stabs_filtered.append(stab)
        for stab in stabs_filtered:
            print("stab", [el.label for el in stab.data_qubits])
            #only take the stabilizers whose data qubits are actually measured.
            detector_idx_lst = []
            flag = False
            for data in stab.data_qubits:
                print("data qubit---", data.label)
                print("zpm", zpm)
                if zpm.m == BasisPrism.Z:
                    try:
                        match = next(
                            m for m in meas_rec_lst
                            if m.meas_type == "MZ"
                            #and m.pipe_prism == prism_pipe
                            and m.z_value == z
                            and m.stabilizer is None #right above, no stabilizer assigned
                            and m.round == r
                            and m.label == data.label
                        )
                    except StopIteration:
                        flag = True
                        break
                elif zpm.m == BasisPrism.X:
                    try:
                        match = next(
                            m for m in meas_rec_lst
                            if m.meas_type == "MX"
                            #and m.pipe_prism == prism_pipe
                            and m.z_value == z
                            and m.stabilizer is None #right above, no stabilizer assigned
                            and m.round == r
                            and m.label == data.label
                        )
                    except StopIteration:
                        flag = True
                        break
                rec_current = match.abs_rec - 1 -circuit.num_measurements
                detector_idx_lst.append(rec_current)

            if flag: #if flag, there was a stabilizer with data qubits that were never measured.
                continue
            if zpm.m == BasisPrism.Z:
                rec_prev = next( #previous stabilizer ancilla
                    m for m in meas_rec_lst
                    if m.meas_type == "MZ"
                    #and m.pipe_prism == prism_pipe
                    and m.z_value == z
                    and m.stabilizer == stab
                    and m.round == r #last stabilizer meas was officially in same round
                ).abs_rec - 1 -circuit.num_measurements
            elif zpm.m == BasisPrism.X:
                rec_prev = next( #previous stabilizer ancilla
                    m for m in meas_rec_lst
                    if m.meas_type == "MX"
                    #and m.pipe_prism == prism_pipe
                    and m.z_value == z
                    and m.stabilizer == stab
                    and m.round == r #last stabilizer meas was officially in same round
                ).abs_rec - 1 -circuit.num_measurements
            detector_idx_lst.append(rec_prev)
            circuit.append("DETECTOR", [
                    stim.target_rec(rec) for rec in detector_idx_lst])
        return circuit

    def create_stim_circuit_bell_multiplexing(
            self,
            rounds,
            p_init: float,
            p_meas: float,
            p_idle: float,
            p_gate2: float,
            cs: CorrelationSurface,
        ) -> stim.Circuit:
        """Create a syndrome extraction circuit based on Bell Multiplexing."""
        #store the measurement record labes together with vital information
        meas_rec_lst: list[MeasRecInfo] = [] #IMPORTANT label starts from 1, not from 0 - i.e. offset by 1 between what is displayed in the circuit svg and the measreclst.
        cnot_order = ["bottom_z", "sides_z", "top_z","bottom_x", "sides_x", "top_x"] #just choose a convention
        circuit = stim.Circuit()

        #only check the relevant horizontal correlation surfaces and whether ver/hor cs
        horizontal_pipes_cs = self.prism_graph.find_ver_hor_correlation_surface(cs)

        for s, z in enumerate(self.z_values):
            print(f"========================z={z}=======================")
            star_ops_x, star_ops_z = self.result_dct[z]["star"]

            current_tick = 0 #re-initialize tick label per z
            #zpm values
            prism_pipes_zpm_temp = self.prism_pipes_to_ZPM[z]
            #stabilizer info etc
            prism_pipes_stabs = self.mapping_full[z]

            #define all current qubits (both ancilla and data) for comparison for idiling noise
            all_current_qubits = []
            for prism_pipe, prism_data in prism_pipes_stabs.items():
                # data qubits
                all_current_qubits.extend(prism_data.positions)
                # ancilla qubits
                if prism_data.stabilizers:
                    for stab in prism_data.stabilizers:
                        if stab.ancilla:
                            all_current_qubits.extend(stab.ancilla)

            #initialization of data qubits
            for prism_pipe in prism_pipes_zpm_temp.keys():
                zpm = prism_pipes_zpm_temp[prism_pipe]
                data_positions = prism_pipes_stabs[prism_pipe].positions
                if zpm.p == BasisPrism.Z:
                    lst = [el.label for el in data_positions]
                    circuit.append("R", lst)
                    circuit.append("X_ERROR", lst, p_init)
                elif zpm.p == BasisPrism.X:
                    lst = [el.label for el in data_positions]
                    circuit.append("RX", lst)
                    circuit.append("Z_ERROR", lst, p_init)
            #r rounds of error correction based on stabilizer_type
            for r in range(rounds):
                #------------initialize all ancillas in that round--------------
                for prism_pipe in prism_pipes_zpm_temp.keys():
                    zpm = prism_pipes_zpm_temp[prism_pipe]
                    stabs = prism_pipes_stabs[prism_pipe].stabilizers
                    data_positions = prism_pipes_stabs[prism_pipe].positions
                    for stab in stabs:
                        #Bell initialization
                        if stab.ancilla is not None:
                            #ancillas are ordered left RX, right RZ
                            label = stab.ancilla[0].label
                            circuit.append("RX", label)
                            circuit.append("Z_ERROR", label, p_init)
                            label = stab.ancilla[1].label
                            circuit.append("R", label)
                            circuit.append("X_ERROR", label, p_init)
                        #--if r!=rounds-1 then add weight-2 stabilizer finalization of previous round--
                        if len(stab.data_qubits) == 2 and r !=0:# and r != rounds-1:
                            lst = [stab.data_qubits[0].label, stab.data_qubits[1].label]
                            print("defold CNOT", lst)
                            if stab.stab_type == "X":
                                circuit.append("CNOT", lst[::-1])
                                circuit.append("DEPOLARIZE2", lst[::-1], p_gate2) #cnot in other direction
                            elif stab.stab_type == "Z":
                                circuit.append("CNOT", lst)
                                circuit.append("DEPOLARIZE2", lst, p_gate2)
                # classical feedback for weight->2 stabs from previous round
                if r != 0:
                    for prism_pipe in prism_pipes_zpm_temp.keys():
                        stabs = prism_pipes_stabs[prism_pipe].stabilizers
                        for stab in stabs:
                            if len(stab.data_qubits) > 2 and stab.stab_type == "XZ": #no classical feedback for single type.
                                neighbor_data = [
                                    qubit for qubit in stab.data_qubits
                                    if Position3DHex.rectangular_neighbor(qubit.rect, stab.ancilla[1].rect)
                                ]
                                rec_mz = next(
                                    m for m in meas_rec_lst
                                    if m.meas_type == "MZ"
                                    and m.pipe_prism == prism_pipe
                                    and m.z_value == z
                                    and m.stabilizer == stab
                                    and m.round == r - 1  # previous round
                                ).abs_rec - 1 - circuit.num_measurements
                                for data in neighbor_data:
                                    circuit.append("CX", [stim.target_rec(rec_mz), data.label])
                #--add idling noise--
                active = self.get_active_qubits_since_last_tick(circuit)
                current_tick = self.append_tick_with_idle_noise(
                    circuit,
                    all_current_qubits,
                    current_tick,
                    p_idle,
                    active)
                #------------CNOT for Bell for ancillas pairs--------------
                for prism_pipe in prism_pipes_zpm_temp.keys():
                    zpm = prism_pipes_zpm_temp[prism_pipe]
                    stabs = prism_pipes_stabs[prism_pipe].stabilizers
                    data_positions = prism_pipes_stabs[prism_pipe].positions
                    for stab in stabs:
                        if stab.ancilla is not None:
                            #Bell initialization CNOT
                            lst = [stab.ancilla[0].label, stab.ancilla[1].label]
                            circuit.append("CNOT", lst)
                            circuit.append("DEPOLARIZE2", lst, p_gate2)
                #--add idling noise--
                active = self.get_active_qubits_since_last_tick(circuit)
                current_tick = self.append_tick_with_idle_noise(
                    circuit,
                    all_current_qubits,
                    current_tick,
                    p_idle,
                    active)
                #--CNOT Gates for SE--
                for direction in cnot_order:
                    for prism_pipe in prism_pipes_zpm_temp.keys():
                        zpm = prism_pipes_zpm_temp[prism_pipe]
                        stabs = prism_pipes_stabs[prism_pipe].stabilizers
                        data_positions = prism_pipes_stabs[prism_pipe].positions
                        for stab in stabs:
                            if stab.stab_type == "XZ":
                                #Z stabilizer
                                if direction in {"bottom_z", "sides_z", "top_z"}:
                                    print("stabtype XZ, Z stab")
                                    entries = getattr(stab, direction.removesuffix("_z"))
                                    print("entries", entries)
                                    for data in entries:
                                        #which ancilla is neighbor?
                                        for ancilla in stab.ancilla:
                                            if Position3DHex.rectangular_neighbor(ancilla.rect, data.rect):
                                                lst = [data.label, ancilla.label]
                                                print("lst", lst)
                                                circuit.append("CNOT", lst)
                                                circuit.append("DEPOLARIZE2", lst, p_gate2)
                                                break
                                #X stabilizer
                                elif direction in {"bottom_x", "sides_x", "top_x"}:
                                    print("stabtype XZ, X stab")
                                    print("direction")
                                    entries = getattr(stab, direction.removesuffix("_x"))
                                    print("entries", entries)
                                    for data in entries:
                                        #which ancilla is neighbor?
                                        for ancilla in stab.ancilla:
                                            if Position3DHex.rectangular_neighbor(ancilla.rect, data.rect):
                                                lst = [ancilla.label, data.label]
                                                print("lst", lst)
                                                circuit.append("CNOT", lst)
                                                circuit.append("DEPOLARIZE2", lst, p_gate2)
                                                break
                            #weight-5 and weight-3, and weight-6 stabilizers on STDW
                            elif stab.stab_type == "X" and len(stab.data_qubits) > 2:
                                print("stab type X, >2", [x.label for x in stab.data_qubits])
                                if direction in {"bottom_x", "sides_x", "top_x"}:
                                    entries = getattr(stab, direction.removesuffix("_x"))
                                    print("direction", direction)
                                    print("entries", entries)
                                    for data in entries:
                                        #which ancilla is neighbor?
                                        for ancilla in stab.ancilla:
                                            if Position3DHex.rectangular_neighbor(ancilla.rect, data.rect):
                                                lst = [ancilla.label, data.label]
                                                print("qubits CNOT", lst)
                                                circuit.append("CNOT", lst)
                                                circuit.append("DEPOLARIZE2", lst, p_gate2)
                                                break
                            elif stab.stab_type == "Z" and len(stab.data_qubits) > 2:
                                print("stab type Z, >2", [x.label for x in stab.data_qubits])
                                if direction in {"bottom_z", "sides_z", "top_z"}:
                                    entries = getattr(stab, direction.removesuffix("_z"))
                                    print("direction", direction)
                                    print("entries", entries)
                                    for data in entries:
                                        #which ancilla is neighbor?
                                        for ancilla in stab.ancilla:
                                            if Position3DHex.rectangular_neighbor(ancilla.rect, data.rect):
                                                lst = [data.label, ancilla.label]
                                                print("qubits CNOT", lst)
                                                circuit.append("CNOT", lst)
                                                circuit.append("DEPOLARIZE2", lst, p_gate2)
                                                break
                    #--add idling noise--
                    active = self.get_active_qubits_since_last_tick(circuit)
                    current_tick = self.append_tick_with_idle_noise(
                        circuit,
                        all_current_qubits,
                        current_tick,
                        p_idle,
                        active)
                #--final bell and fold for weight2--
                for prism_pipe in prism_pipes_zpm_temp.keys():
                    zpm = prism_pipes_zpm_temp[prism_pipe]
                    stabs = prism_pipes_stabs[prism_pipe].stabilizers
                    data_positions = prism_pipes_stabs[prism_pipe].positions
                    for stab in stabs:
                        if len(stab.data_qubits) > 2:
                            #Bell final
                            lst = [stab.ancilla[0].label, stab.ancilla[1].label]
                            circuit.append("CNOT", lst)
                            circuit.append("DEPOLARIZE2", lst, p_gate2)
                        elif len(stab.data_qubits) == 2:
                            #first step of fold for weight-2 stabilizer
                            lst = [stab.data_qubits[0].label, stab.data_qubits[1].label]
                            print("fold", lst)
                            if stab.stab_type == "X":
                                circuit.append("CNOT", lst[::-1])
                                circuit.append("DEPOLARIZE2", lst[::-1], p_gate2) #cnot in other direction
                            elif stab.stab_type == "Z":
                                circuit.append("CNOT", lst)
                                circuit.append("DEPOLARIZE2", lst, p_gate2)
                        else:
                            raise TQECError("weight-2 stabilizer that is not single type - cannot be!")
                #--add idling noise--
                active = self.get_active_qubits_since_last_tick(circuit)
                current_tick = self.append_tick_with_idle_noise(
                    circuit,
                    all_current_qubits,
                    current_tick,
                    p_idle,
                    active)
                #--measure ancilla both for weight-2 and others--
                for prism_pipe in prism_pipes_zpm_temp.keys():
                    zpm = prism_pipes_zpm_temp[prism_pipe]
                    stabs = prism_pipes_stabs[prism_pipe].stabilizers
                    data_positions = prism_pipes_stabs[prism_pipe].positions
                    #measure stabs if higher weight, and also meas stab if weight=2
                    for stab in stabs:
                        print("stab---->", stab.data_qubits)
                        if len(stab.data_qubits) > 2:
                            #measurement
                            if stab.stab_type == "XZ" or stab.stab_type == "X":
                                label = stab.ancilla[0].label
                                circuit.append("Z_ERROR", label, p_meas)
                                circuit.append("MX", label)
                                meas_rec_lst.append(
                                    MeasRecInfo(
                                        meas_type = "MX",
                                        pipe_prism = prism_pipe,
                                        stabilizer = stab,
                                        abs_rec = circuit.num_measurements,
                                        z_value = z,
                                        round = r,
                                        label = label,
                                        tick = current_tick
                                        )
                                    ) #add to record
                            if stab.stab_type == "XZ" or stab.stab_type == "Z":
                                label = stab.ancilla[1].label
                                circuit.append("X_ERROR", label, p_meas)
                                circuit.append("M", label)
                                meas_rec_lst.append(
                                    MeasRecInfo(
                                        meas_type = "MZ",
                                        pipe_prism = prism_pipe,
                                        stabilizer = stab,
                                        abs_rec = circuit.num_measurements,
                                        z_value = z,
                                        round = r,
                                        label = label,
                                        tick = current_tick
                                        )
                                    ) #add to record
                        elif len(stab.data_qubits) == 2:
                            #folded weight-2 stabilizer, perform meas on data qubit
                            label = stab.data_qubits[1].label
                            print("measure within fold", label)
                            if stab.stab_type == "Z":
                                circuit.append("X_ERROR", label, p_meas)
                                circuit.append("M", label)
                                meas_type = "MZ"
                            elif stab.stab_type == "X":
                                circuit.append("Z_ERROR", label, p_meas)
                                circuit.append("MX", label)
                                meas_type = "MX"
                            meas_rec_lst.append(
                                MeasRecInfo(
                                    meas_type = meas_type,
                                    pipe_prism = prism_pipe,
                                    stabilizer = stab,
                                    abs_rec = circuit.num_measurements,
                                    z_value = z,
                                    round = r,
                                    label = label,
                                    tick = current_tick
                                    )
                                ) #add to record
                        else:
                            raise TQECError("weight-2 stabilizer that is not single type - cannot be!")
                #------add final  weight-2 CNOT gate that expands after folding if no further round available------
                if r == rounds-1:
                    flag = False
                    for prism_pipe in prism_pipes_zpm_temp.keys():
                        zpm = prism_pipes_zpm_temp[prism_pipe]
                        stabs = prism_pipes_stabs[prism_pipe].stabilizers
                        for stab in stabs:
                            if len(stab.data_qubits) == 2:
                                flag = True
                                break
                    if flag:
                        print("flag", flag)
                        #--add idling noise as this has to be its own tick--
                        active = self.get_active_qubits_since_last_tick(circuit)
                        current_tick = self.append_tick_with_idle_noise(
                            circuit,
                            all_current_qubits,
                            current_tick,
                            p_idle,
                            active)
                        for prism_pipe in prism_pipes_zpm_temp.keys():
                            zpm = prism_pipes_zpm_temp[prism_pipe]
                            stabs = prism_pipes_stabs[prism_pipe].stabilizers
                            data_positions = prism_pipes_stabs[prism_pipe].positions
                            for stab in stabs:
                                if len(stab.data_qubits) == 2:
                                    lst = [stab.data_qubits[0].label, stab.data_qubits[1].label]
                                    print("additional layer after last round, defold cnot", lst)
                                    if stab.stab_type == "X":
                                        circuit.append("CNOT", lst[::-1])
                                        circuit.append("DEPOLARIZE2", lst[::-1], p_gate2) #cnot in other direction
                                    elif stab.stab_type == "Z":
                                        circuit.append("CNOT", lst)
                                        circuit.append("DEPOLARIZE2", lst, p_gate2)
                    #classical feedback. add this in its own layer as it is not assumed to be physical anyways
                    #important: this must be AFTER folding/unfolding of weight-2 stabilizers, otherwise it may mix them up
                    for prism_pipe in prism_pipes_zpm_temp.keys():
                        zpm = prism_pipes_zpm_temp[prism_pipe]
                        stabs = prism_pipes_stabs[prism_pipe].stabilizers
                        data_positions = prism_pipes_stabs[prism_pipe].positions
                        #measure stabs if higher weight, and also meas stab if weight=2
                        for stab in stabs:
                            if len(stab.data_qubits) > 2 and stab.stab_type == "XZ": #no classical feedback for single type!
                                #add classical feedback X^b, find neighboring data to ancilla[1]
                                #only add classical feedback if actually MZ was done
                                neighbor_data = [
                                    qubit for qubit
                                    in stab.data_qubits # not data_positions as interface qubits are not included then.
                                    if Position3DHex.rectangular_neighbor(qubit.rect, stab.ancilla[1].rect)
                                    ]
                                print("NEIGHBOR DATA FOR CLASSICAL X", [x.label for x in neighbor_data])
                                rec_current = next(
                                    m for m in meas_rec_lst
                                    if m.meas_type == "MZ"
                                    and m.pipe_prism == prism_pipe
                                    and m.z_value == z
                                    and m.stabilizer == stab
                                    and m.round == r
                                ).abs_rec - 1 - circuit.num_measurements
                                print("rec", rec_current + 1 + circuit.num_measurements, rec_current)
                                for data in neighbor_data:
                                    circuit.append("CX", [stim.target_rec(rec_current), data.label]) #classical X
                                    #no error added here because assume that those are classically tracked
                #if a prism pipe has zpm.m != N then data qubits need to be measured
                #after final aditional round for weight-2 stabilizer meas
                for prism_pipe in prism_pipes_zpm_temp.keys():
                    zpm = prism_pipes_zpm_temp[prism_pipe]
                    stabs = prism_pipes_stabs[prism_pipe].stabilizers
                    data_positions = prism_pipes_stabs[prism_pipe].positions
                    lst = [data.label for data in data_positions]
                    if z != max(self.z_values) and r==rounds-1: #if last z values then handled at the end.
                        if zpm.m == BasisPrism.Z:
                            print("meas data qubits Z")
                            circuit.append("X_ERROR", lst, p_meas)
                            circuit.append("M", lst)
                            base_rec = circuit.num_measurements - len(lst)
                            for i, qubit_label in enumerate(lst):
                                meas_rec_lst.append(
                                    MeasRecInfo(
                                        meas_type="MZ",
                                        pipe_prism = prism_pipe,
                                        stabilizer = None,
                                        abs_rec=base_rec + i +1,
                                        z_value=z,
                                        round=r,
                                        label=qubit_label,  # now correctly the individual qubit label
                                        tick=current_tick
                                    )
                                )
                        elif zpm.m == BasisPrism.X:
                            print("meas data qubits MX")
                            circuit.append("Z_ERROR", lst, p_meas)
                            circuit.append("MX", lst)
                            base_rec = circuit.num_measurements - len(lst)
                            print("base_rec", base_rec)
                            for i, qubit_label in enumerate(lst):
                                print("i, label, abs rec", i, qubit_label, base_rec + i)
                                meas_rec_lst.append(
                                    MeasRecInfo(
                                        meas_type="MX",
                                        pipe_prism = prism_pipe,
                                        stabilizer = None,
                                        abs_rec=base_rec + i + 1,
                                        z_value=z,
                                        round=r,
                                        label=qubit_label,  # now correctly the individual qubit label
                                        tick=current_tick
                                    )
                                )
                #--add idling noise--
                active = self.get_active_qubits_since_last_tick(circuit)
                current_tick = self.append_tick_with_idle_noise(
                    circuit,
                    all_current_qubits,
                    current_tick,
                    p_idle,
                    active)

                #---------add detectors based on meas_rec_lst-----------
                #where did the data qubit measurement take place?
                meas_rec_lst_data = self.data_meas_rec_lst_pipe_prism(meas_rec_lst, z, rounds)
                for prism_pipe in prism_pipes_zpm_temp.keys():
                    zpm = prism_pipes_zpm_temp[prism_pipe]
                    stabs = prism_pipes_stabs[prism_pipe].stabilizers
                    if r == 0:
                        print("r=0 case")
                        print("zpm", zpm.m, zpm.p)
                        #only add those default detectors based on zpm.p
                        relevant = [] #this must be reset after each run.
                        if zpm.p == BasisPrism.Z:
                            relevant = [
                                m for m in meas_rec_lst
                                if m.meas_type == "MZ"
                                and m.pipe_prism == prism_pipe
                                and m.z_value == z
                                and m.round == r]
                        elif zpm.p == BasisPrism.X:
                            relevant = [
                                m for m in meas_rec_lst
                                if m.meas_type == "MX"
                                and m.pipe_prism == prism_pipe
                                and m.z_value == z
                                and m.round == r]
                        print("relevant", relevant)
                        for m in relevant:
                            offset = m.abs_rec -1 - circuit.num_measurements
                            print("m abs rec", m.abs_rec)
                            circuit.append("DETECTOR", [stim.target_rec(offset)])
                        if zpm.p == BasisPrism.N and s != 0:
                            print("p = N and not s=0")
                            #IMPORTANT: if basis N, you need to compare to the previous z layer
                            for stab in stabs:
                                changed_shape = self.stabilizer_changed_shape_bell(z, prism_pipe, stab)
                                print("changed_shape", changed_shape)
                                if not changed_shape:
                                    print("not changed shape")
                                    #compare Z stabilizer to previous z layer (stab remained the same)
                                    print("Z")
                                    print("stab", stab)

                                    rec_current = next(
                                        m for m in meas_rec_lst
                                        if m.meas_type == "MZ"
                                        and m.pipe_prism == prism_pipe
                                        and m.z_value == z
                                        and m.stabilizer == stab
                                        and m.round == r
                                    ).abs_rec - 1 - circuit.num_measurements
                                    target_rects = {pe.rect for pe in stab.data_qubits}
                                    rec_prev = next(
                                        m for m in meas_rec_lst
                                        if m.meas_type == "MZ"
                                        and self.pipe_matches(m.pipe_prism, prism_pipe)
                                        and m.z_value == z - 1
                                        and m.stabilizer is not None
                                        and {pe.rect for pe in m.stabilizer.data_qubits} == target_rects
                                        and m.round == rounds - 1
                                    ).abs_rec - 1 - circuit.num_measurements
                                    print("rec current abs", rec_current + 1+ circuit.num_measurements)
                                    print("rec prev abs", rec_prev + 1+ circuit.num_measurements)
                                    circuit.append("DETECTOR", [
                                        stim.target_rec(rec_current), stim.target_rec(rec_prev)])
                                    #compare X stabilizer to previous z layer (stab remained the same)
                                    print("X")
                                    rec_current = next(
                                        m for m in meas_rec_lst
                                        if m.meas_type == "MX"
                                        and m.pipe_prism == prism_pipe
                                        and m.z_value == z
                                        and m.stabilizer == stab
                                        and m.round == r
                                    ).abs_rec - 1 - circuit.num_measurements
                                    target_rects = {pe.rect for pe in stab.data_qubits}
                                    rec_prev = next(
                                        m for m in meas_rec_lst
                                        if m.meas_type == "MX"
                                        and self.pipe_matches(m.pipe_prism, prism_pipe)
                                        and m.z_value == z - 1
                                        and m.stabilizer is not None
                                        and {pe.rect for pe in m.stabilizer.data_qubits} == target_rects
                                        and m.round == rounds - 1
                                    ).abs_rec - 1 - circuit.num_measurements
                                    print("rec current abs", rec_current + 1+ circuit.num_measurements)
                                    print("rec prev abs", rec_prev + 1+ circuit.num_measurements)
                                    circuit.append("DETECTOR", [
                                        stim.target_rec(rec_current), stim.target_rec(rec_prev)])
                                else:
                                    #if stabilizer changed shape but init and
                                    # stabilizer are same basis, add trivial detector
                                    self.add_trivial_detector_changed_shape(
                                        stab, prism_pipe, prism_pipes_stabs,
                                        prism_pipes_zpm_temp, meas_rec_lst, circuit, z, r
                                    )
                                    #triple detector in basis in which we do not initialize
                                    # (i.e. stabilizers which can form horizontal cs)
                                    # e.g. weight-4 in previous layer + current weight-6 and current weight-2
                                    self.add_triple_detector_changed_shape_merge_opposite(
                                        stab,
                                        meas_rec_lst, circuit,
                                        z, r, rounds)
                                    #in the same basis as initialization just add double detectors
                                    self.add_double_detector_changed_shape_merge_same(
                                        stab,
                                        meas_rec_lst,
                                        circuit,
                                        z,
                                        r,
                                        rounds
                                    )
                                    if len(meas_rec_lst_data) != 0:
                                        #triple detector in the basis of data qubit measurements
                                        #compare former weight-6 with current weight-4 and data meas on location of weight-2
                                        #but not weight-2 stabilizer because this does not exist in this basis
                                        #during split
                                        print("meas rec lst data")
                                        for m in meas_rec_lst_data:
                                            print(m.label)
                                        self.add_triple_detector_changed_shape_split_same(
                                            stab,
                                            meas_rec_lst,
                                            circuit,
                                            z,
                                            r,
                                            rounds,
                                            meas_rec_lst_data
                                        )
                                        #triple detector in the opposite basis
                                        #compare former weight-6 with former weight-2 and current weight-4
                                        #during split
                                        self.add_triple_detector_changed_shape_split_opposite(
                                            stab,
                                            meas_rec_lst,
                                            circuit,
                                            z,
                                            r,
                                            rounds,
                                            meas_rec_lst_data
                                        )


                    else: #compare this with previous round meas
                        for stab in stabs:
                            #==usual double detectors that compare current and previous round==
                            if stab.stab_type in {"XZ", "Z"}:
                                # z stabilizer
                                rec_current = next(
                                    m for m in meas_rec_lst
                                    if m.meas_type == "MZ"
                                    and m.pipe_prism == prism_pipe
                                    and m.z_value == z
                                    and m.stabilizer == stab
                                    and m.round == r
                                ).abs_rec - 1 - circuit.num_measurements

                                rec_prev = next(
                                    m for m in meas_rec_lst
                                    if m.meas_type == "MZ"
                                    and m.pipe_prism == prism_pipe
                                    and m.z_value == z
                                    and m.stabilizer == stab
                                    and m.round == r - 1
                                ).abs_rec - 1 - circuit.num_measurements

                                circuit.append("DETECTOR", [
                                    stim.target_rec(rec_current), stim.target_rec(rec_prev)])
                            if stab.stab_type in {"XZ", "X"}:
                                # x stabilizer
                                rec_current = next(
                                    m for m in meas_rec_lst
                                    if m.meas_type == "MX"
                                    and m.pipe_prism == prism_pipe
                                    and m.z_value == z
                                    and m.stabilizer == stab
                                    and m.round == r
                                ).abs_rec - 1 -circuit.num_measurements

                                rec_prev = next(
                                    m for m in meas_rec_lst
                                    if m.meas_type == "MX"
                                    and m.pipe_prism == prism_pipe
                                    and m.z_value == z
                                    and m.stabilizer == stab
                                    and m.round == r - 1
                                ).abs_rec - 1 -circuit.num_measurements

                                circuit.append("DETECTOR", [
                                    stim.target_rec(rec_current), stim.target_rec(rec_prev)])
                    #if, not elif because above else should be done nevertheless too.
                    if r == rounds-1:
                        #==create detectors for the intermediate measurements==
                        circuit = self.create_data_measurement_detectors(
                            prism_pipe, stabs, zpm, meas_rec_lst, circuit, z, r
                            )


            #----------final measurements + OBS--------------
            if z == max(self.z_values):
                #==final measurements==
                for prism_pipe in prism_pipes_zpm_temp.keys():
                    zpm = prism_pipes_zpm_temp[prism_pipe]
                    stabs = prism_pipes_stabs[prism_pipe].stabilizers
                    data_positions = prism_pipes_stabs[prism_pipe].positions
                    if zpm.m == BasisPrism.Z:
                        for data in data_positions:
                            label = data.label
                            circuit.append("X_ERROR", label, p_meas)
                            circuit.append("M", label)
                            meas_rec_lst.append(
                            MeasRecInfo(
                                meas_type = "MZ",
                                pipe_prism = prism_pipe,
                                stabilizer = None,
                                abs_rec = circuit.num_measurements,
                                z_value = z,
                                round = r,
                                label = label,
                                tick = current_tick
                                )
                            ) #add to record
                    elif zpm.m == BasisPrism.X:
                        for data in data_positions:
                            label = data.label
                            circuit.append("Z_ERROR", label, p_meas)
                            circuit.append("MX", label)
                            meas_rec_lst.append(
                            MeasRecInfo(
                                meas_type = "MX",
                                pipe_prism = prism_pipe,
                                stabilizer = None,
                                abs_rec = circuit.num_measurements,
                                z_value = z,
                                round = r,
                                label = label,
                                tick = current_tick
                                )
                            ) #add to record
                    else:
                        raise TQECError("The last z requires data measurements in zpm.m")
                #==final round of detectors based on stabilizers based on zpm.m==
                for prism_pipe in prism_pipes_zpm_temp.keys():
                    zpm = prism_pipes_zpm_temp[prism_pipe]
                    stabs = prism_pipes_stabs[prism_pipe].stabilizers
                    data_positions = prism_pipes_stabs[prism_pipe].positions
                    circuit = self.create_data_measurement_detectors(
                        prism_pipe, stabs, zpm, meas_rec_lst, circuit, z, r
                        )
                    #==correlation surface for observable==
                    num_meas = circuit.num_measurements
                    meas_rec_vertical, meas_rec_horizontal = self.identify_correlation_surface_measurements(horizontal_pipes_cs, meas_rec_lst, num_meas)
                    #==observable==
                    star_ops_x, star_ops_z = self.result_dct[z]["star"]
                    star_ops_x = [[Position3DHex(p.x, p.y, 0) for p in op] for op in star_ops_x]
                    star_ops_z = [[Position3DHex(p.x, p.y, 0) for p in op] for op in star_ops_z]
                    if zpm.m == BasisPrism.Z:
                        data_hex_set = {pe.hex for pe in data_positions if pe.hex is not None}
                        star_op_z = next(
                            (op for op in star_ops_z if set(op) & data_hex_set),
                            None
                        )
                        if star_op_z is not None:
                            obs_targets = []
                            for pos in star_op_z:
                                pos_flat = Position3DHex(pos.x, pos.y, 0)
                                pe = next(p for p in data_positions if p.hex == pos_flat)
                                rec = next(
                                    m for m in meas_rec_lst
                                    if m.meas_type == "MZ"
                                    and m.z_value == z
                                    and m.stabilizer is None
                                    and m.round == r
                                    and m.label == pe.label
                                ).abs_rec - 1 - circuit.num_measurements
                                obs_targets.append(stim.target_rec(rec))
                            #add correlation surface measurements!
                            for (key_type, key_pipe), meas_lst in meas_rec_vertical.items():
                                if key_type == BasisPrism.Z:
                                    obs_targets += meas_lst
                            for (key_type, key_pipe), meas_lst in meas_rec_horizontal.items():
                                if key_type == BasisPrism.Z:
                                    obs_targets += meas_lst
                            circuit.append("OBSERVABLE_INCLUDE", obs_targets, 0)
                        else:
                            raise TQECError("No star op at final z value available.")
                    elif zpm.m == BasisPrism.X:
                        data_hex_set = {pe.hex for pe in data_positions if pe.hex is not None}
                        star_op_x = next(
                            (op for op in star_ops_x if set(op) & data_hex_set),
                            None
                        )
                        if star_op_x is not None:
                            obs_targets = []
                            for pos in star_op_x:
                                pos_flat = Position3DHex(pos.x, pos.y, 0)
                                pe = next(p for p in data_positions if p.hex == pos_flat)
                                rec = next(
                                    m for m in meas_rec_lst
                                    if m.meas_type == "MX"
                                    and m.z_value == z
                                    and m.stabilizer is None
                                    and m.round == r
                                    and m.label == pe.label
                                ).abs_rec - 1 - circuit.num_measurements
                                obs_targets.append(stim.target_rec(rec))
                            for (key_type, key_pipe), meas_lst in meas_rec_vertical.items():
                                if key_type == BasisPrism.X:
                                    obs_targets += meas_lst
                            for (key_type, key_pipe), meas_lst in meas_rec_horizontal.items():
                                if key_type == BasisPrism.X:
                                    obs_targets += meas_lst
                            circuit.append("OBSERVABLE_INCLUDE", obs_targets, 0)
                        else:
                            raise TQECError("No star op at final z value available.")

        return circuit, meas_rec_lst

    def run_all_superdense(
        self,
        rounds,
        p_init: float,
        p_meas: float,
        p_idle: float,
        p_gate2: float,
        cs: CorrelationSurface
        ):
        """Run everything for a superdense circuit."""
        self.retrieve_stabilizers_operators()
        self.reorder_all_stabilizers()
        self.meas_prep_data_qubits()
        _ = self.hex_mapping_to_quadratic()
        circuit, meas_rec_lst = self.create_stim_circuit_bell_multiplexing(
            rounds = rounds,
            p_init = p_init,
            p_meas = p_meas,
            p_idle = p_idle,
            p_gate2 = p_gate2,
            cs = cs
        )
        self.meas_rec_lst = meas_rec_lst
        return circuit



decoder_name = "tesseract"
decoder_dict = tesseract_decoder.make_tesseract_sinter_decoders_dict()

def run_experiment_sinter(
        circuit_builders: list,
        rounds: list[int],
        p_values: list[float],
        num_workers: int = 2,
        max_shots: int = 10_000,
        max_errors: int = 100,
        path: str = "default.csv",
        add_missing_detectors: bool = False,
        se_type: str = "run_all",
        cs = []#empty
) -> list[sinter.TaskStats]:
    tasks = []
    for idx, circuit_builder in enumerate(circuit_builders):
        for p in p_values:
            if se_type == "run_all":
                method = getattr(circuit_builder, se_type)
                circuit = method(
                    rounds=rounds[idx],
                    p_init=p,
                    p_gate2=p,
                    p_meas=p,
                    p_idle=p,
                )
            elif se_type == "run_all_superdense":
                method = getattr(circuit_builder, se_type)
                circuit = method(
                    rounds=rounds[idx],
                    p_init=p,
                    p_gate2=p,
                    p_meas=p,
                    p_idle=p,
                    cs = cs
                )

            if add_missing_detectors:
                circuit = circuit + circuit.missing_detectors(unknown_input=False)
            tasks.append(sinter.Task(
                circuit=circuit,
                json_metadata={'p': p, 'rounds': rounds[idx], 'code_idx': idx, 'd': circuit_builder.d},
            ))

    stats = sinter.collect(
        num_workers=num_workers,
        tasks=tasks,
        custom_decoders=decoder_dict,
        decoders=[decoder_name],
        max_shots=max_shots,
        max_errors=max_errors,
        print_progress=True,
        save_resume_filepath=path
    )
    return stats


def plot_experiment_sinter(stats: list[sinter.TaskStats], d_lst: list[int]):
    fig, ax = plt.subplots()
    sinter.plot_error_rate(
        ax=ax,
        stats=stats,
        x_func=lambda stat: stat.json_metadata['p'],
        group_func=lambda stat: f"d={d_lst[stat.json_metadata['code_idx']]}, r={stat.json_metadata['rounds']}",
    )
    ax.set_xlabel("Physical error rate p")
    ax.set_ylabel("Logical error rate")
    ax.set_title("Logical error rate vs physical error rate")
    ax.loglog()
    ax.grid(True, which='both', ls='--', alpha=0.5)
    ax.legend(title="Code distance")
    plt.tight_layout()
    plt.show()
    return fig
