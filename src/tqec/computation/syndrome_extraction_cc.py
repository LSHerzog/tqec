import re
from collections import defaultdict
from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sinter
import stim
import tesseract_decoder

from tqec.utils.exceptions import TQECError

from tqec.computation.pipe_prism import PrismPipe, PrismPipeKind
from tqec.computation.prism import BasisPrism, Port, Position3DHex, Prism, ZXPrism
from tqec.computation.prism_graph import PrismGraph, StabilizerProductResult


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
        """Construct a mapping of ancilla labels (> mapping labels) with their stabilizer affiliations."""
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
) -> list[sinter.TaskStats]:
    tasks = []
    for idx, circuit_builder in enumerate(circuit_builders):
        for p in p_values:
            circuit = circuit_builder.run_all(
                rounds=rounds[idx],
                p_init=p,
                p_gate2=p,
                p_meas=p,
                p_idle=p,
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
