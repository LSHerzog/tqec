import re
from collections import defaultdict
from dataclasses import dataclass

import networkx as nx
import numpy as np
import stim

from tqec.computation.pipe_prism import PrismPipe, PrismPipeKind
from tqec.computation.prism import BasisPrism, Port, Position3DHex, Prism, ZXPrism
from tqec.computation.prism_graph import PrismGraph


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
            if self.d > 5: #not possible for d=3, d=5
                star_ops_x, star_ops_z = self.prism_graph.star_operator_timeslice(z, self.d)
                result = self.prism_graph.stabilizer_product_timeslice(z, self.d, dct_single_type_stabs, dct_patch_stabilizers, testing = True)
                result_dct[z].update({"star": [star_ops_x, star_ops_z]})
                result_dct[z].update({"product": result})
            #result_dct[z].update({"assignment": self.prism_graph.prism_pipes_to_data_qubits_full})
        self.result_dct = result_dct

    def create_mapping(self):
        """Create a mapping of each data_qubit position to an int."""
        positions = set()
        for z in self.z_values:
            [stabs_x, stabs_z] = self.result_dct[z]["stabs"]
            stabs = stabs_x + stabs_z #double stabs removed
            positions_temp = {item for sublist in stabs for item in sublist}
            #set all z values to 0, because we want the mapping agnostically of z
            positions_temp = {Position3DHex(x = pos.x, y = pos.y, z = 0) for pos in positions_temp}
            positions.update(positions_temp)
        self.mapping = {value: key for key,value in enumerate(positions)}

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

    def create_stim_circuit_naive(
            self,
            #star_idx:int,
            logical_operator: list[Position3DHex],
            rounds,
            after_clifford_depolarization: float,
            before_round_data_depolarization: float,
            after_reset_flip_probability: float,
            before_measure_flip_probability: float,) -> stim.Circuit:
        """Create a stim circuit of the object."""
        #! currently no horizontal correlation surfaces possible yet!
        #! summarize input from correlationsurface directly!
        #! right now, operator_patch only one position, i.e. only straight vertical operators possible
        #! type is "z" or "x"

        circuit = stim.Circuit()

        for z in self.z_values:
            #use both prism_pipes_to_ZPM and prism_pipes_to_data_qubits_full
            prism_pipes_zpm_temp = self.prism_pipes_to_ZPM[z]
            prism_pipes_data_temp = self.prism_graph.prism_pipes_to_data_qubits_full[z]

            #initialize
            for prism_pipe in prism_pipes_zpm_temp.keys():
                zpm = prism_pipes_zpm_temp[prism_pipe]
                data_qubits = prism_pipes_data_temp[prism_pipe]
                mapped_data_qubits = [self.mapping[el] for el in data_qubits]
                if zpm.p == BasisPrism.Z:
                    circuit.append("R", mapped_data_qubits)
                elif zpm.p == BasisPrism.X:
                    circuit.append("RX", mapped_data_qubits)
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


            for _ in range(rounds):
                #! make this a separate method such that you can interchange the syndrome extraction schemes
                mapped_ancilla_qubits = self.mapping_ancillas.values()
                #--------X syndrome extraction----------
                #initialize ancillas in X
                circuit.append("RX", mapped_ancilla_qubits)
                circuit.append("TICK", [])
                for edges in edge_coloring_z.values():
                    #from labels "c+int" find the corresponding stabilizer and the ancilla label
                    #"q+int" describes data qubit label
                    #data qubit label = target, ancilla qubit label = control
                    for edge in edges:
                        q_int = int(edge[0][1:])
                        c_int = int(edge[1][1:])
                        corresponding_stab = self.canonical(stabs_x[c_int])
                        ancilla_int = self.mapping_ancillas[corresponding_stab]
                        circuit.append("CNOT", [ancilla_int, q_int])
                    circuit.append("TICK", [])
                #measure all ancillas in X
                circuit.append("MX", mapped_ancilla_qubits)
                circuit.append("TICK", [])


                #--------Z syndrome extraction----------
                #initialize ancillas in Z
                circuit.append("R", mapped_ancilla_qubits)
                circuit.append("TICK", [])
                for edges in edge_coloring_z.values():
                    #from labels "c+int" find the corresponding stabilizer and the ancilla label
                    #"q+int" describes data qubit label
                    #data qubit label = control, ancilla qubit label = target
                    for edge in edges:
                        q_int = int(edge[0][1:])
                        c_int = int(edge[1][1:])
                        corresponding_stab = self.canonical(stabs_z[c_int])
                        ancilla_int = self.mapping_ancillas[corresponding_stab]
                        circuit.append("CNOT", [q_int, ancilla_int])
                    circuit.append("TICK", [])
                #measure all ancillas in Z
                circuit.append("M", mapped_ancilla_qubits)
                circuit.append("TICK", [])

            offset_start = circuit.num_measurements

            #measure for that prism graph timestep #!ONLY FOR STAR OPS
            #for prism_pipe in prism_pipes_zpm_temp.keys():
            #    zpm = prism_pipes_zpm_temp[prism_pipe]
            #    data_qubits = prism_pipes_data_temp[prism_pipe]
            #    mapped_data_qubits = [self.mapping[el] for el in data_qubits]
            #    final_measurement_order.extend(mapped_data_qubits)
            #    if zpm.m == BasisPrism.Z:
            #        circuit.append("M", mapped_data_qubits)
            #    elif zpm.m == BasisPrism.X:
            #        circuit.append("MX", mapped_data_qubits)


            #since zpm.m should be the same for a layer take last zpm.m
            #zpm = prism_pipes_zpm_temp[prism_pipe]#some previous prism_pipe
            #if zpm.m == BasisPrism.X:
            #    star_ops = self.result_dct[z]["star"][0]
            #elif zpm.m == BasisPrism.Z:
            #    star_ops = self.result_dct[z]["star"][1]

            # find data qubits belonging to the logical operator patches
            obs_qubits = [self.mapping[Position3DHex(p.x, p.y, 0)] for p in logical_operator]

            # only measure the star operator qubits
            if zpm.m == BasisPrism.Z:
                circuit.append("M", obs_qubits)
            elif zpm.m == BasisPrism.X:
                circuit.append("MX", obs_qubits)

            total = circuit.num_measurements

            # build OBS_INCLUDE targets using measurement record offsets
            obs_targets = []
            for i, qubit in enumerate(obs_qubits):
                offset = -(total - offset_start - i)
                obs_targets.append(stim.target_rec(offset))

            circuit.append("OBSERVABLE_INCLUDE", obs_targets, 0)

        return circuit



    def run_all(self,
            logical_operator,
            rounds,
            after_clifford_depolarization: float,
            before_round_data_depolarization: float,
            after_reset_flip_probability: float,
            before_measure_flip_probability: float,):
        """Run all methods."""
        self.retrieve_stabilizers_operators()
        self.create_mapping()
        self.reorder_all_stabilizers()
        self.create_mapping_ancillas()
        self.meas_prep_data_qubits()

        circuit = self.create_stim_circuit_naive(
            logical_operator,
            rounds,after_clifford_depolarization,
            before_round_data_depolarization,
            after_reset_flip_probability,
            before_measure_flip_probability)
        return circuit
