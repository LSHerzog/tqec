import re
from collections import defaultdict
from dataclasses import dataclass

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sinter
import stim
import tesseract_decoder

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
            rounds,
            p_init: float,
            p_meas: float,
            p_idle: float,
            p_gate2: float) -> stim.Circuit:
        """Create a stim circuit of the object."""
        #! currently no horizontal correlation surfaces possible yet!
        #! summarize input from correlationsurface directly!

        #! logical_operator must be an operator on a full patch! otherwise not all detectors created.

        circuit = stim.Circuit()

        horizontal_cs_x_list = []
        horizontal_cs_z_list = []

        for z in self.z_values:
            print(f"building circuit for z={z}...")
            #use both prism_pipes_to_ZPM and prism_pipes_to_data_qubits_full
            prism_pipes_zpm_temp = self.prism_pipes_to_ZPM[z]
            prism_pipes_data_temp = self.prism_graph.prism_pipes_to_data_qubits_full[z]

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
            if self.d>5:
                result = self.result_dct[z]["product"]
                star_ops_x, star_ops_z = self.result_dct[z]["star"]
            else:
                print("No horizontal CS available for d<=5. your circuit will be wrong.")
                star_ops_x, star_ops_z = None, None
                result = None


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
                if self.d>5:
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
                horizontal_cs_x_list.append(horizontal_cs_x)

                #-------x stabilizer detectors---------
                for anc_idx in range(n_ancillas):
                    rec_current = -(n_ancillas - anc_idx)
                    if r == 0:
                        # first round: only deterministic if initialized in X basis
                        if zpm.p == BasisPrism.X:
                            circuit.append("DETECTOR", [stim.target_rec(rec_current)])
                            print("X detector time start:", rec_current)
                    else:
                        # middle rounds: always valid regardless of zpm.p
                        rec_prev = rec_current - meas_per_round
                        circuit.append("DETECTOR", [
                            stim.target_rec(rec_current),
                            stim.target_rec(rec_prev)
                        ])
                        print("X detector bulk:", rec_current, rec_prev)

                #--------Z syndrome extraction----------
                #initialize ancillas in Z
                circuit.append("R", mapped_ancilla_qubits)
                circuit.append("X_ERROR", mapped_ancilla_qubits, p_init)
                circuit.append("TICK", [])

                #collect the measurement indices for a horizontal correlation surface
                horizontal_cs_z = []
                stabilizer_products_z = None
                stabilizer_product_z_ancillas = None
                if self.d>5:
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
                        abs_idx = m_z_start + pos_in_list
                        horizontal_cs_z.append(abs_idx)
                print("horizontal_cs_z", horizontal_cs_z)
                horizontal_cs_z_list.append(horizontal_cs_z)

                #-------z stabilizer detectors---------
                for anc_idx in range(n_ancillas):
                    rec_current = -(n_ancillas - anc_idx)
                    if r == 0:
                        # first round: only deterministic if initialized in Z basis
                        if zpm.p == BasisPrism.Z:
                            circuit.append("DETECTOR", [stim.target_rec(rec_current)])
                            print("Z detector time start:", rec_current)
                    else:
                        # middle rounds: always valid regardless of zpm.p
                        rec_prev = rec_current - meas_per_round
                        circuit.append("DETECTOR", [
                            stim.target_rec(rec_current),
                            stim.target_rec(rec_prev)
                        ])
                        print("Z detector bulk:", rec_current, rec_prev)

            offset_start = circuit.num_measurements

            #check whether some zpm.m is Z or X and then measure accordingly.
            for prism_pipe in prism_pipes_zpm_temp.keys():
                zpm = prism_pipes_zpm_temp[prism_pipe]
                data_qubits = prism_pipes_data_temp[prism_pipe]
                mapped_data_qubits = [self.mapping[Position3DHex(x=el.x, y=el.y, z=0)] for el in data_qubits]
                if zpm.m == BasisPrism.Z:
                    print(f"Measure Z, {prism_pipe}")
                    circuit.append("X_ERROR", mapped_data_qubits, p_meas)
                    circuit.append("M", mapped_data_qubits)
                elif zpm.m == BasisPrism.X:
                    print(f"Measure X, {prism_pipe}")
                    circuit.append("Z_ERROR", mapped_data_qubits, p_meas)
                    circuit.append("MX", mapped_data_qubits)

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
                        #!NO VERTICAL CS INCLUDED YET
                    elif zpm.m == BasisPrism.X:
                        if star_ops_x is not None:
                            current_star = star_ops_x[0]
                            current_hor_lst = horizontal_cs_x_list
                    if current_star is not None:
                        current_star = [Position3DHex(el.x,el.y,0) for el in current_star]

                        print("current star", current_star)
                        print("current_hor_lst", current_hor_lst)

                    obs_targets = []
                    for i, qubit in enumerate(mapped_data_qubits):
                        if current_star is not None:
                            if qubit in current_star:
                                offset = -(total - offset_start - i)
                                obs_targets.append(stim.target_rec(offset))
                                print("added star qubit to OBS", offset)
                        else: #use the whole patch as obs if no star op
                            offset = -(total - offset_start - i)
                            obs_targets.append(stim.target_rec(offset))

                    #add parity of corresponding horizontal cs:
                    if current_hor_lst is not None:
                        for horizontal_cs in current_hor_lst:
                            for el in horizontal_cs:
                                offset = -(total - el)
                                obs_targets.append(stim.target_rec(offset))
                                print("from horizontal CS added", offset)
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
                            rec_last_anc = -(total - offset_start + n_ancillas - anc_idx)
                            stab_data_qubits = [self.mapping[q] for q in stab]
                            data_targets = []
                            for q in stab_data_qubits:
                                if q in mapped_data_qubits:
                                    pos_in_obs = mapped_data_qubits.index(q)
                                    data_targets.append(stim.target_rec(-(total - offset_start - pos_in_obs)))
                            circuit.append("DETECTOR", [stim.target_rec(rec_last_anc)] + data_targets)
                            print("final X detector:", rec_last_anc, data_targets)


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
        rounds: int,
        p_values: list[float],
        num_workers: int = 2,
        max_shots: int = 10_000,
        max_errors: int = 100,
) -> list[sinter.TaskStats]:
    """
    Run a logical error rate experiment for a range of noise levels p using sinter + tesseract.
    All noise parameters are set to p.
    Loops over circuit_builders, logical_operators, and p_values.
    circuit_builders and logical_operators are assumed to be paired (same index = same code).
    """
    tasks = [
        sinter.Task(
            circuit=circuit_builder.run_all(
                rounds=rounds,
                p_init=p,
                p_gate2=p,
                p_meas=p,
                p_idle=p,
            ),
            json_metadata={'p': p, 'rounds': rounds, 'code_idx': idx, 'd': circuit_builder.d},
        )
        for idx, circuit_builder in enumerate(circuit_builders)
        for p in p_values
    ]

    stats = sinter.collect(
        num_workers=num_workers,
        tasks=tasks,
        custom_decoders=decoder_dict,
        decoders=[decoder_name],
        max_shots=max_shots,
        max_errors=max_errors,
        print_progress=True,
    )

    return stats


def plot_experiment_sinter(stats: list[sinter.TaskStats], d_lst: list[int]):
    """Plot sinter tasks."""
    fig, ax = plt.subplots()

    sinter.plot_error_rate(
        ax=ax,
        stats=stats,
        x_func=lambda stat: stat.json_metadata['p'],
        group_func=lambda stat: f"d={d_lst[stat.json_metadata['code_idx']]}",
    )

    ax.set_xlabel("Physical error rate p")
    ax.set_ylabel("Logical error rate")
    ax.set_title("Logical error rate vs physical error rate")
    ax.loglog()
    ax.grid(True, which='both', ls='--', alpha=0.5)
    ax.legend(title="Code distance")
    plt.tight_layout()
    plt.show()
