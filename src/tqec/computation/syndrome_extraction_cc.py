import re
from collections import Counter
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import sinter
import stim
import tesseract_decoder

from tqec.computation.correlation import CorrelationSurface
from tqec.computation.pipe_prism import PrismPipe
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
    """Store infos about stabilizers."""

    data_qubits: list[PositionEntry] | None
    top: list[PositionEntry] | None
    bottom: list[PositionEntry] | None
    sides: list[PositionEntry] | None
    ancilla: list[PositionEntry] | None
    stab_type: str | None


@dataclass
class PrismData:
    """Store data about prisms."""

    positions: list[PositionEntry] = field(default_factory=list)
    stabilizers: list[StabilizerInfo] | None = field(default_factory=list)


@dataclass
class MeasRecInfo:
    """Track measurements in the stim circuit via this class."""

    meas_type: str  # ancilla_mz, ancilla_mx, data_mz, data_mx
    pipe_prism: PrismPipe | Prism
    stabilizer: StabilizerInfo | None
    abs_rec: int
    z_value: int
    round: int
    label: int  # qubit label on which the M is done
    tick: int


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
    """Generate Stim syndrome-extraction circuits for color code pipe diagrams.

    The class translates a :class:`PrismGraph` representation of a computation
    into a fault-tolerant Stim circuit. Central is the method `run_all_superdense`,
    which creates the circuit. As a user, this is the only method which is needed.

    Parameters
    ----------
    prism_graph:
        Prism-graph description of the computation.
    d:
        Code distance of the color code represented by the prism graph.

    Notes
    -----
    The implementation assumes a valid prism graph without open ports and is
    designed for simulations of color-code computations represented by prism
    diagrams. Spatial endings are not yet supported. It is assumed that data
    measurements are performed at the max z value of the diagram.

    """

    def __init__(self, prism_graph: PrismGraph, d: int):
        """Construct syndrome extraction circuits for a prism_graph."""
        self.prism_graph = prism_graph
        self.d = d
        self.z_values = sorted({pos.z for pos in self.prism_graph._graph.nodes})

    def retrieve_stabilizers_operators(self) -> None:
        """Retrieve all stabilizers and logical ooperators for all z in the prism_graph."""
        result_dct = {z: dict() for z in self.z_values}
        # each of those dicts will contain stabilizers, operator products and star operators
        for z in self.z_values:
            (
                stabs_x,
                stabs_z,
                _,
                dct_single_type_stabs,
                dct_patch_stabilizers,
            ) = self.prism_graph.stabilizers_timeslice(z, self.d)
            result_dct[z].update({"stabs": [stabs_x, stabs_z]})
            # if self.d > 5: #not possible for d=3, d=5
            star_ops_x, star_ops_z = self.prism_graph.star_operator_timeslice(z, self.d)
            try:
                result = self.prism_graph.stabilizer_product_timeslice(
                    z, self.d, dct_single_type_stabs, dct_patch_stabilizers, testing=True
                )
            except TQECError as e:
                # for d=3 the 3 coloring cannot be created for a single patch as it
                # does not contain weight 6 stabilizers
                # but we do not care and skip this because for a single patch
                # there's no stabilizer product anyway
                allowed_messages = [
                    "No weight-6 stabilizer found to seed the 3-coloring.",
                    "Could not resolve coloring — stuck with ambiguous boundary stabilizers.",
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
        self.result_dct = result_dct

    @staticmethod
    def canonical(stab: tuple[Position3DHex, ...]) -> tuple[Position3DHex, ...]:
        """Sort positions to make order irrelevant."""
        return tuple(sorted(stab, key=lambda p: (p.x, p.y, p.z)))

    def reorder_stabilizers(self, z: int) -> dict[tuple[Position3DHex, ...], str]:
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

    def reorder_all_stabilizers(self) -> None:
        """Apply reordering for all z."""
        dct_stabilizers_all = {}
        for z in self.z_values:
            dct_stabilizers = self.reorder_stabilizers(z)
            dct_stabilizers_all.update({z: dct_stabilizers})
        self.dct_stabilizers_all = dct_stabilizers_all

    def meas_prep_data_qubits(self) -> None:
        """Find meas/prep instructions for each data qubit for the whole pipe diagram."""
        # retrieve assignment of data qubits per patch / prism
        prism_pipes_to_data_qubits_full = self.prism_graph.prism_pipes_to_data_qubits_full
        # collect ZPM per prism or pipe
        prism_pipes_to_ZPM = dict()  # noqa: N806
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
                temp_dct.update({pipe_prism: zpm})
            prism_pipes_to_ZPM.update({z: temp_dct})
        # note that prism_pipes_to_data_qubits_full and prsim_pipes_to_ZPM
        # should be used together in create_stim_circuit
        self.prism_pipes_to_ZPM = prism_pipes_to_ZPM

    @staticmethod
    def get_zpm_for_stab(
        stab: tuple[Position3DHex, ...],
        prism_pipes_data_temp: dict[Prism | PrismPipe, list[Position3DHex]],
        prism_pipes_zpm_temp: dict[Prism | PrismPipe, ZPM],
        stabs_x: list[list[Position3DHex]] | list[tuple[Position3DHex, ...]],
        stabs_z: list[list[Position3DHex]] | list[tuple[Position3DHex, ...]],
    ) -> ZPM | None:
        """Assign ZPM value from given stab."""
        stab_set = set(stab)

        # normalize to z=0
        stabs_x = [[Position3DHex(el.x, el.y, 0) for el in stab] for stab in stabs_x]
        stabs_z = [[Position3DHex(el.x, el.y, 0) for el in stab] for stab in stabs_z]

        in_x = any(stab_set == set(s) for s in stabs_x)
        in_z = any(stab_set == set(s) for s in stabs_z)

        # normalize pipe vs prism search space
        if in_x and in_z:
            allowed_types = (Prism,)
        else:
            allowed_types = (PrismPipe,)

        for pipe_prism, data_qubits in prism_pipes_data_temp.items():
            if not isinstance(pipe_prism, allowed_types):
                continue

            dq_set = {Position3DHex(q.x, q.y, 0) for q in data_qubits}

            if stab_set & dq_set:
                return prism_pipes_zpm_temp[pipe_prism]

        return None

    @staticmethod
    def assign_stabilizer_info(stab_tup: tuple[Position3DHex, ...]) -> StabilizerInfo:
        """Assign the stabilizer info for a stab given in hex coordinates.

        This covers info about different weight-4 stabilizers and weight-6 stabilizers
        weight-2 stabilizers should not be covered by this.
        """
        # make sure that z=0
        stab = [Position3DHex(pos.x, pos.y, 0) for pos in stab_tup]
        stab_rec = [pos.rectangular_map() for pos in stab]
        y_values = sorted({pos[1] for pos in stab_rec})  # available rec y values
        position_entries = [
            PositionEntry(hex=pos_hex, rect=pos_rec, label=None)
            for pos_hex, pos_rec in zip(stab, stab_rec)
        ]
        if len(stab) == 2:  # minimal assignment for weight-2 stabilizers
            return StabilizerInfo(
                data_qubits=position_entries,
                top=None,
                bottom=None,
                sides=None,
                ancilla=None,
                stab_type=None,
            )
        if len(y_values) == 3:
            # standard simple case
            top = sorted(
                [pe for pe in position_entries if pe.rect[1] == y_values[2]],
                key=lambda pe: pe.rect[0],
            )
            side = sorted(
                [pe for pe in position_entries if pe.rect[1] == y_values[1]],
                key=lambda pe: pe.rect[0],
            )
            bottom = sorted(
                [pe for pe in position_entries if pe.rect[1] == y_values[0]],
                key=lambda pe: pe.rect[0],
            )
            # assign ancillas
            if len(side) == 2:
                a = side[0].rect[0]
                b = side[-1].rect[0]
                ends = sorted([a, b])
                x_values = list(range(ends[0] + 1, ends[1]))
                y_rec = side[0].rect[1]  # y is fix
                ancillas_rec = [(x, y_rec) for x in x_values]
                ancillas = [
                    PositionEntry(hex=None, rect=rec, label=None) for rec in sorted(ancillas_rec)
                ]
            elif len(side) == 1:
                # this is a sideways weight-4 stabilizer which is a bit deformed
                # there are 4 different shapes of that kind and they are covered by the following
                if len(bottom) == 2:
                    y_rec = side[0].rect[1]
                    x_values = [pe.rect[0] for pe in bottom]
                    ancillas_rec = [(x, y_rec) for x in x_values]
                    ancillas = [
                        PositionEntry(hex=None, rect=rec, label=None) for rec in ancillas_rec
                    ]
                elif len(top) == 2:
                    y_rec = side[0].rect[1]
                    x_values = [pe.rect[0] for pe in top]
                    ancillas_rec = [(x, y_rec) for x in x_values]
                    ancillas = [
                        PositionEntry(hex=None, rect=rec, label=None) for rec in ancillas_rec
                    ]
                # this can also be a weight-3 stabilizer
                elif len(bottom) == 1 and len(top) == 1:
                    y_rec = side[0].rect[1]
                    x_values = [bottom[0].rect[0], bottom[0].rect[0] + 1]
                    ancillas_rec = [(x, y_rec) for x in x_values]
                    ancillas = [
                        PositionEntry(hex=None, rect=rec, label=None) for rec in ancillas_rec
                    ]
        elif len(y_values) == 2:
            row_a = sorted(
                [pe for pe in position_entries if pe.rect[1] == y_values[1]],
                key=lambda pe: pe.rect[0],
            )
            row_b = sorted(
                [pe for pe in position_entries if pe.rect[1] == y_values[0]],
                key=lambda pe: pe.rect[0],
            )
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
                if len(side) == 2:  # weight-4 stabilizers
                    # weight-4 stabilizers
                    ends = sorted([side[0].rect[0], side[-1].rect[0]])
                    x_values = list(range(ends[0] + 1, ends[1]))
                    y_rec = side[0].rect[1]
                    ancillas_rec = [(x, y_rec) for x in x_values]
                    ancillas = [
                        PositionEntry(hex=None, rect=rec, label=None) for rec in ancillas_rec
                    ]
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
                # determine top/bottom based on which y values are larger
                if top_bottom[0].rect[1] > y_rec:
                    top = top_bottom
                    bottom = []
                else:
                    top = []
                    bottom = top_bottom
        else:
            raise ValueError("wrong")

        return StabilizerInfo(
            data_qubits=position_entries,
            top=top,
            sides=side,
            bottom=bottom,
            ancilla=ancillas,
            stab_type=None,
        )

    def hex_mapping_to_quadratic(self) -> dict[int, dict[Prism | PrismPipe, PrismData]]:
        """Map the hexagonal layout to a quadratic grid.

        This creates a mapping for both the data and ancilla qubits
        based on Position3DHex.rectangular_map() for the data qubits.
        """
        # first find stabilizerinfo for each stabilizer
        # (no label assignment yet, no prism/pipe assignment yet)
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

        # find integer values for data qubits
        data_idx = 0
        seen: dict[Position3DHex, PositionEntry] = {}
        for z in self.z_values:
            dct_temp: dict[Prism | PrismPipe, PrismData] = {}

            prism_pipes_data = self.prism_graph.prism_pipes_to_data_qubits_full[z]
            # first pass: prisms only
            for pipe_prism, data_qubits in prism_pipes_data.items():
                # if not isinstance(pipe_prism, Prism):
                #    continue
                positions = []
                for pos in data_qubits:
                    flat = Position3DHex(x=pos.x, y=pos.y, z=0)
                    if flat not in seen:
                        position_entry = PositionEntry(
                            hex=flat, rect=flat.rectangular_map(), label=data_idx
                        )
                        positions.append(position_entry)
                        data_idx += 1
                        seen[flat] = position_entry
                    else:
                        positions.append(seen[flat])
                dct_temp.update(
                    {pipe_prism: PrismData(positions=positions.copy(), stabilizers=None)}
                )
            mapping_full.update({z: dct_temp})

        # adapt data_qubit labels in stab_info according to above labels
        rect_to_entry: dict[tuple, PositionEntry] = {pe.rect: pe for pe in seen.values()}
        seen_ancillas: dict[tuple, PositionEntry] = {}
        for z in self.z_values:
            stabilizers = stabilizers_per_layer[z]

            # initialize stabilizers list in each PrismData
            for prism_data in mapping_full[z].values():
                prism_data.stabilizers = []

            for stab_info in stabilizers:
                # assign labels to data qubits in stab_info from seen
                for pe in (
                    stab_info.data_qubits or []
                ):  # empty list in case the stabinfo entry isNone
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
                stab_rects = {
                    pe.rect for pe in (stab_info.data_qubits or [])
                }  # [] to make ty happy
                best_match = None
                best_overlap = 0
                for pipe_prism, prism_data in mapping_full[z].items():
                    prism_rects = {pe.rect for pe in prism_data.positions}
                    overlap = len(stab_rects & prism_rects)
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_match = pipe_prism
                if best_match is not None:
                    prism_data = mapping_full[z][best_match]
                    if prism_data.stabilizers is None:  # to make ty happy
                        prism_data.stabilizers = []
                    prism_data.stabilizers.append(stab_info)

        self.mapping_full = mapping_full
        return mapping_full

    @staticmethod
    def append_tick_with_idle_noise(
        circuit: stim.Circuit,
        all_current_qubits: list[PositionEntry],
        current_tick: int,
        p_idle: float,
        active_qubits: set[int],
    ) -> int:
        """Append idle noise and TICK to circuit."""
        all_qubits = {
            pe.label for pe in all_current_qubits if pe.label is not None
        }  # make ty happy
        idle = all_qubits - active_qubits
        if idle:
            circuit.append("DEPOLARIZE1", list(idle), p_idle)
        circuit.append("TICK", [])
        return current_tick + 1

    @staticmethod
    def get_active_qubits_since_last_tick(circuit: stim.Circuit) -> set[int]:
        """Get all qubit labels that had a gate since the last TICK."""
        active = set()
        for instruction in reversed(circuit):
            if isinstance(instruction, stim.CircuitRepeatBlock):  # because ty
                raise TQECError(
                    "Encountered an unexpected stim.CircuitRepeatBlock while scanning "
                    "backward for the last TICK. Circuit loops are not expected within the "
                    "current active syndrome extraction layer chunk."
                )
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
        # find the matching key as you cannot just take the current prism_pipe
        # and adjust pos. this is not enough
        if isinstance(prism_pipe, Prism):
            matching_key = next(
                (
                    k
                    for k in prism_pipes_stabs_previous
                    if isinstance(k, Prism)
                    and isinstance(prism_pipe_previous, Prism)  # this has to be added for ty
                    and k.position == prism_pipe_previous.position
                ),
                None,
            )
        elif isinstance(prism_pipe, PrismPipe):
            matching_key = next(
                (
                    k
                    for k in prism_pipes_stabs_previous
                    if isinstance(k, PrismPipe)
                    and isinstance(prism_pipe_previous, PrismPipe)  # ty
                    and k.u.position == prism_pipe_previous.u.position
                    and k.v.position == prism_pipe_previous.v.position
                ),
                None,
            )
        if matching_key is not None:  # ty
            stabs_previous = prism_pipes_stabs_previous[matching_key].stabilizers
        else:
            stabs_previous = None
        if not stabs_previous:
            raise ValueError("do not use this if this is the very first z layer.")

        # represent ancillas of current stab as a set of rect tuples for comparison
        if not stab.ancilla:
            raise ValueError("a stabilizer does not have ancillas.")
        current_ancilla_rects = {ancilla.rect for ancilla in stab.ancilla}

        # find the previous stabilizer that shares the same ancilla rect positions
        matching_prev_stab = None
        for prev_stab in stabs_previous:
            if not prev_stab.ancilla:
                continue
            prev_ancilla_rects = {ancilla.rect for ancilla in prev_stab.ancilla}
            if prev_ancilla_rects == current_ancilla_rects:
                matching_prev_stab = prev_stab
                break

        if matching_prev_stab is None:
            return True

        # compare data qubit support via rect tuples
        current_data_rects = {pe.rect for pe in stab.data_qubits or []}  # [] because ty
        prev_data_rects = {pe.rect for pe in matching_prev_stab.data_qubits or []}

        return current_data_rects != prev_data_rects

    def pipe_matches(self, a: Prism | PrismPipe, b: Prism | PrismPipe) -> bool:
        """Compare two prisms/pipes ignoring z coordinate."""
        if type(a) is not type(b):
            return False
        if isinstance(a, Prism) and isinstance(b, Prism):
            return a.position.x == b.position.x and a.position.y == b.position.y
        elif isinstance(a, PrismPipe) and isinstance(b, PrismPipe):
            return (
                a.u.position.x == b.u.position.x
                and a.u.position.y == b.u.position.y
                and a.v.position.x == b.v.position.x
                and a.v.position.y == b.v.position.y
            )
        else:
            return False

    def identify_correlation_surface_measurements(
        self,
        horizontal_pipes_cs: dict[PrismPipe, tuple[BasisPrism, str]],
        meas_rec_lst: list[MeasRecInfo],
        num_meas: int,
    ) -> tuple[
        dict[tuple[BasisPrism, PrismPipe], list[stim.GateTarget]],
        dict[tuple[BasisPrism, PrismPipe], list[stim.GateTarget]],
    ]:
        """Identify measurements that belong to a correlation surface."""
        # vertical correlation surface
        meas_rec_vertical: dict[tuple[BasisPrism, PrismPipe], list[stim.GateTarget]] = {}
        # horizontal correlation surface
        meas_rec_horizontal: dict[tuple[BasisPrism, PrismPipe], list[stim.GateTarget]] = {}

        # Separate hor and ver pipes, then apply the same connected-component logic to both:
        # count endpoint appearances — positions appearing once are real endpoints,
        # positions appearing twice are interior (shared between two pipes in a chain).
        hor_pipes = {p: v for p, v in horizontal_pipes_cs.items() if v[1] == "hor"}
        ver_pipes = {p: v for p, v in horizontal_pipes_cs.items() if v[1] == "ver"}

        def _count_endpoints(pipes):
            counts: Counter = Counter()
            for pipe in pipes:
                counts[pipe.u.position] += 1
                counts[pipe.v.position] += 1
            return counts

        # --- hor component: stabilizer product path between real endpoints ---
        if hor_pipes:
            hor_counts = _count_endpoints(hor_pipes)
            hor_all_positions = set(hor_counts)
            hor_real_endpoints = {pos for pos, cnt in hor_counts.items() if cnt == 1}
            cs_type_hor = next(iter(hor_pipes.values()))[0]
            z_value_hor = next(iter(hor_pipes)).u.position.z
            result = self.result_dct[z_value_hor]["product"]
            if cs_type_hor == BasisPrism.Z:
                stabilizer_product_current = result.stabilizer_products_z
                paths_current = result.paths_z
                meas_type_hor = "MZ"
            elif cs_type_hor == BasisPrism.X:
                stabilizer_product_current = result.stabilizer_products_x
                paths_current = result.paths_x
                meas_type_hor = "MX"
            else:
                raise ValueError("incorrect cs type assignment.")
            tmp: list = []
            seen_recs: set[int] = set()
            for path, cs in zip(paths_current, stabilizer_product_current):
                if {path[0], path[-1]} != hor_real_endpoints:
                    continue
                for stabilizer in cs:
                    stabilizer_flat = {Position3DHex(p.x, p.y, 0) for p in stabilizer}
                    z_val_stab = stabilizer[0].z
                    match = next(
                        (
                            m
                            for m in meas_rec_lst
                            if m.meas_type == meas_type_hor
                            and m.stabilizer is not None
                            and m.z_value == z_val_stab
                            and m.stabilizer.data_qubits is not None
                            and {
                                Position3DHex(pe.hex.x, pe.hex.y, 0)
                                for pe in m.stabilizer.data_qubits
                                if pe.hex is not None
                            }
                            == stabilizer_flat
                        ),
                        None,
                    )
                    if match is not None:
                        rec = match.abs_rec - num_meas - 1
                        if rec not in seen_recs:
                            seen_recs.add(rec)
                            tmp.append(stim.target_rec(rec))
            meas_rec_horizontal[(cs_type_hor, frozenset(hor_all_positions))] = tmp

        # --- ver component: star op measurements from all prisms in the chain ---
        if ver_pipes:
            ver_counts = _count_endpoints(ver_pipes)
            ver_all_prisms = {
                prism
                for pipe in ver_pipes
                for prism in (pipe.u, pipe.v)
            }
            cs_type_ver = next(iter(ver_pipes.values()))[0]
            z_value_ver = next(iter(ver_pipes)).u.position.z
            star_ops_x, star_ops_z = self.result_dct[z_value_ver]["star"]
            if cs_type_ver == BasisPrism.Z:
                star_ops_current = star_ops_z
                meas_type_ver = "MZ"
            elif cs_type_ver == BasisPrism.X:
                star_ops_current = star_ops_x
                meas_type_ver = "MX"
            else:
                raise ValueError("Error during construction of correlation surfaces.")
            if len(star_ops_current) == 0:
                raise ValueError("Mismatch between existing star operators and requested vertical CS.")
            # Collect data qubit measurements from all prisms in the component
            meas_label_abs_temp = []
            for meas_rec_info in meas_rec_lst:
                if (
                    meas_rec_info.pipe_prism in ver_all_prisms
                    and meas_rec_info.stabilizer is None
                    and meas_rec_info.meas_type == meas_type_ver
                ):
                    meas_label_abs_temp.append((meas_rec_info.abs_rec, meas_rec_info.label))
            star_ops_labels = []
            for star_op in star_ops_current:
                star_op_flat_set = {Position3DHex(p.x, p.y, 0) for p in star_op}
                labels = []
                for prism_data in self.mapping_full[z_value_ver].values():
                    for pe in prism_data.positions:
                        if pe.hex in star_op_flat_set:
                            labels.append(pe.label)
                star_ops_labels.append(labels)
            tmp = []
            seen_recs_ver: set[int] = set()
            for star_op_label in star_ops_labels:
                for label in star_op_label:
                    for abs_rec, qubit_label in meas_label_abs_temp:
                        if qubit_label == label:
                            rec = abs_rec - num_meas - 1
                            if rec not in seen_recs_ver:
                                seen_recs_ver.add(rec)
                                tmp.append(stim.target_rec(rec))
            ver_all_positions = {p.position for p in ver_all_prisms}
            meas_rec_vertical[(cs_type_ver, frozenset(ver_all_positions))] = tmp
        return meas_rec_vertical, meas_rec_horizontal

    @staticmethod
    def data_meas_rec_lst_pipe_prism(meas_rec_lst: list[MeasRecInfo], z: int, rounds: int):
        """Collect the pipes of z-1 where data qubit measurements were performed."""
        meas_rec_lst_data = [
            m
            for m in meas_rec_lst
            if m.z_value == z - 1 and m.stabilizer is None and m.round == rounds - 1
        ]
        return meas_rec_lst_data

    def add_triple_detector_changed_shape_split_opposite(
        self,
        stab: StabilizerInfo,
        meas_rec_lst: list[MeasRecInfo],
        circuit: stim.Circuit,
        z: int,
        r: int,
        rounds: int,
        meas_rec_lst_data: list[MeasRecInfo],
    ) -> None:
        """Add `triple` detector after split with the opposite basis as the data qubit meas.

        weight-4 stabilizer of current z layer, with previous weight-2 and weight-6.
        """
        prism_pipes_zpm_previous = self.prism_pipes_to_ZPM[z - 1]
        prism_pipes_stabs_previous = self.mapping_full[z - 1]

        rec_lst = []
        # determine which elements of data_meas_pipe_prism are neighbors of current stab
        # for this we first need to find the previous weight-6 labels
        flag = False
        labels_weight6 = None
        stab_set_label = {el.label for el in (stab.data_qubits or [])}  # ty
        if len(stab_set_label) == 3 or len(stab_set_label) == 5:
            return  # no search if we have weight-3 because those are single type stabilizers
        # but also single type weight-6 operators have to be excluded! this problem occurs only
        # for "horizontal" STDWs if you plot the stabilizers per layer
        if stab.stab_type in {"X", "Z"}:
            return
        for prism_pipe in prism_pipes_zpm_previous.keys():
            stabs = prism_pipes_stabs_previous[prism_pipe].stabilizers
            for stab_temp in stabs or []:  # ty
                stab_temp_labels = {pe.label for pe in (stab_temp.data_qubits or [])}  # [] ty
                if len(stab_temp_labels & stab_set_label) == 4:
                    labels_weight6 = stab_temp_labels
                    flag = True
                    break
            if flag:
                break
        neighbor_meas = [m for m in meas_rec_lst_data if m.label in (labels_weight6 or [])]  # ty
        if len(neighbor_meas) == 0:
            return

        # what is the meas_type of these? take first element and take opposite
        meas_types = [m.meas_type for m in neighbor_meas]
        if not all([meas_types[0] == meas_type for meas_type in meas_types]):
            raise ValueError(
                "There are mixed measurement types on neighboring data qubits of previous z layer."
            )
        if meas_types[0] == "MX":
            meas_type = "MZ"
        elif meas_types[0] == "MZ":
            meas_type = "MX"

        # add rec for current stab:
        rec = (
            next(
                m
                for m in meas_rec_lst
                if m.meas_type == meas_type
                and m.z_value == z
                and m.stabilizer == stab
                and m.round == r
            ).abs_rec
            - 1
            - circuit.num_measurements
        )
        rec_lst.append(rec)

        # add the previous weight-6 stabilizer
        flag = False
        for prism_pipe in prism_pipes_zpm_previous.keys():
            stabs = prism_pipes_stabs_previous[prism_pipe].stabilizers
            for stab_temp in stabs or []:  # ty
                stab_temp_labels = {pe.label for pe in (stab_temp.data_qubits or [])}
                if len(stab_temp_labels & stab_set_label) == 4:
                    rec = (
                        next(
                            m
                            for m in meas_rec_lst
                            if m.meas_type == meas_type
                            and m.z_value == z - 1
                            and m.stabilizer == stab_temp
                            and m.round == rounds - 1
                        ).abs_rec
                        - 1
                        - circuit.num_measurements
                    )
                    rec_lst.append(rec)
                    flag = True
                    break
            if flag:
                break

        # add the weight-2 stabilizer
        flag = False
        for prism_pipe in prism_pipes_zpm_previous.keys():
            stabs = prism_pipes_stabs_previous[prism_pipe].stabilizers
            for stab_temp in stabs or []:
                stab_temp_labels = {pe.label for pe in (stab_temp.data_qubits or [])}
                if (
                    len(stab_temp_labels & (labels_weight6 or set())) == 2
                    and len(stab_temp_labels) == 2
                ):
                    rec = (
                        next(
                            m
                            for m in meas_rec_lst
                            if m.meas_type == meas_type
                            and m.z_value == z - 1
                            and m.stabilizer == stab_temp
                            and m.round == rounds - 1
                        ).abs_rec
                        - 1
                        - circuit.num_measurements
                    )
                    rec_lst.append(rec)
                    flag = True
                    break
            if flag:
                break

        circuit.append("DETECTOR", [stim.target_rec(rec) for rec in rec_lst])

    def add_triple_detector_changed_shape_split_same(
        self,
        stab: StabilizerInfo,
        meas_rec_lst: list[MeasRecInfo],
        circuit: stim.Circuit,
        z: int,
        r: int,
        rounds: int,
        meas_rec_lst_data: list[MeasRecInfo],
    ) -> None:
        """Add `triple` detector after split with the same basis as the data qubit meas.

        Strictly speaking, this is not a `triple` detector because the data qubit measurements are
        not summarized in a stabilizer. Thus there will be more recs in the detector, overall
        four: 1 current weight-4 stabilizer, 1 previous weight-6 stabilizer,
        2 data qubit meas previous.
        """
        prism_pipes_zpm_previous = self.prism_pipes_to_ZPM[z - 1]
        prism_pipes_stabs_previous = self.mapping_full[z - 1]

        rec_lst = []
        # determine which elements of data_meas_pipe_prism are neighbors of current stab
        # for this we first need to find the previous weight-6 labels
        flag = False
        labels_weight6 = None
        stab_set_label = {el.label for el in (stab.data_qubits or [])}  # ty
        if len(stab_set_label) == 3 or len(stab_set_label) == 5:
            return  # no search if we have weight-3 because those are single type stabilizers
        # but also single type weight-6 operators have to be excluded! this problem occurs only
        # for "horizontal" STDWs if you plot the stabilizers per layer
        if stab.stab_type in {"X", "Z"}:
            return
        for prism_pipe in prism_pipes_zpm_previous.keys():
            stabs = prism_pipes_stabs_previous[prism_pipe].stabilizers
            for stab_temp in stabs or []:  # ty
                stab_temp_labels = {pe.label for pe in (stab_temp.data_qubits or [])}
                if len(stab_temp_labels & stab_set_label) == 4:
                    labels_weight6 = stab_temp_labels
                    flag = True
                    break
            if flag:
                break
        neighbor_meas = [m for m in meas_rec_lst_data if m.label in (labels_weight6 or set())]
        if len(neighbor_meas) == 0:
            return

        # what is the meas_type of these? take first element
        meas_types = [m.meas_type for m in neighbor_meas]
        if not all([meas_types[0] == meas_type for meas_type in meas_types]):
            raise ValueError(
                "There are mixed measurement types on neighboring data qubits of previous z layer."
            )
        meas_type = meas_types[0]

        # add rec for current stab:
        rec = (
            next(
                m
                for m in meas_rec_lst
                if m.meas_type == meas_type
                and m.z_value == z
                and m.stabilizer == stab
                and m.round == r
            ).abs_rec
            - 1
            - circuit.num_measurements
        )
        rec_lst.append(rec)

        # add the previous weight-6 stabilizer
        flag = False
        for prism_pipe in prism_pipes_zpm_previous.keys():
            stabs = prism_pipes_stabs_previous[prism_pipe].stabilizers
            for stab_temp in stabs or []:  # ty
                stab_temp_labels = {pe.label for pe in (stab_temp.data_qubits or [])}  # ty
                if len(stab_temp_labels & stab_set_label) == 4:
                    rec = (
                        next(
                            m
                            for m in meas_rec_lst
                            if m.meas_type == meas_type
                            and m.z_value == z - 1
                            and m.stabilizer == stab_temp
                            and m.round == rounds - 1
                        ).abs_rec
                        - 1
                        - circuit.num_measurements
                    )
                    rec_lst.append(rec)
                    flag = True
                    break
            if flag:
                break

        # add the two data qubit measurements.
        for neighbor in neighbor_meas:
            rec = neighbor.abs_rec - 1 - circuit.num_measurements
            rec_lst.append(rec)

        circuit.append("DETECTOR", [stim.target_rec(rec) for rec in rec_lst])

    def add_double_detector_changed_shape_merge_same(
        self,
        stab: StabilizerInfo,
        meas_rec_lst: list[MeasRecInfo],
        circuit: stim.Circuit,
        z: int,
        r: int,
        rounds: int,
    ) -> None:
        """Add a double detector in the initialization basis.

        This compares the current weight-6 stabilizer with the previous weight.4 stabilizer
        it compares the stabilizers of the basis in which zpm.p is done.
        these are naturally okay thus only two stabilizers needed
        """
        stab_labels = {pe.label for pe in (stab.data_qubits or [])}  # ty
        # find the weight-4 stabilizer of the previous layer
        prism_pipes_zpm_previous = self.prism_pipes_to_ZPM[z - 1]
        prism_pipes_stabs_previous = self.mapping_full[z - 1]
        stab_weight4 = None
        flag = False
        for prism_pipe in prism_pipes_zpm_previous.keys():
            stabs = prism_pipes_stabs_previous[prism_pipe].stabilizers
            for stab_temp in stabs or []:  # ty
                stab_temp_labels = {pe.label for pe in (stab_temp.data_qubits or [])}  # ty
                # overlap 4 guarantees correct stab
                if len(stab_temp_labels & stab_labels) == 4 and len(stab_temp_labels) == 4:
                    stab_weight4 = stab_temp
                    flag = True
                    break
            if flag:
                break
        if stab_weight4 is None:
            return

        # find meas type based on initialization of adjacent pipe.
        # both stabilizers here are strictly speaking part of the prism, not the pipe
        # thus no knowledge about pipe initialization
        prism_pipes_zpm_temp = self.prism_pipes_to_ZPM[z]
        prism_pipes_stabs = self.mapping_full[z]
        flag = False
        for prism_pipe in prism_pipes_zpm_temp.keys():
            stabs = prism_pipes_stabs[prism_pipe].stabilizers
            for stab_temp in stabs or []:  # ty
                stab_temp_labels = {pe.label for pe in (stab_temp.data_qubits or [])}  # ty
                zpm = prism_pipes_zpm_temp[prism_pipe]
                # take the weight-2 stabilizer that has overlap with current weight-6
                # this is guaranteed in the adjacent pipe.
                if len(stab_labels & stab_temp_labels) == 2 and len(stab_temp_labels) == 2:
                    # this is the zpm we want
                    if zpm.p == BasisPrism.X:
                        meas_type = "MX"
                    elif zpm.p == BasisPrism.Z:
                        meas_type = "MZ"
                    flag = True
                    break
            if flag:
                break

        rec_lst = []

        # find the correct entry for previous weight-4 stabilizer
        rec = (
            next(
                m
                for m in meas_rec_lst
                if m.meas_type == meas_type
                and m.z_value == z - 1
                and m.stabilizer == stab_weight4
                and m.round == rounds - 1
            ).abs_rec
            - 1
            - circuit.num_measurements
        )
        rec_lst.append(rec)

        # find the correct entry for current weight-6 stabilizer
        rec = (
            next(
                m
                for m in meas_rec_lst
                if m.meas_type == meas_type
                and m.z_value == z
                and m.stabilizer == stab
                and m.round == r
            ).abs_rec
            - 1
            - circuit.num_measurements
        )
        rec_lst.append(rec)

        circuit.append("DETECTOR", [stim.target_rec(rec) for rec in rec_lst])  # inplace replace

    def add_triple_detector_changed_shape_merge_opposite(
        self,
        stab_weight6: StabilizerInfo,
        meas_rec_lst: list[MeasRecInfo],
        circuit: stim.Circuit,
        z: int,
        r: int,
        rounds: int,
    ) -> None:
        """Add a triple detector at the r=0 during a merge.

        this means that we add a detector that compares weight-4 of previous z layer,
        weight-6 and weight-2 of the current layer. the weight-4 and weight-2 are subsets
        of the weight-6 stabilizer
        """
        # usually, weight-4 stabilizer is given as input
        # find weight-6 stabilizer and weight-2 stabilizer
        stab_labels = {pe.label for pe in (stab_weight6.data_qubits or set())}  # ty

        if len(stab_labels) != 6:
            return

        stab_weight4 = None
        stab_weight2 = None
        meas_type = None

        # search for weight-2 stabilizer (requires already found weight-6)
        # determine the meas_type we are looking for. the weight-2 stab is in the pipe,
        # not in the prism, so this must be initialized
        flag = False
        prism_pipes_zpm_temp = self.prism_pipes_to_ZPM[z]
        prism_pipes_stabs = self.mapping_full[z]
        for prism_pipe in prism_pipes_zpm_temp.keys():
            stabs = prism_pipes_stabs[prism_pipe].stabilizers
            zpm = prism_pipes_zpm_temp[prism_pipe]
            for stab in stabs or []:  # ty
                stab_temp_labels = {pe.label for pe in (stab.data_qubits or [])}  # ty
                if stab_temp_labels & stab_labels and len(stab.data_qubits or []) == 2:  # ty
                    stab_weight2 = stab
                    # meas type is opposite of init
                    if zpm.p == BasisPrism.Z:
                        meas_type = "MX"
                    elif zpm.p == BasisPrism.X:
                        meas_type = "MZ"
                    flag = True
                    break
            if flag:
                break

        # find weight-4 stabilizer of previous z layer
        prism_pipes_zpm_previous = self.prism_pipes_to_ZPM[z - 1]
        prism_pipes_stabs_previous = self.mapping_full[z - 1]
        flag = False
        for prism_pipe in prism_pipes_zpm_previous.keys():
            stabs = prism_pipes_stabs_previous[prism_pipe].stabilizers
            for stab in stabs or []:  # ty
                stab_temp_labels = {pe.label for pe in (stab.data_qubits or [])}  # ty
                if stab_labels == stab_temp_labels | set(
                    pe.label
                    for pe in (
                        stab_weight2.data_qubits
                        if (stab_weight2 and stab_weight2.data_qubits)
                        else []
                    )  # ty
                ):
                    stab_weight4 = stab
                    flag = True
                    break
            if flag:
                break
        if (
            stab_weight4 is None
        ):  # this is a weight-6 stabilizer fully in the middle of the interface
            return
        rec_lst = []
        # find the rec labels for the triple detector.
        # weight4 from previous layer
        rec = (
            next(
                m
                for m in meas_rec_lst
                if m.meas_type == meas_type
                and m.z_value == z - 1
                and m.stabilizer == stab_weight4
                and m.round == rounds - 1
            ).abs_rec
            - 1
            - circuit.num_measurements
        )
        rec_lst.append(rec)

        for stab_temp in [stab_weight2, stab_weight6]:
            rec = (
                next(
                    m
                    for m in meas_rec_lst
                    if m.meas_type == meas_type
                    and m.z_value == z
                    and m.stabilizer == stab_temp
                    and m.round == r
                ).abs_rec
                - 1
                - circuit.num_measurements
            )
            rec_lst.append(rec)

        circuit.append("DETECTOR", [stim.target_rec(rec) for rec in rec_lst])  # inplace replace

    def add_trivial_detector_changed_shape(
        self,
        stab: StabilizerInfo,
        prism_pipe: Prism | PrismPipe,
        prism_pipes_stabs: dict[Prism | PrismPipe, list[StabilizerInfo] | None],
        prism_pipes_zpm_temp: dict,
        meas_rec_lst: list[MeasRecInfo],
        circuit: stim.Circuit,
        z: int,
        r: int,
    ) -> None:
        """Add a trivial detector for an XZ stabilizer that changed shape at r=0.

        When a stabilizer changed shape and the initialization basis matches
        one of the syndrome bases, that syndrome measurement is deterministic.
        Only performed if the very first z layer also initialized in the same basis.
        """
        first_z = self.z_values[0]
        first_zpm = self.prism_pipes_to_ZPM[first_z].get(prism_pipe)
        if first_zpm is None:
            # prism_pipe didn't exist at the first z layer, find by position match
            first_zpm = next(
                (
                    zpm
                    for pp, zpm in self.prism_pipes_to_ZPM[first_z].items()
                    if self.pipe_matches(pp, prism_pipe)
                ),
                None,
            )
        if first_zpm is None:
            return

        stab_labels = {pe.label for pe in (stab.data_qubits or [])}
        for candidate_pipe, candidate_prism_data in prism_pipes_stabs.items():
            if not isinstance(candidate_prism_data, PrismData):
                raise ValueError("Expected candidate_prism_data to be a PrismData object.")
            candidate_labels = {pe.label for pe in candidate_prism_data.positions}
            if stab_labels & candidate_labels:
                candidate_zpm = prism_pipes_zpm_temp[candidate_pipe]
                if stab.stab_type == "XZ":
                    if candidate_zpm.p == BasisPrism.X and first_zpm.p == BasisPrism.X:
                        rec = (
                            next(
                                m
                                for m in meas_rec_lst
                                if m.meas_type == "MX"
                                and m.pipe_prism == prism_pipe
                                and m.z_value == z
                                and m.stabilizer == stab
                                and m.round == r
                            ).abs_rec
                            - 1
                            - circuit.num_measurements
                        )
                        circuit.append("DETECTOR", [stim.target_rec(rec)])
                    elif candidate_zpm.p == BasisPrism.Z and first_zpm.p == BasisPrism.Z:
                        rec = (
                            next(
                                m
                                for m in meas_rec_lst
                                if m.meas_type == "MZ"
                                and m.pipe_prism == prism_pipe
                                and m.z_value == z
                                and m.stabilizer == stab
                                and m.round == r
                            ).abs_rec
                            - 1
                            - circuit.num_measurements
                        )
                        circuit.append("DETECTOR", [stim.target_rec(rec)])

    @staticmethod
    def create_data_measurement_detectors(
        stabs: list[StabilizerInfo],
        zpm: ZPM,
        meas_rec_lst: list[MeasRecInfo],
        circuit: stim.Circuit,
        z: int,
        r: int,
    ) -> stim.Circuit:
        """Create detectors based on data qubit measurements.

        i.e. for some stabilizer, take the data qubit measurements of the respective qubits
        and compare to the previous layer's stabilizer ancilla measurements.

        this is for a current pipe/prism.

        #! ATTENTION! in the future, if there may be multiple data qubit measurements
        #! on different pipes/prisms in the same z layer, one has to specify pipe_prism
        #! explicitly. otherwise the wrong stabilizer measurements may be assigned to the detector
        """
        if zpm.m == BasisPrism.N:
            return circuit  # do nothing if no measurement!
        # only go through the stabilizers that are relevant for zpm.m
        stabs_filtered = []
        for stab in stabs:
            if zpm.m == BasisPrism.Z:
                if stab.stab_type in {"XZ", "Z"}:
                    stabs_filtered.append(stab)
            elif zpm.m == BasisPrism.X:
                if stab.stab_type in {"XZ", "X"}:
                    stabs_filtered.append(stab)
        for stab in stabs_filtered:
            # only take the stabilizers whose data qubits are actually measured.
            detector_idx_lst = []
            flag = False
            for data in stab.data_qubits or []:  # ty
                if zpm.m == BasisPrism.Z:
                    try:
                        match = next(
                            m
                            for m in meas_rec_lst
                            if m.meas_type == "MZ"
                            # and m.pipe_prism == prism_pipe
                            and m.z_value == z
                            and m.stabilizer is None  # right above, no stabilizer assigned
                            and m.round == r
                            and m.label == data.label
                        )
                    except StopIteration:
                        flag = True
                        break
                elif zpm.m == BasisPrism.X:
                    try:
                        match = next(
                            m
                            for m in meas_rec_lst
                            if m.meas_type == "MX"
                            # and m.pipe_prism == prism_pipe
                            and m.z_value == z
                            and m.stabilizer is None  # right above, no stabilizer assigned
                            and m.round == r
                            and m.label == data.label
                        )
                    except StopIteration:
                        flag = True
                        break
                rec_current = match.abs_rec - 1 - circuit.num_measurements
                detector_idx_lst.append(rec_current)

            if flag:  # if flag, there was a stabilizer with data qubits that were never measured.
                continue
            if zpm.m == BasisPrism.Z:
                rec_prev = (
                    next(  # previous stabilizer ancilla
                        m
                        for m in meas_rec_lst
                        if m.meas_type == "MZ"
                        # and m.pipe_prism == prism_pipe
                        and m.z_value == z
                        and m.stabilizer == stab
                        and m.round == r  # last stabilizer meas was officially in same round
                    ).abs_rec
                    - 1
                    - circuit.num_measurements
                )
            elif zpm.m == BasisPrism.X:
                rec_prev = (
                    next(  # previous stabilizer ancilla
                        m
                        for m in meas_rec_lst
                        if m.meas_type == "MX"
                        # and m.pipe_prism == prism_pipe
                        and m.z_value == z
                        and m.stabilizer == stab
                        and m.round == r  # last stabilizer meas was officially in same round
                    ).abs_rec
                    - 1
                    - circuit.num_measurements
                )
            detector_idx_lst.append(rec_prev)
            circuit.append("DETECTOR", [stim.target_rec(rec) for rec in detector_idx_lst])
        return circuit

    def add_qubit_coords_to_circuit_quadratic(self, circuit: stim.Circuit) -> stim.Circuit:
        """Prepend QUBIT_COORDS to the circuit using the quadratic (rect) layout."""
        coord_circuit = stim.Circuit()
        for q, (x, y) in sorted(self.qubit_coords_quadratic.items()):
            coord_circuit.append("QUBIT_COORDS", [q], [x, y])
        coord_circuit += circuit
        return coord_circuit

    def assign_qubit_coords_quadratic(self, scale: int = 1):
        """Build quadratic coordinates.

        Build a label -> (x, y) coordinate mapping from the quadratic (rect) layout,
        shifted so all coordinates are non-negative.
        """
        raw_coords = {}

        for prism_pipe_dct in self.mapping_full.values():
            for prism_data in prism_pipe_dct.values():
                for pe in prism_data.positions:
                    if pe.label is not None and pe.rect is not None:
                        x, y = pe.rect
                        coords = (x * scale, y * scale)
                        if pe.label in raw_coords and raw_coords[pe.label] != coords:
                            raise ValueError(
                                f"Inconsistent coords for label {pe.label}: "
                                f"{raw_coords[pe.label]} vs {coords}"
                            )
                        raw_coords[pe.label] = coords

                if prism_data.stabilizers:
                    for stab in prism_data.stabilizers:
                        if stab.ancilla:
                            for anc_pe in stab.ancilla:
                                if anc_pe.label is not None and anc_pe.rect is not None:
                                    x, y = anc_pe.rect
                                    coords = (x * scale, y * scale)
                                    if (
                                        anc_pe.label in raw_coords
                                        and raw_coords[anc_pe.label] != coords
                                    ):
                                        raise ValueError(
                                            f"Inconsistent coords for ancilla label {anc_pe.label}:"
                                            f" {raw_coords[anc_pe.label]} vs {coords}"
                                        )
                                    raw_coords[anc_pe.label] = coords
        #flip signs such that crumble represents this in alignment to our plots
        self.qubit_coords_quadratic = {
            label: (abs(x), abs(y)) for label, (x, y) in raw_coords.items()
        }
        return self.qubit_coords_quadratic

    def find_reachable_via_horizontal(
        self, cs: CorrelationSurface, prisms_in_obs: list[Prism]
    ) -> set[Position3DHex]:
        """Determine which of the observables (star operator on which patch) receives corrections.

        This means that we have parities from measurements at the vertical/horizontal correlation
        surfaces of horizontal pipes. They cannot be multiplied anywhere.
        This method gives you the correct output for teleportation and CNOT circuits.
        But it does NOT generalize to other structures!
        """
        zx = self.prism_graph.to_zx_graph()
        reachable_via_horizontal: set[Position3DHex] = set()

        cs_positions = {zx.positions[v] for v in cs.span_vertices}

        min_z_in_cs = min(p.z for p in cs_positions)
        max_z_in_cs = max(p.z for p in cs_positions)

        min_z_xy = {(p.x, p.y) for p in cs_positions if p.z == min_z_in_cs}
        max_z_xy = {(p.x, p.y) for p in cs_positions if p.z == max_z_in_cs}

        all_graph_positions_max_z = [p for p in zx.positions.values() if p.z == max_z_in_cs]

        if len(all_graph_positions_max_z) == 1:
            shared_xy = max_z_xy  # single node in full graph at max_z: always reachable
        else:
            shared_xy = min_z_xy & max_z_xy

        for prism in prisms_in_obs:
            xy = (prism.position.x, prism.position.y)
            if xy in shared_xy:
                reachable_via_horizontal.add(Position3DHex(prism.position.x, prism.position.y, 0))
        return reachable_via_horizontal

    def build_syndrome_extraction_round(
        self,
        circuit: stim.Circuit,
        r: int,
        rounds: int,
        z: int,
        prism_pipes_zpm_temp: dict[Prism | PrismPipe, ZPM],
        prism_pipes_stabs: dict,  # too lazy to find detailed type that does not mess around with ty
        cnot_order: list[str],
        meas_rec_lst: list[MeasRecInfo],
        p_init: float,
        p_gate2: float,
        p_meas: float,
        p_idle: float,
        all_current_qubits: list[PositionEntry],
        current_tick: int,
    ) -> tuple[int, list[MeasRecInfo], stim.Circuit]:
        """Construct a single round of stabilizer measurements.

        For weight-3, 5, 6 we use the superdense scheme and for weight-2 stabilizers
        the inline folding scheme is applied.

        This method also adds idling noise etc., ticks and X gates based on superdense
        syndrome extraction's measurement outcomes.

        The inline folding and superdense SE is slightly shifted towards each other such
        that it works out with minimal overhead.

        """
        # ------------initialize all ancillas in that round--------------
        for prism_pipe in prism_pipes_zpm_temp.keys():  # noqa: PLC0206
            zpm = prism_pipes_zpm_temp[prism_pipe]
            stabs = prism_pipes_stabs[prism_pipe].stabilizers
            data_positions = prism_pipes_stabs[prism_pipe].positions
            for stab in stabs:
                # Bell initialization
                if stab.ancilla is not None:
                    # ancillas are ordered left RX, right RZ
                    label = stab.ancilla[0].label
                    circuit.append("RX", label)
                    circuit.append("Z_ERROR", label, p_init)
                    label = stab.ancilla[1].label
                    circuit.append("R", label)
                    circuit.append("X_ERROR", label, p_init)
                # --if r!=rounds-1 then add weight-2 stabilizer finalization of previous round--
                if len(stab.data_qubits) == 2 and r != 0:  # and r != rounds-1:
                    lst = [stab.data_qubits[0].label, stab.data_qubits[1].label]
                    # defold CNOT
                    if stab.stab_type == "X":
                        circuit.append("CNOT", lst[::-1])
                        circuit.append("DEPOLARIZE2", lst[::-1], p_gate2)  # cnot in other direction
                    elif stab.stab_type == "Z":
                        circuit.append("CNOT", lst)
                        circuit.append("DEPOLARIZE2", lst, p_gate2)
        # classical feedback for weight->2 stabs from previous round
        if r != 0:
            for prism_pipe in prism_pipes_zpm_temp.keys():
                stabs = prism_pipes_stabs[prism_pipe].stabilizers
                for stab in stabs:
                    if len(stab.data_qubits) > 2 and stab.stab_type == "XZ":
                        # no classical feedback for single type.
                        neighbor_data = [
                            qubit
                            for qubit in stab.data_qubits
                            if Position3DHex.rectangular_neighbor(qubit.rect, stab.ancilla[1].rect)
                        ]
                        rec_mz = (
                            next(
                                m
                                for m in meas_rec_lst
                                if m.meas_type == "MZ"
                                and m.pipe_prism == prism_pipe
                                and m.z_value == z
                                and m.stabilizer == stab
                                and m.round == r - 1  # previous round
                            ).abs_rec
                            - 1
                            - circuit.num_measurements
                        )
                        for data in neighbor_data:
                            circuit.append("CX", [stim.target_rec(rec_mz), data.label])
        # --add idling noise--
        active = self.get_active_qubits_since_last_tick(circuit)
        current_tick = self.append_tick_with_idle_noise(
            circuit, all_current_qubits, current_tick, p_idle, active
        )
        # ------------CNOT for Bell for ancillas pairs--------------
        for prism_pipe in prism_pipes_zpm_temp.keys():  # noqa: PLC0206
            zpm = prism_pipes_zpm_temp[prism_pipe]
            stabs = prism_pipes_stabs[prism_pipe].stabilizers
            data_positions = prism_pipes_stabs[prism_pipe].positions
            for stab in stabs:
                if stab.ancilla is not None:
                    # Bell initialization CNOT
                    lst = [stab.ancilla[0].label, stab.ancilla[1].label]
                    circuit.append("CNOT", lst)
                    circuit.append("DEPOLARIZE2", lst, p_gate2)
        # --add idling noise--
        active = self.get_active_qubits_since_last_tick(circuit)
        current_tick = self.append_tick_with_idle_noise(
            circuit, all_current_qubits, current_tick, p_idle, active
        )
        # --CNOT Gates for SE--
        for direction in cnot_order:
            for prism_pipe in prism_pipes_zpm_temp.keys():  # noqa: PLC0206
                zpm = prism_pipes_zpm_temp[prism_pipe]
                stabs = prism_pipes_stabs[prism_pipe].stabilizers
                data_positions = prism_pipes_stabs[prism_pipe].positions
                for stab in stabs:
                    if stab.stab_type == "XZ":
                        # Z stabilizer
                        if direction in {"bottom_z", "sides_z", "top_z"}:
                            entries = getattr(stab, direction.removesuffix("_z"))
                            for data in entries:
                                # which ancilla is neighbor?
                                for ancilla in stab.ancilla:
                                    if Position3DHex.rectangular_neighbor(ancilla.rect, data.rect):
                                        lst = [data.label, ancilla.label]
                                        circuit.append("CNOT", lst)
                                        circuit.append("DEPOLARIZE2", lst, p_gate2)
                                        break
                        # X stabilizer
                        elif direction in {"bottom_x", "sides_x", "top_x"}:
                            entries = getattr(stab, direction.removesuffix("_x"))
                            for data in entries:
                                # which ancilla is neighbor?
                                for ancilla in stab.ancilla:
                                    if Position3DHex.rectangular_neighbor(ancilla.rect, data.rect):
                                        lst = [ancilla.label, data.label]
                                        circuit.append("CNOT", lst)
                                        circuit.append("DEPOLARIZE2", lst, p_gate2)
                                        break
                    # weight-5 and weight-3, and weight-6 stabilizers on STDW
                    elif stab.stab_type == "X" and len(stab.data_qubits) > 2:
                        if direction in {"bottom_x", "sides_x", "top_x"}:
                            entries = getattr(stab, direction.removesuffix("_x"))
                            for data in entries:
                                # which ancilla is neighbor?
                                for ancilla in stab.ancilla:
                                    if Position3DHex.rectangular_neighbor(ancilla.rect, data.rect):
                                        lst = [ancilla.label, data.label]
                                        circuit.append("CNOT", lst)
                                        circuit.append("DEPOLARIZE2", lst, p_gate2)
                                        break
                    elif stab.stab_type == "Z" and len(stab.data_qubits) > 2:
                        if direction in {"bottom_z", "sides_z", "top_z"}:
                            entries = getattr(stab, direction.removesuffix("_z"))
                            for data in entries:
                                # which ancilla is neighbor?
                                for ancilla in stab.ancilla:
                                    if Position3DHex.rectangular_neighbor(ancilla.rect, data.rect):
                                        lst = [data.label, ancilla.label]
                                        circuit.append("CNOT", lst)
                                        circuit.append("DEPOLARIZE2", lst, p_gate2)
                                        break
            # --add idling noise--
            active = self.get_active_qubits_since_last_tick(circuit)
            current_tick = self.append_tick_with_idle_noise(
                circuit, all_current_qubits, current_tick, p_idle, active
            )
        # --final bell and fold for weight2--
        for prism_pipe in prism_pipes_zpm_temp.keys():  # noqa: PLC0206
            zpm = prism_pipes_zpm_temp[prism_pipe]
            stabs = prism_pipes_stabs[prism_pipe].stabilizers
            data_positions = prism_pipes_stabs[prism_pipe].positions
            for stab in stabs:
                if len(stab.data_qubits) > 2:
                    # Bell final
                    lst = [stab.ancilla[0].label, stab.ancilla[1].label]
                    circuit.append("CNOT", lst)
                    circuit.append("DEPOLARIZE2", lst, p_gate2)
                elif len(stab.data_qubits) == 2:
                    # first step of fold for weight-2 stabilizer
                    lst = [stab.data_qubits[0].label, stab.data_qubits[1].label]
                    # fold cnot
                    if stab.stab_type == "X":
                        circuit.append("CNOT", lst[::-1])
                        circuit.append("DEPOLARIZE2", lst[::-1], p_gate2)  # cnot in other direction
                    elif stab.stab_type == "Z":
                        circuit.append("CNOT", lst)
                        circuit.append("DEPOLARIZE2", lst, p_gate2)
                else:
                    raise TQECError("weight-2 stabilizer that is not single type - cannot be!")
        # --add idling noise--
        active = self.get_active_qubits_since_last_tick(circuit)
        current_tick = self.append_tick_with_idle_noise(
            circuit, all_current_qubits, current_tick, p_idle, active
        )
        # --measure ancilla both for weight-2 and others--
        for prism_pipe in prism_pipes_zpm_temp.keys():  # noqa: PLC0206
            zpm = prism_pipes_zpm_temp[prism_pipe]
            stabs = prism_pipes_stabs[prism_pipe].stabilizers
            data_positions = prism_pipes_stabs[prism_pipe].positions
            # measure stabs if higher weight, and also meas stab if weight=2
            for stab in stabs:
                if len(stab.data_qubits) > 2:
                    # measurement
                    if stab.stab_type in {"XZ", "X"}:
                        label = stab.ancilla[0].label
                        circuit.append("Z_ERROR", label, p_meas)
                        circuit.append("MX", label)
                        meas_rec_lst.append(
                            MeasRecInfo(
                                meas_type="MX",
                                pipe_prism=prism_pipe,
                                stabilizer=stab,
                                abs_rec=circuit.num_measurements,
                                z_value=z,
                                round=r,
                                label=label,
                                tick=current_tick,
                            )
                        )  # add to record
                    if stab.stab_type in {"XZ", "Z"}:
                        label = stab.ancilla[1].label
                        circuit.append("X_ERROR", label, p_meas)
                        circuit.append("M", label)
                        meas_rec_lst.append(
                            MeasRecInfo(
                                meas_type="MZ",
                                pipe_prism=prism_pipe,
                                stabilizer=stab,
                                abs_rec=circuit.num_measurements,
                                z_value=z,
                                round=r,
                                label=label,
                                tick=current_tick,
                            )
                        )  # add to record
                elif len(stab.data_qubits) == 2:
                    # folded weight-2 stabilizer, perform meas on data qubit
                    label = stab.data_qubits[1].label
                    # measure in fold
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
                            meas_type=meas_type,
                            pipe_prism=prism_pipe,
                            stabilizer=stab,
                            abs_rec=circuit.num_measurements,
                            z_value=z,
                            round=r,
                            label=label,
                            tick=current_tick,
                        )
                    )  # add to record
                else:
                    raise TQECError("weight-2 stabilizer that is not single type - cannot be!")
        # ------add final  weight-2 CNOT gate that expands-------
        # ------after folding if no further round available------
        if r == rounds - 1:
            flag = False
            for prism_pipe in prism_pipes_zpm_temp.keys():  # noqa: PLC0206
                zpm = prism_pipes_zpm_temp[prism_pipe]
                stabs = prism_pipes_stabs[prism_pipe].stabilizers
                for stab in stabs:
                    if len(stab.data_qubits) == 2:
                        flag = True
                        break
            if flag:
                # --add idling noise as this has to be its own tick--
                active = self.get_active_qubits_since_last_tick(circuit)
                current_tick = self.append_tick_with_idle_noise(
                    circuit, all_current_qubits, current_tick, p_idle, active
                )
                for prism_pipe in prism_pipes_zpm_temp.keys():  # noqa: PLC0206
                    zpm = prism_pipes_zpm_temp[prism_pipe]
                    stabs = prism_pipes_stabs[prism_pipe].stabilizers
                    data_positions = prism_pipes_stabs[prism_pipe].positions
                    for stab in stabs:
                        if len(stab.data_qubits) == 2:
                            lst = [stab.data_qubits[0].label, stab.data_qubits[1].label]
                            if stab.stab_type == "X":
                                circuit.append("CNOT", lst[::-1])
                                circuit.append("DEPOLARIZE2", lst[::-1], p_gate2)
                                # cnot in other direction, thus ::-1
                            elif stab.stab_type == "Z":
                                circuit.append("CNOT", lst)
                                circuit.append("DEPOLARIZE2", lst, p_gate2)
            # classical feedback. add this in its own layer
            # as it is not assumed to be physical anyways
            # important: this must be AFTER folding/unfolding of weight-2 stabilizers,
            # otherwise it may mix them up
            for prism_pipe in prism_pipes_zpm_temp.keys():  # noqa: PLC0206
                zpm = prism_pipes_zpm_temp[prism_pipe]
                stabs = prism_pipes_stabs[prism_pipe].stabilizers
                data_positions = prism_pipes_stabs[prism_pipe].positions
                # measure stabs if higher weight, and also meas stab if weight=2
                for stab in stabs:
                    if len(stab.data_qubits) > 2 and stab.stab_type == "XZ":
                        # no classical feedback for single type!
                        # add classical feedback X^b, find neighboring data to ancilla[1]
                        # only add classical feedback if actually MZ was done
                        neighbor_data = [
                            qubit
                            for qubit in stab.data_qubits
                            if Position3DHex.rectangular_neighbor(qubit.rect, stab.ancilla[1].rect)
                        ]
                        rec_current = (
                            next(
                                m
                                for m in meas_rec_lst
                                if m.meas_type == "MZ"
                                and m.pipe_prism == prism_pipe
                                and m.z_value == z
                                and m.stabilizer == stab
                                and m.round == r
                            ).abs_rec
                            - 1
                            - circuit.num_measurements
                        )
                        for data in neighbor_data:
                            # classically controlled X gate
                            circuit.append("CX", [stim.target_rec(rec_current), data.label])
                            # no error added here because assume that those are classically tracked
        # if a prism pipe has zpm.m != N then data qubits need to be measured
        # after final additional round for weight-2 stabilizer meas
        for prism_pipe in prism_pipes_zpm_temp.keys():  # noqa: PLC0206
            zpm = prism_pipes_zpm_temp[prism_pipe]
            stabs = prism_pipes_stabs[prism_pipe].stabilizers
            data_positions = prism_pipes_stabs[prism_pipe].positions
            lst = [data.label for data in data_positions]
            if (
                z != max(self.z_values) and r == rounds - 1
            ):  # if last z values then handled at the end.
                if zpm.m == BasisPrism.Z:
                    # meas data qubits Z
                    circuit.append("X_ERROR", lst, p_meas)
                    circuit.append("M", lst)
                    base_rec = circuit.num_measurements - len(lst)
                    for i, qubit_label in enumerate(lst):
                        meas_rec_lst.append(
                            MeasRecInfo(
                                meas_type="MZ",
                                pipe_prism=prism_pipe,
                                stabilizer=None,
                                abs_rec=base_rec + i + 1,
                                z_value=z,
                                round=r,
                                label=qubit_label,  # now correctly the individual qubit label
                                tick=current_tick,
                            )
                        )
                elif zpm.m == BasisPrism.X:
                    # meas data qubits X
                    circuit.append("Z_ERROR", lst, p_meas)
                    circuit.append("MX", lst)
                    base_rec = circuit.num_measurements - len(lst)
                    for i, qubit_label in enumerate(lst):
                        meas_rec_lst.append(
                            MeasRecInfo(
                                meas_type="MX",
                                pipe_prism=prism_pipe,
                                stabilizer=None,
                                abs_rec=base_rec + i + 1,
                                z_value=z,
                                round=r,
                                label=qubit_label,  # now correctly the individual qubit label
                                tick=current_tick,
                            )
                        )
        # --add idling noise--
        active = self.get_active_qubits_since_last_tick(circuit)
        current_tick = self.append_tick_with_idle_noise(
            circuit, all_current_qubits, current_tick, p_idle, active
        )

        return current_tick, meas_rec_lst, circuit

    def add_detector_annotation(
        self,
        circuit: stim.Circuit,
        meas_rec_lst: list[MeasRecInfo],
        initialized_qubits: list[PositionEntry],
        prism_pipes_zpm_temp: dict,
        prism_pipes_stabs: dict,
        z: int,
        r: int,
        rounds: int,
        s: int,
    ) -> stim.Circuit:
        """Add detector annotations.

        The detector annotations correspond to the measurements
        generated in the current syndrome-extraction round.
        """
        meas_rec_lst_data = self.data_meas_rec_lst_pipe_prism(meas_rec_lst, z, rounds)
        for prism_pipe in prism_pipes_zpm_temp.keys():  # noqa: PLC0206
            zpm = prism_pipes_zpm_temp[prism_pipe]
            stabs = prism_pipes_stabs[prism_pipe].stabilizers
            if r == 0:
                # only add those default detectors based on zpm.p
                relevant = []  # this must be reset after each run.
                if zpm.p == BasisPrism.Z:
                    relevant = [
                        m
                        for m in meas_rec_lst
                        if m.meas_type == "MZ"
                        and m.pipe_prism == prism_pipe
                        and m.z_value == z
                        and m.round == r
                    ]
                elif zpm.p == BasisPrism.X:
                    relevant = [
                        m
                        for m in meas_rec_lst
                        if m.meas_type == "MX"
                        and m.pipe_prism == prism_pipe
                        and m.z_value == z
                        and m.round == r
                    ]
                for m in relevant:
                    offset = m.abs_rec - 1 - circuit.num_measurements
                    circuit.append("DETECTOR", [stim.target_rec(offset)])
                if zpm.p == BasisPrism.N and s != 0:
                    # IMPORTANT: if basis N, you need to compare to the previous z layer
                    stabs_temp = [
                        el
                        for el in stabs
                        if len(el.data_qubits) != 3
                        and len(el.data_qubits) != 5
                        and len(el.data_qubits) != 2
                    ]
                    for stab in stabs_temp:
                        changed_shape = self.stabilizer_changed_shape_bell(z, prism_pipe, stab)
                        if not changed_shape:
                            # skip the stabilizer if it contains a data qubit
                            # which was just re-initialized
                            if any(q in initialized_qubits for q in stab.data_qubits):
                                continue
                            # compare Z stabilizer to previous z layer (stab remained the same)
                            rec_current = (
                                next(
                                    m
                                    for m in meas_rec_lst
                                    if m.meas_type == "MZ"
                                    and m.pipe_prism == prism_pipe
                                    and m.z_value == z
                                    and m.stabilizer == stab
                                    and m.round == r
                                ).abs_rec
                                - 1
                                - circuit.num_measurements
                            )
                            target_rects = {pe.rect for pe in stab.data_qubits}
                            rec_prev = (
                                next(
                                    m
                                    for m in meas_rec_lst
                                    if m.meas_type == "MZ"
                                    and self.pipe_matches(m.pipe_prism, prism_pipe)
                                    and m.z_value == z - 1
                                    and m.stabilizer is not None
                                    and {
                                        pe.rect
                                        for pe in (
                                            m.stabilizer.data_qubits
                                            if (m.stabilizer and m.stabilizer.data_qubits)
                                            else []
                                        )
                                    }
                                    == target_rects  # ty
                                    and m.round == rounds - 1
                                ).abs_rec
                                - 1
                                - circuit.num_measurements
                            )
                            circuit.append(
                                "DETECTOR",
                                [stim.target_rec(rec_current), stim.target_rec(rec_prev)],
                            )
                            # compare X stabilizer to previous z layer (stab remained the same)
                            rec_current = (
                                next(
                                    m
                                    for m in meas_rec_lst
                                    if m.meas_type == "MX"
                                    and m.pipe_prism == prism_pipe
                                    and m.z_value == z
                                    and m.stabilizer == stab
                                    and m.round == r
                                ).abs_rec
                                - 1
                                - circuit.num_measurements
                            )
                            target_rects = {pe.rect for pe in stab.data_qubits}
                            rec_prev = (
                                next(
                                    m
                                    for m in meas_rec_lst
                                    if m.meas_type == "MX"
                                    and self.pipe_matches(m.pipe_prism, prism_pipe)
                                    and m.z_value == z - 1
                                    and m.stabilizer is not None
                                    and {
                                        pe.rect
                                        for pe in (
                                            m.stabilizer.data_qubits
                                            if (m.stabilizer and m.stabilizer.data_qubits)
                                            else []
                                        )
                                    }
                                    == target_rects  # ty
                                    and m.round == rounds - 1
                                ).abs_rec
                                - 1
                                - circuit.num_measurements
                            )
                            circuit.append(
                                "DETECTOR",
                                [stim.target_rec(rec_current), stim.target_rec(rec_prev)],
                            )
                        else:
                            # if stabilizer changed shape but init and
                            # stabilizer are same basis, add trivial detector
                            self.add_trivial_detector_changed_shape(
                                stab,
                                prism_pipe,
                                prism_pipes_stabs,
                                prism_pipes_zpm_temp,
                                meas_rec_lst,
                                circuit,
                                z,
                                r,
                            )
                            # triple detector in basis in which we do not initialize
                            # (i.e. stabilizers which can form horizontal cs)
                            # e.g. weight-4 in previous layer
                            # + current weight-6 and current weight-2
                            self.add_triple_detector_changed_shape_merge_opposite(
                                stab, meas_rec_lst, circuit, z, r, rounds
                            )
                            # in the same basis as initialization just add double detectors
                            self.add_double_detector_changed_shape_merge_same(
                                stab, meas_rec_lst, circuit, z, r, rounds
                            )
                            if len(meas_rec_lst_data) != 0:
                                # triple detector in the basis of data qubit measurements
                                # compare former weight-6 with current weight-4 and data meas
                                # on location of weight-2
                                # but not weight-2 stabilizer
                                # because this does not exist in this basis during split
                                self.add_triple_detector_changed_shape_split_same(
                                    stab, meas_rec_lst, circuit, z, r, rounds, meas_rec_lst_data
                                )
                                # triple detector in the opposite basis
                                # compare former weight-6 with former weight-2 and current weight-4
                                # during split
                                self.add_triple_detector_changed_shape_split_opposite(
                                    stab, meas_rec_lst, circuit, z, r, rounds, meas_rec_lst_data
                                )
            else:  # compare this with previous round meas
                for stab in stabs:
                    # ==usual double detectors that compare current and previous round==
                    if stab.stab_type in {"XZ", "Z"}:
                        # z stabilizer
                        rec_current = (
                            next(
                                m
                                for m in meas_rec_lst
                                if m.meas_type == "MZ"
                                and m.pipe_prism == prism_pipe
                                and m.z_value == z
                                and m.stabilizer == stab
                                and m.round == r
                            ).abs_rec
                            - 1
                            - circuit.num_measurements
                        )

                        rec_prev = (
                            next(
                                m
                                for m in meas_rec_lst
                                if m.meas_type == "MZ"
                                and m.pipe_prism == prism_pipe
                                and m.z_value == z
                                and m.stabilizer == stab
                                and m.round == r - 1
                            ).abs_rec
                            - 1
                            - circuit.num_measurements
                        )

                        circuit.append(
                            "DETECTOR", [stim.target_rec(rec_current), stim.target_rec(rec_prev)]
                        )
                    if stab.stab_type in {"XZ", "X"}:
                        # x stabilizer
                        rec_current = (
                            next(
                                m
                                for m in meas_rec_lst
                                if m.meas_type == "MX"
                                and m.pipe_prism == prism_pipe
                                and m.z_value == z
                                and m.stabilizer == stab
                                and m.round == r
                            ).abs_rec
                            - 1
                            - circuit.num_measurements
                        )

                        rec_prev = (
                            next(
                                m
                                for m in meas_rec_lst
                                if m.meas_type == "MX"
                                and m.pipe_prism == prism_pipe
                                and m.z_value == z
                                and m.stabilizer == stab
                                and m.round == r - 1
                            ).abs_rec
                            - 1
                            - circuit.num_measurements
                        )

                        circuit.append(
                            "DETECTOR", [stim.target_rec(rec_current), stim.target_rec(rec_prev)]
                        )
            # if, not elif because above else should be done nevertheless too.
            if r == rounds - 1:
                # ==create detectors for the intermediate measurements==
                circuit = self.create_data_measurement_detectors(
                    stabs, zpm, meas_rec_lst, circuit, z, r
                )
        return circuit

    def add_final_meas_and_obs(
        self,
        circuit: stim.Circuit,
        meas_rec_lst: list[MeasRecInfo],
        prism_pipes_zpm_temp: dict,
        prism_pipes_stabs: dict,
        horizontal_pipes_cs_lst: list,
        prisms_in_obs_lst: list,
        cs_lst: list,
        z: int,
        r: int,
        current_tick: int,
        p_meas: float,
    ) -> tuple[stim.Circuit, list[MeasRecInfo]]:
        """Add final measurements and observables.

        This applied final data qubit measurements, parts of which are
        included in the logical observables.
        """
        # ==final measurements==
        for prism_pipe in prism_pipes_zpm_temp.keys():  # noqa: PLC0206
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
                            meas_type="MZ",
                            pipe_prism=prism_pipe,
                            stabilizer=None,
                            abs_rec=circuit.num_measurements,
                            z_value=z,
                            round=r,
                            label=label,
                            tick=current_tick,
                        )
                    )  # add to record
            elif zpm.m == BasisPrism.X:
                for data in data_positions:
                    label = data.label
                    circuit.append("Z_ERROR", label, p_meas)
                    circuit.append("MX", label)
                    meas_rec_lst.append(
                        MeasRecInfo(
                            meas_type="MX",
                            pipe_prism=prism_pipe,
                            stabilizer=None,
                            abs_rec=circuit.num_measurements,
                            z_value=z,
                            round=r,
                            label=label,
                            tick=current_tick,
                        )
                    )  # add to record
            else:
                raise TQECError("The last z requires data measurements in zpm.m")

        #!TODO ALSO TAKE STAR OPERATORS OF PREVIOUS Z INTO ACCOUNT
        # track which CS measurement records have already been included per observable.
        cs_targets_seen_per_obs: dict[int, set[int]] = {i: set() for i in range(len(cs_lst))}
        for obs_idx, cs in enumerate(cs_lst):
            prisms_in_obs = prisms_in_obs_lst[obs_idx]
            horizontal_pipes_cs = horizontal_pipes_cs_lst[obs_idx]
            if cs is None:  # memory experiment case
                reachable_via_horizontal = set()
            else:
                reachable_via_horizontal = self.find_reachable_via_horizontal(cs, prisms_in_obs)

            # ==final round of detectors based on stabilizers based on zpm.m==
            for prism_pipe in prism_pipes_zpm_temp.keys():  # noqa: PLC0206
                zpm = prism_pipes_zpm_temp[prism_pipe]
                stabs = prism_pipes_stabs[prism_pipe].stabilizers
                data_positions = prism_pipes_stabs[prism_pipe].positions
                circuit = self.create_data_measurement_detectors(
                    stabs, zpm, meas_rec_lst, circuit, z, r
                )
            # ==correlation surface for observable==
            # Computed once per observable after all detectors for this z are done.
            num_meas = circuit.num_measurements
            meas_rec_vertical, meas_rec_horizontal = (
                self.identify_correlation_surface_measurements(
                    horizontal_pipes_cs,
                    meas_rec_lst,
                    num_meas,
                )
            )
            # ==observable==
            # Loop over prisms_in_obs (not prism_pipes_zpm_temp) so that we use each
            # prism's own z-value for star ops and meas_rec lookups.
            obs_targets = []
            for prism in prisms_in_obs:
                prism_z = prism.position.z
                if isinstance(prism.kind, ZXPrism):
                    zpm = ZPM(z=prism_z, p=prism.kind.prep, m=prism.kind.meas)
                else:
                    continue
                print("zpm", zpm)
                data_positions = self.mapping_full[prism_z][prism].positions
                star_ops_x, star_ops_z = self.result_dct[prism_z]["star"]
                star_ops_x = [[Position3DHex(p.x, p.y, 0) for p in op] for op in star_ops_x]
                star_ops_z = [[Position3DHex(p.x, p.y, 0) for p in op] for op in star_ops_z]
                involved_data_qubits = []
                for obs_prism in prisms_in_obs:
                    involved_data_qubits += self.mapping_full[obs_prism.position.z][obs_prism].positions
                data_hex_set = {pe.hex for pe in data_positions if pe.hex is not None}
                involved_hex_set = {pe.hex for pe in involved_data_qubits if pe.hex is not None}
                star_ops_z_current = [
                    op for op in star_ops_z if set(op) & data_hex_set and set(op) & involved_hex_set
                ]
                star_ops_x_current = [
                    op for op in star_ops_x if set(op) & data_hex_set and set(op) & involved_hex_set
                ]
                data_positions_hex = {el.hex for el in data_positions}
                star_ops_z_current = [
                    [hex_pos for hex_pos in op if hex_pos in data_positions_hex]
                    for op in star_ops_z_current
                    if set(op) & data_hex_set and set(op) & data_positions_hex
                ]
                star_ops_x_current = [
                    [hex_pos for hex_pos in op if hex_pos in data_positions_hex]
                    for op in star_ops_x_current
                    if set(op) & data_hex_set and set(op) & data_positions_hex
                ]

                if zpm.m == BasisPrism.Z:
                    print("zpm m = Z")
                    if not star_ops_z_current:
                        effective_star_ops = star_ops_x_current
                    else:
                        effective_star_ops = star_ops_z_current
                    for star_op_z in effective_star_ops:
                        for pos in star_op_z:
                            pos_flat = Position3DHex(pos.x, pos.y, 0)
                            pe = next(p for p in data_positions if p.hex == pos_flat)
                            rec = (
                                next(
                                    m
                                    for m in meas_rec_lst
                                    if m.meas_type == "MZ"
                                    and m.z_value == prism_z
                                    and m.stabilizer is None
                                    and m.round == r
                                    and m.label == pe.label
                                ).abs_rec
                                - 1
                                - circuit.num_measurements
                            )
                            obs_targets.append(stim.target_rec(rec))
                        star_op_flat_set = set(star_op_z)  # already flattened to z=0
                        star_prism_position = next(
                            (
                                obs_prism.position
                                for obs_prism in prisms_in_obs
                                if any(
                                    pe.hex is not None
                                    and Position3DHex(pe.hex.x, pe.hex.y, 0) in star_op_flat_set
                                    for pe in self.mapping_full[obs_prism.position.z][obs_prism].positions
                                )
                            ),
                            None,
                        )
                        if star_prism_position is not None:
                            star_prism_position = Position3DHex(
                                star_prism_position.x, star_prism_position.y, 0
                            )
                        needs_cs_correction = (
                            star_prism_position is not None
                            and star_prism_position in reachable_via_horizontal
                        )
                        print("needs_cs_correction Z", needs_cs_correction)
                        if needs_cs_correction:
                            seen = cs_targets_seen_per_obs[obs_idx]
                            for (key_type, key_pipe), meas_lst in meas_rec_vertical.items():
                                if key_type == BasisPrism.Z:
                                    for t in meas_lst:
                                        if t.value not in seen:
                                            seen.add(t.value)
                                            obs_targets.append(t)
                                            print("added vertical cs", t)
                            for (key_type, key_pipe), meas_lst in meas_rec_horizontal.items():
                                if key_type == BasisPrism.Z:
                                    for t in meas_lst:
                                        if t.value not in seen:
                                            seen.add(t.value)
                                            obs_targets.append(t)
                                            print("added horizontal cs", t)
                        #    circuit.append("OBSERVABLE_INCLUDE", obs_targets, obs_idx)
                        #elif star_prism_position is not None:
                        #    circuit.append("OBSERVABLE_INCLUDE", obs_targets, obs_idx)
                        #else:
                        #    continue
                elif zpm.m == BasisPrism.X:
                    print("zpm m = X")
                    if not star_ops_x_current:
                        #if measurement color changed, there was no star op assigned
                        effective_star_ops = star_ops_z_current
                    else:
                        effective_star_ops = star_ops_x_current
                    for star_op_x in effective_star_ops:
                        for pos in star_op_x:
                            pos_flat = Position3DHex(pos.x, pos.y, 0)
                            pe = next(p for p in data_positions if p.hex == pos_flat)
                            rec = (
                                next(
                                    m
                                    for m in meas_rec_lst
                                    if m.meas_type == "MX"
                                    and m.z_value == prism_z
                                    and m.stabilizer is None
                                    and m.round == r
                                    and m.label == pe.label
                                ).abs_rec
                                - 1
                                - circuit.num_measurements
                            )
                            obs_targets.append(stim.target_rec(rec))
                        star_op_flat_set = set(star_op_x)  # already flattened to z=0
                        star_prism_position = next(
                            (
                                obs_prism.position
                                for obs_prism in prisms_in_obs
                                if any(
                                    pe.hex is not None
                                    and Position3DHex(pe.hex.x, pe.hex.y, 0) in star_op_flat_set
                                    for pe in self.mapping_full[obs_prism.position.z][obs_prism].positions
                                )
                            ),
                            None,
                        )
                        if star_prism_position is not None:
                            star_prism_position = Position3DHex(
                                star_prism_position.x, star_prism_position.y, 0
                            )
                        needs_cs_correction = (
                            star_prism_position is not None
                            and star_prism_position in reachable_via_horizontal
                        )
                        print("needs_cs_correction X", needs_cs_correction)
                        if needs_cs_correction:
                            seen = cs_targets_seen_per_obs[obs_idx]
                            for (key_type, key_pipe), meas_lst in meas_rec_vertical.items():
                                if key_type == BasisPrism.X:
                                    for t in meas_lst:
                                        if t.value not in seen:
                                            seen.add(t.value)
                                            obs_targets.append(t)
                            for (key_type, key_pipe), meas_lst in meas_rec_horizontal.items():
                                if key_type == BasisPrism.X:
                                    for t in meas_lst:
                                        if t.value not in seen:
                                            seen.add(t.value)
                                            obs_targets.append(t)
                            #circuit.append("OBSERVABLE_INCLUDE", obs_targets, obs_idx)
                        #elif star_prism_position is not None:
                            #circuit.append("OBSERVABLE_INCLUDE", obs_targets, obs_idx)
                        #else:
                        #    continue
                #obs_targets will contain elements that are from star operators but actually
                #already part of the correlation surface, we do not want these and filter them out
            obs_targets = list(set(obs_targets))
            circuit.append("OBSERVABLE_INCLUDE", obs_targets, obs_idx)
        return circuit, meas_rec_lst

    def create_stim_circuit_bell_multiplexing(
        self,
        rounds,
        p_init: float,
        p_meas: float,
        p_idle: float,
        p_gate2: float,
        cs_lst: list[CorrelationSurface],
    ) -> tuple[stim.Circuit, list[MeasRecInfo]]:
        """Create a syndrome extraction circuit based on Bell Multiplexing."""
        # store the measurement record labes together with vital information
        # IMPORTANT label starts from 1, not from 0 - i.e. offset by 1 between what
        # is displayed in the circuit svg and the measreclst.
        meas_rec_lst: list[MeasRecInfo] = []
        # just choose a convention for the CNOT order
        cnot_order = ["bottom_z", "sides_z", "top_z", "bottom_x", "sides_x", "top_x"]
        circuit = stim.Circuit()

        # special case if memory experiment
        memory_experiment = (
            len(cs_lst) == 0
            and len(self.prism_graph.prisms) == 1
            and len(self.prism_graph.pipes) == 0
        )
        if memory_experiment:
            sole_prism = next(iter(self.prism_graph.prisms))

            horizontal_pipes_cs_lst = [{}]
            prisms_in_obs_lst = [[sole_prism]]

            # run the observable loop once
            cs_lst = []
        else:
            # only check the relevant horizontal correlation surfaces and whether ver/hor cs
            horizontal_pipes_cs_lst = []
            for cs in cs_lst:
                horizontal_pipes_cs = self.prism_graph.find_ver_hor_correlation_surface(cs)
                horizontal_pipes_cs_lst.append(horizontal_pipes_cs)

            # later, for OBS_INCLUDE you need to know which prisms in the final z layer
            # are actually involved in the given correlation surfaces
            zx = self.prism_graph.to_zx_graph()
            max_z = max(self.z_values)

            #prisms_in_obs_lst = []
            #for cs in cs_lst:
            #    vertex_ids = cs.span_vertices
            #    positions_in_cs = {vid: zx.positions[vid] for vid in vertex_ids}
            #    prisms_in_cs = [
            #        prism
            #        for prism in self.prism_graph.prisms
            #        if prism.position in positions_in_cs.values()
            #    ]
            #    prisms_in_obs = [prism for prism in prisms_in_cs if prism.position.z == max_z]
            #    prisms_in_obs_lst.append(prisms_in_obs)
            #!update collection of prisms_in_obs_lst if obs can be somewhere else than max_z
            zx = self.prism_graph.to_zx_graph()
            prisms_in_obs_lst = []
            for cs in cs_lst:
                vertex_ids = cs.span_vertices
                positions_in_cs = {vid: zx.positions[vid] for vid in vertex_ids}
                prisms_in_cs = [
                    prism
                    for prism in self.prism_graph.prisms
                    if prism.position in positions_in_cs.values()
                ]
                prisms_in_obs = []
                for prism in prisms_in_cs:
                    # 1. Use the p2v dictionary for an instant lookup of the vertex id
                    v_id = zx.p2v.get(prism.position)
                    # 2. Access the internal PyZX graph via `zx.g` to check the number of neighbors
                    if v_id is not None and len(zx.g.neighbors(v_id)) == 1:
                        prisms_in_obs.append(prism)
                prisms_in_obs_lst.append(prisms_in_obs)
                print("prism in obs", prisms_in_obs)

        for s, z in enumerate(self.z_values):
            current_tick = 0  # re-initialize tick label per z
            # zpm values
            prism_pipes_zpm_temp = self.prism_pipes_to_ZPM[z]
            # stabilizer info etc
            prism_pipes_stabs = self.mapping_full[z]

            # define all current qubits (both ancilla and data) for comparison for idiling noise
            all_current_qubits = []
            for prism_pipe, prism_data in prism_pipes_stabs.items():
                # data qubits
                all_current_qubits.extend(prism_data.positions)
                # ancilla qubits
                if prism_data.stabilizers:
                    for stab in prism_data.stabilizers:
                        if stab.ancilla:
                            all_current_qubits.extend(stab.ancilla)

            # initialization of data qubits
            # need to track the currently initialized qubits to avoid incorrect detectors at r=0
            initialized_qubits = []
            for prism_pipe in prism_pipes_zpm_temp.keys():
                zpm = prism_pipes_zpm_temp[prism_pipe]
                data_positions = prism_pipes_stabs[prism_pipe].positions
                if zpm.p == BasisPrism.Z:
                    lst = [el.label for el in data_positions]
                    initialized_qubits += data_positions
                    circuit.append("R", [qubit for qubit in lst if qubit is not None])
                    circuit.append("X_ERROR", [qubit for qubit in lst if qubit is not None], p_init)
                elif zpm.p == BasisPrism.X:
                    lst = [el.label for el in data_positions]
                    initialized_qubits += data_positions
                    circuit.append("RX", [qubit for qubit in lst if qubit is not None])
                    circuit.append("Z_ERROR", [qubit for qubit in lst if qubit is not None], p_init)
            # r rounds of error correction based on stabilizer_type
            for r in range(rounds):
                current_tick, meas_rec_lst, circuit = self.build_syndrome_extraction_round(
                    circuit=circuit,
                    r=r,
                    rounds=rounds,
                    z=z,
                    prism_pipes_zpm_temp=prism_pipes_zpm_temp,
                    prism_pipes_stabs=prism_pipes_stabs,
                    cnot_order=cnot_order,
                    meas_rec_lst=meas_rec_lst,
                    p_init=p_init,
                    p_gate2=p_gate2,
                    p_meas=p_meas,
                    p_idle=p_idle,
                    all_current_qubits=all_current_qubits,
                    current_tick=current_tick,
                )

                # ---------add detectors based on meas_rec_lst-----------
                circuit = self.add_detector_annotation(
                    circuit=circuit,
                    meas_rec_lst=meas_rec_lst,
                    initialized_qubits=initialized_qubits,
                    prism_pipes_zpm_temp=prism_pipes_zpm_temp,
                    prism_pipes_stabs=prism_pipes_stabs,
                    z=z,
                    r=r,
                    rounds=rounds,
                    s=s,
                )

            # ----------final measurements + OBS--------------
            if z == max(self.z_values):
                circuit, meas_rec_lst = self.add_final_meas_and_obs(
                    circuit=circuit,
                    meas_rec_lst=meas_rec_lst,
                    prism_pipes_zpm_temp=prism_pipes_zpm_temp,
                    prism_pipes_stabs=prism_pipes_stabs,
                    horizontal_pipes_cs_lst=horizontal_pipes_cs_lst,
                    prisms_in_obs_lst=prisms_in_obs_lst,
                    cs_lst=cs_lst,
                    z=z,
                    r=r,
                    current_tick=current_tick,
                    p_meas=p_meas,
                )

        return circuit, meas_rec_lst

    def run_all_superdense(
        self,
        rounds,
        p_init: float,
        p_meas: float,
        p_idle: float,
        p_gate2: float,
        cs_lst: list[CorrelationSurface],
    ):
        """Run everything for a superdense circuit."""
        self.retrieve_stabilizers_operators()
        self.reorder_all_stabilizers()
        self.meas_prep_data_qubits()
        _ = self.hex_mapping_to_quadratic()
        circuit, meas_rec_lst = self.create_stim_circuit_bell_multiplexing(
            rounds=rounds,
            p_init=p_init,
            p_meas=p_meas,
            p_idle=p_idle,
            p_gate2=p_gate2,
            cs_lst=cs_lst,
        )
        self.meas_rec_lst = meas_rec_lst
        self.assign_qubit_coords_quadratic()
        circuit = self.add_qubit_coords_to_circuit_quadratic(circuit)
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
    cs_lst=[],  # empty
) -> list[sinter.TaskStats]:
    """Run experiments with sinter."""
    tasks = []
    for idx, circuit_builder in enumerate(circuit_builders):
        for p in p_values:
            se_type = "run_all_superdense"
            method = getattr(circuit_builder, se_type)
            circuit = method(
                rounds=rounds[idx], p_init=p, p_gate2=p, p_meas=p, p_idle=p, cs_lst=cs_lst
            )

            if add_missing_detectors:
                circuit = circuit + circuit.missing_detectors(unknown_input=False)
            tasks.append(
                sinter.Task(
                    circuit=circuit,
                    json_metadata={
                        "p": p,
                        "rounds": rounds[idx],
                        "code_idx": idx,
                        "d": circuit_builder.d,
                        "code": "color_code",
                    },
                )
            )

    stats = sinter.collect(
        num_workers=num_workers,
        tasks=tasks,
        custom_decoders=decoder_dict,
        decoders=[decoder_name],
        max_shots=max_shots,
        max_errors=max_errors,
        print_progress=True,
        save_resume_filepath=path,
        count_observable_error_combos=True,
    )
    return stats


def plot_experiment_sinter(stats: list[sinter.TaskStats], d_lst: list[int]):
    """Stats is what is plotted.

    If you have multiple observables that were tracked separately,
    you have to apply split_counts_for_observables(stats) first and
    plot the respective observable's outcome.

    Args:
        stats (list[sinter.TaskStats]): sinter output
        d_lst (list[int]): distances for plotting

    """
    fig, ax = plt.subplots()
    sinter.plot_error_rate(
        ax=ax,
        stats=stats,
        x_func=lambda stat: stat.json_metadata["p"],
        group_func=lambda stat: (
            f"d={d_lst[stat.json_metadata['code_idx']]}, r={stat.json_metadata['rounds']}"
        ),
    )
    ax.set_xlabel("Physical error rate p")
    ax.set_ylabel("Logical error rate")
    ax.set_title("Logical error rate vs physical error rate")
    ax.loglog()
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend(title="Code distance")
    plt.tight_layout()
    plt.show()
    return fig
