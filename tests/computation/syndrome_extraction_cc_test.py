import pytest
import stim

from tqec.computation.pipe_prism import PrismPipe, PrismPipeKind
from tqec.computation.prism import BasisPrism, Position3DHex, Prism, ZXPrism
from tqec.computation.prism_graph import PrismGraph
from tqec.computation.syndrome_extraction_cc import (
    ZPM,
    MeasRecInfo,
    PositionEntry,
    SyndromeExtractionStimCC,
)


def test_zpm_from_string_parses_valid_string():
    zpm = ZPM.from_string("12XZ")
    assert zpm == ZPM(z=12, p=BasisPrism.X, m=BasisPrism.Z)
    assert str(zpm) == "12XZ"  # roundtrip


def test_zpm_from_string_rejects_malformed_string():
    with pytest.raises(ValueError, match="expected format"):
        ZPM.from_string("12X")  # missing one letter


def test_zpm_from_string_rejects_invalid_letter():
    with pytest.raises(ValueError, match="P field"):
        ZPM.from_string("12QZ")


def test_zpm_from_string_accepts_n_basis():
    zpm = ZPM.from_string("0NN")
    assert zpm.p == BasisPrism.N
    assert zpm.m == BasisPrism.N


def test_canonical_sorts_by_x_then_y_then_z():
    a = Position3DHex(2, 1, 0)
    b = Position3DHex(0, 0, 0)
    c = Position3DHex(1, 5, 0)
    assert SyndromeExtractionStimCC.canonical((a, b, c)) == (b, c, a)


def test_canonical_is_order_independent():
    a = Position3DHex(2, 1, 0)
    b = Position3DHex(0, 0, 0)
    assert SyndromeExtractionStimCC.canonical((a, b)) == SyndromeExtractionStimCC.canonical((b, a))


@pytest.fixture
def two_patch_setup():
    """A prism (ZZ) connected via an X-pipe to a second prism, disjoint data qubits."""  # noqa: D401
    p1 = Prism(position=Position3DHex(0, 0, 0), kind=ZXPrism.from_str("ZZ"))
    p2 = Prism(position=Position3DHex(2, 0, 0), kind=ZXPrism.from_str("ZZ"))
    pipe = PrismPipe(p1, p2, PrismPipeKind(hor=BasisPrism.X, ver=BasisPrism.Z))

    dq_p1 = [Position3DHex(0, 0, 0), Position3DHex(1, 1, 0)]
    dq_pipe = [Position3DHex(2, 0, 0), Position3DHex(3, 1, 0)]

    data_temp = {p1: dq_p1, pipe: dq_pipe}
    zpm_p1 = ZPM(z=0, p=BasisPrism.Z, m=BasisPrism.Z)
    zpm_pipe = ZPM(z=0, p=BasisPrism.X, m=BasisPrism.X)
    zpm_temp = {p1: zpm_p1, pipe: zpm_pipe}

    return dq_p1, dq_pipe, data_temp, zpm_temp, zpm_p1, zpm_pipe


def test_get_zpm_for_stab_in_both_x_and_z_searches_prisms(two_patch_setup):
    dq_p1, _, data_temp, zpm_temp, zpm_p1, _ = two_patch_setup
    stab = tuple(dq_p1)
    result = SyndromeExtractionStimCC.get_zpm_for_stab(
        stab, data_temp, zpm_temp, stabs_x=[stab], stabs_z=[stab]
    )
    assert result == zpm_p1


def test_get_zpm_for_stab_only_in_x_searches_pipes(two_patch_setup):
    _, dq_pipe, data_temp, zpm_temp, _, zpm_pipe = two_patch_setup
    stab = tuple(dq_pipe)
    result = SyndromeExtractionStimCC.get_zpm_for_stab(
        stab, data_temp, zpm_temp, stabs_x=[stab], stabs_z=[]
    )
    assert result == zpm_pipe


def test_get_zpm_for_stab_no_overlap_returns_none(two_patch_setup):
    _, _, data_temp, zpm_temp, _, _ = two_patch_setup
    stab = (Position3DHex(99, 99, 0),)
    result = SyndromeExtractionStimCC.get_zpm_for_stab(
        stab, data_temp, zpm_temp, stabs_x=[], stabs_z=[]
    )
    assert result is None


def test_get_zpm_for_stab_partial_overlap_is_sufficient(two_patch_setup):
    _, dq_pipe, data_temp, zpm_temp, _, zpm_pipe = two_patch_setup
    stab = (dq_pipe[0], Position3DHex(50, 50, 0))
    result = SyndromeExtractionStimCC.get_zpm_for_stab(
        stab, data_temp, zpm_temp, stabs_x=[], stabs_z=[]
    )
    assert result == zpm_pipe


def test_assign_stabilizer_info_weight2_minimal_assignment():
    stab = (Position3DHex(0, 0, 0), Position3DHex(1, 1, 0))
    info = SyndromeExtractionStimCC.assign_stabilizer_info(stab)
    assert info.top is None and info.sides is None and info.bottom is None and info.ancilla is None
    assert len(info.data_qubits) == 2


def test_assign_stabilizer_info_weight6_hexagon():
    coords = [(0, 0), (1, 1), (2, 0), (3, -1), (2, -2), (1, -1)]
    stab = tuple(Position3DHex(x, y, 0) for x, y in coords)
    info = SyndromeExtractionStimCC.assign_stabilizer_info(stab)
    assert sorted(pe.rect for pe in info.top) == [(0, 0), (1, 0)]
    assert sorted(pe.rect for pe in info.sides) == [(-1, -1), (2, -1)]
    assert sorted(pe.rect for pe in info.bottom) == [(0, -2), (1, -2)]
    assert sorted(pe.rect for pe in info.ancilla) == [(0, -1), (1, -1)]


def test_assign_stabilizer_info_weight4_standard():
    coords = [(0, 0), (1, 1), (2, 0), (1, -1)]
    stab = tuple(Position3DHex(x, y, 0) for x, y in coords)
    info = SyndromeExtractionStimCC.assign_stabilizer_info(stab)
    assert len(info.top) == 2
    assert len(info.sides) == 2
    assert info.bottom == []
    assert sorted(pe.rect for pe in info.ancilla) == [(0, -1), (1, -1)]


def test_assign_stabilizer_info_weight3():
    coords = [(0, 0), (1, 1), (1, -1)]
    stab = tuple(Position3DHex(x, y, 0) for x, y in coords)
    info = SyndromeExtractionStimCC.assign_stabilizer_info(stab)
    assert len(info.top) == 2
    assert len(info.sides) == 1
    assert len(info.data_qubits) == 3


def _entries(labels):
    return [PositionEntry(hex=None, rect=(i, 0), label=label) for i, label in enumerate(labels)]


def test_append_tick_with_idle_noise_adds_noise_on_idle_qubits_and_increments_tick():
    circuit = stim.Circuit()
    tick = SyndromeExtractionStimCC.append_tick_with_idle_noise(
        circuit, _entries([0, 1, 2]), current_tick=3, p_idle=0.01, active_qubits={0, 1}
    )
    depolarize = next(instr for instr in circuit if instr.name == "DEPOLARIZE1")
    assert [t.value for t in depolarize.targets_copy()] == [2]
    assert tick == 4


def test_append_tick_with_idle_noise_no_noise_when_all_active():
    circuit = stim.Circuit()
    SyndromeExtractionStimCC.append_tick_with_idle_noise(
        circuit, _entries([0, 1]), current_tick=0, p_idle=0.02, active_qubits={0, 1}
    )
    assert [instr.name for instr in circuit] == ["TICK"]


def test_get_active_qubits_since_last_tick_collects_targets_after_tick():
    circuit = stim.Circuit()
    circuit.append("H", [5])
    circuit.append("TICK")
    circuit.append("CX", [0, 1])
    circuit.append("M", [2])
    assert SyndromeExtractionStimCC.get_active_qubits_since_last_tick(circuit) == {0, 1, 2}


@pytest.fixture
def se():
    """Bypass __init__ since pipe_matches doesn't touch self state."""
    return SyndromeExtractionStimCC.__new__(SyndromeExtractionStimCC)


def _prism(x, y, z, kind="ZZ"):
    return Prism(position=Position3DHex(x, y, z), kind=ZXPrism.from_str(kind))


def _pipe(ux, uy, uz, vx, vy, vz):
    return PrismPipe(
        _prism(ux, uy, uz), _prism(vx, vy, vz), PrismPipeKind(hor=BasisPrism.X, ver=BasisPrism.Z)
    )


def test_pipe_matches_prisms_ignore_z_and_kind(se):
    assert se.pipe_matches(_prism(0, 0, 0, "ZZ"), _prism(0, 0, 5, "XX")) is True


def test_pipe_matches_prisms_different_xy_no_match(se):
    assert se.pipe_matches(_prism(0, 0, 0), _prism(1, 1, 0)) is False


def test_pipe_matches_pipes_ignore_z(se):
    assert se.pipe_matches(_pipe(0, 0, 0, 1, 1, 0), _pipe(0, 0, 9, 1, 1, 9)) is True


def test_pipe_matches_prism_vs_pipe_never_matches(se):
    assert se.pipe_matches(_prism(0, 0, 0), _pipe(0, 0, 0, 1, 1, 0)) is False


def _meas_rec(z_value, stabilizer, round_, label=0):
    return MeasRecInfo(
        meas_type="data_mz",
        pipe_prism=None,
        stabilizer=stabilizer,
        abs_rec=0,
        z_value=z_value,
        round=round_,
        label=label,
        tick=0,
    )


def test_data_meas_rec_lst_pipe_prism_filters_on_all_conditions():
    matching = _meas_rec(z_value=0, stabilizer=None, round_=2)
    wrong_round = _meas_rec(z_value=0, stabilizer=None, round_=1)
    has_stabilizer = _meas_rec(z_value=0, stabilizer="STAB", round_=2)
    wrong_z = _meas_rec(z_value=1, stabilizer=None, round_=2)

    result = SyndromeExtractionStimCC.data_meas_rec_lst_pipe_prism(
        [matching, wrong_round, has_stabilizer, wrong_z], z=1, rounds=3
    )
    assert result == [matching]


def test_data_meas_rec_lst_pipe_prism_no_matches_returns_empty_list():
    lst = [_meas_rec(z_value=5, stabilizer=None, round_=2)]
    assert SyndromeExtractionStimCC.data_meas_rec_lst_pipe_prism(lst, z=1, rounds=3) == []


def test_run_all_superdense_two_patches_circuit_is_well_formed():
    g = PrismGraph("two-patches")
    prisms = [
        (Position3DHex(2, 2, -1), "ZN", "init"),
        (Position3DHex(2, 2, 0), "NX", "left"),
        (Position3DHex(3, 1, 0), "XN", "right"),
        (Position3DHex(3, 1, 1), "NZ", "final"),
    ]
    for pos, kind, label in prisms:
        g.add_prism(pos, kind, label)

    pipe_kind = PrismPipeKind(hor=BasisPrism.N, ver=BasisPrism.N)
    g.add_pipe(prisms[0][0], prisms[1][0], pipe_kind)
    pipe_kind = PrismPipeKind(hor=BasisPrism.X, ver=BasisPrism.Z)
    g.add_pipe(prisms[1][0], prisms[2][0], pipe_kind)
    pipe_kind = PrismPipeKind(hor=BasisPrism.N, ver=BasisPrism.N)
    g.add_pipe(prisms[2][0], prisms[3][0], pipe_kind)

    d = 7
    rounds = d
    p = 0.001
    cs_lst = g.find_correlation_surfaces()

    se = SyndromeExtractionStimCC(g, d)
    circuit = se.run_all_superdense(
        rounds,
        p,
        p,
        p,
        p,
        cs_lst=cs_lst,
    )

    # the DEM must be buildable
    dem = circuit.detector_error_model()
    assert dem is not None

    # code distance check
    assert len(circuit.shortest_graphlike_error()) == 7

    # no missing detectors
    missing = circuit.missing_detectors(unknown_input=False)
    assert len(missing) == 0
