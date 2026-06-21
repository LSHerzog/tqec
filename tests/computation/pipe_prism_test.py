import pytest

from tqec.computation.pipe_prism import PrismPipe, PrismPipeKind
from tqec.computation.prism import BasisPrism, Position3DHex, Prism, ZXPrism
from tqec.utils.exceptions import TQECError


def test_prism_pipe_kind() -> None:
    kind_spatial = PrismPipeKind(hor=BasisPrism.X, ver=BasisPrism.Z, has_hadamard=False)
    assert str(kind_spatial) == "XZ"
    assert kind_spatial.is_spatial
    assert not kind_spatial.is_temporal

    kind_temporal = PrismPipeKind(hor=BasisPrism.N, ver=BasisPrism.N)
    assert str(kind_temporal) == "NN"
    assert kind_temporal.is_temporal
    assert not kind_temporal.is_spatial

def test_prism_pipe_from_prisms_validation() -> None:
    # Using your explicit valid hex neighbor setup
    u_base = Prism(Position3DHex(0, 0, 0), ZXPrism.from_str("XX"))
    v_spatial_neighbor = Prism(Position3DHex(1, 1, 0), ZXPrism.from_str("XX"))

    with pytest.raises(ValueError, match="If hor=N also ver must be N\\."):
        PrismPipe.from_prisms(u_base, v_spatial_neighbor, PrismPipeKind(BasisPrism.N, BasisPrism.X))
    with pytest.raises(ValueError, match="If ver=N also hor must be N\\."):
        PrismPipe.from_prisms(u_base, v_spatial_neighbor, PrismPipeKind(BasisPrism.X, BasisPrism.N))

    far_prism = Prism(Position3DHex(5, 5, 0), ZXPrism.from_str("XX"))
    with pytest.raises(TQECError, match="The prisms must be neighbours to create a pipe\\."):
        PrismPipe.from_prisms(u_base, far_prism, PrismPipeKind(BasisPrism.X, BasisPrism.Z))

    with pytest.raises(
            ValueError,
            match="the pipe must be temporal,i\\.e\\. is allowed to differ only in pos\\.Z"
        ):
            PrismPipe.from_prisms(u_base,
                                  v_spatial_neighbor,
                                  PrismPipeKind(BasisPrism.N, BasisPrism.N)
                                  )

def test_prism_pipe_basis_matching() -> None:
    u_mismatch = Prism(Position3DHex(0, 0, 0), ZXPrism.from_str("ZX"))
    v_neighbor = Prism(Position3DHex(1, 1, 0), ZXPrism.from_str("XX"))
    spatial_kind = PrismPipeKind(BasisPrism.X, BasisPrism.Z)  # hor=X

    with pytest.raises(ValueError, match="prep of v must be same as hor of pipe"):
        PrismPipe.from_prisms(u_mismatch, v_neighbor, spatial_kind)

    u_temporal_invalid = Prism(Position3DHex(0, 0, 0), ZXPrism.from_str("XX"))
    v_temporal_valid = Prism(Position3DHex(0, 0, 1), ZXPrism.from_str("NX"))
    temporal_kind = PrismPipeKind(BasisPrism.N, BasisPrism.N)

    with pytest.raises(
            ValueError,
            match="The meas face that touches the temporal pipe must be N\\."
        ):
            PrismPipe.from_prisms(u_temporal_invalid, v_temporal_valid, temporal_kind)

    u_base = Prism(Position3DHex(0, 0, 0), ZXPrism.from_str("XX"))
    v_spatial_neighbor = Prism(Position3DHex(1, 1, 0), ZXPrism.from_str("XX"))
    with pytest.raises(
        ValueError,
            match="A spatial pipe must have different basis walls\\. You got hor=X, ver=X"
        ):
            PrismPipe.from_prisms(
                u_base,
                v_spatial_neighbor,
                PrismPipeKind(hor=BasisPrism.X, ver=BasisPrism.X)
            )


def test_valid_prism_pipe_instantiation() -> None:
    u_prism = Prism(Position3DHex(0, 0, 1), ZXPrism.from_str("NX"))
    v_prism = Prism(Position3DHex(0, 0, 0), ZXPrism.from_str("XN"))
    temporal_kind = PrismPipeKind(BasisPrism.N, BasisPrism.N)

    pipe = PrismPipe.from_prisms(u_prism, v_prism, temporal_kind)

    assert pipe.u.position.z == 0
    assert pipe.v.position.z == 1


def test_direction_connecting_bdry() -> None:
    u_even = Prism(Position3DHex(2, 2, 0), ZXPrism.from_str("XX"))

    v_c = Prism(Position3DHex(3, 1, 0), ZXPrism.from_str("XX"))
    pipe_c = PrismPipe.from_prisms(u_even, v_c, PrismPipeKind(BasisPrism.X, BasisPrism.Z))
    assert pipe_c.direction_connecting_bdry() == "b"

    v_b = Prism(Position3DHex(1, 3, 0), ZXPrism.from_str("XX"))
    pipe_b = PrismPipe.from_prisms(u_even, v_b, PrismPipeKind(BasisPrism.X, BasisPrism.Z))
    assert pipe_b.direction_connecting_bdry() == "a"

    v_a = Prism(Position3DHex(3, 3, 0), ZXPrism.from_str("XX"))
    pipe_a = PrismPipe.from_prisms(u_even, v_a, PrismPipeKind(BasisPrism.X, BasisPrism.Z))
    assert pipe_a.direction_connecting_bdry() == "c"
