from tqec.computation.prism import BasisPrism, Port, Position3DHex, Prism, ZXPrism


def test_zx_prism_kind() -> None:
    assert len(ZXPrism.all_kinds()) == 6
    kind = ZXPrism.from_str("XX")
    assert str(kind) == "XX"
    assert kind.as_tuple() == (BasisPrism.X, BasisPrism.X)

    kind_mixed = ZXPrism.from_str("XN")
    assert str(kind_mixed) == "XN"
    assert kind_mixed.prep == BasisPrism.X
    assert kind_mixed.meas == BasisPrism.N


def test_zx_prism() -> None:
    prism = Prism(Position3DHex(0, 0, 0), ZXPrism.from_str("XX"))
    assert prism.is_zx_prism
    assert not prism.is_port
    assert str(prism) == "XX(0, 0, 0)"
    assert prism.to_dict() == {
        "position": (0, 0, 0),
        "kind": "XX",
        "label": "",
    }


def test_port_prism() -> None:
    prism = Prism(Position3DHex(0, 0, 0), Port(), "p")
    assert prism.is_port
    assert not prism.is_zx_prism
    assert str(prism) == "PORT(0, 0, 0)"

    assert prism == Prism(Position3DHex(0, 0, 0), Port(), "p")
    assert prism.to_dict() == {
        "position": (0, 0, 0),
        "kind": "PORT",
        "label": "p",
    }


def test_prism_from_dict() -> None:
    prism_dict = {
        "position": (0, 0, 0),
        "kind": "XX",
        "label": "",
    }
    assert Prism.from_dict(prism_dict) == Prism(Position3DHex(0, 0, 0), ZXPrism.from_str("XX"))