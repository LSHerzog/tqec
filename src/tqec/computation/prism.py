"""Defines the :py:class:`~tqec.computation.prism.Prism` class."""

from __future__ import annotations

from dataclasses import astuple, dataclass
from enum import Enum
from typing import Any

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

    #!todo add methods like shift_by, shift_in_direction, is_neighbor etc.

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
