from prism import Prism, ZXPrism, BasisPrism, Position3DHex
from dataclasses import dataclass
from tqec.utils.exceptions import TQECError

@dataclass(frozen=True)
class PrismPipeKind:
    """Pipe Kind for Prism stuff.

    hor and ver must be either X or Z, N is only allowed for the prisms but
    not for prism pipes (spatial). For temporal pipes, you need hor=ver="N".
    """

    hor: BasisPrism | None
    ver: BasisPrism | None
    has_hadamard: bool = False

    def __str__(self) -> str:
        return "".join(
            basis.value if basis is not None else "O" for basis in (self.hor, self.ver)
        ) + ("H" if self.has_hadamard else "")

    @property
    def is_temporal(self) -> bool:
        """If hor=ver=N, it is a temporal pipe."""
        return self.hor is BasisPrism.N and self.hor is BasisPrism.N

    @property
    def is_spatial(self) -> bool:
        """Spatial if both hor and ver not N."""
        return not self.is_temporal


@dataclass(frozen=True)
class PrismPipe:
    """Connection of two Prisms.

    Horizontal Pipes are the STDWs
    Vertical Pipes are just placeholder in time

    Attributes:
        u: The cube at the head of the pipe. The position of u will be guaranteed to be less than v.
        v: The cube at the tail of the pipe. The position of v will be guaranteed to be greater
            than u.
        kind: The kind of the pipe.

    """

    u: Prism
    v: Prism
    kind: PrismPipeKind

    #!todo add __post_init__

    @staticmethod
    def from_prisms(u: Prism, v: Prism, kind: PrismPipeKind):
        """Create pipe between two given prisms with specific ver/hor basis."""
        if not u.is_zx_prism and not v.is_zx_prism:
            raise TQECError("At least one cube must be a ZX cube to infer the pipe kind.")
        u, v = (u, v) if u.position.z < v.position.z else (v, u) #!ENOUGH TO THINK ONLY ABOUT TIME ORDERING? 
        if not u.position.is_neighbour(v.position):
            raise TQECError("The prisms must be neighbours to create a pipe.")

        hor = kind.hor
        ver = kind.ver

        if hor == BasisPrism.N and ver != BasisPrism.N:
            raise ValueError("If hor=N also ver must be N.")
        if hor != BasisPrism.N and ver == BasisPrism.N:
            raise ValueError("If ver=N also hor must be N.")

        #temporal pipe -> check whether u and v need temporal connection
        if hor == BasisPrism.N and ver == BasisPrism.N:
            if u.position.x != v.position.x or u.position.y!=v.position.y:
                raise ValueError("If hor=ver=N, then the pipe must be temporal,i.e. "
                "is allowed to differ only in pos.Z")
        if u.position.z != v.position.z:
            if hor != BasisPrism.N or ver != BasisPrism.N:
                raise ValueError("If temporal pipe (u.position.z!=v.position.z) " \
                "the hor=ver=N is necessary.")
        #spatial pipe -> check whether u and v need spatial connection + do the meas/prep colors fit?
        elif hor == BasisPrism.X and ver == BasisPrism.Z:
            if u.position.z != v.position.z:
                raise ValueError("hor=X and ver=Z must be a spatial pipe.")
        elif hor == BasisPrism.Z and ver == BasisPrism.X:
            if u.position.z != v.position.z:
                raise ValueError("hor=Z and ver=X must be a spatial pipe.")

        #make sure that the pipe colors fit the meas/prep colors of the prisms
        if kind.is_spatial:
            if isinstance(u.kind, ZXPrism):
                if u.kind.prep in (BasisPrism.X, BasisPrism.Z) and u.kind.prep != hor:
                    raise ValueError("prep of u must be same as hor of pipe")
                if u.kind.meas in (BasisPrism.X, BasisPrism.Z) and u.kind.meas != hor:
                    raise ValueError("meas of u must be same as hor of pipe")
            if isinstance(v.kind, ZXPrism):
                if v.kind.prep in (BasisPrism.X, BasisPrism.Z) and v.kind.prep != hor:
                    raise ValueError("prep of v must be same as hor of pipe")
                if v.kind.meas in (BasisPrism.X, BasisPrism.Z) and v.kind.meas != hor:
                    raise ValueError("meas of v must be same as hor of pipe")

        if kind.is_temporal:
            #v touches pipe with prep face and u touches pipe with meas face. 
            # thus these faces should be N
            if isinstance(u.kind, ZXPrism):
                if u.kind.meas != BasisPrism.N:
                    raise ValueError("The prep face that touches the temporal pipe must be N.")
                if v.kind.prep != BasisPrism.N:
                    raise ValueError("The meas face touching the temporal pipe must be N.")


        pipe_kind = PrismPipeKind(hor = hor, ver = ver)
        return PrismPipe(u, v, pipe_kind)


