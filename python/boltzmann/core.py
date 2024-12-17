from __future__ import annotations
import logging
import numpy as np


from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence, Type

from boltzmann.utils.option import Some, to_opt, map_opt, Option, unwrap


logger = logging.getLogger(__name__)


Array1dT = np.ndarray | Sequence[int] | Sequence[float]


def to_array1d(arr: Array1dT) -> np.ndarray:
    arr = np.squeeze(np.array(arr))
    assert np.ndim(arr) == 1, f"Expected 1d array for arr! [{arr.shape=!r}]"
    return arr


DimsT = list[int] | int | None


def to_dims(dims: DimsT) -> list[int]:
    match dims:
        case None:
            return []
        case int():
            return [dims]
        case list():
            return dims
        case _:
            raise TypeError(f"Expected None, int or list[int] for dims. [{dims=}]")


@dataclass
class Domain:
    """
    Describes the domain grid and provides functions for creating arrays.

    The semantics are that each cell has a size of `dx`, the `lower` & `upper` arrays are the lower
    and upper _edges_ of the cells respectively.
    This means `N = (upper - lower) / dx`, and if `lower == upper` then we have no cells.
    The centre of the cells are located at `(x, y) = ((i + 0.5) * dx, (j + 0.5) * dx)`.
    """

    lower: np.ndarray
    upper: np.ndarray
    counts: np.ndarray
    dx: float
    dims: int

    @staticmethod
    def make(
        lower: Option[Array1dT] = None,
        upper: Option[Array1dT] = None,
        counts: Option[Array1dT] = None,
        dx: Option[float] = None,
    ) -> Domain:
        lower_ = map_opt(lower, to_array1d)
        upper_ = map_opt(upper, to_array1d)
        counts_ = map_opt(counts, to_array1d)
        dx = to_opt(dx)

        eps = 1e-8

        # When specified, dx is used to infer the others.
        # The lower bound is always implicitly zero if not specified.
        # After this match ll, uu, and cc should be non-None, we don't care about dx.
        match (lower_, upper_, counts_, dx):
            case Some(ll), Some(uu), Some(cc), None:
                pass
            case None, Some(uu), Some(cc), None:
                ll = np.zeros_like(uu)
            case Some(ll), Some(uu), None, Some(d):
                cc = (uu - ll + eps) / d
            case Some(ll), None, Some(cc), Some(d):
                uu = ll + cc * d
            case None, None, Some(cc), Some(d):
                uu = d * cc
                ll = np.zeros_like(uu)
            case None, Some(uu), None, Some(d):
                cc = uu / d
                ll = np.zeros_like(uu)
            case _:
                raise ValueError("Invalid argument combination!")

        lower = ll
        upper = uu
        counts = cc

        # sanity check values

        assert lower.ndim == 1
        assert lower.shape == upper.shape
        assert lower.shape == counts.shape
        assert upper.shape == counts.shape
        assert all(upper > lower)
        assert all(counts > 0)
        dx_ = (upper - lower) / counts
        assert np.allclose(dx_, dx_[0])

        counts = np.array(counts, dtype=np.int32)

        return Domain(lower, upper, counts, dx_[0], len(counts))

    def get_dim(self, dim: int) -> np.ndarray:
        assert dim < self.dims, f"Invalid dim! [{dim=}, {self.dims=}]"
        return np.linspace(self.lower[dim], self.upper[dim], self.counts[dim])

    @property
    def x(self) -> np.ndarray:
        return self.get_dim(0)

    @property
    def y(self) -> np.ndarray:
        return self.get_dim(1)

    @property
    def z(self) -> np.ndarray:
        return self.get_dim(2)

    def get_extent(self, dim: int) -> float:
        assert dim < self.dims, f"Invalid dim! [{dim=}, {self.dims=}]"
        return self.upper[dim] - self.lower[dim]

    @property
    def width(self) -> float:
        return self.get_extent(0)

    @property
    def height(self) -> float:
        return self.get_extent(1)

    @property
    def depth(self) -> float:
        return self.get_extent(2)

    def make_array(
        self,
        dims: DimsT = None,
        dtype: Type[np.float32] | Type[np.int32] = np.float32,
        fill: float | int = 0.0,
    ) -> np.ndarray:
        dims = to_dims(dims)
        dims = [int(np.prod(self.counts + 2))] + dims  # + 2 for periodic BCs
        return np.full(dims, fill, dtype)

    def unflatten(self, arr: np.ndarray, rev: bool = True) -> np.ndarray:
        """
        Expands the collapsed spatial index into one for x, y, and (if needed) z.

        For the simulation data is just stored in a flat array where the spatial indices are collapsed,
        this function undoes that collapsing to let us slice into the array along desired axes.

        NOTE: One perculiarity of our data layout is that (in 2D) the y axis is first.

        This function is idemponent.

        .. code-block::

            rho = dom.make_array()    # rho.shape == [nx*nx]
            rho = dom.unflatten(rho)  # rho.shape == [ny, nx]
            vel = dom.make_array(2)   # vel.shape == [nx*nx, 2]
            vel = dom.unflatten(vel)  # vel.shape == [ny, nx, 2]
        """
        n = arr.shape[1:]
        c = tuple(self.counts + 2)

        # looks already unflattened
        if (len(arr.shape) >= self.dims) and (
            (arr.shape[: self.dims] == c) or (arr.shape[: self.dims] == c[::-1])
        ):
            return arr

        if rev:
            c = c[::-1]

        return np.reshape(arr, c + n)


@dataclass
class TimeMeta:
    """
    Contains parameters for output interval and simulation duration.
    """

    dt_output: float
    t_max: float
    output_count: int

    def batch_steps(self, dt_step: float) -> int:
        return int(self.dt_output / dt_step + 1e-8)

    @staticmethod
    def make(
        *,
        dt_output: float | None = None,
        t_max: float | None = None,
        output_count: int | None = None,
    ) -> "TimeMeta":
        d_ = to_opt(dt_output)
        t_ = to_opt(t_max)
        i_ = to_opt(output_count)

        match (d_, t_, i_):
            case (None, Some(t), Some(i)):
                d = t / i
            case (Some(d), None, Some(i)):
                t = i * d
            case (Some(d), Some(t), None):
                i = int(t / d + 1e-8)
            case _:
                raise ValueError(
                    f"Must specify exactly two arguments! [{dt_output=}, {t_max=}, {output_count=}]"
                )

        return TimeMeta(d, t, i)


@dataclass
class Scales:
    """
    Characteristic scale of fundamental units.
    Used to convert between physical units (SI) and simulation (lattice) units.
    """

    dx: float
    dt: float
    dm: float

    @staticmethod
    def make(
        *,
        dx: float | None = None,
        dt: float | None = None,
        cs: float | None = None,
        dm: float = 1.0,
    ) -> Scales:
        x_ = to_opt(dx)
        t_ = to_opt(dt)
        c_ = to_opt(cs)

        match (x_, t_, c_):
            case (None, Some(t), Some(c)):
                x = (c * np.sqrt(3)) * t
            case (Some(x), None, Some(c)):
                t = x / (c * np.sqrt(3))
            case (Some(x), Some(t), None):
                pass
            case _:
                msg = f"Must specify exactly two of dx, dt and cs! [{dx=}, {dt=}, {cs=}]"
                raise ValueError(msg)

        return Scales(x, t, dm)

    @property
    def cs(self) -> float:
        """
        NOTE: Only use this if you _really_ mean the speed of sound.
              If you are just converting a velocity to physical units this is the wrong thing.
        """
        return (self.dx / self.dt) / np.sqrt(3)

    def to_lattice_units(
        self, value: float | np.ndarray, L: int = 0, T: int = 0, M: int = 0
    ) -> float | np.ndarray:
        return value * pow(self.dx, -L) * pow(self.dt, -T) * pow(self.dm, -M)

    def to_physical_units(
        self, value: float | np.ndarray, L: int = 0, T: int = 0, M: int = 0
    ) -> float | np.ndarray:
        return value * pow(self.dx, L) * pow(self.dt, T) * pow(self.dm, M)


VELOCITY = dict(L=1, T=-1)
ACCELERATION = dict(L=1, T=-2)
DENSITY = dict(M=1, L=-3)


@dataclass
class FluidMeta:
    rho: float
    mu: float
    nu: float

    @staticmethod
    def make(
        *, rho: float | None = None, mu: float | None = None, nu: float | None = None
    ) -> FluidMeta:
        r_ = to_opt(rho)
        m_ = to_opt(mu)
        n_ = to_opt(nu)

        match (r_, m_, n_):
            case (None, Some(m), Some(n)):
                r = m / n
            case (Some(r), None, Some(n)):
                m = r * n
            case (Some(r), Some(m), None):
                n = m / r
            case _:
                msg = f"Must specify exactly two of rho, mu and nu! [{rho=}, {mu=}, {nu=}]"
                raise ValueError(msg)

        return FluidMeta(r, m, n)

    def to_lattice_units(self, scale: Scales) -> FluidMeta:
        """
        Convert the unit scales.
        """
        return FluidMeta.make(
            rho=self.rho / (scale.dm / scale.dx**3),
            nu=self.nu / (scale.cs**2 * scale.dt),
        )

    def to_physical_units(self, scale: Scales) -> FluidMeta:
        """
        Convert the unit scales.
        """
        return FluidMeta.make(
            rho=self.rho * (scale.dm / scale.dx**3),
            nu=self.nu * (scale.cs**2 * scale.dt),
        )


WATER = FluidMeta.make(mu=0.001, rho=1000)


@dataclass
class SimulationMeta:
    """
    SimulationMeta is supposed to the all of the parameters needed for the simulation,
    however I don't like it.

    Some of this stuff is fine like output config, domain meta and scale,
    but some of it is rather specific to the particular simulation.
    """

    domain: Domain
    time: TimeMeta
    scales: Scales

    # These two are less generic so not sure they should live here?
    # We may want multiple fluids which even have different timestamps.
    fluid: FluidMeta

    # We have BGK & TRT params baked in here: they should be separate too.
    tau_pos_lu: float = field(init=False)
    tau_neg_lu: float = field(init=False)
    w_pos_lu: float = field(init=False)
    w_neg_lu: float = field(init=False)

    def __post_init__(self) -> None:
        nu_lu = self.fluid.to_lattice_units(self.scales).nu

        l_ = 0.25  # magic parameter
        tau_pos_lu = nu_lu + 0.5
        tau_neg_lu = l_ / nu_lu + 0.5

        if tau_pos_lu < 0.6:
            # stability can suffer but sim will be accurate
            logger.warning(f"Small value for tau! [{tau_pos_lu=}]")

        if tau_pos_lu > 1.0:
            # error grows with tau and this leads to very inaccurate simulations
            raise ValueError(f"Large value for tau! [{tau_pos_lu=}]")

        w_pve_lu = 1.0 / tau_pos_lu
        w_nve_lu = 1.0 / tau_neg_lu

        self.tau_pos_lu = tau_pos_lu
        self.tau_neg_lu = tau_neg_lu
        self.w_pos_lu = w_pve_lu
        self.w_neg_lu = w_nve_lu


# Cell type
class CellType(Enum):
    FLUID = 0
    BC_WALL = 1
    BC_VELOCITY = 2
    FIXED_PRESSURE = 3


class Model:
    def __init__(self, ws: list[float], qs: list[list[float]]) -> None:
        assert len(ws) == len(qs)

        js = np.array([qs.index([-q[0], -q[1]]) for q in qs])

        self.ws = np.array(ws, dtype=np.float32)
        self.qs = np.array(qs, dtype=np.int32)
        self.js = np.array(js, dtype=np.int32)

        self.Q = self.qs.shape[0]
        self.D = self.qs.shape[1]


# NOTE: order is not arbitrary
#       1. rest velocity at index zero
#       2. pairs of opposite velocities follow
D2Q9 = Model(
    [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
    [
        [0, 0],
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
        [1, 1],
        [-1, -1],
        [-1, 1],
        [1, -1],
    ],
)

# NOTE: order is not arbitrary
#       1. rest velocity at index zero
#       2. pairs of opposite velocities follow
D2Q5 = Model(
    [1 / 3, 1 / 6, 1 / 6, 1 / 6, 1 / 6],
    [
        [0, 0],
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
    ],
)


def calc_equilibrium(
    vel: np.ndarray,
    rho: np.ndarray,
    feq: np.ndarray,
    model: Model,
):
    """
    Velocity is measured in lattice-units.
    """
    vv = np.sum(vel**2, axis=-1)
    for i in range(model.Q):
        w = model.ws[i]
        q = model.qs[i]
        qv = vel @ q
        feq[:, i] = rho * w * (1 + 3.0 * qv + 4.5 * qv**2 - (3.0 / 2.0) * vv)
