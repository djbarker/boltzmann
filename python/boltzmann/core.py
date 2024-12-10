from __future__ import annotations
import logging
import numpy as np


from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from boltzmann.utils.option import Some, to_opt


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
class DomainMeta:
    """
    Describes the domain grid.

    The semantics are that each cell has a size of `dx`, the `lower` & `upper` arrays are the lower
    and upper _edges_ of the cells respectively.
    This means `N = (upper - lower) / dx`, and if `lower == upper` then we have no cells.
    The centre of the cells are located at `(x, y) = ((i + 0.5) * dx, (j + 0.5) * dx)`.
    """

    lower: np.ndarray
    upper: np.ndarray
    dx: float
    dims: int = field(init=False)
    counts: np.ndarray = field(init=False)

    def __post_init__(self):
        self.lower = to_array1d(self.lower)
        self.upper = to_array1d(self.upper)
        assert self.lower.ndim == 1
        assert self.lower.shape == self.upper.shape
        self.dims = len(self.lower)
        self.counts = np.array(
            (self.upper - self.lower) / self.dx,
            dtype=np.int32,
        )

        # sanity check values
        assert self.dx > 0
        for i, (l, u, c) in enumerate(zip(self.lower, self.upper, self.counts)):
            dx_ = (u - l) / c
            assert (
                abs(dx_ - self.dx) < 1e-8
            ), f"dx differs from dimension 0 for dimension {i}. [{l=}, {u=}, {c=} => {dx_=}]"

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

    @staticmethod
    def with_dx_and_counts(dx: float, counts: Array1dT) -> "DomainMeta":
        counts = to_array1d(counts)
        return DomainMeta(np.zeros_like(counts), counts * dx, dx)

    @staticmethod
    def with_extent_and_counts(
        lower: Array1dT, upper: Array1dT, counts: Array1dT
    ) -> "DomainMeta":
        lower = to_array1d(lower)
        upper = to_array1d(upper)
        counts = to_array1d(counts)
        assert lower.shape == counts.shape
        assert upper.shape == counts.shape
        dx = (upper[0] - lower[0]) / (counts[0] - 1)
        return DomainMeta(lower, upper, dx)

    def make_array(
        self,
        dims: DimsT = None,
        dtype: np.dtype = np.float32,
        fill: float | int = 0.0,
    ) -> np.ndarray:
        dims = to_dims(dims)
        dims = list(self.counts + 2) + dims  # + 2 for periodic BCs
        return np.full(dims, fill, dtype)


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
                x = c * np.sqrt(3) * t
            case (Some(x), None, Some(c)):
                x = x / np.sqrt(3)
                t = x / c
            case (Some(x), Some(t), None):
                x = x / np.sqrt(3)
            case _:
                msg = (
                    f"Must specify exactly two of dx, dt and cs! [{dx=}, {dt=}, {cs=}]"
                )
                raise ValueError(msg)

        return Scales(x, t, dm)

    @property
    def cs(self) -> float:
        # Remember, dx already contains the factor 1/sqrt(3).
        return self.dx / self.dt

    @property
    def inv(self) -> Scales:
        return Scales(1 / self.dx, 1 / self.dt, 1 / self.dm)

    def rescale(self, value: float, L: int = 0, T: int = 0, M: int = 0) -> float:
        """
        Correctly rescale a dimensional quantity.
        Dimensions are specified by passing non-zero values to the kwargs L, T and M.
        """
        return value * pow(self.dx, -L) * pow(self.dt, -T) * pow(self.dm, -M)


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

    def rescale(self, scale: Scales) -> FluidMeta:
        """
        Convert the unit scales.
        """
        return FluidMeta.make(
            rho=self.rho / (scale.dm / scale.dx**3),
            nu=self.nu / (scale.cs**2 * scale.dt),
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

    domain: DomainMeta
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
        nu_lu = self.fluid.rescale(self.scales).nu

        l_ = 0.25  # magic parameter
        tau_pos_lu = nu_lu + 0.5
        tau_neg_lu = l_ / nu_lu + 0.5

        if tau_pos_lu < 0.6:
            logger.warning(f"Small value for tau! [{tau_pos_lu=}]")

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
