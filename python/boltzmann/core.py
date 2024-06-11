import logging
import numpy as np


from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence, Protocol


log = logging.getLogger(__name__)


ExtentT = np.ndarray | Sequence[Sequence[float]]


def to_extent(extent: ExtentT) -> np.ndarray:
    extent = np.array(extent)
    assert (
        extent.ndim == 2 and extent.shape[1] == 2
    ), f"Expected array of pairs for domain extents! [extent.shape={extent.shape!r}]"
    return extent


CountsT = np.ndarray | list[int]


def to_counts(counts: CountsT) -> np.ndarray:
    counts = np.squeeze(np.array(counts))
    assert np.ndim(counts) == 1, f"Expected 1d array for counts! [{counts.shape=!r}]"
    return counts


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

    extent_si: np.ndarray
    dx_si: float
    dims: int = field(init=False)
    counts: np.ndindex = field(init=False)

    def __post_init__(self):
        self.extent_si = to_extent(self.extent_si)
        self.dims = self.extent_si.shape[0]
        self.counts = np.array(
            (self.extent_si[:, 1] - self.extent_si[:, 0]) / self.dx_si,
            dtype=np.int32,
        )

        # sanity check values
        assert self.dx_si > 0
        for i, (e, c) in enumerate(zip(self.extent_si, self.counts)):
            dx_ = (e[1] - e[0]) / c
            assert (
                abs(dx_ - self.dx_si) < 1e-8
            ), f"dx differs from dimension 0 for dimension {i}. [{e=}, {c=} => {dx_=}]"

    def get_dim(self, dim: int) -> np.ndarray:
        assert dim <= self.dims, f"Invalid dim! [{dim=}, {self.dims=}]"
        return np.linspace(self.extent_si[dim, 0], self.extent_si[dim, 1], self.counts[dim])

    @property
    def x(self) -> np.ndarray:
        return self.get_dim(0)

    @property
    def y(self) -> np.ndarray:
        return self.get_dim(1)

    @property
    def z(self) -> np.ndarray:
        return self.get_dim(2)

    @staticmethod
    def with_dx_and_counts(dx_si: float, counts: CountsT) -> "DomainMeta":
        counts = to_counts(counts)
        extent_si = np.vstack([np.zeros_like(counts), counts])
        return DomainMeta(dx_si, extent_si)

    @staticmethod
    def with_extent_and_counts(extent_si: ExtentT, counts: CountsT) -> "DomainMeta":
        extent_si = to_extent(extent_si)
        counts = to_counts(counts)
        assert extent_si.shape[0] == counts.shape[0]
        dx_si = (extent_si[0][1] - extent_si[0][0]) / (counts[0] - 1)
        return DomainMeta(extent_si=extent_si, dx_si=dx_si)

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
class FluidMeta:
    mu_si: float
    rho_si: float
    nu_si: float = field(init=False)

    def __post_init__(self):
        self.nu_si = self.mu_si / self.rho_si


FluidMeta.WATER = FluidMeta(mu_si=0.001, rho_si=1000)


@dataclass
class SimulationMeta:

    domain: DomainMeta
    fluid: FluidMeta
    dt_si: float

    cs_si: float = field(init=False)
    tau_pos_lu: float = field(init=False)
    tau_neg_lu: float = field(init=False)
    w_pos_lu: float = field(init=False)
    w_neg_lu: float = field(init=False)

    def __post_init__(self):
        self.cs_si = (1.0 / np.sqrt(3.0)) * (self.domain.dx_si / self.dt_si)

        nu_lu = self.fluid.nu_si / (self.cs_si**2 * self.dt_si)

        l_ = 0.25  # magic parameter
        tau_pos_lu = nu_lu + 0.5
        tau_neg_lu = l_ / nu_lu + 0.5

        if tau_pos_lu < 0.6:
            log.warning(f"Small value for tau! [{tau_pos_lu=}]")

        w_pve_lu = 1.0 / tau_pos_lu
        w_nve_lu = 1.0 / tau_neg_lu

        self.tau_pos_lu = tau_pos_lu
        self.tau_neg_lu = tau_neg_lu
        self.w_pos_lu = w_pve_lu
        self.w_neg_lu = w_nve_lu

    @staticmethod
    def with_cs(domain: DomainMeta, fluid: FluidMeta, cs_si: float) -> "SimulationMeta":
        dt = domain.dx_si / (np.sqrt(3.0) * cs_si)
        return SimulationMeta(domain, fluid, dt)


# Cell types
class CellType(Enum):
    FLUID = 0
    BC_WALL = 1
    BC_VELOCITY = 2
    FIXED_PRESSURE = 3


class Model:
    def __init__(self, ws: list[float], qs: list[list[float]]):
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
