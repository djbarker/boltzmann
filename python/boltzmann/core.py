import numpy as np
import numba


from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence, Protocol


ExtentT = np.ndarray | Sequence[Sequence[float]]


def to_extent(extent: ExtentT) -> np.ndarray:
    extent = np.array(extent)
    assert (
        np.ndim(extent) == 2 and extent.shape[1] == 2
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


class DomainMeta:

    def __init__(self, dx: float, extent: ExtentT):
        extent = to_extent(extent)
        self.dx = dx
        self.dims = extent.shape[0]
        self.extent = extent
        self.counts = np.array((extent[:, -1] - extent[:, 0]) / dx, dtype=np.int32)

        # sanity check values
        assert dx > 0
        for i, (e, c) in enumerate(zip(self.extent, self.counts)):
            dx_ = (e[1] - e[0]) / c
            assert (
                abs(dx_ - dx) < 1e-8
            ), f"dx differs from dimension 0 for dimension {i}. [{e=}, {c=} => {dx_=}]"

    def get_dim(self, dim: int) -> np.ndarray:
        assert dim <= self.dims, f"Invalid dim! [{dim=}, {self.dims=}]"
        return np.linspace(
            self.extent[dim, 0] - self.dx, self.extent[dim, 1] + self.dx, self.counts[dim] + 2
        )

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
    def with_dx_and_counts(dx: float, counts: CountsT) -> "DomainMeta":
        counts = to_counts(counts)
        extent = np.vstack([np.zeros_like(counts), counts])
        return DomainMeta(dx, extent)

    @staticmethod
    def with_extent_and_counts(extent: ExtentT, counts: CountsT) -> "DomainMeta":
        extent = to_extent(extent)
        counts = to_counts(counts)
        assert extent.shape[0] == counts.shape[0]
        dx = (extent[0][1] - extent[0][0]) / counts[0]
        return DomainMeta(dx, extent)

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
    mu: float
    rho: float
    nu: float = field(init=False)

    def __post_init__(self):
        self.nu = self.mu / self.rho


class SimulationMeta:

    def __init__(self, domain: DomainMeta, fluid: FluidMeta, dt: float):
        self.domain = domain
        self.fluid = fluid  # TODO: this & tau will need to change for multiphase
        self.dt = dt
        self.c = domain.dx / dt
        self.tau = 3 * fluid.nu * (1 / (self.c * domain.dx)) + 0.5

        assert 0.5 < self.tau, f"Invalid relaxation time! [tau={self.tau}]"


# Cell types
class CellType(Enum):
    FLUID = 0
    BC_WALL = 1
    FIXED_VELOCITY = 2
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


D2Q9 = Model(
    [4 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 9, 1 / 36, 1 / 36, 1 / 36, 1 / 36],
    [[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1], [-1, -1], [1, -1]],
)
