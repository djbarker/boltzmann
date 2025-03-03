from __future__ import annotations
import logging
import numpy as np


from dataclasses import dataclass
from enum import Enum
from math import sqrt
from typing import Sequence, overload

from boltzmann.utils.option import Some, to_opt, map_opt, Option

__all__ = [
    "CellFlags",
    "calc_lbm_params",
    "check_lbm_params",
]

# ------------------------------------
# Re-export the rust module here.

import boltzmann.boltzmann  # type: ignore
from boltzmann.boltzmann import *  # type: ignore  # noqa: F403

if hasattr(boltzmann.boltzmann, "__all__"):
    __all__.extend(boltzmann.boltzmann.__all__)

# ------------------------------------

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
    Describes the domain grid.

    The semantics are that each cell has a size of `dx`, the `lower` & `upper` arrays are the lower
    and upper _edges_ of the cells respectively.
    This means `N = (upper - lower) / dx`, and if `lower == upper` then we have no cells.
    The centre of the cells are thus offset by `dx/2`, so cell with index `(i, j)` is located at

        `(x, y) = ((i + 0.5) * dx, (j + 0.5) * dx)`

    For the simple two-array implementation it's quite efficient to pad the arrays by one at the end
    of each dimension. Then we can stream into/from these with simple offsets (not worrying about
    periodicity), then copy the data across as needed. This does not work for in-place updates,
    where we would need to update the padded values too, but only for the directions that stream
    back into domain, necessitating a check of whether we're in the periodic copy or not.
    Ultimately it's probably worth doing this but that's a TODO.
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
        return np.linspace(
            self.lower[dim] + self.dx / 2,
            self.upper[dim] - self.dx / 2,
            self.counts[dim],
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

    def meshgrid(self) -> tuple[np.ndarray, ...]:
        """
        Return a meshgrid of the domain.

        .. code-block: python

            dom = Domain.make(lower=[-0.5, 0], upper=[0.5, 1.0], counts=[100, 100])
            XX, YY = dom.meshgrid()

        """
        return np.meshgrid(*[self.get_dim(i) for i in range(self.dims)], indexing="ij")


@dataclass
class TimeMeta:
    """
    Contains parameters for output interval and simulation duration.
    """

    dt_step: float
    dt_output: float
    t_max: float
    output_count: int

    @property
    def batch_steps(self) -> int:
        return int(self.dt_output / self.dt_step + 1e-8)

    @staticmethod
    def make(
        *,
        dt_step: float,
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
                    f"Must specify dt_step and exactly two other arguments! [{dt_output=}, {t_max=}, {output_count=}]"
                )

        return TimeMeta(dt_step, d, t, i)


@dataclass
class UnitConverter:
    """
    Convert a specific dimensional quantity between physical & lattice units.
    """

    to_physical: float
    to_lattice: float

    @overload
    def to_lattice_units(self, value: float) -> float: ...

    @overload
    def to_lattice_units(self, value: np.ndarray) -> np.ndarray: ...

    @overload
    def to_physical_units(self, value: float) -> float: ...

    @overload
    def to_physical_units(self, value: np.ndarray) -> np.ndarray: ...

    def to_lattice_units(self, value: float | np.ndarray) -> float | np.ndarray:
        """
        Convert the given dimensional value(s) into lattice-units using the correct conversion factor.

        NOTE: Be very careful if assigning the result of this since it creates a copy.
              You probably want to assign the elements rather than the whole array, e.g

              >>> vel_ = converter.to_lattice_units(vel_)     # Wrong! vel_ will have a new buffer.
              >>> vel_[:] = converter.to_lattice_units(vel_)  # Right! Write to the vel_ buffer.
        """
        return value * self.to_lattice

    def to_physical_units(self, value: float | np.ndarray) -> float | np.ndarray:
        """
        Convert the given dimensional value(s) into physical-units using the correct conversion factor.

        NOTE: Be very careful if assigning the result of this since it creates a copy.
              You probably want to assign the elements rather than the whole array, e.g

              >>> vel_ = converter.to_physical_units(vel_)     # Wrong! vel_ will have a new buffer.
              >>> vel_[:] = converter.to_physical_units(vel_)  # Right! Write to the vel_ buffer.
        """
        return value * self.to_physical


@dataclass
class Scales:
    """
    Characteristic scales of fundamental units.
    Used to convert between physical units (SI) and simulation (lattice) units.

    To convert you must use this to construct a `UnitConverter` object for the given powers.
    Convenience properties exist for the three most common cases; distance, velocity & acceleration.

    .. code-block: python

        scales = Scales.make(dx=dx, dt=dt)
        vel_si = 1.0  # [m/s]

        # The statements below are equivalent:
        vel_lu = scales.converter(L=1, T=-1).to_lattice_units(vel_si)
        vel_lu = scales.velocity.to_lattice_units(vel_si)
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

    def converter(self, L: int = 0, T: int = 0, M: int = 0) -> UnitConverter:
        """
        Construct a `UnitConverter` object for the given dimensionality.
        """
        factor = pow(self.dx, L) * pow(self.dt, T) * pow(self.dm, M)
        return UnitConverter(
            to_physical=factor,
            to_lattice=1.0 / factor,
        )

    @property
    def distance(self) -> UnitConverter:
        return self.converter(L=1, T=0)

    @property
    def velocity(self) -> UnitConverter:
        return self.converter(L=1, T=-1)

    @property
    def acceleration(self) -> UnitConverter:
        return self.converter(L=1, T=-2)


class CellFlags:
    FLUID = 0
    WALL = 1
    FIXED_FLUID_VELOCITY = 2
    FIXED_FLUID_DENSITY = 4
    FIXED_SCALAR_VALUE = 8
    FIXED_FLUID = FIXED_FLUID_VELOCITY | FIXED_FLUID_DENSITY


def check_lbm_params(Re: float, L: float, tau: float, M_max: float = 0.1):
    """
    Check if the chosen parameters are likely to be stable or not with BGK collision operator.
    """

    # Maximum value of tau implied by Mach condition.
    tau_mach = np.sqrt(3) * M_max * L / Re + 0.5

    # Minimum value of L implied by the BGK stability condition.
    L_stab = Re / 24

    eps = 1e-8
    assert L + eps > L_stab, f"BGK stability condition violated. {L=} < {L_stab}"
    assert tau < tau_mach + eps, f"Mach condition violated. {tau=} > {tau_mach}"


def calc_lbm_params(
    # params
    Re: float,
    L: float,
    tau: float | None = None,
    # bounds
    M_max: float = 0.1,
    tau_max: float = 1.0,
    tau_min: float = 0.5,
) -> tuple[float, float]:
    """
    Choose the values of tau and u which maximize simulation speed & verify stability.
    If you specify tau that value will be used instead.
    """

    if tau is None:
        tau = sqrt(3) * M_max * L / Re + 0.5  # Maximum permissible by Mach condition
        tau = min(max(tau, tau_min), tau_max)  # Hard bounds on tau

    check_lbm_params(Re, L, tau, M_max)

    nu = (tau - 0.5) / 3
    u = Re * nu / L

    return (tau, u)
