from __future__ import annotations

import numpy as np

from dataclasses import dataclass, field
from typing import Sequence, overload

from boltzmann.utils.option import Option, Some, map_opt, to_opt


Array1dT = np.ndarray | Sequence[int] | Sequence[float]


def _to_array1d(arr: Array1dT) -> np.ndarray:
    arr = np.squeeze(np.array(arr))
    assert np.ndim(arr) == 1, f"Expected 1d array for arr! [{arr.shape=!r}]"
    return arr


DimsT = list[int] | int | None


def _to_dims(dims: DimsT) -> list[int]:
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

    .. role:: python(code)
        :language: python

    The semantics are that each cell has a size of ``dx``, the ``lower`` & ``upper`` arrays are the lower
    and upper *edges* of the cells respectively.
    This means :python:`N = (upper - lower) / dx`, and if :python:`lower == upper` then we have no cells.
    The centre of the cells are thus offset by ``dx/2``, so the centre of the cell with index ``(i, j)`` is located at

    .. code-block:: python

        (x, y) = ((i + 0.5) * dx, (j + 0.5) * dx)

    .. note::

        For the simple two-array implementation it's quite efficient to pad the arrays by one at the end
        of each dimension. Then we can stream into/from these with simple offsets (not worrying about
        periodicity), then copy the data across as needed. This does not work for in-place updates,
        where we would need to update the padded values too, but only for the directions that stream
        back into domain, necessitating a check of whether we're in the periodic copy or not.
        Ultimately it's probably worth doing this but that's a **TODO**.

    """

    lower: np.ndarray  #: Lower bound of the domain in each dimension.
    upper: np.ndarray  #: Upper bound of the domain in each dimension.
    counts: np.ndarray  #: Number of cells in each dimension.
    dx: float  #: Physical extent of each cell.
    dims: int  #: Number of dimensions. (:python:`== len(counts)`)

    @staticmethod
    def make(
        lower: Option[Array1dT] = None,
        upper: Option[Array1dT] = None,
        counts: Option[Array1dT] = None,
        dx: Option[float] = None,
    ) -> Domain:
        """
        Construct a :py:class:`Domain` object from a valid combination of the parameters.
        Valid combinations are those that are sufficient to infer the others.

        Specific valid combinations are:

        * ``lower``, ``upper`` & ``counts`` - ``dx`` is inferred as ``(upper - lower) / counts``.
        * ``lower``, ``upper`` & ``dx`` - ``counts`` is inferred as ``(upper - lower) / dx``.
        * ``lower``, ``counts`` & ``dx`` - ``upper`` is inferred as ``lower + counts * dx``.
        * ``upper``, ``counts`` & ``dx`` - ``lower`` is inferred as ``upper - counts * dx``.

        In the above ``lower`` can be omitted and it will be inferred as zero(s).

        .. note::

            The cell size ``dx`` must be equal in all dimensions.
            This means that the aspect ratio(s) of ``counts`` must match the aspect ratio(s) of the extent.
            This method will throw a :py:exc:`ValueError` if this is not the case.

        """
        lower_ = map_opt(lower, _to_array1d)
        upper_ = map_opt(upper, _to_array1d)
        counts_ = map_opt(counts, _to_array1d)
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
            case None, Some(uu), Some(cc), Some(d):
                ll = uu - cc * d
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
        """
        Positions of the cell centres along the specified dimension.

        :param dim: The dimension to return the cell positions for.
        """
        assert dim < self.dims, f"Invalid dim! [{dim=}, {self.dims=}]"
        return np.linspace(
            self.lower[dim] + self.dx / 2,
            self.upper[dim] - self.dx / 2,
            self.counts[dim],
        )

    @property
    def x(self) -> np.ndarray:
        """
        Positions of cell centers along the x-axis.
        """
        return self.get_dim(0)

    @property
    def y(self) -> np.ndarray:
        """
        Positions of cell centers along the y-axis.
        """
        return self.get_dim(1)

    @property
    def z(self) -> np.ndarray:
        """
        Positions of cell centers along the z-axis.
        """
        return self.get_dim(2)

    def get_extent(self, dim: int) -> float:
        """
        Size of the domain in the specified dimension.
        """
        assert dim < self.dims, f"Invalid dim! [{dim=}, {self.dims=}]"
        return self.upper[dim] - self.lower[dim]

    @property
    def width(self) -> float:
        """
        Size of the domain along the x-axis.
        """
        return self.get_extent(0)

    @property
    def height(self) -> float:
        """
        Size of the domain along the y-axis.
        """
        return self.get_extent(1)

    @property
    def depth(self) -> float:
        """
        Size of the domain along the z-axis.
        """
        return self.get_extent(2)

    def meshgrid(self) -> tuple[np.ndarray, ...]:
        """
        Return a meshgrid of the domain.
        Useful for setting initial conditions based on physical parameters.

        .. code-block: python

            dom = Domain.make(lower=[-0.5, 0], upper=[0.5, 1.0], counts=[100, 100])
            XX, YY = dom.meshgrid()

        """
        return np.meshgrid(*[self.get_dim(i) for i in range(self.dims)], indexing="ij")


@dataclass
class UnitConverter:
    """
    Convert a specific dimensional quantity between physical & lattice units.
    """

    _to_physical: float

    @overload
    def to_lattice_units(self, value: float) -> float: ...

    @overload
    def to_lattice_units(self, value: np.ndarray) -> np.ndarray: ...

    def to_lattice_units(self, value: float | np.ndarray) -> float | np.ndarray:
        """
        Convert the given dimensional value(s) into lattice-units using the correct conversion factor.

        .. warning::
            Be careful if assigning the result of this since it creates a copy.
            You probably want to assign the elements rather than the whole array, e.g

            >>> vel_ = converter.to_lattice_units(vel_)     # Wrong! vel_ will have a new buffer.
            >>> vel_[:] = converter.to_lattice_units(vel_)  # Right! Write to the vel_ buffer.
        """
        return value / self._to_physical

    @overload
    def to_physical_units(self, value: float) -> float: ...

    @overload
    def to_physical_units(self, value: np.ndarray) -> np.ndarray: ...

    def to_physical_units(self, value: float | np.ndarray) -> float | np.ndarray:
        """
        Convert the given dimensional value(s) into physical-units using the correct conversion factor.

        .. warning::
            Be careful if assigning the result of this since it creates a copy.
            You probably want to assign the elements rather than the whole array, e.g

            >>> vel_ = converter.to_physical_units(vel_)     # Wrong! vel_ will have a new buffer.
            >>> vel_[:] = converter.to_physical_units(vel_)  # Right! Write to the vel_ buffer.
        """
        return value * self._to_physical


@dataclass
class Scales:
    """
    Characteristic scales of fundamental units.
    Used to convert between physical units (SI) and simulation (lattice) units.

    To convert you must use this to construct a :py:class:`UnitConverter` object for the given powers.
    Convenience properties exist for the three most common cases; distance, velocity & acceleration.

    .. code-block:: python

        scales = Scales.make(dx=dx, dt=dt)
        vel_si = 1.0  # [m/s]

        # The statements below are equivalent:
        vel_lu = scales.converter(L=1, T=-1).to_lattice_units(vel_si)
        vel_lu = scales.velocity.to_lattice_units(vel_si)
    """

    dx: float  #: Lattice spacing.
    dt: float  #: Time step.
    dm: float = field(default=1)  #: Mass scale (usually unneeded.)

    def converter(self, L: int = 0, T: int = 0, M: int = 0) -> UnitConverter:
        """
        Construct a :py:class:`UnitConverter` object for the given dimensionality.
        """
        return UnitConverter(
            _to_physical=pow(self.dx, L) * pow(self.dt, T) * pow(self.dm, M),
        )

    @property
    def time(self) -> UnitConverter:
        return self.converter(L=0, T=1)

    @property
    def distance(self) -> UnitConverter:
        """
        Convert distances between physical and lattice units.
        """
        return self.converter(L=1, T=0)

    @property
    def velocity(self) -> UnitConverter:
        """
        Convert speeds between physical and lattice units.
        """
        return self.converter(L=1, T=-1)

    @property
    def acceleration(self) -> UnitConverter:
        """
        Convert accelerations between physical and lattice units.
        """
        return self.converter(L=1, T=-2)
