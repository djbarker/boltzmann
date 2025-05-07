"""
Containts classes which look like the core classes but do not involve the extension module.
This is useful for debugging where you suspect the extension module is causing an issue.

.. note::
    In order to use this you need stop the :py:mod:`boltzmann.core` and :py:mod:`boltzmann.simulation` modules from importing the extension module.
"""

import numpy as np

from dataclasses import dataclass

CountsT = int | tuple[int] | tuple[int, int] | tuple[int, int]


@dataclass
class MockCells:
    """
    Looks like :py:class:`~boltzmann.core.Cells` but does not involve any Rust.
    """

    counts: tuple[int, ...]
    flags: np.ndarray
    size_bytes: int

    @property
    def count(self) -> int:
        return int(np.prod(self.counts))


@dataclass
class MockFluid:
    """
    Looks like :py:class:`~boltzmann.core.Fluid` but does not involve any Rust.
    """

    f: np.ndarray
    vel: np.ndarray
    rho: np.ndarray
    size_bytes: int


@dataclass
class MockSimulation:
    """
    Looks like :py:class:`~boltzmann.core.Simulation` but does not involve any Rust.
    """

    cells: MockCells
    fluid: MockFluid
    iteration: int = 0
    device_info: str = "### MOCK SIM ###"

    @staticmethod
    def make(counts: CountsT) -> "MockSimulation":
        if isinstance(counts, int):
            counts = (counts,)

        cells = MockCells(
            counts=counts,
            flags=np.zeros(counts, dtype=np.uint8),
            size_bytes=int(np.prod(counts) * np.dtype(np.uint8).itemsize),
        )

        match counts:
            case (x,):
                fsize = (x, 3)
                vsize = (x,)
            case (x, y):
                fsize = (x, y, 9)
                vsize = (x, y, 2)
            case (x, y, z):
                fsize = (x, y, z, 27)
                vsize = (x, y, z, 3)
            case _:
                raise ValueError("Invalid dimensionality for counts")

        fluid = MockFluid(
            f=np.zeros(fsize, dtype=np.float32),
            vel=np.zeros(vsize, dtype=np.float32),
            rho=np.ones(counts, dtype=np.float32),
            size_bytes=int(
                cells.count * np.dtype(np.float32).itemsize * (fsize[-1] + vsize[-1] + 1)
            ),
        )

        return MockSimulation(
            cells=cells,
            fluid=fluid,
        )

    @property
    def size_bytes(self) -> int:
        return self.cells.size_bytes + self.fluid.size_bytes

    def iterate(self, iters: int):
        self.iteration += iters
