import argparse as ap
import json
import logging
import sys
import time
import datetime

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from boltzmann.core import (
    CellType,
    SimulationMeta,
    calc_equilibrium,
)
from boltzmann.utils.logger import PerfInfo, tick

from boltzmann_rs import Fluid, Scalar, Domain


logger = logging.getLogger(__name__)


# @dataclass(init=False)
# class Cells:
#     cells: np.ndarray

#     def __init__(self, domain: Domain) -> None:
#         self.cells = domain.make_array(dtype=np.int32, fill=CellType.FLUID.value)

#     def save(self, base: Path):
#         np.save(base / "chk.cells.npy", self.cells)

#     def load(self, base: Path):
#         self.cells[:] = np.load(base / "chk.cells.npy")

#     @property
#     def size_bytes(self) -> int:
#         return self.cells.nbytes


# @dataclass(init=False)
# class Field:
#     name: str
#     model: Model

#     f: np.ndarray

#     def __init__(self, name: str, model: Model, domain: Domain) -> None:
#         self.name = name
#         self.model = model
#         self.f = domain.make_array(model.Q)

#     def save(self, base: Path):
#         np.save(base / f"chk.{self.name}.npy", self.f)

#     def load(self, base: Path):
#         self.f[:] = np.load(base / f"chk.{self.name}.npy")

#     @property
#     def size_bytes(self) -> int:
#         return self.f.nbytes


# @dataclass(init=False)
# class FluidField(Field):
#     rho: np.ndarray
#     vel: np.ndarray

#     def __init__(self, name: str, model: Model, domain: Domain) -> None:
#         super().__init__(name, model, domain)
#         self.rho = domain.make_array(fill=1)
#         self.vel = domain.make_array(model.D)

#     def load(self, base: Path):
#         super().load(base)
#         self.macro()  # recalc momements

#     def macro(self):
#         # careful to repopulate existing arrays
#         self.rho[:] = np.sum(self.f, axis=-1)
#         self.vel[:] = np.dot(self.f, self.model.qs) / self.rho[:, np.newaxis]

#     def equilibrate(self):
#         calc_equilibrium(self.vel, self.rho, self.f, self.model)

#     @property
#     def size_bytes(self) -> int:
#         return super().size_bytes + self.rho.nbytes + self.vel.nbytes


# @dataclass(init=False)
# class ScalarField(Field):
#     val: np.ndarray

#     def __init__(self, name: str, model: Model, domain: Domain) -> None:
#         super().__init__(name, model, domain)
#         self.val = domain.make_array()

#     def load(self, base: Path):
#         super().load(base)
#         self.macro()  # recalc momements

#     def macro(self):
#         # careful to repopulate existing arrays
#         self.val[:] = np.sum(self.f, axis=-1)

#     def equilibrate(self, vel: np.ndarray):
#         """
#         Like Fluid.equilibrate but velocity field is externally imposed.
#         """
#         assert vel.shape[-1] == self.model.D, "Dimension mismatch!"
#         calc_equilibrium(vel, self.val, self.f, self.model)

#     @property
#     def size_bytes(self) -> int:
#         return super().size_bytes + self.val.nbytes


def save_fluid(base: Path, fluid: Fluid):
    """
    Save the `Fluid` object arrays.
    """
    np.save(base / "chk.f.npy", fluid.f)


def load_fluid(base: Path, fluid: Fluid):
    """
    Load the `Fluid` object arrays.
    """
    fluid.f[:] = np.load(base / "chk.f.npy")


def save_scalar(base: Path, scalar: Scalar):
    """
    Save the `Scalar` object arrays.
    """
    np.save(base / "chk.g.npy", scalar.g)


def load_scalar(base: Path, scalar: Scalar):
    """
    Load the `Scalar` object arrays.
    """
    scalar.g[:] = np.load(base / "chk.g.npy")


def save_domain(base: Path, domain: Domain):
    """
    Save the `Domain` object arrays.
    """
    np.save(base / "chk.cells.npy", domain.cells)


def load_domain(base: Path, domain: Domain):
    """
    Load the `Domain` object arrays.
    """
    domain.cells[:] = np.load(base / "chk.cells.npy")


class SimulationLoop(Protocol):
    def loop_for(self, steps: int): ...
    def write_output(self, base: Path, step: int): ...
    def write_checkpoint(self, base: Path): ...
    def read_checkpoint(self, base: Path): ...


class SimulationRunner:
    def __init__(
        self,
        base: Path,
        meta: SimulationMeta,
        loop: SimulationLoop,
        step: int = 0,
    ):
        if base.exists():
            assert base.is_dir(), f"Base path must be a directory! [{base=}]"
        else:
            base.mkdir()

        self.base = base
        self.meta = meta
        self.loop = loop
        self.step = step
        self.last = datetime.datetime.now()

    @staticmethod
    def load_checkpoint(
        base: Path, meta: SimulationMeta, loop: SimulationLoop
    ) -> "SimulationRunner":
        logger.info(f"Reading checkpoint from '{base}'")

        # read meta-data
        with open(base / "chk.meta.json", "r") as fin:
            step = json.loads(fin.read())["step"]

        # read fields
        loop.read_checkpoint(base)

        return SimulationRunner(base, meta, loop, step)

    def run(self, *, write_checkpoints: bool = True):
        batch_iters = self.meta.time.batch_steps(self.meta.scales.dt)
        batch_iters = 2 * int(batch_iters / 2 + 0.5)  # iters must be even due to AA pattern
        logger.info(f"{batch_iters} iters/output")
        logger.info(f"{np.prod(self.meta.domain.counts) / 1e6:,.2f}m cells")

        if self.step > 0:
            logger.info(f"Resuming from step {self.step}")
        else:
            self.loop.write_output(self.base, self.step)
            logger.info("Wrote output 0")

        cell_count = int(np.prod(self.meta.domain.counts))
        perf_total = PerfInfo()
        max_i = self.meta.time.output_count
        for i in range(self.step + 1, max_i):
            # give the cpu fans a chance ...
            curr = datetime.datetime.now()
            diff = curr - self.last
            if diff > datetime.timedelta(minutes=4, seconds=30):
                time.sleep(30)
            self.last = curr

            timer = tick()

            self.loop.loop_for(batch_iters)

            # tock() before checkpointing so it's not included in the mlups calculation

            perf_batch = timer.tock(events=cell_count * batch_iters)
            perf_total = perf_total + perf_batch

            mlups_batch = perf_batch.events / (1e6 * perf_batch.seconds)
            mlups_total = perf_total.events / (1e6 * perf_total.seconds)

            self.loop.write_output(self.base, i)

            if write_checkpoints:
                # write meta-data
                meta = {"step": i}
                with open(self.base / "chk.meta.json", "w") as fout:
                    fout.write(json.dumps(meta))

                # write fields
                self.loop.write_checkpoint(self.base)

            perf_out = timer.tock() - perf_batch

            logger.info(
                f"Batch {i}: {mlups_batch:7.2f} MLUP/s, sim {perf_batch.seconds:.1f}s, out {perf_out.seconds:.1f}s"
            )


def run_sim_cli(sim: SimulationMeta, loop: SimulationLoop):
    """
    The 'main' method for running sims and handling cmd line args.
    """

    # If it looks like we're in an ipython session, ignore command line arguments.
    try:
        get_ipython()  # noqa: F821
        args = []
    except:
        args = sys.argv[1:]

    parser = ap.ArgumentParser()
    parser.add_argument("--base", type=str, default="out")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-checkpoint", action="store_true")
    args = parser.parse_args(args)

    base = Path(args.base)

    if args.resume:
        logger.info("Resuming")
        Runner = SimulationRunner.load_checkpoint
    else:
        Runner = SimulationRunner

    write_checkpoints = not args.no_checkpoint

    Runner(base, sim, loop).run(write_checkpoints=write_checkpoints)
