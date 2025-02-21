from abc import ABC, abstractmethod
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

from boltzmann.core import SimulationMeta
from boltzmann.utils.logger import PerfInfo, dotted, tick

from boltzmann_rs import Simulation, Fluid, Scalar, Cells


logger = logging.getLogger(__name__)


def save_fluid(base: Path, fluid: Fluid):
    """
    Save the `Fluid` object arrays.
    """
    np.save(base / "chk.f.npy", fluid.f)
    np.save(base / "chk.rho.npy", fluid.rho)
    np.save(base / "chk.vel.npy", fluid.vel)


def load_fluid(base: Path, fluid: Fluid):
    """
    Load the `Fluid` object arrays.
    """
    fluid.f[:] = np.load(base / "chk.f.npy")
    fluid.rho[:] = np.load(base / "chk.rho.npy")
    fluid.vel[:] = np.load(base / "chk.vel.npy")


def save_scalar(base: Path, scalar: Scalar, name: str):
    """
    Save the `Scalar` object arrays.
    """
    np.save(base / f"chk.{name}.g.npy", scalar.g)
    np.save(base / f"chk.{name}.val.npy", scalar.val)


def load_scalar(base: Path, scalar: Scalar, name: str):
    """
    Load the `Scalar` object arrays.
    """
    scalar.g[:] = np.load(base / f"chk.{name}.g.npy")
    scalar.val[:] = np.load(base / f"chk.{name}.val.npy")


@dataclass
class SimulationLoop(ABC):
    sim: Simulation  # type: ignore

    @abstractmethod
    def loop_for(self, steps: int): ...

    @abstractmethod
    def write_output(self, base: Path, step: int): ...

    @abstractmethod
    def write_checkpoint(self, base: Path): ...

    @abstractmethod
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
        batch_iters = self.meta.time.batch_steps
        batch_iters = 2 * int(batch_iters / 2 + 0.5)  # iters must be even due to AA pattern

        dotted(logger, "Memory usage", f"{self.loop.sim.size_bytes / 1e6:.2f}MB")
        dotted(logger, "Iters / output", batch_iters)
        dotted(logger, "Cells", f"{np.prod(self.meta.domain.counts) / 1e6}M")

        if self.step > 0:
            logger.info(f"Resuming from step {self.step}")
        else:
            self.loop.write_output(self.base, self.step)
            logger.info("Wrote output 0")

        timer_total = tick()

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

            t = timer_total.elapsed
            t = str(t).split(".")[0]

            per = timer_total.elapsed / (i - self.step)
            rem = max_i - i
            eta = datetime.datetime.now() + rem * per

            logger.info(
                f"Batch {i}:  {t}  {mlups_batch:7.2f} MLUP/s, sim {perf_batch.seconds:.1f}s, out {perf_out.seconds:.1f}s, eta {eta:%H:%M:%S}"
            )


def run_sim_cli(sim: SimulationMeta, loop: SimulationLoop, base: str | Path = "out"):
    """
    The 'main' method for running sims and handling cmd line args.
    """

    # If it looks like we're in an ipython session, ignore command line arguments.
    try:
        get_ipython()  # noqa: F821
        args = []
    except NameError:
        args = sys.argv[1:]

    parser = ap.ArgumentParser()
    parser.add_argument("--base", type=str, default=base)
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
