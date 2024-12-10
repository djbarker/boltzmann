from dataclasses import field
import json
import logging
from pathlib import Path
import numpy as np

from dataclasses import dataclass
from typing import Protocol


from boltzmann.core import SimulationMeta
from boltzmann.utils.logger import PerfInfo, tick


logger = logging.getLogger(__name__)


@dataclass
class Cells:
    cells: np.ndarray

    def save(self, base: Path):
        assert base.is_dir(), "Base path must be a directory!"
        np.save(base / "chk.cells.npy", self.cells)

    def load(self, base: Path):
        assert base.is_dir(), "Base path must be a directory!"
        self.cells = np.load(base / "chk.cells.npy")


@dataclass
class Field:
    name: str
    f1: np.ndarray
    f2: np.ndarray = field(init=False)

    def __post_init__(self):
        self.f2 = self.f1.copy()

    def save(self, base: Path):
        assert base.is_dir(), "Base path must be a directory!"
        np.save(base / f"chk.{self.name}.npy", self.f1)

    def load(self, base: Path):
        assert base.is_dir(), "Base path must be a directory!"
        self.f1 = np.load(base / f"chk.{self.name}.npy")
        self.f2[:] = self.f1[:]


@dataclass
class FluidField(Field):
    rho: np.ndarray
    vel: np.ndarray

    def __post_init__(self):
        super().__post_init__()  # Field.__post_init_()
        assert self.f1.shape[:-1] == self.rho.shape
        assert self.f1.shape[:-1] == self.vel.shape[:-1]


@dataclass
class ScalarField(Field):
    val: np.ndarray

    def __post_init__(self):
        assert self.f1.shape[:-1] == self.val.shape


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
        assert base.is_dir(), f"Base path must be a directory! [{base=}]"
        self.base = base
        self.meta = meta
        self.loop = loop
        self.step = step

    @staticmethod
    def load_checkpoint(
        base: Path, meta: SimulationMeta, loop: SimulationLoop
    ) -> "SimulationRunner":
        # read meta-data
        with open(base / "chk.meta.json", "r") as fin:
            step = json.loads(fin.read())["step"]

        # read fields
        loop.read_checkpoint(base)

        return SimulationRunner(base, meta, loop, step)

    def run(self):
        batch_iters = self.meta.time.batch_steps(self.meta.scales.dt)
        logger.info(f"{batch_iters} iters/output")

        self.loop.write_output(self.base, 0)

        perf_total = PerfInfo()
        out_i = self.meta.time.output_count
        for i in range(1, out_i):
            perf_batch = tick()

            self.loop.loop_for(batch_iters)

            # tock() before checkpointing so it's not included in the mlups calculation

            cell_count = int(np.prod(self.meta.domain.counts))
            perf_batch = perf_batch.tock(events=cell_count * batch_iters)
            perf_total = perf_total + perf_batch

            mlups_batch = perf_batch.events / (1e6 * perf_batch.seconds)
            mlups_total = perf_total.events / (1e6 * perf_total.seconds)

            self.loop.write_output(self.base, i)

            # write meta-data
            meta = {"step": i}
            with open(self.base / "chk.meta.json", "w") as fout:
                fout.write(json.dumps(meta))

            # write fields
            self.loop.write_checkpoint(self.base)

            logger.info(f"Batch {i}: {mlups_batch:.2f} mlups, {mlups_total=:.2f}")
