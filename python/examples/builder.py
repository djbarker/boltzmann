from __future__ import annotations

import logging
import sys
import time

from pathlib import Path

from boltzmann.core import Simulation
from boltzmann.simulation import SimulationScript, IterInfo
from boltzmann.utils.logger import basic_config


if __name__ != "__main__":
    sys.exit(0)

basic_config()
logger = logging.getLogger(__name__)

meta = IterInfo.make(dt=0.001, dt_output=0.1, count=100)
with SimulationScript([2000, 2000], 1 / 0.51, meta, "out") as sim:

    @sim.init
    def init(sim: Simulation):
        sim.fluid.vel[10:20, :, 1] = 0.1

    @sim.out
    def output(sim: Simulation, output_dir: Path, iter: int):
        pass
