from __future__ import annotations

import logging
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path

from boltzmann.simulation import SimulationScript, IterInfo
from boltzmann.utils.logger import basic_config


basic_config()
logger = logging.getLogger(__name__)

meta = IterInfo.make(dt=0.001, dt_output=0.1, count=100)
with (script := SimulationScript([200, 100], 1 / 0.51, meta, "out")) as sim:

    @script.init
    def init():
        sim.fluid.vel[:, 40:60, 0] = 0.1
        sim.fluid.rho[:] += 0.1 * np.random.uniform(-1, 1, sim.fluid.rho.shape)

    @script.out
    def output(output_dir: Path, iter: int):
        dvydx = np.diff(sim.fluid.vel[..., 1], axis=0)[:, :-1]
        dvxdy = np.diff(sim.fluid.vel[..., 0], axis=1)[:-1, :]
        curl = dvydx - dvxdy
        plt.imshow(curl.T, cmap="RdBu")
        plt.show()
