"""
A 3D version of jets.py.
"""

# %%

import logging
import matplotlib.pyplot as plt
import numpy as np

from boltzmann.core import Simulation, check_lbm_params
from boltzmann.utils.logger import basic_config, dotted, timed
from boltzmann.utils.vtkio import VtiWriter


basic_config()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# Create the simulation.
tau = 0.505
cnt = [400, 400, 400]
sim = Simulation("gpu", cnt, 1 / tau)
col1 = sim.add_tracer(1 / tau)
col2 = sim.add_tracer(1 / tau)

cell_cnt = np.prod(cnt)
dotted(logger, "Cell count", f"{cell_cnt / 1e6:.1f} million")
dotted(logger, "Memory usage", f"{sim.size_bytes / 1e6:.1f} MB")

# Set some initial velocity and fix those cells to make a little jet.

X, Y, Z = [np.linspace(-1, 1, cnt[i]) for i in range(3)]
XX, YY, ZZ = np.meshgrid(X, Y, Z, indexing="ij")

sim.fluid.rho[:] += np.random.uniform(-0.01, 0.01, sim.fluid.rho.shape)

mask = (np.abs(XX + 0.3) < 0.05) & ((YY**2 + (ZZ - 0.02) ** 2) < 0.08**2)
sim.fluid.vel[mask, 0] = 0.05
sim.cells.flags[mask] |= 6
sim.cells.flags[mask] |= 8
col1.val[mask] = 1.0

mask = (np.abs(XX - 0.3) < 0.05) & ((YY**2 + (ZZ + 0.02) ** 2) < 0.08**2)
sim.fluid.vel[mask, 0] = -0.05
sim.cells.flags[mask] |= 6
sim.cells.flags[mask] |= 8
col2.val[mask] = 1.0

# %% Run it.


def calc_q():
    global q

    duxdx = np.diff(sim.fluid.vel[..., 0], axis=0)[:, :-1, :-1]
    duydx = np.diff(sim.fluid.vel[..., 1], axis=0)[:, :-1, :-1]
    duzdx = np.diff(sim.fluid.vel[..., 2], axis=0)[:, :-1, :-1]
    duxdy = np.diff(sim.fluid.vel[..., 0], axis=1)[:-1:, :, :-1]
    duydy = np.diff(sim.fluid.vel[..., 1], axis=1)[:-1:, :, :-1]
    duzdy = np.diff(sim.fluid.vel[..., 2], axis=1)[:-1:, :, :-1]
    duxdz = np.diff(sim.fluid.vel[..., 0], axis=2)[:-1:, :-1, :]
    duydz = np.diff(sim.fluid.vel[..., 1], axis=2)[:-1:, :-1, :]
    duzdz = np.diff(sim.fluid.vel[..., 2], axis=2)[:-1:, :-1, :]

    jxx = 0.5 * (duxdx + duxdx)
    jxy = 0.5 * (duxdy + duydx)
    jxz = 0.5 * (duxdz + duzdx)
    jyx = 0.5 * (duydx + duxdy)
    jyy = 0.5 * (duydy + duydy)
    jyz = 0.5 * (duydz + duzdy)
    jzx = 0.5 * (duzdx + duxdz)
    jzy = 0.5 * (duzdy + duydz)
    jzz = 0.5 * (duzdz + duzdz)

    sxx = 0.5 * (duxdx - duxdx)
    sxy = 0.5 * (duxdy - duydx)
    sxz = 0.5 * (duxdz - duzdx)
    syx = 0.5 * (duydx - duxdy)
    syy = 0.5 * (duydy - duydy)
    syz = 0.5 * (duydz - duzdy)
    szx = 0.5 * (duzdx - duxdz)
    szy = 0.5 * (duzdy - duydz)
    szz = 0.5 * (duzdz - duzdz)

    # fmt: off
    q = 0.5*(
        (jxx**2 + jxy**2 + jxz**2 + jyx**2 + jyy**2 + jyz**2 + jzx**2 + jzy**2 + jzz**2) 
      - (sxx**2 + sxy**2 + sxz**2 + syx**2 + syy**2 + syz**2 + szx**2 + szy**2 + szz**2)
    )
    # fmt: on


def write_out(i: int):
    with VtiWriter(f"output_{i:06d}.vti", [c - 1 for c in cnt]) as writer:
        writer.add_data("velocity", sim.fluid.vel[:-1, :-1, :-1])
        writer.add_data("density", sim.fluid.rho[:-1, :-1, :-1])
        writer.add_data("qcriterion", q)
        writer.add_data("col1", col1.val[:-1, :-1, :-1])
        writer.add_data("col2", col2.val[:-1, :-1, :-1])
        writer.set_default("velocity")


calc_q()
write_out(0)

for i in range(1, 200):
    with timed(logger, f"Step {i}", events=cell_cnt * 50):
        sim.iterate(50)

    calc_q()
    write_out(i)

# %%
