"""
Almost the simplest possible example of setting up & running a simulation.
It purely runs the lattice Boltzmann simulation, with no mapping to physical units or careful choice of parameters.
It does, however, show using a tracer and setting some boundary conditions.
"""

# %%

import matplotlib.pyplot as plt
import numpy as np

from boltzmann.core import Simulation
from boltzmann.vtkio import VtiWriter

# Create the simulation.
tau = 0.51
cnt = [257, 103]
sim = Simulation("cpu", cnt, 9, 1 / tau)
col = sim.add_tracer(5, 1 / tau)

# Set some initial velocity and fix those cells to make a little jet.
sx = slice(20, 25)
sy = slice(50, 55)
sim.fluid.vel[sx, sy, 0] = 0.1
sim.cells.flags[sx, sy] |= 6
col.val[sx, sy] = 1.0
sim.cells.flags[sx, sy] |= 8

sx = slice(50, 55)
sy = slice(20, 25)
sim.fluid.vel[sx, sy, 1] = 0.1
sim.cells.flags[sx, sy] |= 6
col.val[sx, sy] = 1.0
sim.cells.flags[sx, sy] |= 8

sx = slice(45, 60)
sy = slice(45, 60)
sim.cells.flags[sx, sy] |= 1  # wall


# %% Run it.

sim.iterate(250)

with VtiWriter("output.vti", cnt) as writer:
    writer.add_data("velocity", sim.fluid.vel)
    writer.add_data("density", sim.fluid.rho)

# %% Plot it.
vmag = np.sqrt(np.sum(sim.fluid.vel**2, -1))
plt.imshow(vmag.T, interpolation="none", origin="lower", vmin=0, vmax=0.11)
plt.show()

plt.imshow(col.val.T, interpolation="none", origin="lower", vmin=0, vmax=1)
plt.show()

# %%
