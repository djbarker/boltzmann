# The most simple possible example of setting up & running a simulation.
# It purely runs the lattice Boltzmann simulation with no mapping to physical units or careful choice of LBM parameters.

# %%

import matplotlib.pyplot as plt
import numpy as np

from boltzmann_rs import Simulation

# Create the simulation.
tau = 0.52
cnt = [257, 103]
sim = Simulation("cpu", cnt, 9, 1 / tau)

# Set some initial velocity and fix those cells to make a little jet.
sx = slice(20, 25, 1)
sy = slice(50, 55, 1)
sim.fluid.vel[sx, sy, 0] = 0.1
sim.cells.cell_type[sx, sy] |= 6  # "magic" number

sx = slice(50, 55)
sy = slice(20, 25)
sim.fluid.vel[sx, sy, 1] = 0.1
sim.cells.cell_type[sx, sy] |= 6  # "magic" number


# sim.fluid.vel[:10, :10, 1] = 0.1
# sim.fluid.vel[20:30, :30, 1] = 0.1
# sim.fluid.vel[40:50, :50, 1] = 0.1
# sim.fluid.vel[60:70, :70, 1] = 0.1

# %%  Run it.
sim.iterate(2000)

# %% Plot it.
vmag = np.sqrt(np.sum(sim.fluid.vel**2, -1))
plt.imshow(vmag.T, interpolation="none", origin="lower", vmax=0.11)

# %%
