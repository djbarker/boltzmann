# %%

import numpy as np
import matplotlib.pyplot as plt

from boltzmann.core import Simulation, CellType

np.random.seed(42)

# Create the simulation
tau = 0.52
cnt = [1500, 1500]
sim = Simulation("gpu", cnt, 1 / tau)
red = sim.add_tracer(1 / tau)
grn = sim.add_tracer(1 / tau)

# Set some initial conditions
n = 40
nn = [cnt[0] // n, cnt[1] // n]
sx = slice(cnt[0] // 3, cnt[0] // 3 + nn[0])
sy = slice(cnt[1] // 2 - nn[1], cnt[1] // 2 + nn[1])
sim.fluid.vel[sx, sy, 0] = 0.1
red.val[sx, sy] = 1.0
sim.cells.flags[sx, sy] |= CellType.FIXED_FLUID_VELOCITY.value
sim.cells.flags[sx, sy] |= CellType.FIXED_SCALAR_VALUE.value

sx = slice(2 * cnt[0] // 3 - nn[0], 2 * cnt[0] // 3)
sy = slice(cnt[1] // 2 - nn[1], cnt[1] // 2 + nn[1])
sim.fluid.vel[sx, sy, 0] = -0.1
grn.val[sx, sy] = 1.0
sim.cells.flags[sx, sy] |= CellType.FIXED_FLUID_VELOCITY.value
sim.cells.flags[sx, sy] |= CellType.FIXED_SCALAR_VALUE.value

sim.fluid.rho[:] += 0.01 * np.random.uniform(-1, 1, sim.fluid.rho.shape)

# %%

tprev = 0
for i, tcurr in enumerate([3000]):  # , 10000, 15000, 20000]):
    # Run it
    sim.iterate(tcurr - tprev)
    tprev = tcurr

    # Plot it
    rgb = np.zeros((*sim.fluid.rho.shape, 3))
    rgb[..., 0] = red.val * 0.9 + grn.val * 0.7
    rgb[..., 1] = red.val * 0.9
    rgb[..., 2] = grn.val * 0.9
    rgb = np.swapaxes(rgb, 0, 1)
    plt.imshow(rgb, origin="lower", interpolation="none")
    plt.axis("off")
    plt.savefig(f"jets_{i}.png", dpi=500, bbox_inches="tight", pad_inches=0)
    plt.show()

# %%
