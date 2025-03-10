# %%
import matplotlib as mpl

mpl.use("Agg")


import matplotlib.pyplot as plt
import numpy as np

from boltzmann.core import Simulation, CellFlags

sim = Simulation("cpu", [100, 100], 1 / 0.51)
temp = sim.add_tracer("temp", 1 / 0.51)
grav = np.array([0.1, 1], np.float32)
sim.add_boussinesq_coupling(temp, 1, 1, grav)
# sim.set_gravity(grav)

temp.val[:] = 1.0
temp.val[50, 50] = 2.0
sim.cells.flags[50, 50] = CellFlags.FIXED_SCALAR_VALUE

sim.iterate(100)

plt.imshow(temp.val.T)
plt.savefig("out/temp.png")
plt.close()

vmag = np.sqrt(np.sum(sim.fluid.vel**2, -1))
plt.imshow(vmag.T)
plt.savefig("out/vmag.png")
plt.close()

print(vmag.min(), vmag.max())

# %%
