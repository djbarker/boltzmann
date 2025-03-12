# %%

import numpy as np
import matplotlib.pyplot as plt

from boltzmann.core import Simulation

np.random.seed(42)

# Create the simulation
tau = 0.51
sim = Simulation("cpu", [200, 100], 1 / tau)

# Set some initial condition
sim.fluid.vel[:, 40:60, 0] = 0.1
sim.fluid.rho[:] += 0.1 * np.random.uniform(-1, 1, sim.fluid.rho.shape)

# Run it
sim.iterate(3000)

# Plot it
grad = np.gradient(sim.fluid.vel)
curl = grad[1][..., 0] - grad[0][..., 1]
plt.imshow(curl.T, cmap="RdBu")
plt.show()

# %%
