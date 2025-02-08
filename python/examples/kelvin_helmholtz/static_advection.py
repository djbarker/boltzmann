# %%
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats.qmc import PoissonDisk

# %%

nx = 1000
ny = 1002
x = np.linspace(0, 1, nx)
y = np.linspace(-0.5, 0.5, ny)
XX, YY = np.meshgrid(x, y, indexing="ij")

VVx = np.zeros_like(XX)
VVx[:, : ny // 2] = -1
VVx[:, ny // 2 :] = +1

VVx = np.tanh(YY / 0.25)

VVy = 0.5 * np.exp(-((YY / 0.25) ** 2)) * np.sin(2 * np.pi * XX)

VV = np.sqrt(VVx**2 + VVy**2)

fig = plt.figure()
ax = plt.gca()
ax.set_aspect("equal")

# pdsk = PoissonDisk(2, radius=0.1)
# xy = pdsk.fill_space()
# xy[:, 1] -= 0.5  # samples from [0, 1]^2

xx = np.concatenate(
    [
        np.linspace(0, 1, 24)[1:-1],
        np.zeros(24)[1:-1],
        np.ones(24)[1:-1],
    ]
)

yy = np.concatenate(
    [
        np.zeros(24)[1:-1],
        np.linspace(0, 1, 24)[1:-1] - 0.5,
        np.linspace(0, 1, 24)[1:-1] - 0.5,
    ]
)

xy = np.array([xx, yy]).T

# ax.scatter(xy[:, 0], xy[:, 1])

from matplotlib.patches import ArrowStyle

ax.streamplot(
    XX.T,
    YY.T,
    VVx.T,
    VVy.T,
    color=VV.T,
    cmap=plt.cm.viridis,
    broken_streamlines=True,
    start_points=xy,
    arrowstyle=ArrowStyle.Fancy(head_length=0.4, head_width=0.4, tail_width=0.4),
    maxlength=1000,
    density=10,
)

# %%

plt.quiver(
    XX.T[::50, ::50],
    YY.T[::50, ::50],
    VVx.T[::50, ::50],
    VVy.T[::50, ::50],
    pivot="mid",
)

# %%
plt.scatter(xy[:, 0], xy[:, 1])
# %%
