# %% Imports

import logging
from boltzmann.units import Domain, Scales
import matplotlib.pyplot as plt
import numpy as np
import sys

from pathlib import Path
from dataclasses import dataclass, field

from boltzmann.utils.logger import basic_config, dotted
from boltzmann.core import CellFlags, calc_lbm_params_si, Simulation, bgk, trt
from boltzmann.simulation import IterInfo, NoCheckpoints, run_sim

basic_config()

logger = logging.getLogger(__name__)
logger.info("Starting")


def u_analytical(y, t, W: float, g: float, nu: float, n: int = 20):
    """
    Time dependent analytical solution.
    """
    u = (g / (2 * nu)) * y * (W - y)
    for i in range(0, n):
        j = 2 * i + 1
        k = np.pi * j / W
        u += (g / nu) * (2 * ((-1) ** j - 1) / (k**3 * W)) * np.exp(-nu * k**2 * t) * np.sin(k * y)

    return u


# %% Params

# pipe width [m]
L_si = 0.01

# density [kg/m^3]
rho_si = 1000

# viscosity [Pas]
mu_si = 0.001

# Desired Reynold's number
Re = 100

# choose parameterisation
L = 100

# implied acceleration & pressure drop
g_si = 8 * Re * (mu_si / rho_si) ** 2 / L_si**3
dP_si = rho_si * g_si

# max infinite time velocity
nu_si = mu_si / rho_si
u_si = (g_si / nu_si) * (L_si**2) / 8
Re = u_si * L_si / nu_si
T_si = 25 * L_si / u_si
# T_si = np.log(2) * L_si**2 / (np.pi**2 * nu_si)

logger.info(f"Max velocity ............ {u_si} m/s")
logger.info(f"Characteristic time ..... {T_si} s")
logger.info(f"Reynolds number ......... {Re:,f}")
logger.info(f"Pressure drop ........... {dP_si / 1000} kPa/m")
logger.info(f"Body acceleration ....... {g_si} m/s^2")

dx = L_si / L
tau, u = calc_lbm_params_si(dx, u_si, L_si, nu_si)
dt = (u / u_si) * dx

logger.info(f"dx ...................... {dx}")
logger.info(f"dt ...................... {dt}")
logger.info(f"u_LU .................... {u:.4f}")
logger.info(f"L_LU .................... {L:.1f}")
logger.info(f"tau_LU .................. {tau:.3f}")

domain = Domain.make(upper=[L_si, L_si / 4], dx=dx)
scales = Scales(dx, dt)
iters = IterInfo.make(
    dt=dt,
    dt_output=T_si,
    t_max=T_si * 25,
)

g_lu = scales.acceleration.to_lattice_units(g_si)
g_lu = np.array([0, g_lu], dtype=np.float32)

# %% Initialize arrays

logger.info("Initializing arrays")

omega = trt(tau)
# omega = trt(tau, tau)
# omega = bgk(tau)

cnt = domain.counts.astype(np.int32)
sim = Simulation("gpu", cnt, omega)

sim.cells.flags[+0, :] = CellFlags.WALL
sim.cells.flags[-1, :] = CellFlags.WALL

sim.fluid.vel[:, :, 0] = (g_si / (2 * nu_si)) * domain.xx * (L_si - domain.xx)
sim.fluid.vel[:] = scales.velocity.to_lattice_units(sim.fluid.vel)

sim.set_gravity(g_lu)


# %% Run simulation loop

for iter in run_sim(sim, iters, "out", checkpoints=NoCheckpoints()):
    # Generate output plots:

    fig = plt.figure(figsize=(6, 3))
    ax = fig.gca()
    plt.minorticks_on()
    ax.grid(True, which="both", linewidth=0.5, color="#CCCCCC")

    # Steady state.
    u = (1 / 2) * (g_si / nu_si) * domain.x * (L_si - domain.x)
    ax.plot(100 * domain.x, 100 * u, "k--", linewidth=1.5, label="Analytical")

    # Transient.
    # u = u_analytical(domain.x, step * time.dt_output, L_si, g_si, nu_si)
    # ax.plot(100 * domain.x, 100 * u, "g--", linewidth=1.5, label="Analytical")

    # Simulation.
    u = scales.velocity.to_physical_units(sim.fluid.vel)
    u = np.sqrt(np.sum(u**2, axis=-1))[:, 10]
    ax.plot(
        100 * domain.x,
        100 * u,
        linewidth=2.5,
        c="#7d39aa",
        alpha=0.8,
        label="Simulation",
    )

    ax.legend(edgecolor="none")
    # ax.set_ylim(ymin=0)
    # ax.set_xlim(0, 100 * L_si)
    ax.set_xlabel(r"$y$ [cm]")
    ax.set_ylabel(r"$u(y)$ [cm/sec]")
    plt.tight_layout()
    plt.savefig(f"out/velprof_{iter:06d}.png", dpi=200)
    fig.clear()
    plt.close()

    if False:
        with VtiWriter(str(f"out/data_{iter:06d}.vti"), domain) as writer:
            writer.add_data("density", sim.fluid.rho)
            writer.add_data("velocity", sim.fluid.vel * scales.dx / scales.dt, default=True)
            writer.add_data("cell", sim.domain.cell_type)


# %% Save final output

vel_[:] = scales.velocity.to_physical_units(vel_)
np.save(f"out/final.{tau}.{L}.npy", vel_)
sys.exit(0)

# %% Combine outputs into one plot.

from danpy.plotting import style

style()

fig = plt.figure(figsize=(5, 3))
ax = fig.gca()
plt.minorticks_on()
ax.grid(True, which="both", linewidth=0.5, color="#CCCCCC")


# Steady state.
x = np.linspace(0, L_si, 250)
u = (1 / 2) * (g_si / nu_si) * x * (L_si - x)
ax.plot(100 * x, 100 * u, "k--", linewidth=1, label="Analytical", zorder=10)

for i, (tau, L) in enumerate([(0.55, 100), (0.51, 100), (0.55, 200), (0.51, 200)]):
    # Simulation.
    vel_ = np.load(f"out/final.{tau}.{L}.npy")
    x = np.linspace(0, L_si, vel_.shape[0])
    u = np.sqrt(np.sum(vel_**2, axis=-1))[:, 1]
    ax.plot(100 * x, 100 * u, linewidth=1.5, alpha=0.8, label=f"Params #{i + 1}")

ax.legend(edgecolor="none")
ax.set_ylim(ymin=0)
ax.set_xlim(0, 100 * L_si)
ax.set_xlabel(r"$y$ [cm]")
ax.set_ylabel(r"$u(y)$ [cm/sec]")
plt.tight_layout()
plt.savefig("out/velprof_combined.svg")
fig.clear()
plt.close()

# %%
