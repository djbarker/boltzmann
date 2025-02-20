# %% Imports

import logging
import matplotlib.pyplot as plt
import numpy as np
import sys

from pathlib import Path
from dataclasses import dataclass, field

from boltzmann.utils.logger import basic_config, dotted
from boltzmann.core import (
    ACCELERATION,
    VELOCITY,
    CellType,
    Domain,
    Scales,
    SimulationMeta,
    TimeMeta,
    check_lbm_params,
)
from boltzmann.simulation import run_sim_cli
from boltzmann.vtkio import VtiWriter
from boltzmann_rs import Simulation

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
Re = 1000

# choose parameterisation
L = 200
tau = 0.51

# implied acceleration & pressure drop
g_si = 8 * Re * (mu_si / rho_si) ** 2 / L_si**3
dP_si = rho_si * g_si

# max infinite time velocity
nu_si = mu_si / rho_si
u_si = (g_si / nu_si) * (L_si**2) / 8
Re = u_si * L_si / nu_si
# T_si = 25 * L_si / u_si
T_si = np.log(2) * L_si**2 / (np.pi**2 * nu_si)

logger.info(f"Max velocity ............ {u_si} m/s")
logger.info(f"Characteristic time ..... {T_si} s")
logger.info(f"Reynolds number ......... {Re:,f}")
logger.info(f"Pressure drop ........... {dP_si / 1000} kPa/m")
logger.info(f"Body acceleration ....... {g_si} m/s^2")

nu = (tau - 0.5) / 3
u = Re * nu / L
dx = L_si / L
dt = (u / u_si) * dx

logger.info(f"dx ...................... {dx}")
logger.info(f"dt ...................... {dt}")
logger.info(f"u_LU .................... {u:.4f}")
logger.info(f"L_LU .................... {L:.1f}")
logger.info(f"nu_LU ................... {nu:.4f}")
logger.info(f"tau_LU .................. {tau:.3f}")

check_lbm_params(L, u, tau, M_max=0.3)

domain = Domain.make(upper=[L_si, L_si / 4], dx=dx)
scales = Scales.make(dx=dx, dt=dt)
time = TimeMeta.make(
    dt_step=dt,
    dt_output=T_si,
    t_max=T_si * 25,
)
meta = SimulationMeta(domain, time)

logger.info(f"Cell counts ............. {tuple(domain.counts)}")
logger.info(f"Iters / output .......... {time.batch_steps}")
logger.info(f"Reynolds number ......... {Re:,.0f}")

g_lu = scales.to_lattice_units(g_si, **ACCELERATION)
g_lu = np.array([0, g_lu], dtype=np.float32)

# %% Initialize arrays

logger.info("Initializing arrays")

cnt = domain.counts.astype(np.int32)
sim = Simulation(cnt, q=9, omega_ns=1 / tau)

cells_ = domain.unflatten(sim.domain.cell_type)
cells_[+0, :] = CellType.WALL.value
cells_[-1, :] = CellType.WALL.value

vel_ = domain.unflatten(sim.fluid.vel)
vel_[:, :, 0] = ((g_si / (2 * nu_si)) * domain.x * (L_si - domain.x))[:, None]
vel_[:] = scales.to_lattice_units(vel_, **VELOCITY)

mem_mb = (sim.domain.size_bytes + sim.fluid.size_bytes) / 1e6
logger.info(f"Memory usage: {mem_mb:,.2f} MB")

sim.set_gravity(g_lu)

# Important!
sim.finalize(True)

# %% Define simulation loop


@dataclass
class PlanePoiseuille:
    vti: bool = field(default=False)

    def loop_for(self, steps: int):
        if np.any(~np.isfinite(sim.fluid.f)):
            raise ValueError("Non-finite value in f.")

        sim.iterate(steps)

        logger.info(f"u_max = {100 * np.max(np.sum(vel_**2, axis=-1)) / u_si:.2f} %")

    def write_output(self, base: Path, step: int):
        # assert vel_.base is sim.fluid.vel

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
        u = np.sqrt(np.sum(scales.to_physical_units(vel_, **VELOCITY) ** 2, axis=-1))[:, 1]
        ax.plot(100 * domain.x, 100 * u, linewidth=2.5, c="#7d39aa", alpha=0.8, label="Simulation")

        ax.legend(edgecolor="none")
        # ax.set_ylim(ymin=0)
        # ax.set_xlim(0, 100 * L_si)
        ax.set_xlabel(r"$y$ [cm]")
        ax.set_ylabel(r"$u(y)$ [cm/sec]")
        plt.tight_layout()
        plt.savefig(base / f"velprof_{step:06d}.png", dpi=200)
        fig.clear()
        plt.close()

        if self.vti:
            with VtiWriter(str(base / f"data_{step:06d}.vti"), domain) as writer:
                writer.add_data("density", sim.fluid.rho)
                writer.add_data("velocity", sim.fluid.vel * scales.dx / scales.dt, default=True)
                writer.add_data("cell", sim.domain.cell_type)

    def write_checkpoint(self, base: Path):
        pass

    def read_checkpoint(self, base: Path):
        pass


# %% Main Loop

run_sim_cli(meta, PlanePoiseuille(vti=False))

# %% Save final output

vel_[:] = scales.to_physical_units(vel_, **VELOCITY)
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
