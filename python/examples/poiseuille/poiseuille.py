# %% Imports

import logging
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from pprint import pprint
from dataclasses import asdict

from boltzmann.utils.logger import basic_config
from boltzmann.core import (
    Domain,
    Scales,
    SimulationMeta,
    FluidMeta,
    CellType,
    TimeMeta,
    ACCELERATION,
)
from boltzmann.simulation import load_fluid, run_sim_cli, save_fluid
from boltzmann.vtkio import VtiWriter
from boltzmann_rs import Simulation

basic_config()

logger = logging.getLogger(__name__)
logger.info("Starting")


# %% Params

# pipe width [m]
l_si = 0.01

# gravitational acceleration [m/s^2]
g_si = 0.2

# max infinite time velocity
mu_si = 0.01
rho_si = 1000
nu_si = mu_si / rho_si
v_si = (rho_si * g_si / mu_si) * (l_si**2) / 8
logger.info(f"Max vel.: {v_si:.04f} m/s")

re_no = v_si * l_si * rho_si / mu_si
logger.info(f"Reynolds no.:  {re_no:,.0f}")

dx = l_si / (24 * 8)

# Max Mach number implied dt
Mmax = 0.1
cs = v_si / Mmax
dt_mach = np.sqrt(3) * dx / cs

# Max tau implied dt
tau_max = 0.7
dt_err = (1 / 3) * (tau_max - 0.5) * dx**2 / nu_si

dt = min(dt_err, dt_mach) * 0.5

# Characteristic time
T = np.log(2) * l_si**2 / (np.pi**2 * nu_si)

domain = Domain.make(upper=[l_si, l_si / 4], dx=dx)
assert domain.dx == dx
fluid_meta = FluidMeta.make(nu=nu_si, rho=rho_si)
scales = Scales.make(dx=dx, dt=dt)
time_meta = TimeMeta.make(
    dt_output=T * 0.25,
    t_max=T * 10,
)
sim_meta = SimulationMeta(domain=domain, time=time_meta, scales=scales, fluid=fluid_meta)

logger.info(f"\n{pprint(asdict(sim_meta), sort_dicts=False, width=10)}")

g_lu = scales.to_lattice_units(g_si, **ACCELERATION)
g_lu = np.array([0, g_lu], dtype=np.float32)

# %% Initialize arrays

logger.info("Initializing arrays")

omega_ns = sim_meta.w_pos_lu
cnt = domain.counts.astype(np.int32)
sim = Simulation(cnt, q=9, omega_ns=omega_ns)

cells_ = domain.unflatten(sim.domain.cell_type)
cells_[+0, :] = CellType.WALL.value
cells_[-1, :] = CellType.WALL.value

vel_ = domain.unflatten(sim.fluid.vel)

mem_mb = (sim.domain.size_bytes + sim.fluid.size_bytes) / 1e6
logger.info(f"Memory usage: {mem_mb:,.2f} MB")

sim.set_gravity(g_lu)

# Important!
sim.finalize()

# %% Define simulation loop


class PlanePoiseuille:
    def __init__(self, vti: bool = False) -> None:
        self.vti = vti

    def loop_for(self, steps: int):
        if np.any(~np.isfinite(sim.fluid.f)):
            raise ValueError("Non-finite value in f.")

        sim.iterate(steps)

        logger.info(f"u_max = {100 * np.max(np.sum(vel_**2, axis=-1)) / v_si:.2f} %")

    def write_output(self, base: Path, step: int):
        # assert vel_.base is sim.fluid.vel

        fig = plt.figure(figsize=(6, 3))
        ax = fig.gca()
        plt.minorticks_on()
        ax.grid(True, which="both", linewidth=0.5, color="#CCCCCC")
        w = domain.width
        x = domain.x
        u = (1 / 2) * (g_si / fluid_meta.nu) * x * (w - x)
        ax.plot(100 * x, 100 * u, "k--", linewidth=1.5, label="Analytical")
        ax.plot(
            100 * x,
            100 * np.sqrt(np.sum((vel_ * scales.dx / scales.dt) ** 2, axis=-1))[:, 1],
            linewidth=2.5,
            c="#7d39aa",
            alpha=0.8,
            label="Simulation",
        )
        ax.legend(edgecolor="none")
        ax.set_ylim(ymin=0)
        ax.set_xlim(0, 1)
        ax.set_xlabel(r"$y$ [cm]")
        ax.set_ylabel(r"$u(y)$ [cm/sec]")
        plt.tight_layout()
        plt.savefig(base / f"velprof_{step:06d}.png", dpi=200)
        fig.clear()

        if self.vti:
            with VtiWriter(str(base / f"data_{step:06d}.vti"), domain) as writer:
                writer.add_data("density", sim.fluid.rho)
                writer.add_data("velocity", sim.fluid.vel * scales.dx / scales.dt, default=True)
                writer.add_data("cell", sim.domain.cell_type)

    def write_checkpoint(self, base: Path):
        save_fluid(base, sim.fluid)

    def read_checkpoint(self, base: Path):
        load_fluid(base, sim.fluid)


# %% Main Loop

run_sim_cli(sim_meta, PlanePoiseuille())

# %%
