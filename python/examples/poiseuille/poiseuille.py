# %% Imports

import logging
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from pprint import pprint
from dataclasses import asdict, dataclass, field

from boltzmann.utils.logger import basic_config
from boltzmann.core import (
    Domain,
    Scales,
    SimulationMeta,
    FluidMeta,
    CellType,
    TimeMeta,
    ACCELERATION,
    VELOCITY,
    D2Q9 as D2Q9_py,
)
from boltzmann.simulation import Cells, FluidField, run_sim_cli
from boltzmann.vtkio import VtiWriter

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

domain = Domain.make(upper=[l_si, l_si / 4], dx=dx)
assert domain.dx == dx
fluid_meta = FluidMeta.make(nu=nu_si, rho=rho_si)
scales = Scales.make(dx=dx, dt=dt)
time_meta = TimeMeta.make(
    dt_output=0.5,
    # dt_output=dt,
    output_count=25,
)
sim = SimulationMeta(domain=domain, time=time_meta, scales=scales, fluid=fluid_meta)

logger.info(f"\n{pprint(asdict(sim), sort_dicts=False, width=10)}")

g_lu = scales.to_lattice_units(g_si, **ACCELERATION)
g_lu = np.array([0, g_lu], dtype=np.float32)


# %% Initialize arrays

logger.info("Initializing arrays")

cells = Cells(domain)
fluid = FluidField(
    "fluid",
    D2Q9_py,
    domain,
)

cells_ = domain.unflatten(cells.cells)
cells_[:, 1] = CellType.BC_WALL.value
cells_[:, -2] = CellType.BC_WALL.value

# flag arrays
# TODO: this is duped between sims
is_wall = (cells.cells == CellType.BC_WALL.value).astype(np.int32)
is_fixed = (cells.cells != CellType.FLUID.value).astype(np.int32)

# set wall velocity to zero
# (just for nice output, doesn't affect the simulation)
vel_ = domain.unflatten(fluid.vel)
vel_[cells_ == CellType.BC_WALL.value, :] = 0

# Important convert velocity to lattice units / timestep
# TODO: should be somewhere inside the simulation class
#        ... actually no. We may want to just do everything in lattice-units
#        ... what about some sort of with statement?
# NOTE: for this sim it doesn't matter because vel is zero.
vel_[:] = scales.to_lattice_units(vel_, **VELOCITY)

# TODO: automating this too couldn't hurt!
fluid.equilibrate()

# %% Compile

from boltzmann.bz_numba import (  # noqa: E402
    NumbaDomain,
    NumbaParams,
    D2Q9 as D2Q9_nb,
    loop_for_2,
)

# make numba objects
domain_nb = NumbaDomain(domain.counts)
params_nb = NumbaParams(dt, domain.dx, sim.w_pos_lu, sim.w_neg_lu, g_lu)

# %% Define simulation loop


@dataclass
class PlanePoiseuille:
    vti: bool = field(default=False)

    def loop_for(self, steps: int):
        if np.any(~np.isfinite(fluid.f1)):
            raise ValueError("Non-finite value in f.")

        loop_for_2(
            steps,
            fluid.rho,
            fluid.vel,
            fluid.f1,
            fluid.f2,
            is_wall,
            is_fixed,
            params_nb,
            domain_nb,
            D2Q9_nb,
        )

        logger.info(f"u_max = {np.max(np.sum(vel_**2, axis=-1)):.4f}")

    def write_output(self, base: Path, step: int):
        assert vel_.base is fluid.vel

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
            100 * np.sqrt(np.sum((vel_ * scales.dx / scales.dt) ** 2, axis=-1))[1, 1:-1],
            linewidth=2.5,
            c="#7d39aa",
            alpha=0.8,
            label="Simulation",
        )
        ax.legend(edgecolor="none")
        ax.set_ylim(ymin=0)
        ax.set_xlim(0, 1)
        ax.set_xlabel(r"$y$ [cm]")
        ax.set_ylabel(r"$u_x(y)$ [cm/sec]")
        plt.tight_layout()
        plt.savefig(base / f"velprof_{step:06d}.png", dpi=200)
        fig.clear()

        if self.vti:
            with VtiWriter(str(base / f"data_{step:06d}.vti"), domain) as writer:
                writer.add_data("density", fluid.rho)
                writer.add_data("velocity", fluid.vel * scales.dx / scales.dt, default=True)
                writer.add_data("cell", cells.cells)

    def write_checkpoint(self, base: Path):
        fluid.save(base)
        cells.save(base)

    def read_checkpoint(self, base: Path):
        fluid.load(base)
        cells.load(base)


# %% Main Loop

run_sim_cli(sim, PlanePoiseuille())
