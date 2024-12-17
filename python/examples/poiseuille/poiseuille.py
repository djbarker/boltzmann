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
    DomainMeta,
    Scales,
    SimulationMeta,
    FluidMeta,
    CellType,
    TimeMeta,
    ACCELERATION,
    VELOCITY,
)
from boltzmann.simulation import Cells, FluidField, SimulationRunner
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

dx = l_si / (24 * 6)

# Max Mach number implied dt
Mmax = 0.1
cs = v_si / Mmax
dt_mach = np.sqrt(3) * dx / cs

# Max tau implied dt
tau_max = 0.7
dt_err = (1 / 3) * (tau_max - 0.5) * dx**2 / nu_si

dt = min(dt_err, dt_mach) * 0.5

dom_meta = DomainMeta.make(upper=[l_si, l_si / 4], dx=dx)
assert dom_meta.dx == dx
fluid_meta = FluidMeta.make(nu=nu_si, rho=rho_si)
scales = Scales.make(dx=dx, dt=dt)
time_meta = TimeMeta.make(dt_output=0.5, output_count=25)
sim_meta = SimulationMeta(domain=dom_meta, time=time_meta, scales=scales, fluid=fluid_meta)

logger.info(f"\n{pprint(asdict(sim_meta), sort_dicts=False, width=10)}")


g_lu = scales.to_lattice_units(g_si, **ACCELERATION)
g_lu = np.array([0, g_lu], dtype=np.float32)


# %% Compile

logger.info("Compiling using Numba...")

from boltzmann.impl2 import (  # noqa: E402
    loop_for_2,
    unflatten,
    PeriodicDomain,
    NumbaParams,
    D2Q9,
)

# make numba objects
pidx = PeriodicDomain(dom_meta.counts)
params = NumbaParams(scales.dt, scales.dx, sim_meta.w_pos_lu, sim_meta.w_neg_lu, g_lu)

# %% Initialize arrays

logger.info("Initializing arrays")

fluid = FluidField(
    "fluid",
    f1=make_array(pidx, 9),
    vel=make_array(pidx, 2),
    rho=make_array(pidx, fill=fluid_meta.rho),
)

cells = Cells(
    cells=make_array(pidx, dtype=np.int32),
)

cells_ = unflatten(pidx, cells.cells)
cells_[:, 1] = CellType.BC_WALL.value
cells_[:, -2] = CellType.BC_WALL.value

# flag arrays
# TODO: this is duped between sims
is_wall = (cells.cells == CellType.BC_WALL.value).astype(np.int32)
is_fixed = (cells.cells != CellType.FLUID.value).astype(np.int32)

# set wall velocity to zero
# (just for nice output, doesn't affect the simulation)
vel_ = unflatten(pidx, fluid.vel)
vel_[cells_ == CellType.BC_WALL.value, 0] = 0

# Important convert velocity to lattice units / timestep
# TODO: should be somewhere inside the simulation class
# NOTE: for this sim doesn't matter since velocity starts zero everywhere!
vel_[:] = scales.to_lattice_units(vel_, **VELOCITY)

# initial f is equilibrium for desired values of v and rho
calc_equilibrium(fluid.vel, fluid.rho, fluid.f1, D2Q9)
fluid.f2[:] = fluid.f1[:]


@dataclass
class PlanePoiseuille:
    vti: bool = field(default=False)

    def loop_for(self, steps: int):
        if np.any(~np.isfinite(fluid.f1)):
            raise ValueError("Non-finite value in f.")

        loop_for_2(
            steps,
            fluid.vel,
            fluid.rho,
            fluid.f1,
            fluid.f2,
            is_wall,
            is_fixed,
            params,
            pidx,
            D2Q9,
        )

        logger.info(f"u_max = {np.max(np.sum(vel_**2, axis=-1)):.4f}")

    def write_output(self, base: Path, step: int):
        assert vel_.base is fluid.vel

        fig = plt.figure(figsize=(6, 3))
        ax = fig.gca()
        plt.minorticks_on()
        ax.grid(True, which="both", linewidth=0.5, color="#CCCCCC")
        w = dom_meta.width
        x = dom_meta.x
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
            with VtiWriter(str(base / f"data_{step:06d}.vti"), pidx) as writer:
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

base = Path("out")

if base.exists():
    assert base.is_dir()
else:
    base.mkdir()

logger.info("Creating simulation runner")
runner = SimulationRunner(base, sim_meta, PlanePoiseuille())

logger.info("Starting simulation loop")
runner.run()
