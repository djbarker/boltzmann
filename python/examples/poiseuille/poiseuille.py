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

dom_meta = DomainMeta.with_extent_and_counts(lower=[0, 0], upper=[0.01, 0.01], counts=[400, 100])
fluid_meta = FluidMeta.make(mu=0.1, rho=1000)
scales = Scales.make(dx=dom_meta.dx, cs=1.0)
time_meta = TimeMeta.make(dt_output=0.1, output_count=10)
sim_meta = SimulationMeta(domain=dom_meta, time=time_meta, scales=scales, fluid=fluid_meta)

logger.info(f"\n{pprint(asdict(sim_meta), sort_dicts=False, width=10)}")

# pipe width [m]
l_si = dom_meta.width

# gravitational acceleration [m/s^2]
g_si = 0.1
g_lu = scales.rescale(g_si, **ACCELERATION)

# max infinite time velocity
v_si = (g_si / fluid_meta.nu) * (l_si**2) / 8
logger.info(f"Max vel.: {v_si:.04f} m/s")

re_no = v_si * l_si / fluid_meta.nu
logger.info(f"Reynolds no.:  {re_no:,.0f}")

g_lu = np.array([0, g_lu], dtype=np.float32)


# %% Compile

logger.info("Compiling using Numba...")

from boltzmann.impl2 import (  # noqa: E402
    calc_equilibrium,
    loop_for_2,
    make_array,
    unflatten,
    PeriodicDomain,
    NumbaParams,
    D2Q9,
)


# make numba objects
pidx = PeriodicDomain(dom_meta.counts)
params = NumbaParams(scales.dt, scales.dx, scales.cs, sim_meta.w_pos_lu, sim_meta.w_neg_lu, g_lu)

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
is_wall = (cells.cells == CellType.BC_WALL.value).astype(np.int32)
is_fixed = (cells.cells != CellType.FLUID.value).astype(np.int32)

# set wall velocity to zero
# (just for nice output, doesn't affect the simulation)
vel_ = unflatten(pidx, fluid.vel)
vel_[cells_ == CellType.BC_WALL.value, 0] = 0


# initial f is equilibrium for desired values of v and rho
calc_equilibrium(fluid.vel, fluid.rho, fluid.f1, params.cs_si, D2Q9)
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

        fig = plt.figure()
        ax = fig.gca()
        plt.minorticks_on()
        ax.grid(True, which="both", linewidth=0.5, color="#CCCCCC")
        w = dom_meta.width
        x = dom_meta.x
        u = (1 / 2) * (g_si / fluid_meta.nu) * x * (w - x)
        ax.plot(100 * x, 100 * u, "k--")
        ax.plot(
            100 * x,
            100 * np.sqrt(np.sum(vel_**2, axis=-1))[10, 1:-1],
            linewidth=2,
            c="#7d39aa",
            alpha=0.8,
        )
        ax.set_ylim(ymin=0)
        ax.set_xlim(0, 1)
        ax.set_xlabel(r"$y$ [cm]")
        ax.set_ylabel(r"$u_x(y)$ [cm/sec]")
        plt.savefig(base / f"velprof_{step:06d}.svg")
        fig.clear()

        if self.vti:
            with VtiWriter(str(base / f"data_{step:06d}.vti"), pidx) as writer:
                writer.add_data("density", fluid.rho)
                writer.add_data("velocity", fluid.vel, default=True)
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
