# %% Imports

import logging
import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

from pathlib import Path
from pprint import pprint
from dataclasses import asdict

from boltzmann.utils.logger import basic_config
from boltzmann.core import (
    VELOCITY,
    Domain,
    Scales,
    SimulationMeta,
    FluidMeta,
    CellType,
    TimeMeta,
    D2Q9 as D2Q9_py,
    D2Q5 as D2Q5_py,
    calc_curl_2d,
    upstream_indices,
)
from boltzmann.utils.mpl import PngWriter
from boltzmann.simulation import Cells, FluidField, ScalarField, run_sim_cli
from boltzmann_rs import loop_for_advdif_2d


basic_config()
logger = logging.getLogger(__name__)
logger.info("Starting")

# %% Params

# dimensions [m]
y_si = 0.5
x_si = 1.5

# flow velocity [m/s]
v0_si = 1.0

vmax_vmag = 1.6 * v0_si
vmax_curl = 30 * v0_si
vmax_conc = 1.025


nu_si = 1e-4
rho_si = 1000

re_no = v0_si * y_si / nu_si
logger.info(f"Reynolds no.:  {re_no:,.0f}")

# geometry
# scale = 35
scale = 10
dx = 1.0 / (100 * scale)
upper = np.array([x_si, y_si])

# Max Mach number implied dt
Mmax = 0.1
cs = v0_si / Mmax
dt_mach = np.sqrt(3) * dx / cs

# Max tau implied dt
tau_max = 0.6
dt_err = (1 / 3) * (tau_max - 0.5) * dx**2 / nu_si

dt = min(dt_err, dt_mach)

# dimensionless time does not depend on viscosity, purely on distances
fps = 30
out_dx_si = x_si / (5 * fps)  # want to output when flow has moved this far
sim_dx_si = v0_si * dt  # flow moves this far in dt (i.e. one iteration)
n = out_dx_si / sim_dx_si
n = int(n + 1e-8)
out_dt_si = dt * n

domain = Domain.make(upper=upper, dx=dx)
scales = Scales.make(dx=dx, dt=dt)
time_meta = TimeMeta.make(
    dt_output=out_dt_si,
    # dt_output=dt,
    output_count=250,
)
fluid_meta = FluidMeta.make(nu=nu_si, rho=rho_si)
sim = SimulationMeta(domain, time_meta, scales, fluid_meta)

indices = upstream_indices(domain, D2Q9_py)

logger.info(f"\n{pprint(asdict(sim), sort_dicts=False, width=10)}")

# %% Initialize arrays

logger.info("Initializing arrays")

cells = Cells(domain)
fluid = FluidField("fluid", D2Q9_py, domain)
tracer = ScalarField("tracer", D2Q5_py, domain)

# introduce slight randomness to initial density
np.random.seed(42)
rho_ = domain.unflatten(fluid.rho)
rho_[:, :] = fluid_meta.rho
# rho_[10:-10, 10:-10] *= 1 + 0.001 * (np.random.uniform(size=rho_[10:-10, 10:-10].shape) - 0.5)

# fixed velocity in- & out-flow
# (only need to specify one due to periodicity)
cells_ = domain.unflatten(cells.cells)
cells_[:, +0] = CellType.BC_VELOCITY.value  # bottom
cells_[:, -1] = CellType.BC_VELOCITY.value  # top

# set upper half's velocity & concentration
vel_ = domain.unflatten(fluid.vel)
vel_[:, domain.counts[1] // 2 :, 0] = -v0_si
vel_[:, : domain.counts[1] // 2, 0] = +v0_si

# perturb velocity
n = 10
l = x_si / n  # wavelength
vy_si = v0_si * 0.001 * np.sin(2 * np.pi * (domain.x / l))
vel_[:, 1:-1, 1] = vy_si[:, None]

conc_ = domain.unflatten(tracer.val)
conc_[:, domain.counts[1] // 2 :] = 1.0

# flag arrays
# TODO: this is duped between sims
is_wall = cells.cells == CellType.BC_WALL.value
is_fixed = cells.cells != CellType.FLUID.value

# set wall velocity to zero
# (just for nice output, doesn't affect the simulation)
vel_[cells_ == CellType.BC_WALL.value, 0] = 0

# IMPORTANT: convert velocity to lattice units / timestep
vel_[:] = scales.to_lattice_units(vel_, **VELOCITY)

fluid.equilibrate()
tracer.equilibrate(fluid.vel)

mem_mb = (cells.size_bytes + fluid.size_bytes + tracer.size_bytes) / 1e6
logger.info(f"Memory usage: {mem_mb:,.2f} Mb")


# %% Define simulation loop

FONT = dict(
    fontsize=14,
    fontfamily="sans-serif",
    fontstyle="italic",
    fontweight="bold",
)


def png_writer(path: Path, data: np.ndarray, **kwargs) -> PngWriter:
    fig_kwargs = kwargs.pop("fig_kwargs", {})
    fig_kwargs["dpi"] = 300
    return PngWriter(path, domain, cells_, data, fig_kwargs=fig_kwargs, **kwargs)


def annotate(ax1, text: str, **kwargs):
    ax1.annotate(
        text,
        (y_si * 0.025, y_si * 0.975),
        xytext=(0.25, -1.1),
        textcoords="offset fontsize",
        **kwargs,
        **FONT,
    )


def write_png_vmag(path: Path):
    vmag = np.sqrt(np.sum(scales.to_physical_units(vel_, **VELOCITY) ** 2, axis=-1))
    vmag = np.tanh(vmag / vmax_vmag) * vmax_vmag

    with png_writer(
        path,
        vmag,
        cmap=plt.cm.inferno,  # type: ignore
        vmin=0,
        vmax=vmax_vmag,
        fig_kwargs={
            "facecolor": "#888888",
        },
    ) as fig:
        ax1 = fig.gca()
        n = 70
        s = domain.counts[0] // n
        xx, yy = np.meshgrid(
            domain.x[s // 2 :: s],
            domain.y[s // 2 :: s],
        )
        vx = np.squeeze(vel_[s // 2 :: s, s // 2 :: s, 0::2]).T
        vy = np.squeeze(vel_[s // 2 :: s, s // 2 :: s, 1::2]).T
        vm = np.sqrt(vx**2 + vy**2)
        vx = np.where(vm > 0, vx / vm, 0)
        vy = np.where(vm > 0, vy / vm, 0)
        ax1.scatter(
            xx.flatten(),
            yy.flatten(),
            c="#AAAAAA",
            s=1,
            marker=".",
            edgecolors="none",
            alpha=0.7,
        )
        ax1.quiver(
            xx,
            yy,
            vx,
            vy,
            scale=1.5 * n,
            scale_units="width",
            headwidth=0,
            headlength=0,
            headaxislength=0,
            # pivot="mid",
            width=3,
            units="dots",
            color="#AAAAAA",
            alpha=0.7,
        )
        annotate(ax1, "Velocity", c="w")


def write_png_curl(path: Path):
    vort = calc_curl_2d(fluid.vel, cells.cells, indices)  # in LU
    vort = vort * ((dx / dt) / dx)  # in SI
    vort = np.tanh(vort / vmax_curl) * vmax_curl

    with png_writer(
        path,
        vort,
        cmap=plt.cm.RdBu,  # type: ignore
        vmin=-vmax_curl,
        vmax=vmax_curl,
        fig_kwargs={
            "facecolor": "#E0E0E0",
        },
    ) as fig:
        ax1 = fig.gca()
        annotate(ax1, "Vorticity", c="k")


def write_png_conc(path: Path):
    with png_writer(
        path,
        conc_,
        cmap=plt.cm.viridis,  # type: ignore
        vmin=0,
        vmax=vmax_conc,
        fig_kwargs={
            "facecolor": "#E0E0E0",
        },
    ) as fig:
        ax1 = fig.gca()
        annotate(ax1, "Tracer", c="w")


class KelvinHelmholtz:
    def loop_for(self, steps: int):
        if steps % 2 != 0:
            steps += 1  # even steps please

        if np.any(~np.isfinite(fluid.f1)) or np.any(~np.isfinite(tracer.f1)):
            raise ValueError("Non-finite value in f.")

        # if np.any(fluid.f1 < 0) or np.any(tracer.f1 < 0):
        #     raise ValueError("Negative value in f.")

        # loop_for_2_advdif(
        #     steps,
        #     fluid.rho,
        #     fluid.vel,
        #     fluid.f1,
        #     fluid.f2,
        #     D2Q9_nb,
        #     tracer.val,
        #     tracer.f1,
        #     tracer.f2,
        #     D2Q5_nb,
        #     is_wall,
        #     is_fixed,
        #     params_nb,
        #     domain_nb,
        # )

        omega_ns = sim.w_pos_lu
        omega_ad = sim.w_pos_lu

        loop_for_advdif_2d(
            steps,
            fluid.f1,
            fluid.rho,
            fluid.vel,
            tracer.f1,
            tracer.val,
            is_wall,
            is_fixed,
            indices,
            domain.counts,
            omega_ns,
            omega_ad,
        )

    def write_output(self, base: Path, step: int):
        assert vel_.base is fluid.vel
        write_png_curl(base / f"curl_{step:06d}.png")
        write_png_conc(base / f"conc_{step:06d}.png")
        write_png_vmag(base / f"vmag_{step:06d}.png")

    def write_checkpoint(self, base: Path):
        # cells.save(base)  # these are defined above (as are mask arrays is_wall etc)
        fluid.save(base)
        tracer.save(base)

    def read_checkpoint(self, base: Path):
        # cells.load(base)
        fluid.load(base)
        tracer.load(base)


# %% Main Loop

run_sim_cli(sim, KelvinHelmholtz(), write_checkpoints=False)

# render with
# ffmpeg -i out/curl_%06d.png -i out/vmag_%06d.png -c:v libx264 -crf 10 -r 30 -filter_complex "[1]pad=iw:ih+2:0:2[v1];[0][v1]vstack=inputs=2" -y kh.mp4
# ffmpeg -i out/conc_%06d.png -i out/curl_%06d.png -i out/vmag_%06d.png -c:v libx264 -crf 10 -r 30 -filter_complex "[1]pad=iw:ih+2:0:2[v1];[2]pad=iw:ih+2:0:2[v2];[0][v1][v2]vstack=inputs=3" -y kh4.mp4
