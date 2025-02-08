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
)
from boltzmann.utils.mpl import PngWriter
from boltzmann.simulation import run_sim_cli, load_fluid, load_scalar, save_fluid, save_scalar
from boltzmann_rs import (
    calc_curl_2d,
    Simulation,
)


basic_config()
logger = logging.getLogger(__name__)
logger.info("Starting")

# %% Params

# https://www.sciencedirect.com/science/article/pii/S0898122118306278

# dimensions [m]
aspect_ratio = 2
y_si = 2.0
x_si = y_si * aspect_ratio

# flow velocity [m/s]
v0_si = 1.0

# vmax_vmag = 1.6 * v0_si
vmax_vmag = 1.3 * v0_si
vmax_curl = 30 * v0_si
vmax_conc = 1.025

nu_si = 1e-4
rho_si = 1000

# geometry
scale = 10
# scale = 5
dx = 1.0 / (100 * scale)
upper = np.array([x_si, y_si])

re_no = v0_si * (10 * dx) / nu_si
logger.info(f"Reynolds no.:  {re_no:,.0f}")


# Max Mach number implied dt
Mmax = 0.05
cs = v0_si / Mmax
dt_mach = np.sqrt(3) * dx / cs

# Max tau implied dt
tau_max = 0.55
dt_err = (1 / 3) * (tau_max - 0.5) * dx**2 / nu_si

dt = min(dt_err, dt_mach)

# dimensionless time does not depend on viscosity, purely on distances
fps = 30
out_dx_si = x_si / (20 * fps)  # want to output when flow has moved this far
sim_dx_si = v0_si * dt  # flow moves this far in dt (i.e. one iteration)
n = out_dx_si / sim_dx_si
n = int(n + 1e-8)
out_dt_si = dt * n

domain = Domain.make(upper=upper, dx=dx)
scales = Scales.make(dx=dx, dt=dt)
time_meta = TimeMeta.make(
    dt_output=out_dt_si,
    # dt_output=dt,
    output_count=1200,
)
fluid_meta = FluidMeta.make(nu=nu_si, rho=rho_si)
sim_meta = SimulationMeta(domain, time_meta, scales, fluid_meta)

logger.info(f"\n{pprint(asdict(sim_meta), sort_dicts=False, width=10)}")

# %% Initialize arrays

logger.info("Initializing simulation")

omega_ns = sim_meta.w_pos_lu
omega_ad = sim_meta.w_pos_lu

cnt = domain.counts.astype(np.int32)

sim = Simulation(cnt, q=9, omega_ns=omega_ns)
sim.add_tracer(q=5, omega_ad=omega_ad)

# fixed velocity in- & out-flow
# (only need to specify one due to periodicity)
cells_ = domain.unflatten(sim.domain.cell_type)
cells_[:, +0] = CellType.FIXED.value  # bottom
cells_[:, -1] = CellType.FIXED.value  # top

# set velocity
vel_ = domain.unflatten(sim.fluid.vel)
vel_[:, : domain.counts[1] // 2, 0] = +v0_si  # lower half
vel_[:, domain.counts[1] // 2 :, 0] = -v0_si  # upper half

# perturb velocity
n = 2
l = x_si / int(2 * n * aspect_ratio)  # wavelength
vy_si = v0_si * 0.001 * np.sin(2 * np.pi * (domain.x / l))
vy_si += v0_si * 0.00001 * np.sin(2 * np.pi * (domain.x / (l * 2)))
vy_si += v0_si * 0.0000001 * np.sin(2 * np.pi * (domain.x / (l * 4)))
vel_[:, 1:-1, 1] = vy_si[:, None]

# set concentration
conc_ = domain.unflatten(sim.tracer.val)
conc_[:, : domain.counts[1] // 2] = 1.0  # lower half

# IMPORTANT: convert velocity to lattice units / timestep
vel_[:] = scales.to_lattice_units(vel_, **VELOCITY)

sim.finalize()

mem_mb = (sim.domain.size_bytes + sim.fluid.size_bytes + sim.tracer.size_bytes) / 1e6
logger.info(f"Memory usage: {mem_mb:,.2f} MB")


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
    # vmag = np.sqrt(np.sum(scales.to_physical_units(vel_, **VELOCITY) ** 2, axis=-1))
    vmag = scales.to_physical_units(vel_, **VELOCITY)[:, :, 1]
    vmag = np.tanh(vmag / vmax_vmag) * vmax_vmag

    with png_writer(
        path,
        vmag,
        # cmap=plt.cm.inferno,  # type: ignore
        # vmin=0,
        cmap=plt.cm.Spectral,  # type: ignore
        vmin=-vmax_vmag,
        vmax=vmax_vmag,
        fig_kwargs={
            "facecolor": "#888888",
        },
    ) as fig:
        ax1 = fig.gca()
        # Plot the quivers.
        if False:
            n = 70
            s = domain.counts[0] // n
            xx, yy = np.meshgrid(
                domain.x[s // 2 :: s],
                domain.y[s // 2 :: s],
            )
            vx = np.squeeze(vel_[s // 2 :: s, s // 2 :: s, 0::2]).T[:, ::-1]
            vy = np.squeeze(vel_[s // 2 :: s, s // 2 :: s, 1::2]).T[:, ::-1]
            vm = np.sqrt(vx**2 + vy**2)
            vx = np.where(vm > 0, vx / vm, 0)
            vy = np.where(vm > 0, vy / vm, 0)
            ax1.scatter(
                xx.flatten(),
                yy.flatten(),
                c="#AAAAAA",
                s=3,
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
                width=2,
                units="dots",
                color="#AAAAAA",
                alpha=0.7,
            )
        annotate(ax1, "Velocity", c="w")


def write_png_curl(path: Path):
    vort = np.zeros_like(sim.fluid.rho)
    calc_curl_2d(sim.fluid.vel, sim.domain.cell_type, cnt, vort)  # in LU
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
        cmap=plt.cm.cividis,  # type: ignore
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
        if np.any(~np.isfinite(sim.fluid.f)) or np.any(~np.isfinite(sim.tracer.g)):
            raise ValueError("Non-finite value in f.")

        sim.iterate(steps)

    def write_output(self, base: Path, step: int):
        write_png_curl(base / f"curl_{step:06d}.png")
        write_png_conc(base / f"conc_{step:06d}.png")
        write_png_vmag(base / f"vmag_{step:06d}.png")

    def write_checkpoint(self, base: Path):
        save_fluid(base, sim.fluid)
        save_scalar(base, sim.tracer)

    def read_checkpoint(self, base: Path):
        load_fluid(base, sim.fluid)
        load_scalar(base, sim.tracer)
        sim.finalize()


# %% Main Loop

run_sim_cli(sim_meta, KelvinHelmholtz())

# render with
# FPS=30 ffmpeg -framerate $FPS -i out/conc_%06d.png -framerate $FPS -i out/curl_%06d.png -c:v libx264 -crf 10 -filter_complex "[1]pad=iw:ih+2:0:2[v1];[0][v1]vstack=inputs=2" -y kh.mp4
# FPS=30 ffmpeg -framerate $FPS -i out/conc_%06d.png -framerate $FPS -i out/curl_%06d.png -framerate $FPS -i out/vmag_%06d.png -c:v libx264 -crf 10 -filter_complex "[1]pad=iw:ih+2:0:2[v1];[2]pad=iw:ih+2:0:2[v2];[0][v1][v2]vstack=inputs=3" -y kh4.mp4

import sys

sys.exit()

# %%

import logging
import matplotlib.pyplot as plt
import numpy as np

import boltzmann_rs as bz

from boltzmann.utils.logger import basic_config, time

basic_config()
logger = logging.getLogger(__file__)

cnt = np.array([1000, 1000], dtype=np.int32)
sim = bz.Simulation(cnt, q=9, omega_ns=0.51)
sim.add_tracer(q=5, omega_ad=0.51)
v = sim.fluid.vel.reshape([1000, 1000, 2])
r = sim.fluid.rho.reshape([1000, 1000])
C = sim.tracer.val.reshape([1000, 1000])
v[:, 350:450, 0] = 0.1
v[:, 350:450, 1] = -0.05
# r[10, 10] = 1.01
x = np.linspace(-2, 2, 1000)
XX, YY = np.meshgrid(x, x)
RR = np.sqrt(XX**2 + YY**2)
C[:] = np.exp(-((RR - 0.7) ** 2) / (2 * 0.01))
sim.finalize()

vmag = np.sqrt(np.sum(v**2, axis=-1))
vmax = vmag.max()
plt.imshow(vmag.T, vmin=0, vmax=vmax, interpolation="none")
plt.colorbar()
plt.show()

plt.imshow(C.T, vmax=1)
plt.colorbar()
plt.show()

# vort = np.zeros_like(sim.fluid.rho)
# bz.calc_curl_2d(sim.fluid.vel, sim.domain.cell_type, cnt, vort)
# vort_ = vort.reshape([1000, 1000])
# plt.imshow(vort_)
# plt.show()

print(r.min(), r.max())

# %%

cells = sim.fluid.rho.shape[0]
steps = 200
with time(logger, cells * steps):
    sim.iterate(steps)

vmag = np.sqrt(np.sum(v**2, axis=-1))
plt.imshow(vmag.T, vmin=0, vmax=vmax, interpolation="none")
plt.colorbar()
plt.show()

plt.imshow(C.T, vmax=1.05)
plt.colorbar()
plt.show()

print(r.min(), r.max())
# %%
