# %% Imports

import logging
import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt

from PIL import ImageDraw, ImageFont
from pathlib import Path
from pprint import pprint
from dataclasses import asdict

from boltzmann.utils.logger import basic_config, time
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

verbose = False

# %% Params

# https://www.sciencedirect.com/science/article/pii/S0898122118306278

# dimensions [m]
aspect_ratio = 2
y_si = 2.0
x_si = y_si * aspect_ratio

# flow velocity [m/s]
# v0_si = 1.0
v0_si = 5.0

# vmax_vmag = 1.6 * v0_si
vmax_vmag = 1.3 * v0_si
vmax_curl = 30 * v0_si
vmax_conc = 1.025

nu_si = 1e-6
rho_si = 1000

# geometry
scale = 30
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

# allocate extra arrays for plotting to avoid reallocating each step
vmag_ = np.zeros_like(conc_)
curl_ = np.zeros_like(conc_)

# IMPORTANT: convert velocity to lattice units / timestep
vel_[:] = scales.to_lattice_units(vel_, **VELOCITY)

sim.finalize()

mem_mb = (sim.domain.size_bytes + sim.fluid.size_bytes + sim.tracer.size_bytes) / 1e6
logger.info(f"Memory usage: {mem_mb:,.2f} MB")


# %% Define simulation loop

# FONT = dict(
#     fontsize=14,
#     fontfamily="sans-serif",
#     fontstyle="italic",
#     fontweight="bold",
# )

fsz = domain.counts[1] // 15
fox = domain.counts[1] // 60
foy = 0

try:
    FONT = ImageFont.truetype("/home/dan/micromamba/envs/boltzmann/fonts/Inconsolata-Bold.ttf", fsz)
except OSError as e:
    raise OSError("Couldn't load True-Type Font file") from e


def png_writer(path: Path, data: np.ndarray, **kwargs) -> PngWriter:
    return PngWriter(path, domain, cells_, data, **kwargs)


class KelvinHelmholtz:
    def loop_for(self, steps: int):
        if np.any(~np.isfinite(sim.fluid.f)) or np.any(~np.isfinite(sim.tracer.g)):
            raise ValueError("Non-finite value in f.")

        sim.iterate(steps)

    def write_output(self, base: Path, step: int):
        global conc_, curl_, vmag_
        with time(logger, "calc curl", silent=not verbose):
            curl__ = curl_.reshape(sim.fluid.rho.shape)
            calc_curl_2d(sim.fluid.vel, sim.domain.cell_type, cnt, curl__)  # in LU
            curl_[:] = curl_ * ((dx / dt) / dx)  # in SI
            curl_[:] = np.tanh(curl_ / vmax_curl) * vmax_curl

        with time(logger, "calc vmag", silent=not verbose):
            vmag_[:] = np.sqrt(np.sum(vel_**2, -1))
            vmag_[:] = scales.to_physical_units(vmag_, **VELOCITY)
            vmag_[:] = np.tanh(vmag_ / vmax_vmag) * vmax_vmag

        with time(logger, "writing output", 1, silent=not verbose):
            with png_writer(
                base / f"vmag_{step:06d}.png",
                vmag_,
                cmap=plt.cm.inferno,  # type: ignore
                vmin=0,
                vmax=vmax_vmag,
            ) as img:
                draw = ImageDraw.Draw(img)
                draw.text((fox, foy), "Velocity", (255, 255, 255), font=FONT)

            with png_writer(
                base / f"conc_{step:06d}.png",
                conc_,
                cmap=plt.cm.cividis,  # type: ignore
                vmin=0,
                vmax=vmax_conc,
            ) as img:
                draw = ImageDraw.Draw(img)
                draw.text((fox, foy), "Tracer", (255, 255, 255), font=FONT)

            with png_writer(
                base / f"curl_{step:06d}.png",
                curl_,
                cmap=plt.cm.RdBu,  # type: ignore
                vmin=-vmax_curl,
                vmax=vmax_curl,
            ) as img:
                draw = ImageDraw.Draw(img)
                draw.text((fox, foy), "Vorticity", (0, 0, 0), font=FONT)

    def write_checkpoint(self, base: Path):
        with time(logger, "writing checkpoint", 1, silent=not verbose):
            save_fluid(base, sim.fluid)
            save_scalar(base, sim.tracer)

    def read_checkpoint(self, base: Path):
        load_fluid(base, sim.fluid)
        load_scalar(base, sim.tracer)
        sim.finalize()


# %% Main Loop

run_sim_cli(sim_meta, KelvinHelmholtz())

# render with
# FPS=30 ffmpeg -framerate $FPS -i out/conc_%06d.png -framerate $FPS -i out/curl_%06d.png                                      -c:v libx264 -crf 10 -filter_complex "[1]pad=iw:ih+2:0:2[v1];[0][v1]vstack=inputs=2" -y kh.mp4
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
logger = logging.getLogger("kh")

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
