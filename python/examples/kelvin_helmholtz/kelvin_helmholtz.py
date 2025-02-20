# %% Imports

import logging
from typing import Literal
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
from PIL import ImageDraw, ImageFont
from pathlib import Path
from pprint import pformat
from dataclasses import asdict

from boltzmann.utils.logger import basic_config, time, dotted
from boltzmann.core import (
    VELOCITY,
    Domain,
    Scales,
    SimulationMeta,
    CellType,
    TimeMeta,
    calc_lbm_params,
)
from boltzmann.utils.mpl import PngWriter
from boltzmann.simulation import (
    SimulationLoop,
    run_sim_cli,
    load_fluid,
    load_scalar,
    save_fluid,
    save_scalar,
)
from boltzmann_rs import (
    calc_curl_2d,
    Simulation,
)


basic_config()
logger = logging.getLogger(__name__)
logger.info("Starting")

verbose = False

# Make custom colormap for vorticity.

colors = plt.cm.Spectral(np.linspace(0, 1, 11))
colors[6, :] = 1.0  # Make center white not cream.
nodes = np.linspace(0, 1, len(colors))
mycmap = LinearSegmentedColormap.from_list("mycmap", list(zip(nodes, colors)))

# %% Params

# https://www.sciencedirect.com/science/article/pii/S0898122118306278

# dimensions [m]
aspect_ratio = 2
y_si = 2.0
x_si = y_si * aspect_ratio
d_si = y_si / 25  # transition region width [m]
u_si = 2.5  # flow velocity [m/s]
nu_si = 1e-5  # kinematic viscosity [m^2/s]

Re = u_si * d_si / nu_si

# LBM parameters
L = 4000

(tau, u) = calc_lbm_params(Re, L, M_max=0.1)
dx = y_si / L
dt = (u / u_si) * dx

dotted(logger, "Reynolds number", Re)
dotted(logger, "tau", tau)
dotted(logger, "u", u)
dotted(logger, "L", L)
dotted(logger, "dx", dx)
dotted(logger, "dt", dt)

scales = Scales.make(dx=dx, dt=dt)

# dimensionless time does not depend on viscosity, purely on distances
fps = 30
out_dx_si = x_si / (20 * fps)  # want to output when flow has moved this far
sim_dx_si = u_si * dt  # flow moves this far in dt (i.e. one iteration)
n = out_dx_si / sim_dx_si
n = int(n + 1e-8)
out_dt_si = dt * n

time_meta = TimeMeta.make(
    dt_step=dt,
    dt_output=out_dt_si,
    output_count=1200,
)


# geometry
lower = np.array([0 * x_si, -y_si / 2.0])
upper = np.array([1 * x_si, +y_si / 2.0])
domain = Domain.make(lower=lower, upper=upper, dx=dx)

sim_meta = SimulationMeta(domain, time_meta)

logger.info(f"\n{pformat(asdict(sim_meta), sort_dicts=False, width=10)}")
dotted(logger, "Iters / output", int(out_dt_si / dt))

# color scheme constants
# vmax_vmag = 1.6 * v0_si
vmax_vmag = 1.3 * u_si
vmax_curl = u_si / d_si
vmax_qcrit = 0.2 * (u_si / d_si) ** 2
vmax_conc = 1.00


# %% Initialize arrays

omega_ns = 1 / tau
omega_ad = 1 / 0.52

cnt = domain.counts.astype(np.int32)

with time(logger, "Creating simulation"):
    sim = Simulation(cnt, q=9, omega_ns=omega_ns)
    tracer = sim.add_tracer(q=5, omega_ad=omega_ad)

with time(logger, "Setting initial values"):
    # fixed velocity in- & out-flow
    cells_ = domain.unflatten(sim.domain.cell_type)
    cells_[:, +0] = CellType.FIXED.value  # bottom
    cells_[:, -1] = CellType.FIXED.value  # top

    # set velocity
    vel_ = domain.unflatten(sim.fluid.vel)
    vel_[:, :, 0] = u_si * np.tanh(domain.y / d_si)[None, :]

    # perturb velocity
    vy_si = np.zeros_like(domain.x)
    vy_si += u_si * np.sin(2 * np.pi * (domain.x / x_si) * 1) * 0.001
    vy_si += u_si * np.sin(2 * np.pi * (domain.x / x_si) * 2) * 0.0001
    vy_si += u_si * np.sin(2 * np.pi * (domain.x / x_si) * 3) * 0.00001
    vy_si += u_si * np.sin(2 * np.pi * (domain.x / x_si) * 4) * 0.000001
    vy_si += u_si * np.sin(2 * np.pi * (domain.x / x_si) * 5) * 0.0000001
    vel_[:, 1:-1, 1] = vy_si[:, None]

    # set concentration
    conc_ = domain.unflatten(tracer.val)
    conc_[:, : domain.counts[1] // 2] = 1.0  # lower half
    conc_[: domain.counts[0] // 2, domain.counts[1] // 2 :] = 0.5  # upper-left half

    # allocate extra arrays for plotting to avoid reallocating each step
    vmag_ = np.zeros_like(conc_)
    curl_ = np.zeros_like(conc_)
    qcrit_ = np.zeros_like(vel_[:, :, 0])

    # IMPORTANT: convert velocity to lattice units / timestep
    vel_[:] = scales.to_lattice_units(vel_, **VELOCITY)

with time(logger, "finalizing"):
    sim.finalize(True)


# %% Define simulation loop

fsz = domain.counts[1] // 15
fox = domain.counts[1] // 60
foy = 0

try:
    FONT = ImageFont.truetype("/home/dan/micromamba/envs/boltzmann/fonts/Inconsolata-Bold.ttf", fsz)
except OSError as e:
    raise OSError("Couldn't load True-Type Font file") from e


def write_png(
    path: Path, data: np.ndarray, label: str, background: Literal["dark", "light"], **kwargs
):
    outx = 2000

    col = (255, 255, 255) if background == "dark" else (0, 0, 0)

    with PngWriter(path, outx, domain, cells_, data, **kwargs) as img:
        draw = ImageDraw.Draw(img)
        draw.text((fox, foy), label, col, font=FONT)


class KelvinHelmholtz(SimulationLoop):
    def loop_for(self, steps: int):
        if np.any(~np.isfinite(sim.fluid.f)) or np.any(~np.isfinite(tracer.g)):
            raise ValueError("Non-finite value detected.")

        self.sim.iterate(steps)

    def write_output(self, base: Path, step: int):
        global conc_, curl_, vmag_
        with time(logger, "calc curl", silent=not verbose):
            curl__ = curl_.reshape(sim.fluid.rho.shape)
            qcrit__ = qcrit_.reshape(sim.fluid.rho.shape)
            calc_curl_2d(sim.fluid.vel, sim.domain.cell_type, cnt, curl__, qcrit__)  # in LU
            curl_[:] = curl_ * ((dx / dt) / dx)  # in SI
            curl_[:] = np.tanh(curl_ / vmax_curl) * vmax_curl
            qcrit_[:] = qcrit_ * ((dx / dt) / dx) ** 2  # in SI
            qcrit_[:] = np.tanh(qcrit_ / vmax_qcrit) * vmax_qcrit

        with time(logger, "calc vmag", silent=not verbose):
            vmag_[:] = np.sqrt(np.sum(vel_**2, -1))
            vmag_[:] = scales.to_physical_units(vmag_, **VELOCITY)
            vmag_[:] = ((np.tanh(2 * (vmag_ / vmax_vmag) - 1) + 1) / 2) * vmax_vmag

        with time(logger, "writing output", 1, silent=not verbose):
            write_png(
                base / f"vmag_{step:06d}.png",
                vmag_,
                "Velocity",
                "dark",
                cmap="inferno",
                vmax=vmax_vmag,
            )

            write_png(
                base / f"conc_{step:06d}.png",
                conc_,
                "Tracer",
                "dark",
                cmap="cividis",
                vmin=0,
                vmax=vmax_conc,
            )

            write_png(
                base / f"curl_{step:06d}.png",
                curl_,
                "Vorticity",
                "light",
                cmap="RdBu",
                # cmap=mycmap,
                vmin=-vmax_curl,
                vmax=vmax_curl,
            )

    def write_checkpoint(self, base: Path):
        with time(logger, "writing checkpoint", 1, silent=not verbose):
            save_fluid(base, sim.fluid)
            save_scalar(base, tracer, "tracer")

    def read_checkpoint(self, base: Path):
        load_fluid(base, sim.fluid)
        load_scalar(base, tracer, "tracer")
        sim.finalize(False)


# %% Main Loop

run_sim_cli(sim_meta, KelvinHelmholtz(sim))

# render with
# export FPS=30; ffmpeg -framerate $FPS -i out/conc_%06d.png -framerate $FPS -i out/curl_%06d.png                                      -c:v libx264 -crf 10 -filter_complex "[1]pad=iw:ih+2:0:2[v1];[0][v1]vstack=inputs=2" -y kh.mp4
# export FPS=30; ffmpeg -framerate $FPS -i out/conc_%06d.png -framerate $FPS -i out/curl_%06d.png -framerate $FPS -i out/vmag_%06d.png -c:v libx264 -crf 10 -filter_complex "[1]pad=iw:ih+2:0:2[v1];[2]pad=iw:ih+2:0:2[v2];[0][v1][v2]vstack=inputs=3" -y kh4.mp4

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
sim.finalize(True)

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
