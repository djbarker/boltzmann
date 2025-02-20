# %% Imports

import logging
from typing import Literal
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import distance_transform_edt
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

# %% Params

enable_tracer = True

# dimensions [m]
aspect_ratio = 2
y_si = 4.0
x_si = y_si * aspect_ratio

d_si = 1.5 * y_si / 10.0  # Cylinder diameter.

# flow velocity [m/s]
u_si = 2.0

# kinematic viscosity [m^2/s]
nu_si = 1e-4

Re = u_si * d_si / nu_si

# LBM parameterization
L = 4000

(tau, u) = calc_lbm_params(Re, L, 0.55, M_max=0.1)
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
out_dx_si = d_si / 15  # want to output when flow has moved this far
sim_dx_si = u_si * dt  # flow moves this far in dt (i.e. one iteration)
n = out_dx_si / sim_dx_si
n = int(n + 1e-8)
out_dt_si = dt * n

time_meta = TimeMeta.make(
    dt_step=dt,
    dt_output=out_dt_si,
    # dt_output=dt,
    output_count=800,
)


# geometry
lower = np.array([0 * x_si, -y_si / 2.0])
upper = np.array([1 * x_si, +y_si / 2.0])
domain = Domain.make(lower=lower, upper=upper, dx=dx)

sim_meta = SimulationMeta(domain, time_meta)

logger.info(f"Params: \n{pformat(asdict(sim_meta), sort_dicts=False, width=10)}")
dotted(logger, "Iters / output", int(out_dt_si / dt))

# color scheme constants
l_si = d_si / 10  # Characteristic length scale for color normalization.
vmax_vmag = 1.6 * u_si
vmax_curl = u_si / l_si
vmax_qcrit = 0.2 * (u_si / l_si) ** 2
vmax_conc = 1.00

# %% Initialize arrays

omega_ns = 1 / tau
omega_ad = 1 / 0.52

cnt = domain.counts.astype(np.int32)

with time(logger, "Creating simulation"):
    sim = Simulation(cnt, q=9, omega_ns=omega_ns)

if enable_tracer:
    with time(logger, "Adding tracers"):
        tracer_R = sim.add_tracer(q=5, omega_ad=omega_ad)
        tracer_G = sim.add_tracer(q=5, omega_ad=omega_ad)
        tracer_B = sim.add_tracer(q=5, omega_ad=omega_ad)

with time(logger, "Setting initial values"):
    # fixed velocity in- & out-flow
    cells_ = domain.unflatten(sim.domain.cell_type)
    cells_[+0, :] = CellType.FIXED.value  # left
    cells_[-1, :] = CellType.FIXED.value  # right

    # cylinder
    XX, YY = np.meshgrid(domain.x, domain.y, indexing="ij")
    cx = x_si / 5.0
    cy = 0.0
    RR = (XX - cx) ** 2 + (YY - cy) ** 2
    cells_[RR < (d_si / 2) ** 2] = CellType.WALL.value

    # Ramp the velocity from zero to flow velocity moving away from the walls.
    WW = 1 - 1 * (cells_ == CellType.WALL.value)
    DD = distance_transform_edt(WW).clip(0, int((y_si / 10) / domain.dx))
    DD = DD / np.max(DD)
    vel_ = domain.unflatten(sim.fluid.vel)
    vel_[:, :, 0] = -u_si * DD

    # perturb velocity
    vy_si = np.exp(-(((domain.x - cx) / 10) ** 2)) * u_si * 0.01
    vel_[:, :, 1] = vy_si[:, None]

    # Set up tracers.
    if enable_tracer:
        # TODO: It would be nice to be able to set FIXED the concentration only
        tracer_R_ = domain.unflatten(tracer_R.val)
        tracer_G_ = domain.unflatten(tracer_G.val)
        tracer_B_ = domain.unflatten(tracer_B.val)

        darken = 1.0
        n_streamers = 22
        dj = int(domain.counts[1] / 150.0)
        for i in range(n_streamers):
            c = plt.cm.hsv(i / (n_streamers))
            yj = int((domain.counts[1] / n_streamers) * (i + 0.5))
            xs = slice(0, dj)
            ys = slice(yj - dj // 2, yj + dj // 2)
            tracer_R_[xs, ys] = c[0] * darken
            tracer_G_[xs, ys] = c[1] * darken
            tracer_B_[xs, ys] = c[2] * darken
            cells_[xs, ys] = CellType.FIXED.value

        cells_[xs, :] = CellType.FIXED.value
        tracer_ = np.zeros([vel_.shape[0], vel_.shape[1], 3])

    # allocate extra arrays for plotting to avoid reallocating each step
    vmag_ = np.zeros_like(vel_[:, :, 0])
    curl_ = np.zeros_like(vel_[:, :, 0])
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


def smooth(x: np.ndarray, a: float = 10) -> np.ndarray:
    return x - np.log(1 + np.exp(a * (x - 1))) / a + np.log(1 + np.exp(a * (-x - 1))) / a


class VortexStreet(SimulationLoop):
    def loop_for(self, steps: int):
        if (
            np.any(~np.isfinite(sim.fluid.f))
            or np.any(~np.isfinite(tracer_R.g))
            or np.any(~np.isfinite(tracer_G.g))
            or np.any(~np.isfinite(tracer_B.g))
        ):
            raise ValueError("Non-finite value detected.")

        sim.iterate(steps)

    def write_output(self, base: Path, step: int):
        global conc_, curl_, qcrit_, vmag_, tracer_

        # First gather the various data to output.

        with time(logger, "calc curl", silent=not verbose):
            curl__ = curl_.reshape(sim.fluid.rho.shape)
            qcrit__ = qcrit_.reshape(sim.fluid.rho.shape)
            calc_curl_2d(sim.fluid.vel, sim.domain.cell_type, cnt, curl__, qcrit__)  # in LU
            curl_[:] = curl_ * ((dx / dt) / dx)  # in SI
            curl_[:] = np.tanh(curl_ / vmax_curl) * vmax_curl
            qcrit_[:] = qcrit_ * ((dx / dt) / dx) ** 2  # in SI
            qcrit_[:] = np.tanh(qcrit_ / vmax_qcrit) * vmax_qcrit

        with time(logger, "calc vmag", silent=not verbose):
            vmag_[:] = np.sqrt(np.sum(vel_**2, axis=-1))
            vmag_[:] = scales.to_physical_units(vmag_, **VELOCITY)
            vmag_[:] = np.tanh(vmag_ / vmax_vmag) * vmax_vmag

        tracer_[:, :, 0] = tracer_R_
        tracer_[:, :, 1] = tracer_G_
        tracer_[:, :, 2] = tracer_B_

        # Then write out the PNG files.

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
                base / f"curl_{step:06d}.png",
                curl_,
                "Vorticity",
                "light",
                cmap="RdBu",
                vmin=-vmax_curl,
                vmax=vmax_curl,
            )

            write_png(
                base / f"qcrit_{step:06d}.png",
                qcrit_,
                "Q-Criterion",
                "dark",
                cmap="viridis_r",
                vmin=-vmax_qcrit,
                vmax=vmax_qcrit,
            )

            if enable_tracer:
                write_png(
                    base / f"cols_{step:06d}.png",
                    tracer_,
                    "Tracer",
                    "dark",
                    vmax=0.4,
                )

    def write_checkpoint(self, base: Path):
        with time(logger, "writing checkpoint", 1, silent=not verbose):
            save_fluid(base, sim.fluid)
            if enable_tracer:
                save_scalar(base, tracer_R, "red")
                save_scalar(base, tracer_G, "grn")
                save_scalar(base, tracer_B, "blu")

    def read_checkpoint(self, base: Path):
        load_fluid(base, sim.fluid)
        if enable_tracer:
            load_scalar(base, tracer_R, "red")
            load_scalar(base, tracer_G, "grn")
            load_scalar(base, tracer_B, "blu")
        sim.finalize(False)


# %% Main Loop

run_sim_cli(sim_meta, VortexStreet(sim))

# render with
# export FPS=30; ffmpeg -framerate $FPS -i out/conc_%06d.png -framerate $FPS -i out/curl_%06d.png                                      -c:v libx264 -crf 10 -filter_complex "[1]pad=iw:ih+2:0:2[v1];[0][v1]vstack=inputs=2" -y kh.mp4
# export FPS=30; ffmpeg -framerate $FPS -i out/conc_%06d.png -framerate $FPS -i out/curl_%06d.png -framerate $FPS -i out/vmag_%06d.png -c:v libx264 -crf 10 -filter_complex "[1]pad=iw:ih+2:0:2[v1];[2]pad=iw:ih+2:0:2[v2];[0][v1][v2]vstack=inputs=3" -y kh4.mp4

# see: https://trac.ffmpeg.org/wiki/Encode/H.264
# export FPS=30; ffmpeg -framerate $FPS -i out/cols_%06d.png -framerate $FPS -i out/curl_%06d.png -framerate $FPS -i out/vmag_%06d.png -framerate $FPS -i out/qcrit_%06d.png -filter_complex "[0]pad=iw+5:ih+5:iw:ih[tl];[1]pad=iw:ih+5:0:ih[tr];[2]pad=iw+5:ih:iw:0[bl];[3]pad=iw:ih:0:0[br];[tl][tr][bl][br]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]" -map "[v]" -c:v libx264 -crf 23 -tune animation -y vs4.mp4
