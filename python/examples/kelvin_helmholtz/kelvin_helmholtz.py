# %% Imports

import logging
from typing import Literal
from boltzmann.simulation import TimeMeta, parse_cli, run_sim
from boltzmann.units import Domain, Scales
import numpy as np

from PIL import ImageDraw, ImageFont
from pathlib import Path

from boltzmann.utils.logger import basic_config, time, dotted
from boltzmann.core import (
    CellFlags,
    calc_lbm_params_lu,
    calc_lbm_params_si,
)
from boltzmann.core import calc_curl_2d, Simulation  # type: ignore
from boltzmann.utils.mpl import PngWriter


basic_config()
logger = logging.getLogger(__name__)


# %% Params

# https://www.sciencedirect.com/science/article/pii/S0898122118306278

# dimensions [m]
aspect_ratio = 2
y_si = 2.0
x_si = y_si * aspect_ratio
d_si = y_si / 50  # transition region width [m]
u_si = 2.5  # flow velocity [m/s]
nu_si = 1e-5  # kinematic viscosity [m^2/s]

Re = int(u_si * d_si / nu_si + 1e-8)

# LBM parameters
L = 4000
D = L * (d_si / y_si)
dx = y_si / L

(tau, u) = calc_lbm_params_si(dx, u_si, y_si, nu_si, slack=10.0)
dt = (u / u_si) * dx

dotted(logger, "Reynolds number", Re)
dotted(logger, "tau", tau)
dotted(logger, "u", u)
dotted(logger, "L", L)
dotted(logger, "dx", dx)
dotted(logger, "dt", dt)

scales = Scales.make(dx=dx, dt=dt)

# dimensionless time does not depend on viscosity, purely on distances
out_dx_si = x_si / 250  # want to output when flow has moved this far
sim_dx_si = u_si * dt  # flow moves this far in dt (i.e. one iteration)
n = out_dx_si / sim_dx_si
n = int(n + 1e-8)
out_dt_si = dt * n

time_meta = TimeMeta.make(
    dt_step=dt,
    dt_output=out_dt_si,
    output_count=1200,
)

dotted(logger, "Steps/output", f"{out_dt_si / dt:,.0f}")
# raise ValueError

# geometry
lower = np.array([0 * x_si, -y_si / 2.0])
upper = np.array([1 * x_si, +y_si / 2.0])
domain = Domain.make(lower=lower, upper=upper, dx=dx)

# color scheme constants
# vmax_vmag = 1.6 * v0_si
vmax_vmag = 1.3 * u_si
vmax_curl = 0.6 * u_si / d_si
vmax_qcrit = 0.2 * (u_si / d_si) ** 2
vmax_conc = 1.00


# %% Initialize arrays

omega_ns = 1 / tau
omega_ad = 1 / min(0.51, tau)

cnt = domain.counts

args = parse_cli(base=f"out_re{Re}")

# Either load the simulatin or create it
if args.resume:
    with time(logger, "Loading simulation"):
        sim = Simulation.load_checkpoint(args.dev, str(args.base / "checkpoint.mpk"))
        tracer = sim.get_tracer(0)
else:
    with time(logger, "Creating simulation"):
        sim = Simulation(args.dev, cnt, omega_ns=omega_ns)
        tracer = sim.add_tracer(omega_ad=omega_ad)

    with time(logger, "Setting initial values"):
        # fixed velocity in- & out-flow
        sim.cells.flags[:, +0] = CellFlags.FIXED_FLUID | CellFlags.FIXED_SCALAR_VALUE  # bottom
        sim.cells.flags[:, -1] = CellFlags.FIXED_FLUID | CellFlags.FIXED_SCALAR_VALUE  # top

        # set velocity
        sim.fluid.vel[:, :, 0] = u_si * np.tanh(domain.y / d_si)[None, :]

        # perturb velocity
        vy_si = np.zeros_like(domain.x)
        vy_si += u_si * np.sin(2 * np.pi * (domain.x / x_si) * 1) * 0.000001
        # vy_si += u_si * np.sin(2 * np.pi * (domain.x / x_si) * 2) * 0.00001
        # vy_si += u_si * np.sin(2 * np.pi * (domain.x / x_si) * 3) * 0.0001
        # vy_si += u_si * np.sin(2 * np.pi * (domain.x / x_si) * 4) * 0.001
        vy_si += u_si * np.sin(2 * np.pi * (domain.x / x_si) * 5) * 0.001
        sim.fluid.vel[:, 1:-1, 1] = vy_si[:, None]

        # set concentration
        tracer.val[:, : domain.counts[1] // 2] = 1.0  # lower half

        # IMPORTANT: convert velocity to lattice units / timestep
        sim.fluid.vel[:] = scales.velocity.to_lattice_units(sim.fluid.vel)


# %% Define simulation loop

# allocate extra arrays for plotting to avoid reallocating each step
vmag_ = np.zeros_like(sim.fluid.rho)
curl_ = np.zeros_like(sim.fluid.rho)
qcrit_ = np.zeros_like(sim.fluid.rho)


def write_png(
    path: Path,
    data: np.ndarray,
    label: str,
    background: Literal["dark", "light"],
    **kwargs,
):
    outx = 2000

    # colours
    tcol = (255, 255, 255) if background == "dark" else (0, 0, 0)
    bcol = (0, 0, 0) if background == "dark" else (255, 255, 255)

    # font
    fsz = domain.counts[1] // 15
    fox = domain.counts[1] // 60
    foy = 0
    f = ImageFont.truetype("NotoSans-BoldItalic", fsz)

    with PngWriter(path, outx, sim.cells.flags, data, **kwargs) as img:
        draw = ImageDraw.Draw(img)
        draw.text((fox, foy), label, tcol, font=f, stroke_width=fsz // 15, stroke_fill=bcol)


def write_output(base: Path, iter: int):
    global curl_, vmag_, qcrit_
    with time(logger, "calc curl"):
        calc_curl_2d(sim.fluid.vel, sim.cells.flags, cnt, curl_, qcrit_)  # in LU
        curl_[:] = curl_ * ((dx / dt) / dx)  # in SI
        curl_[:] = np.tanh(curl_ / vmax_curl) * vmax_curl

    with time(logger, "calc vmag"):
        vmag_[:] = np.sqrt(np.sum(sim.fluid.vel**2, -1))
        vmag_[:] = scales.velocity.to_physical_units(vmag_)
        vmag_[:] = ((np.tanh(2 * (vmag_ / vmax_vmag) - 1) + 1) / 2) * vmax_vmag

    with time(logger, "writing output"):
        write_png(
            base / f"vmag_{iter:06d}.png",
            vmag_,
            "Velocity",
            "dark",
            cmap="inferno",
            vmax=vmax_vmag,
        )

        write_png(
            base / f"conc_{iter:06d}.png",
            tracer.val,
            "Tracer",
            "dark",
            cmap="cividis",
            vmin=0,
            vmax=vmax_conc,
        )

        write_png(
            base / f"curl_{iter:06d}.png",
            curl_,
            "Vorticity",
            "light",
            cmap="RdBu_r",
            # cmap=mycmap,
            vmin=-vmax_curl,
            vmax=vmax_curl,
        )


# %% Main Loop

run_sim(args.base, time_meta, sim, write_output, write_checkpoints=not args.no_checkpoint)
