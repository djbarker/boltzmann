"""
https://en.wikipedia.org/wiki/Rayleigh%E2%80%93Taylor_instability
"""

import logging
import numpy as np

from PIL import ImageFont, ImageDraw, Image
from pathlib import Path


from boltzmann.core import CellFlags, calc_lbm_params_si
from boltzmann.simulation import SimulationScript, IterInfo
from boltzmann.units import Domain, Scales
from boltzmann.utils.logger import basic_config, dotted
from boltzmann.utils.mpl import PngWriter, OrangeBlue_r, InkyBlueRed
from boltzmann.utils.postproc import calc_curl

np.random.seed(42)

basic_config()
logger = logging.getLogger(__name__)

# system dimensions
x_si = 1.0  # [m]
y_si = 2.0  # [m]

dx = y_si / 10000  # [m]

nu_si = 1e-4

domain = Domain.make(
    lower=[0 * x_si, -0.5 * y_si],
    upper=[1 * x_si, +0.5 * y_si],
    dx=dx,
)
dt = 0.0005
scales = Scales(dx=dx, dt=dt)
u_si = scales.velocity.to_physical_units(0.1)
(tau, u) = calc_lbm_params_si(dx, u_si, x_si, nu_si)

dotted(logger, "tau", tau)
dotted(logger, "u", u)

T0 = 0.0  # Reference temperature [C]
dT = 0.01  # Pertruabtion magnitude [C]
alpha = -0.1  # Thermal coupling constant [m/s^2/K]
alpha = scales.acceleration.to_lattice_units(alpha)

dotted(logger, "alpha", alpha)

# font
fsz = domain.counts[1] // 30
fox = domain.counts[1] // 60
foy = 0
f = ImageFont.truetype("NotoSans-BoldItalic", fsz)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)


def label(img: Image.Image, label: str):
    draw = ImageDraw.Draw(img)
    draw.text(
        (fox, foy),
        label,
        _WHITE,
        font=f,
        stroke_width=fsz // 15,
        stroke_fill=_BLACK,
    )


# img size
outx = 2000

dT_ = dT * 1.0
dt_out = 0.2
iter = IterInfo(int(dt_out / dt), 1000)
with (script := SimulationScript(domain.counts, 1 / 0.505, iter, "out")) as sim:

    @script.init
    def init():
        temp = sim.add_tracer("temp", 1 / 0.505)
        grav = np.array([0, -1], np.float32)
        sim.add_boussinesq_coupling(temp, alpha, T0, grav)

        # set initial temperature
        temp.val[:, :] = T0 - dT * np.tanh((domain.yy - 0.05 * np.cos(2 * np.pi * domain.xx)) * 100)

        # walls
        sim.cells.flags[:, +0] = CellFlags.WALL
        sim.cells.flags[:, -1] = CellFlags.WALL

    @script.out
    def output(output_dir: Path, iter: int):
        temp = sim.get_tracer("temp")

        with PngWriter(
            output_dir / f"temp_{iter:06d}.png",
            outx,
            sim.cells.flags,
            temp.val,
            # "RdBu_r",
            InkyBlueRed,
            vmin=T0 - dT_,
            vmax=T0 + dT_,
        ) as img:
            label(img, "Temperature")

        vmax = 0.05
        vmag = np.sqrt(np.sum(sim.fluid.vel**2, -1))
        vmag = np.tanh(vmag / vmax) * vmax
        with PngWriter(
            output_dir / f"vmag_{iter:06d}.png",
            outx,
            sim.cells.flags,
            vmag,
            "viridis",
            vmin=0,
            vmax=vmax,
        ) as img:
            label(img, "Velocity")

        cmax = 0.0012
        curl = calc_curl(sim.fluid.vel)
        curl = np.tanh(curl / cmax) * cmax
        with PngWriter(
            output_dir / f"curl_{iter:06d}.png",
            outx,
            sim.cells.flags,
            curl,
            OrangeBlue_r,
            vmin=-cmax,
            vmax=cmax,
        ) as img:
            label(img, "Vorticity")
