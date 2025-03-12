import numpy as np

from PIL import ImageFont, ImageDraw, Image
from pathlib import Path


from boltzmann.core import CellFlags
from boltzmann.simulation import SimulationScript, IterInfo
from boltzmann.utils.logger import basic_config
from boltzmann.utils.mpl import PngWriter, OrangeBlue_r
from boltzmann.utils.postproc import calc_curl

np.random.seed(42)

basic_config()

nx = 5000  # Cell counts.
ny = 3000

T0 = 0.0  # Reference temperature
dT = 0.01  # Pertruabtion magnitude
alpha = -0.001  # Thermal coupling constant

# font
fsz = ny // 30
fox = ny // 60
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

dT_ = dT * 0.5
iter = IterInfo(250, 3000)
with (script := SimulationScript([nx, ny], 1 / 0.505, iter, "out")) as sim:

    @script.init
    def init():
        temp = sim.add_tracer("temp", 1 / 0.505)
        grav = np.array([0, -1], np.float32)
        sim.add_boussinesq_coupling(temp, alpha, T0, grav)

        # perturb density
        sim.fluid.rho[:] += np.random.uniform(-0.01, 0.01, sim.fluid.rho.shape)

        # set value to 1 everywhere initially
        temp.val[:] = T0

        # heat & cold source
        temp.val[:, :+3] = T0 + dT
        temp.val[:, -3:] = T0 - dT
        sim.cells.flags[:, :+3] = CellFlags.FIXED_SCALAR_VALUE
        sim.cells.flags[:, -3:] = CellFlags.FIXED_SCALAR_VALUE

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
            OrangeBlue_r,
            vmin=T0 - dT_,
            vmax=T0 + dT_,
        ) as img:
            label(img, "Temperature")

        vmax = 0.08
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

        # cmax = 0.08
        # curl = calc_curl(sim.fluid.vel)
        # curl = np.tanh(curl / cmax) * cmax
        # with PngWriter(
        #     output_dir / f"curl_{iter:06d}.png",
        #     outx,
        #     sim.cells.flags,
        #     curl,
        #     "RdBu",
        #     vmin=-cmax,
        #     vmax=cmax,
        # ) as img:
        #     label(img, "Vorticity")
