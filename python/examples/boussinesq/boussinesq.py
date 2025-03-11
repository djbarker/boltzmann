import numpy as np

from PIL import ImageFont, ImageDraw
from pathlib import Path


from boltzmann.core import CellFlags
from boltzmann.simulation import SimulationScript, IterInfo
from boltzmann.utils.logger import basic_config
from boltzmann.utils.mpl import PngWriter, OrangeBlue_r

basic_config()

n = 2000  # Cell count.

T0 = 0.0  # Reference temperature
dT = 0.01  # Pertruabtion magnitude
alpha = -0.001  # Thermal coupling constant


# font
fsz = n // 25
fox = n // 60
foy = 0
f = ImageFont.truetype("NotoSans-BoldItalic", fsz)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)

dT_ = dT * 0.25
iter = IterInfo(500, 1500)
with (script := SimulationScript([n, n], 1 / 0.51, iter, "out")) as sim:

    @script.init
    def init():
        temp = sim.add_tracer("temp", 1 / 0.52)
        grav = np.array([0, -1], np.float32)
        sim.add_boussinesq_coupling(temp, alpha, T0, grav)

        # set value to 1 everywhere initially
        temp.val[:] = T0

        # heat source
        n_ = n // 4
        w = 4
        temp.val[n_ - w : n_ + w, n_] = T0 + dT
        sim.cells.flags[n_ - w : n_ + w, n_] = CellFlags.FIXED_SCALAR_VALUE

        # cold source
        n_ = 3 * n // 4
        temp.val[n_ - w : n_ + w, n_] = T0 - dT
        sim.cells.flags[n_ - w : n_ + w, n_] = CellFlags.FIXED_SCALAR_VALUE

        # walls at top & bottom
        sim.cells.flags[:, +0] = CellFlags.WALL | CellFlags.FIXED_SCALAR_VALUE
        sim.cells.flags[:, -1] = CellFlags.WALL | CellFlags.FIXED_SCALAR_VALUE

    @script.out
    def output(output_dir: Path, iter: int):
        temp = sim.get_tracer("temp")

        with PngWriter(
            output_dir / f"temp_{iter:06d}.png",
            1000,
            sim.cells.flags,
            temp.val,
            "RdBu_r",
            vmin=T0 - dT_,
            vmax=T0 + dT_,
        ) as img:
            draw = ImageDraw.Draw(img)
            draw.text(
                (fox, foy),
                "Temperature",
                _WHITE,
                font=f,
                stroke_width=fsz // 15,
                stroke_fill=_BLACK,
            )

        vmag = np.sqrt(np.sum(sim.fluid.vel**2, -1))
        with PngWriter(
            output_dir / f"vmag_{iter:06d}.png",
            1000,
            sim.cells.flags,
            vmag,
            "viridis",
            vmin=0,
            vmax=0.03,
        ) as img:
            draw = ImageDraw.Draw(img)
            draw.text(
                (fox, foy),
                "Velocity",
                _WHITE,
                font=f,
                stroke_width=fsz // 15,
                stroke_fill=_BLACK,
            )
