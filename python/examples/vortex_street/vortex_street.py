# %% Imports

from concurrent.futures import ThreadPoolExecutor
import logging
from typing import Literal
import numpy as np

from scipy.ndimage import distance_transform_edt
from PIL import ImageDraw, ImageFont
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

from boltzmann.utils.logger import basic_config, time, dotted
from boltzmann.core import VELOCITY, Domain, Scales, CellType, TimeMeta, calc_lbm_params
from boltzmann.utils.mpl import PngWriter
from boltzmann.simulation import parse_cli, run_sim
from boltzmann_rs import calc_curl_2d, Simulation  # type: ignore

basic_config()
logger = logging.getLogger(__name__)


def make_cmap(name: str, colors: list) -> LinearSegmentedColormap:
    nodes = np.linspace(0, 1, len(colors))
    return LinearSegmentedColormap.from_list(name, list(zip(nodes, colors)))


colors = [
    "#ffe359",
    "#ff8000",
    "#734c26",
    "#1c1920",
    "#1c1920",
    "#1c1920",
    "#265773",
    "#0f82b8",
    "#8fceff",
]

nodes = [0.0, 0.16666667, 0.33333333, 0.49, 0.5, 0.51, 0.66666667, 0.83333333, 1.0]

OrangeBlue = LinearSegmentedColormap.from_list("OrangeBlue", list(zip(nodes, colors)))

# %% Params

# dimensions [m]
aspect_ratio = 2
y_si = 4.0
x_si = y_si * aspect_ratio

d_si = 1.5 * y_si / 10.0  # Cylinder diameter.

Re = 10000  # Reynolds number [1]
nu_si = 1e-4  # kinematic viscosity [m^2/s]
u_si = Re * nu_si / d_si  # flow velocity [m/s]

# LBM parameterization
L = 3000

D = L * (d_si / y_si)
(tau, u) = calc_lbm_params(Re, D, M_max=0.1)
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

# color scheme constants
l_si = d_si / 10  # Characteristic length scale for color normalization.
vmax_vmag = 1.6 * u_si
vmax_curl = u_si / l_si
vmax_qcrit = 0.2 * (u_si / l_si) ** 2
vmax_conc = 1.00

# %% Initialize arrays

omega_ns = 1 / tau
omega_ad = 1 / min(0.52, tau)

cnt = domain.counts.astype(np.int32)

args = parse_cli(base=f"out_re{Re}")

# Either load the simulation or create it.
if args.resume:
    with time(logger, "Loading simulation"):
        sim = Simulation.load_checkpoint(args.dev, str(args.base / "checkpoint.mpk"))
        tracer_R = sim.get_tracer(0)
        tracer_G = sim.get_tracer(1)
        # tracer_B = sim.get_tracer(2)
else:
    with time(logger, "Creating simulation"):
        sim = Simulation(args.dev, cnt, q=9, omega_ns=omega_ns)
        tracer_R = sim.add_tracer(q=5, omega_ad=omega_ad)
        tracer_G = sim.add_tracer(q=5, omega_ad=omega_ad)
        # tracer_B = sim.add_tracer(q=5, omega_ad=omega_ad)


# allocate extra arrays for plotting to avoid reallocating each step
vmag_ = np.zeros_like(sim.fluid.rho[:, :])
curl_ = np.zeros_like(sim.fluid.rho[:, :])
qcrit_ = np.zeros_like(sim.fluid.rho[:, :])
tracer_ = np.zeros([sim.fluid.vel.shape[0], sim.fluid.vel.shape[1], 3])

if not args.resume:
    with time(logger, "Setting initial values"):
        # fixed velocity in- & out-flow
        sim.cells.cell_type[+0, :] = CellType.FIXED_FLUID.value  # left
        sim.cells.cell_type[-1, :] = CellType.FIXED_FLUID.value  # right
        sim.cells.cell_type[+0, :] |= CellType.FIXED_SCALAR_VALUE.value  # left
        sim.cells.cell_type[-1, :] |= CellType.FIXED_SCALAR_VALUE.value  # right

        # cylinder
        XX, YY = np.meshgrid(domain.x, domain.y, indexing="ij")
        cx = x_si / 6.0
        cy = 0.0
        RR = (XX - cx) ** 2 + (YY - cy) ** 2
        sim.cells.cell_type[RR < (d_si / 2) ** 2] = CellType.WALL.value

        # Ramp the velocity from zero to flow velocity moving away from the walls.
        WW = 1 - 1 * (sim.cells.cell_type == CellType.WALL.value)
        DD = distance_transform_edt(WW).clip(0, int((y_si / 10) / domain.dx))  # type: ignore
        DD = DD / np.max(DD)
        sim.fluid.vel[:, :, 0] = u_si * DD

        # perturb velocity
        vy_si = -np.exp(-(((domain.x - cx) / 10) ** 2)) * u_si * 0.01
        sim.fluid.vel[:, :, 1] = vy_si[:, None]

        # Set up tracers.

        # darken = 1.0
        # n_streamers = 22
        # dj = int(domain.counts[1] / 150.0)
        # for i in range(n_streamers):
        #     c = plt.cm.hsv(i / (n_streamers))
        #     yj = int((domain.counts[1] / n_streamers) * (i + 0.5))
        #     xs = slice(0, dj)
        #     ys = slice(yj - dj // 2, yj + dj // 2)
        #     tracer_R_[xs, ys] = c[0] * darken
        #     tracer_G_[xs, ys] = c[1] * darken
        #     tracer_B_[xs, ys] = c[2] * darken
        #     cells_[xs, ys] = CellType.FIXED.value
        #
        # cells_[xs, :] = CellType.FIXED.value

        mask = distance_transform_edt(WW)
        mask = (0 < mask) & (mask <= 2)  # type: ignore
        # mask = (mask == 1) & (mask == 2)
        tracer_R.val[:, : domain.counts[1] // 2][mask[:, : domain.counts[1] // 2]] = 1.0
        tracer_G.val[:, domain.counts[1] // 2 :][mask[:, domain.counts[1] // 2 :]] = 1.0
        sim.cells.cell_type[mask] |= CellType.FIXED_SCALAR_VALUE.value

        sx = slice(2 * domain.counts[0] // 5, 2 * domain.counts[0] // 5 + 10)
        sy = slice(domain.counts[1] // 2 - 5, domain.counts[1] // 2 + 5)
        tracer_R.val[sx, sy] = 1.0
        sim.cells.cell_type[sx, sy] |= CellType.FIXED_SCALAR_VALUE.value

        sx = slice(2 * domain.counts[0] // 3, 2 * domain.counts[0] // 3 + 10)
        sy = slice(domain.counts[1] // 2 - 5, domain.counts[1] // 2 + 5)
        tracer_G.val[sx, sy] = 1.0
        sim.cells.cell_type[sx, sy] |= CellType.FIXED_SCALAR_VALUE.value

        # IMPORTANT: convert velocity to lattice units / timestep
        sim.fluid.vel[:] = scales.to_lattice_units(sim.fluid.vel, **VELOCITY)


# %% Define simulation output

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

    tcol = (255, 255, 255) if background == "dark" else (0, 0, 0)
    bcol = (0, 0, 0) if background == "dark" else (255, 255, 255)

    with PngWriter(path, outx, sim.cells.cell_type, data, **kwargs) as img:
        draw = ImageDraw.Draw(img)
        draw.text((fox, foy), label, tcol, font=FONT, stroke_width=fsz // 15, stroke_fill=bcol)


def smooth(x: np.ndarray, a: float = 10) -> np.ndarray:
    return x - np.log(1 + np.exp(a * (x - 1))) / a + np.log(1 + np.exp(a * (-x - 1))) / a


# output is slow so we parallelize it
executor = ThreadPoolExecutor(max_workers=10)


def write_output(base: Path, iter: int):
    global conc_, curl_, qcrit_, vmag_, tracer_

    # First gather the various data to output.

    with time(logger, "calc curl"):
        curl__ = curl_.reshape(sim.fluid.rho.shape)
        qcrit__ = qcrit_.reshape(sim.fluid.rho.shape)
        calc_curl_2d(sim.fluid.vel, sim.cells.cell_type, cnt, curl__, qcrit__)  # in LU
        curl_[:] = curl_ * ((dx / dt) / dx)  # in SI
        curl_[:] = np.tanh(curl_ / vmax_curl) * vmax_curl
        qcrit_[:] = qcrit_ * ((dx / dt) / dx) ** 2  # in SI
        qcrit_[:] = np.tanh(qcrit_ / vmax_qcrit) * vmax_qcrit

    with time(logger, "calc vmag"):
        vmag_[:] = np.sqrt(np.sum(sim.fluid.vel**2, axis=-1))
        vmag_[:] = scales.to_physical_units(vmag_, **VELOCITY)
        vmag_[:] = np.tanh(vmag_ / vmax_vmag) * vmax_vmag

    tracer_[:, :, 0] = tracer_R.val
    tracer_[:, :, 1] = tracer_G.val
    # tracer_[:, :, 2] = tracer_B_

    # Then write out the PNG files.

    with time(logger, "writing output", 1):
        futs = []
        futs.append(
            executor.submit(
                lambda: write_png(
                    base / f"vmag_{iter:06d}.png",
                    vmag_,
                    "Velocity",
                    "dark",
                    cmap="inferno",
                    vmax=vmax_vmag,
                )
            )
        )

        futs.append(
            executor.submit(
                lambda: write_png(
                    base / f"curl_{iter:06d}.png",
                    curl_,
                    "Vorticity",
                    # "light",
                    # cmap="RdBu",
                    "dark",
                    cmap=OrangeBlue,
                    vmin=-vmax_curl,
                    vmax=vmax_curl,
                )
            )
        )

        futs.append(
            executor.submit(
                lambda: write_png(
                    base / f"cols_{iter:06d}.png",
                    tracer_,
                    "Tracer",
                    "dark",
                    vmax=0.4,
                )
            )
        )

        # join
        _ = [f.result() for f in futs]

        # write_png(
        #     base / f"qcrit_{iter:06d}.png",
        #     qcrit_,
        #     "Q-Criterion",
        #     "dark",
        #     cmap=OrangeBlue,
        #     vmin=-vmax_qcrit,
        #     vmax=vmax_qcrit,
        # )


# %% Main Loop

run_sim(args.base, time_meta, sim, write_output, write_checkpoints=not args.no_checkpoint)


# render with
# export FPS=30; ffmpeg -framerate $FPS -i out/conc_%06d.png -framerate $FPS -i out/curl_%06d.png                                      -c:v libx264 -crf 10 -filter_complex "[1]pad=iw:ih+2:0:2[v1];[0][v1]vstack=inputs=2" -y kh.mp4
# export FPS=30; ffmpeg -framerate $FPS -i out/conc_%06d.png -framerate $FPS -i out/curl_%06d.png -framerate $FPS -i out/vmag_%06d.png -c:v libx264 -crf 10 -filter_complex "[1]pad=iw:ih+2:0:2[v1];[2]pad=iw:ih+2:0:2[v2];[0][v1][v2]vstack=inputs=3" -y kh4.mp4

# see: https://trac.ffmpeg.org/wiki/Encode/H.264
# export FPS=30; ffmpeg -framerate $FPS -i out/cols_%06d.png -framerate $FPS -i out/curl_%06d.png -framerate $FPS -i out/vmag_%06d.png -framerate $FPS -i out/qcrit_%06d.png -filter_complex "[0]pad=iw+5:ih+5:iw:ih[tl];[1]pad=iw:ih+5:0:ih[tr];[2]pad=iw+5:ih:iw:0[bl];[3]pad=iw:ih:0:0[br];[tl][tr][bl][br]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]" -map "[v]" -c:v libx264 -crf 23 -tune animation -y vs4.mp4
