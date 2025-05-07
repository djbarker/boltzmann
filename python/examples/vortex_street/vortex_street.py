# %% Imports

import logging
import numpy as np

from pathlib import Path
from PIL import ImageDraw, ImageFont, Image
from scipy.ndimage import distance_transform_edt, maximum_filter, minimum_filter, binary_dilation
from typing import Literal

from boltzmann.core import CellFlags, calc_lbm_params_si
from boltzmann.core import Simulation  # type: ignore
from boltzmann.simulation import IterInfo, parse_args, run_sim
from boltzmann.units import Domain, Scales
from boltzmann.utils.logger import basic_config, timed, dotted
from boltzmann.utils.mpl import PngWriter
from boltzmann.utils.postproc import calc_curl, calc_vmag, calc_stream_func

basic_config()
logger = logging.getLogger(__name__)

# %% Params

# dimensions [m]
aspect_ratio = 3
y_si = 4.0
x_si = y_si * aspect_ratio

d_si = 0.15 * y_si  # Cylinder diameter.
Re = 10  # Reynolds number [1]
nu_si = 1e-4  # kinematic viscosity [m^2/s]
L = 1000  # Cell count in the y direction [1]

# implied params
u_si = Re * nu_si / d_si  # => flow velocity [m/s]
dx = y_si / L  # => spatial resolution [m]

# calc LBM params
(tau, u) = calc_lbm_params_si(dx, u_si, d_si, nu_si, M_max=0.1)
dt = (u / u_si) * dx

dotted(logger, "Reynolds number", Re)
dotted(logger, "tau", tau)
dotted(logger, "u", u)
dotted(logger, "L", L)
dotted(logger, "dx", dx)
dotted(logger, "dt", dt)

scales = Scales(dx=dx, dt=dt)

# dimensionless time does not depend on viscosity, purely on distances
out_dx_si = d_si / 20  # want to output when flow has moved this far
sim_dx_si = u_si * dt  # flow moves this far in dt (i.e. one iteration)
n = out_dx_si / sim_dx_si
n = int(n + 1e-8)
out_dt_si = dt * n

meta = IterInfo.make(
    dt=dt,
    dt_output=out_dt_si,
    count=800,
)

# geometry
lower = np.array([0 * x_si, -y_si / 2.0])
upper = np.array([1 * x_si, +y_si / 2.0])
domain = Domain.make(lower=lower, upper=upper, dx=dx)

# cylinder center
cx = d_si * 2.5
cy = 0.0

# color scheme constants
l_si = d_si / 10  # Characteristic length scale for color normalization.
vmax_vmag = 1.8 * u_si
vmax_curl = 0.8 * u_si / l_si
vmax_qcrit = 0.2 * (u_si / l_si) ** 2
vmax_conc = 1.00

# %% Initialize arrays


def pve(x: np.ndarray) -> np.ndarray:
    """
    Clip an array below by zero.
    """
    return np.where(x < 0, 0, x)


velocity_init_mode: Literal["uniform", "tapered", "stream"] = "stream"

omega_ns = 1 / tau
omega_ad = 1 / min(0.52, tau)

args = parse_args(out_dir=f"out_re{Re}")

# Either load the simulation or create it.
if args.resume:
    sim = Simulation.load_checkpoint(args.device, str(args.out_dir / "checkpoint.mpk"))
else:
    sim = Simulation(args.device, domain.counts, omega_ns=omega_ns)

    # Fixed velocity & tracer in- & out-flow.
    sim.cells.flags[+0, :] = CellFlags.FIXED_FLUID  # left
    sim.cells.flags[-1, :] = CellFlags.FIXED_FLUID  # right
    sim.cells.flags[+0, :] |= CellFlags.FIXED_SCALAR_VALUE  # left
    sim.cells.flags[-1, :] |= CellFlags.FIXED_SCALAR_VALUE  # right

    # Cylinder geometry.
    XX, YY = np.meshgrid(domain.x, domain.y, indexing="ij")
    RR = np.sqrt((XX - cx) ** 2 + (YY - cy) ** 2)
    sim.cells.flags[RR < (d_si / 2)] = CellFlags.WALL

    # Use EDT not SDF because we may have more complex shapes than a cylinder.
    WW = 1 - 1 * (sim.cells.flags == CellFlags.WALL)
    DD = distance_transform_edt(WW) * domain.dx  # type: ignore

    match velocity_init_mode:
        case "uniform":
            sim.fluid.vel[..., 0] = u_si
        case "tapered":
            D = 1.0 * d_si
            sim.fluid.vel[:, :, 0] = u_si * (np.where(DD > D, D, DD) / (D))
        case "stream":
            stream = domain.yy / domain.dx
            stream *= 1 - np.exp(-((DD / (y_si / 10)) ** 2))
            sim.fluid.vel[..., 0] = +u_si * np.gradient(stream, axis=1)
            sim.fluid.vel[..., 1] = -u_si * np.gradient(stream, axis=0)
        case _:
            raise ValueError(velocity_init_mode)

    # Perturb y-velocity slightly to speed onset of shedding.
    vy_si = -np.exp(-(((domain.x - cx) / 10) ** 2)) * u_si * 0.01
    sim.fluid.vel[..., 1] += vy_si[:, None]

    # NOTE: Convert velocity to lattice units / timestep.
    sim.fluid.vel[:] = scales.velocity.to_lattice_units(sim.fluid.vel)


# %% Define simulation output

# allocate extra arrays for plotting to avoid reallocating each step
vmag_ = np.zeros_like(sim.fluid.rho[:, :])
curl_ = np.zeros_like(sim.fluid.rho[:, :])
stream_ = np.zeros_like(sim.fluid.rho[:, :])
arr_i8_ = np.zeros_like(sim.fluid.rho[:, :], dtype=np.int8)
tracer_ = np.zeros([sim.fluid.vel.shape[0], sim.fluid.vel.shape[1], 3])

# output params
fsz = domain.counts[1] // 15
fox = domain.counts[1] // 60
foy = 0
font = ImageFont.truetype("NotoSans-BoldItalic", fsz)
outy = 1000
outx = outy * aspect_ratio


def write_png(
    path: Path,
    data: np.ndarray,
    label: str,
    background: Literal["dark", "light"],
    **kwargs,
):
    tcol = (255, 255, 255) if background == "dark" else (0, 0, 0)
    bcol = (0, 0, 0) if background == "dark" else (255, 255, 255)

    with PngWriter(path, outx, sim.cells.flags, data, **kwargs) as img:
        draw = ImageDraw.Draw(img)
        draw.text((fox, foy), label, tcol, font=font, stroke_width=fsz // 15, stroke_fill=bcol)


for iter in run_sim(sim, meta, args.out_dir, args.checkpoints):
    with timed(logger, "writing output", 1):
        with timed(logger, "calc curl"):
            curl_[:] = calc_curl(sim.fluid.vel, scales=scales)

        # Calculate current dimensionless time.
        if False:
            T = d_si / u_si
            tau_ = (iter * out_dt_si) / T

            write_png(
                args.out_dir / f"density_{iter:06d}.png",
                sim.fluid.rho,
                f"Density\nτ = {tau_:7.2f}",
                "light",
                cmap="InkyBlueRed",
                vmin=0.95,
                vmax=1.05,
            )

        write_png(
            args.out_dir / f"curl_{iter:06d}.png",
            curl_,
            "Vorticity",
            "dark",
            cmap="OrangeBlue",
            vmin=-vmax_curl,
            vmax=vmax_curl,
        )

        lines = dict(linewidths=1, linestyles="solid", negative_linestyles="solid")

        def imshow_with_contours(
            vel: np.ndarray,
            data: np.ndarray | None,
            levels: np.ndarray,
            mode: Literal["mult", "add", "smart"],
            fname: str,
            *,
            vmax: float | None = None,
            vmin: float | None = None,
            cmap: str = "inferno",
        ):
            """ """
            vmag_[:] = calc_vmag(vel, scales=scales)
            stream_[:] = calc_stream_func(vel, scales=scales, origin=(0, y_si / 2))

            vmag_[sim.cells.flags == CellFlags.WALL] = np.nan
            stream_[sim.cells.flags == CellFlags.WALL] = np.nan

            data = data or vmag_
            vmin = vmin or np.nanmin(data)
            vmax = vmax or np.nanmax(data)

            with PngWriter(
                args.out_dir / fname,
                outx,
                sim.cells.flags,
                vmag_,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            ) as img:
                # Quantize stream function & detect edges.
                width = L // 300  # Pixels.
                arr_i8_[:] = np.digitize(stream_, levels)
                arr_i8_[:] = (maximum_filter(arr_i8_, 3) - minimum_filter(arr_i8_, 3)) != 0
                arr_i8_[:] = binary_dilation(arr_i8_, iterations=(width - 2) // 2)

                # Combine stream contours with image.
                buf1 = np.array(img).astype(np.float64) / 255.0
                buf2 = arr_i8_.astype(np.float64).T[
                    ..., None
                ]  # Zero where we have no contours, one where we do.

                if mode == "add":
                    buf1 = buf1 + 0.3 * buf2
                elif mode == "mult":
                    buf1 = buf1 * (1 + buf2)
                elif mode == "smart":
                    # This mode attempts to make ligher areas darker (or at least not overexposed) and
                    # darker areas lighter.
                    # The results heavily depend on the mapping parameters a & b which are the values
                    # zero and full brightness get mapped to respectively.
                    #
                    # NOTE: This works on each colour channel separately. To do a better job we could
                    #       map into HSL and work just on the L channel.

                    # -- additive blending
                    # a = 0.1
                    # b = 0.8
                    # m = b - a
                    # buf3 = m * buf1 + a
                    # buf1 = buf1 * (1 - buf2) + buf3 * buf2

                    # -- multiplicative blending
                    a = 1.5
                    b = 1.0
                    m = b - a
                    buf3 = m * buf1 + a
                    buf1 = buf1 * (1 - buf2) + buf3 * buf2 * buf1

                # Ignore the alpha channel.
                # buf1[..., 3] = 1.0

                # Replace the original image.
                img_ = Image.fromarray((buf1 * 255).astype(np.uint8))
                img.paste(img_, (0, 0))

        # -- Fixed reference frame.

        smax_lu = u * L / 2
        smax_si = scales.converter(L=2, T=-1).to_physical_units(smax_lu)
        levels = np.linspace(-1, 1, 30)
        levels = np.abs(levels) ** 2 * np.sign(levels) * smax_si

        imshow_with_contours(
            sim.fluid.vel,
            None,
            levels,
            "smart",
            f"vmag_fixed_{iter:06d}.png",
            vmin=0,
            vmax=vmax_vmag,
        )

        # -- Comoving reference frame.

        uu = np.array([u, 0])[None, None, :]
        smax_si *= 0.3
        levels = np.linspace(-1, 1, 30)
        levels = np.abs(levels) ** 2 * np.sign(levels) * smax_si

        imshow_with_contours(
            sim.fluid.vel - uu,
            None,
            levels,
            "smart",
            f"vmag_comov_{iter:06d}.png",
            vmin=0,
            vmax=vmax_vmag * 0.6,
        )


# render with
# export FPS=30; ffmpeg -framerate $FPS -i out/conc_%06d.png -framerate $FPS -i out/curl_%06d.png                                      -c:v libx264 -crf 10 -filter_complex "[1]pad=iw:ih+2:0:2[v1];[0][v1]vstack=inputs=2" -y kh.mp4
# export FPS=30; ffmpeg -framerate $FPS -i out/conc_%06d.png -framerate $FPS -i out/curl_%06d.png -framerate $FPS -i out/vmag_%06d.png -c:v libx264 -crf 10 -filter_complex "[1]pad=iw:ih+2:0:2[v1];[2]pad=iw:ih+2:0:2[v2];[0][v1][v2]vstack=inputs=3" -y kh4.mp4

# see: https://trac.ffmpeg.org/wiki/Encode/H.264
# export FPS=30; ffmpeg -framerate $FPS -i out/cols_%06d.png -framerate $FPS -i out/curl_%06d.png -framerate $FPS -i out/vmag_%06d.png -framerate $FPS -i out/qcrit_%06d.png -filter_complex "[0]pad=iw+5:ih+5:iw:ih[tl];[1]pad=iw:ih+5:0:ih[tr];[2]pad=iw+5:ih:iw:0[bl];[3]pad=iw:ih:0:0[br];[tl][tr][bl][br]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]" -map "[v]" -c:v libx264 -crf 23 -tune animation -y vs4.mp4
# export FPS=30; ffmpeg -framerate $FPS -i out/cols_%06d.png -framerate $FPS -i out/curl_%06d.png -framerate $FPS -i out/vmag_%06d.png -filter_complex "[1]pad=iw:ih+2:0:2[v1];[2]pad=iw:ih+2:0:2[v2];[0][v1][v2]vstack=inputs=3" -c:v libx264 -crf 23 -tune animation -y vs_re1000.mp4
