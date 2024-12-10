# %% Imports

import logging
import numpy as np
import os

from numba import set_num_threads
from multiprocessing import cpu_count
from pprint import pprint
from dataclasses import asdict
from typing import Literal

from boltzmann.utils.logger import tick, PerfInfo, basic_config
from boltzmann.core import DomainMeta, SimulationMeta, FluidMeta, D2Q9, D2Q5, CellType
from boltzmann.utils.mpl import PngWriter

# be nice
set_num_threads(cpu_count() - 2)

logger = logging.getLogger(__name__)

basic_config(logger)

logger.info("Starting")

# %% Params

# dimensions [m]
y_si = 0.5
x_si = 1.5

# flow velocity [m/s]
v0_si = 1

vmax_vmag = 1.6 * v0_si
vmax_curl = 30 * v0_si
vmax_conc = 1.025

# nu_si = 2e-4  # re = 50
# nu_si = 1e-5  # re= 1000
nu_si = 1e-4
rho_si = 1000
mu_si = nu_si * rho_si

re_no = v0_si * y_si / nu_si
logger.info(f"Reynolds no.:  {re_no:,.0f}")

cs_mult = 10.0
cs_si = v0_si * cs_mult

scale = 30
# scale = 6
cnts = [int(100 * scale * x_si) + 1, int(00 * scale * y_si) + 1]
dom = DomainMeta.with_extent_and_counts(extent_si=[[0, x_si], [0, y_si]], counts=cnts)
fld = FluidMeta(mu_si, rho_si)
sim = SimulationMeta.with_cs(domain=dom, fluid=fld, cs=cs_si)

logger.info(f"\n{pprint(asdict(sim), sort_dicts=False, width=10)}")

# dimensionless time does not depend on viscosity, purely on distances
out_dx_si = x_si / v0_si / 400.0  # want to output when flow has moved this far
sim_dx_si = v0_si * sim.dt  # flow moves this far in dt (i.e. one iteration)
n = out_dx_si / sim_dx_si
n = int(n + 1e-8)
out_dt_si = sim.dt * n

logger.info(f"Steps per output: {n=}")


# %% Compile

logger.info("Compiling using Numba...")

from boltzmann.impl2 import *

# make numba objects
pidx = PeriodicDomain(dom.counts)
g_lu = np.array([0.0, 0.0], dtype=np.float32)
params = NumbaParams(sim.dt, dom.dx, sim.cs, sim.w_pos_lu, sim.w_neg_lu, g_lu)

# %% Initialize arrays

cell = make_array(pidx, dtype=np.int32)

# D2Q9 for concentration
f1 = make_array(pidx, 9)
f2 = make_array(pidx, 9)
vel_si = make_array(pidx, 2)
rho_si = make_array(pidx, fill=fld.rho)

# D2Q5 for concentration
g1 = make_array(pidx, 5)
g2 = make_array(pidx, 5)
conc = make_array(pidx)

# introduce slight randomness to initial density
np.random.seed(42)
# rho *= 1 + 0.0001 * (np.random.uniform(size=rho.shape) - 0.5)
rho_si_ = unflatten(pidx, rho_si)
rho_si_[10:-10, 10:-10] *= 1 + 0.001 * (np.random.uniform(size=rho_si_[10:-10, 10:-10].shape) - 0.5)

x = np.pad(dom.x, (1, 1), mode="edge")
y = np.pad(dom.y, (1, 1), mode="edge")
XX, YY = np.meshgrid(x, y)

# XX = XX.flatten()
# YY = YY.flatten()
XX = flatten(pidx, XX)
YY = flatten(pidx, YY)

# fixed velocity in- & out-flow
# (only need to specify one due to periodicity)
cell_ = unflatten(pidx, cell)
cell_[1, :] = CellType.BC_VELOCITY.value  # bottom
cell_[-2, :] = CellType.BC_VELOCITY.value  # top

cell_[dom.counts[1] // 2 + 10 :, 0] = CellType.BC_VELOCITY.value
cell_[: dom.counts[1] // 2 - 10, 0] = CellType.BC_VELOCITY.value

# set upper half's velocity & concentration
vel_si_ = unflatten(pidx, vel_si)
vel_si_[dom.counts[1] // 2 :, :, 0] = v0_si
vel_si_[: dom.counts[1] // 2, :, 0] = -v0_si

conc_ = unflatten(pidx, conc)
conc_[dom.counts[1] // 2 :, :] = 1.0

# ensure cell types match across boundary
pidx.copy_periodic(cell)

# flag arrays
is_wall = (cell == CellType.BC_WALL.value).astype(np.int32)
is_fixed = (cell != CellType.FLUID.value).astype(np.int32)

# initial f is equilibrium for desired values of v and rho
calc_equilibrium(vel_si, rho_si, f1, np.float32(params.cs_si), D2Q9)
f2[:] = f1[:]

# initial g is equilibrium for desired value of conc
calc_equilibrium_advdif(vel_si, conc, g1, np.float32(params.cs_si), D2Q5)
g2[:] = g1[:]


# %% Define VTK output function

from boltzmann.vtkio import VtiWriter


def write_vti(
    path: str,
    vel_si: np.ndarray,
    rho_si: np.ndarray,
    curl_si: np.ndarray,
    conc: np.ndarray,
    cell: np.ndarray,
    f: np.ndarray,
    pidx: PeriodicDomain,
    params: NumbaParams,
    *,
    save_f: bool = False,
):
    with VtiWriter(path, pidx) as writer:
        writer.add_data("Velocity", vel_si, default=True)
        writer.add_data("Density", rho_si)
        writer.add_data("Vorticity", curl_si)
        writer.add_data("Tracer", conc)
        writer.add_data("CellType", cell)
        if save_f:
            writer.add_data("F", f)


# %% Write PNGs

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt


FONT = dict(
    fontsize=14,
    fontfamily="sans-serif",
    fontstyle="italic",
    fontweight="bold",
)


def png_writer(path: str, data: np.ndarray, **kwargs) -> PngWriter:
    return PngWriter(path, dom, pidx, cell, data, **kwargs)


def annotate(ax1, text: str, **kwargs):
    ax1.annotate(
        text,
        (x_si * 0.05, x_si * 0.95),
        xytext=(0.25, -1.1),
        textcoords="offset fontsize",
        **kwargs,
        **FONT,
    )


def write_png_vmag(path: str):
    vmag = np.sqrt(np.sum(vel_si**2, axis=-1))
    vmag = np.tanh(vmag / vmax_vmag) * vmax_vmag

    with png_writer(
        path,
        vmag,
        cmap=plt.cm.inferno,
        vmin=0,
        vmax=vmax_vmag,
        fig_kwargs={
            "facecolor": "#888888",
        },
    ) as fig:
        ax1 = fig.gca()
        annotate(ax1, "Velocity", c="w")


def write_png_curl(path: str):
    vort = np.tanh(curl_si / vmax_curl) * vmax_curl

    with png_writer(
        path,
        vort,
        cmap=plt.cm.RdBu,
        vmin=-vmax_curl,
        vmax=vmax_curl,
        fig_kwargs={
            "facecolor": "#E0E0E0",
        },
    ) as fig:
        ax1 = fig.gca()
        annotate(ax1, "Vorticity", c="k")


def write_png_conc(path: str):
    with png_writer(
        path,
        conc,
        cmap=plt.cm.viridis,
        vmin=0,
        vmax=vmax_conc,
        fig_kwargs={
            "facecolor": "#E0E0E0",
        },
    ) as fig:
        ax1 = fig.gca()
        annotate(ax1, "Tracer", c="w")


# %% Main Loop

import time

path_out = f"out"
tmpl_out = f"{path_out}/out_{{:06d}}.vti"

path = os.path.dirname(tmpl_out)
if not os.path.exists(path):
    os.makedirs(path)


# write out the zeroth timestep
pref = f"example2_re{re_no:.0f}"
curl_si = calc_curl_2d(pidx, vel_si, cell, params.dx_si)
# write_vti(tmpl_out.format(0), vel_si, rho_si, curl_si, cell, f1, pidx, params)
write_png_vmag(f"{path_out}/vmag_{0:06d}.png")
write_png_curl(f"{path_out}/curl_{0:06d}.png")
write_png_conc(f"{path_out}/conc_{0:06d}.png")

t = 0.0
out_i = 1
out_t = out_dt_si
max_i = 500

batch_i = int((out_dt_si + 1e-8) // sim.dt)

logger.info(f"{batch_i:,d} iters/output")

# start from checkpoint
if False:

    with open(f"{path_out}/checkpoint_info.txt") as fin:
        out_i = int(fin.readline().split("=")[-1])
        t = out_i * out_dt_si

    logger.info(f"Starting from checkpoint {out_i=} {t=}")

    # load distribution functions
    f1 = np.load(f"{path_out}/checkpoint_f1.npy")
    g1 = np.load(f"{path_out}/checkpoint_g1.npy")

    f2[:] = f1[:]
    g2[:] = g1[:]

    # re-calc macroscopic
    # qs = np.ascontiguousarray(D2Q9.qs_f32.T)
    # rho_si = np.sum(f1, axis=-1)
    # vel_si = np.dot(qs, f1) * params.cs_si / rho_si
    # conc = np.sum(g1, axis=-1)


perf_total = PerfInfo()

for out_i in range(out_i, max_i + 1):
    perf_batch = tick()

    if np.any(~np.isfinite(f1)):
        raise ValueError(f"Non-finite value in f.")

    loop_for_2_advdif(
        batch_i, rho_si, vel_si, f1, f2, D2Q9, conc, g1, g2, D2Q5, is_wall, is_fixed, params, pidx
    )

    perf_batch = perf_batch.tock(events=np.prod(dom.counts) * batch_i)
    perf_total = perf_total + perf_batch

    mlups_batch = perf_batch.events / (1e6 * perf_batch.seconds)
    mlups_total = perf_total.events / (1e6 * perf_total.seconds)

    curl_si = calc_curl_2d(pidx, vel_si, cell, params.dx_si)

    write_png_vmag(f"{path_out}/vmag_{out_i:06d}.png")
    write_png_curl(f"{path_out}/curl_{out_i:06d}.png")
    write_png_conc(f"{path_out}/conc_{out_i:06d}.png")

    # if out_i % 10 == 0:
    # write_vti(f"{path_out}/checkpoint_vtk.vti", vel_si, rho_si, curl_si, cell, f1, pidx, params)
    np.save(f"{path_out}/checkpoint_f1", f1, allow_pickle=False)
    np.save(f"{path_out}/checkpoint_g1", g1, allow_pickle=False)
    with open(f"{path_out}/checkpoint_info.txt", "w") as fout:
        fout.write(f"{out_i=}")

    logger.info(f"Wrote {out_i=} {out_t=:.3f}, {mlups_batch=:.2f}, {mlups_total=:.2f}\r")

    if out_i % 10 == 0:
        time.sleep(60)

    out_t += out_dt_si

# render with
# ffmpeg -i out/curl_%06d.png -i out/vmag_%06d.png -c:v libx264 -crf 10 -r 30 -filter_complex "[1]pad=iw:ih+2:0:2[v1];[0][v1]vstack=inputs=2" -y kh.mp4
# ffmpeg -i out/conc_%06d.png -i out/curl_%06d.png -i out/vmag_%06d.png -c:v libx264 -crf 10 -r 30 -filter_complex "[1]pad=iw:ih+2:0:2[v1];[2]pad=iw:ih+2:0:2[v2];[0][v1][v2]vstack=inputs=3" -y kh4.mp4
