# %% Imports

import logging
import numpy as np
import os

from pprint import pprint
from dataclasses import asdict
from scipy.signal import convolve2d
from scipy.ndimage import distance_transform_edt

from boltzmann.utils.logger import tick, PerfInfo, basic_config
from boltzmann.core import DomainMeta, SimulationMeta, FluidMeta, D2Q9, CellType

log = logging.getLogger(__name__)

basic_config(log)

log.info("Starting")


def voronoi(
    points: np.ndarray, extent: np.ndarray, counts: np.ndarray, width: float, upsample: int = 4
) -> np.ndarray:
    assert points.ndim == 2
    assert points.shape[1] == 2
    assert extent.shape == (2, 2)
    assert counts.shape == (2,)

    counts_ = counts * upsample

    II = (counts_[0] * (points[:, 0] - extent[0, 0]) / (extent[0, 1] - extent[0, 0])).astype(
        np.int64
    )
    JJ = (counts_[1] * (points[:, 1] - extent[1, 0]) / (extent[1, 1] - extent[1, 0])).astype(
        np.int64
    )

    WW = np.zeros(tuple(counts_))
    WW[II, JJ] = 1

    WW = np.tile(WW, (3, 3))

    KK = np.array(
        [
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0],
        ],
    )

    DD = distance_transform_edt(1 - WW)
    CC = convolve2d(DD, KK) > 0

    # TODO: downsample here

    VV = distance_transform_edt(1 - CC) < (width * upsample)

    # chop out middle of periodic tiling
    VV = VV[counts_[0] : 2 * counts_[0], counts_[1] : 2 * counts_[1]]

    return VV


# %% Params

nu_si = 3e-5
rho_si = 1000
mu_si = nu_si * rho_si

cs_si = 1.0

dom = DomainMeta.with_extent_and_counts(extent_si=[[-0.1, 0.1], [-0.1, 0.1]], counts=[751, 751])
fld = FluidMeta(mu_si, rho_si)
sim = SimulationMeta.with_cs(domain=dom, fluid=fld, cs=cs_si)

log.info(f"\n{pprint(asdict(sim), sort_dicts=False, width=18)}")

n = 100
out_dt_si = sim.dt * n

# gravitational acceleration [m/s^2]
g_si = 0.1
g_lu = g_si * (sim.dt / sim.cs)

log.info(f"Steps per output: {n=}")


# %% Compile

log.info("Compiling using Numba...")

from boltzmann.impl2 import *

# make numba objects
pidx = PeriodicDomain(dom.counts)
g_lu = np.array([0.0, -g_lu], dtype=np.float32)
params = NumbaParams(sim.dt, dom.dx, sim.cs, sim.w_pos_lu, sim.w_neg_lu, g_lu)

# %% Initialize arrays

f1 = make_array(pidx, 9)
f2 = make_array(pidx, 9)
vel_si = make_array(pidx, 2)
rho_si = make_array(pidx, fill=fld.rho)
cell = make_array(pidx, dtype=np.int32)

np.random.seed(42)
points = np.random.uniform(size=(100, 2))
extent = np.array([[0, 1], [0, 1]])
counts = np.array([750, 750])

VV = voronoi(points, extent, counts, 12, upsample=1)

cell_ = unflatten(pidx, cell)
cell_[1:-1, 1:-1] = 1 - VV


# ensure cell types match across boundary
pidx.copy_periodic(cell)

# flag arrays
is_wall = (cell == CellType.BC_WALL.value).astype(np.int32)
update_vel = (cell == CellType.FLUID.value).astype(np.int32)

# initial f is equilibrium for desired values of v and rho
calc_equilibrium(vel_si, rho_si, f1, np.float32(params.cs_si), D2Q9)
f2[:] = f1[:]


# %% Define VTK output function

from boltzmann.vtkio import VtiWriter


def write_vti(
    path: str,
    vel_si: np.ndarray,
    rho_si: np.ndarray,
    curl_si: np.ndarray,
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
        writer.add_data("CellType", cell)
        if save_f:
            writer.add_data("F", f)


# %% Write PNGs

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt


def write_png(
    path: str,
    vel: np.ndarray,
    rho: np.ndarray,
    vort: np.ndarray,
    cell: np.ndarray,
    show: bool = False,
):
    vmag = np.sqrt(np.sum(vel**2, axis=-1))
    vmag = unflatten(pidx, vmag)[1:-1, 1:-1]
    cell = unflatten(pidx, cell)[1:-1, 1:-1]
    vmag = np.where(cell == 1, np.nan, vmag)

    ar = (dom.extent[0, 1] - dom.extent[0, 0]) / (dom.extent[1, 1] - dom.extent[1, 0])

    fig = plt.figure(figsize=(8, 8 / ar), facecolor="#252525")
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.imshow(vmag, vmin=0, vmax=0.04, cmap=plt.cm.inferno)

    fnt = dict(
        fontsize=18,
        fontfamily="sans-serif",
        fontstyle="italic",
        fontweight="bold",
    )

    # ax1.annotate(f"Re = {int(re+1e-8):,d}", (40, 90), c="w", **fnt)

    plt.axis("off")

    if not show:
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0)
        fig.clear()
        plt.close()


# %% Main Loop

import time

path_out = f"out/"
tmpl_out = f"{path_out}/out_{{:06d}}.vti"

path = os.path.dirname(tmpl_out)
if not os.path.exists(path):
    os.makedirs(path)


# write out the zeroth timestep
curl_si = calc_curl_2d(pidx, vel_si, cell, params.dx_si)
# write_vti(tmpl_out.format(0), vel_si, rho_si, curl_si, cell, f1, pidx, params)
write_png(f"{path_out}/out_{0:06d}.png", vel_si, rho_si, curl_si, cell)

t = 0.0
out_i = 1
out_t = out_dt_si
max_i = 50

batch_i = int((out_dt_si + 1e-8) // sim.dt)

log.info(f"{batch_i:,d} iters/output")

perf_total = PerfInfo()

for out_i in range(1, max_i):
    perf_batch = tick()

    if np.any(~np.isfinite(f1)):
        raise ValueError(f"Non-finite value in f.")

    loop_for_2(batch_i, vel_si, rho_si, f1, f2, is_wall, update_vel, params, pidx, D2Q9)

    perf_batch = perf_batch.tock(events=np.prod(dom.counts) * batch_i)
    perf_total = perf_total + perf_batch

    mlups_batch = perf_batch.events / (1e6 * perf_batch.seconds)
    mlups_total = perf_total.events / (1e6 * perf_total.seconds)

    curl_si = calc_curl_2d(pidx, vel_si, cell, params.dx_si)
    # write_vti(tmpl_out.format(out_i), vel_si, rho_si, curl_si, cell, f1, pidx, params)
    write_png(f"{path_out}/out_{out_i:06d}.png", vel_si, rho_si, curl_si, cell)

    if out_i % 10 == 0:
        write_vti(f"{path_out}/checkpoint_vtk.vti", vel_si, rho_si, curl_si, cell, f1, pidx, params)
        np.save(f"{path_out}/checkpoint_f1", f1, allow_pickle=False)
        with open(f"{path_out}/checkpoint_info.txt", "w") as fout:
            fout.write(f"{out_i=}")

    # just to be nice to my cpu fan
    if out_i % 10 == 0:
        time.sleep(30)

    vmax_si = np.max(np.sqrt(np.sum(vel_si**2, -1)))

    log.info(f"{out_i=}, {out_t=:.3f} s, {mlups_batch:.2f} MLU/s, {vmax_si=:.4f} m/s\r")

    out_t += out_dt_si
