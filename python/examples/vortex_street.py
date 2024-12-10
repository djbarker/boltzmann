# %% Imports

import logging
import numpy as np
import os

from pprint import pprint
from dataclasses import asdict
from typing import Literal

from boltzmann.utils.logger import tick, PerfInfo, basic_config
from boltzmann.core import DomainMeta, SimulationMeta, FluidMeta, D2Q9, CellType

log = logging.getLogger(__name__)

basic_config(log)

log.info("Starting")

# %% Params

# cylinder centre & radius [m]
cx_si, cy_si = -0.125, 0
r_si = 0.01

# flow velocity [m/s]
v0_si = 1
# v0 = 0.5

vmax_vmag = 1.8
vmax_curl = 350

# nu_si = 2e-4  # re = 50
# nu_si = 1e-5  # re= 1000
nu_si = 1e-4
rho_si = 1000
mu_si = nu_si * rho_si

re_no = v0_si * r_si / nu_si
log.info(f"Reynolds no.:  {re_no:,.0f}")

cs_mult = 20.0
cs_si = v0_si * cs_mult

dom = DomainMeta.with_extent_and_counts(extent_si=[[-0.2, 0.3], [-0.1, 0.1]], counts=[1876, 751])
fld = FluidMeta(mu_si, rho_si)
sim = SimulationMeta.with_cs(domain=dom, fluid=fld, cs=cs_si)

log.info(f"\n{pprint(asdict(sim), sort_dicts=False, width=10)}")

# dimensionless time does not depend on viscosity, purely on distances
out_dx_si = r_si / 8.0  # want to output when flow has moved this far
sim_dx_si = v0_si * sim.dt  # flow moves this far in dt (i.e. one iteration)
n = out_dx_si / sim_dx_si
n = int(n + 1e-8)
out_dt_si = sim.dt * n

log.info(f"Steps per output: {n=}")


# %% Compile

log.info("Compiling using Numba...")

from boltzmann.impl2 import *

# make numba objects
pidx = PeriodicDomain(dom.counts)
g_lu = np.array([0.0, 0.0], dtype=np.float32)
params = NumbaParams(sim.dt, dom.dx, sim.cs, sim.w_pos_lu, sim.w_neg_lu, g_lu)

# %% Initialize arrays

f1 = make_array(pidx, 9)
f2 = make_array(pidx, 9)
vel_si = make_array(pidx, 2)
rho_si = make_array(pidx, fill=fld.rho)
cell = make_array(pidx, dtype=np.int32)

# introduce slight randomness to initial density
np.random.seed(42)
# rho *= 1 + 0.0001 * (np.random.uniform(size=rho.shape) - 0.5)
rho_si_ = unflatten(pidx, rho_si)
rho_si_[:, 3:5] *= 1 + 0.0001 * (np.random.uniform(size=rho_si_[:, 3:5].shape) - 0.5)

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
cell_[:, 1] = CellType.BC_VELOCITY.value


# reduce velocity around cylinder centre to reduce initial sound waves
vel_si[:, 0] = v0_si * (
    1 - np.exp(-(((YY - cy_si) / (r_si * 4)) ** 2) - ((XX - cx_si) / (r_si * 4)) ** 2)
)

# --- Cylinder
mask = ((XX - cx_si) ** 2 + (YY - cy_si) ** 2) < r_si**2.0
mask = mask.flatten()
cell[mask] = CellType.BC_WALL.value

# --- Tapered
# xs = np.linspace(cx_si, cx_si + 0.05, 500)
# rs = np.linspace(r_si, 0.0, 500)
# for x, r_si in zip(xs, rs):
#     mask = ((XX - x) ** 2 + (YY - 0.0) ** 2) < r_si**2.0
#     mask = mask.flatten()
#     cell[mask] = CellType.BC_WALL.value

# --- Square
# cell[:] = CellType.BC_WALL.value * ((np.abs(xx - cx) < r) * (np.abs(yy - cy) < r))

# --- Wedge
# cell[:] = CellType.BC_WALL.value * (
#     (np.abs(xx - cx) < r) * (np.abs(yy - cy) < 0.5 * (xx - (cx - r)))
# )

# ramp velocity from zero to flow velocity moving away from the walls

from scipy.ndimage import distance_transform_edt

WW = 1 - 1 * (cell == CellType.BC_WALL.value)
WW = unflatten(pidx, WW)
DD = distance_transform_edt(WW).clip(0, 75)
DD = DD / np.max(DD)
DD = flatten(pidx, DD)
vel_si[:, 0] = v0_si * DD


# ensure cell types match across boundary
pidx.copy_periodic(cell)

# flag arrays
is_wall = (cell == CellType.BC_WALL.value).astype(np.int32)
update_vel = (cell == CellType.FLUID.value).astype(np.int32)

# set wall velocity to zero
# (just for nice output, doesn't affect the simulation)
# vel[cell == CellType.BC_WALL.value, 0] = 0

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
    re: float,
    vel: np.ndarray,
    vort: np.ndarray,
    cell: np.ndarray,
    what: Literal["vmag", "curl"],
    show: bool = False,
):
    vmag = np.sqrt(np.sum(vel**2, axis=-1))
    vmag = unflatten(pidx, vmag)[1:-1, 1:-1]
    vort = unflatten(pidx, vort)[1:-1, 1:-1]
    cell = unflatten(pidx, cell)[1:-1, 1:-1]

    vmag = np.where(cell == 1, np.nan, vmag)
    vort = np.where(cell == 1, np.nan, vort)

    vmag = np.tanh(vmag / vmax_vmag) * vmax_vmag
    vort = np.tanh(vort / vmax_curl) * vmax_curl

    match what:
        case "vmag":
            col = "w"
            fcol = "#E0E0E0"
            cmap = plt.cm.inferno
            vmin = 0
            vmax = vmax_vmag * 0.95
            arr = vmag
        case "curl":
            col = "k"
            fcol = "#888888"
            cmap = plt.cm.RdBu
            vmin = -vmax_curl * 0.95
            vmax = vmax_curl * 0.95
            arr = vort
        case _:
            raise ValueError(f"{what=!r}")

    ar = (dom.extent[0, 1] - dom.extent[0, 0]) / (dom.extent[1, 1] - dom.extent[1, 0])

    fig = plt.figure(figsize=(8, 8 / ar), facecolor=fcol)
    ax1 = fig.add_subplot(1, 1, 1)

    ax1.imshow(arr, vmin=vmin, vmax=vmax, cmap=cmap, interpolation="none")

    fnt = dict(
        fontsize=18,
        fontfamily="sans-serif",
        fontstyle="italic",
        fontweight="bold",
    )

    ax1.annotate(f"Re = {int(re+1e-8):,d}", (40, 90), c=col, **fnt)

    plt.axis("off")

    if not show:
        fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0)
        fig.clear()
        plt.close()


# %% Main Loop

import time

path_out = f"vortex_street/re{re_no:.0f}"
tmpl_out = f"{path_out}/out_{{:06d}}.vti"

path = os.path.dirname(tmpl_out)
if not os.path.exists(path):
    os.makedirs(path)


# write out the zeroth timestep
pref = f"example2_re{re_no:.0f}"
curl_si = calc_curl_2d(pidx, vel_si, cell, params.dx_si)
# write_vti(tmpl_out.format(0), vel_si, rho_si, curl_si, cell, f1, pidx, params)
write_png(f"{path_out}/vmag_{0:06d}.png", re_no, vel_si, curl_si, cell, "vmag")
write_png(f"{path_out}/curl_{0:06d}.png", re_no, vel_si, curl_si, cell, "curl")

t = 0.0
out_i = 1
out_t = out_dt_si
max_i = 1000

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
    write_png(f"{path_out}/vmag_{out_i:06d}.png", re_no, vel_si, curl_si, cell, "vmag")
    write_png(f"{path_out}/curl_{out_i:06d}.png", re_no, vel_si, curl_si, cell, "curl")

    if out_i % 10 == 0:
        write_vti(f"{path_out}/checkpoint_vtk.vti", vel_si, rho_si, curl_si, cell, f1, pidx, params)
        np.save(f"{path_out}/checkpoint_f1", f1, allow_pickle=False)
        with open(f"{path_out}/checkpoint_info.txt", "w") as fout:
            fout.write(f"{out_i=}")

    # just to be nice to my cpu fan
    if out_i % 10 == 0:
        time.sleep(30)

    log.info(f"Wrote {out_i=} {out_t=:.3f}, {mlups_batch=:.2f}, {mlups_total=:.2f}\r")

    out_t += out_dt_si
