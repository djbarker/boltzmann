# %% Imports

import logging

import numpy as np

from boltzmann.utils.logger import tick, PerfInfo, basic_config
from boltzmann.core import DomainMeta, SimulationMeta, FluidMeta, D2Q9, CellType
from boltzmann.vtkio import VtiWriter

log = logging.getLogger(__name__)

basic_config(log)


# %% Params

dom = DomainMeta.with_extent_and_counts(extent=[[-0.2, 0.2], [-0.1, 0.1]], counts=[1000, 500])
fld = FluidMeta.WATER
sim = SimulationMeta.with_cs(domain=dom, fluid=fld, cs=1)

log.info(f"{dom.dx=}, {dom.extent=}, {dom.counts=}, cells={np.prod(dom.counts):,d}")
log.info(f"{fld.mu=}, {fld.rho=}, {fld.nu=}")
log.info(f"{sim.tau=}, {sim.c=}, {sim.dt=}")

if sim.tau < 0.6:
    log.warning(f"Small value for tau! [tau={sim.tau}]")

l_ = 0.25
nu_ = fld.nu / (sim.c * dom.dx)
wpve = 1.0 / (nu_ + 0.5)
wnve = 1.0 / (l_ / nu_ + 0.5)

log.info(f"{1/wpve=} {1/wnve=} {nu_=}")

# cylinder centre & radius [m]
cx, cy = -0.1, 0
r = 0.01

# flow velocity [m/s]
v0 = 0.01
# v0 = 0.5

re_no = v0 * r / fld.nu
log.info(f"Reynolds no.:  {re_no:,.0f}")


# %% Compile

log.info("Compiling using Numba...")

from boltzmann.impl2 import *


# make numba objects
pidx = PeriodicDomain(dom.counts)
params = NumbaParams(sim.dt, dom.dx, sim.c, sim.tau, wpve, wnve)
d2q9 = NumbaModel(D2Q9.ws, D2Q9.qs, D2Q9.js, pidx.counts)

# %% Initialize arrays

f1 = make_array(pidx, 9)
f2 = make_array(pidx, 9)
vel = make_array(pidx, 2)
rho = make_array(pidx, fill=fld.rho)
cell = make_array(pidx, dtype=np.int32)
curl = make_array(pidx)

# introduce slight randomness to initial density
np.random.seed(42)
# rho *= 1 + 0.0001 * (np.random.uniform(size=rho.shape) - 0.5)

x = np.pad(dom.x, (1, 1), mode="edge")
y = np.pad(dom.y, (1, 1), mode="edge")
XX, YY = np.meshgrid(x, y)

# XX = XX.flatten()
# YY = YY.flatten()
XX = flatten(pidx, XX)
YY = flatten(pidx, YY)


# fixed velocity in- & out-flow
# (only need to specify one due to periodicity)
cell[make_slice_y1d(pidx, 1)] = CellType.BC_VELOCITY.value


# reduce velocity around cylinder centre to reduce initial sound waves
vel[:, 0] = v0 * (1 - np.exp(-(((YY - cy) / (r * 4)) ** 2) - ((XX - cx) / (r * 4)) ** 2))

# --- Cylinder
# cell[:] = CellType.BC_WALL.value * (((XX - cx) ** 2 + (YY - cy) ** 2) < r**2.0)

# --- Tapered
xs = np.linspace(cx, cx + 0.05, 500)
rs = np.linspace(r, 0.0, 500)
for x, r in zip(xs, rs):
    mask = ((XX - x) ** 2 + (YY - 0.0) ** 2) < r**2.0
    mask = mask.flatten()
    cell[mask] = CellType.BC_WALL.value

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
DD = distance_transform_edt(WW).clip(0, 100)
DD = DD / np.max(DD)
DD = flatten(pidx, DD)
vel[:, 0] = v0 * DD


# ensure cell types match across boundary
pidx.copy_periodic(cell)

# flag arrays
is_wall = (cell == CellType.BC_WALL.value).astype(np.int32)
update_vel = (cell == CellType.FLUID.value).astype(np.int32)

# set wall velocity to zero
# (just for nice output, doesn't affect the simulation)
# vel[cell == CellType.BC_WALL.value, 0] = 0

# initial f is equilibrium for desired values of v and rho
calc_equilibrium(vel, rho, f1, np.float32(params.cs), d2q9)
f2[:] = f1[:]


# %% Define VTK output function


def write_vti(
    path: str,
    v: np.ndarray,
    rho: np.ndarray,
    curl: np.ndarray,
    cell: np.ndarray,
    f: np.ndarray,
    pidx: PeriodicDomain,
    params: NumbaParams,
    *,
    save_f: bool = False,
):
    with VtiWriter(path, pidx) as writer:
        writer.add_data("Velocity", v, default=True)
        writer.add_data("Density", rho)
        writer.add_data("Vorticity", curl)
        writer.add_data("CellType", cell)
        if save_f:
            writer.add_data("F", f)


# %% Write PNGs


def write_png(path: str, vel: np.ndarray, rho: np.ndarray, vort: np.ndarray, show: bool = False):
    pass  # some Qt error with numba :(


#     vmag = np.sqrt(np.sum(vel**2, axis=-1))

#     vort = np.squeeze(vort)
#     rho = np.squeeze(rho)
#     vel = np.squeeze(vel)

#     fig = plt.figure(figsize=(8, 4.5))
#     ax1 = fig.add_subplot(2, 2, 1)
#     ax2 = fig.add_subplot(2, 2, 2)
#     ax3 = fig.add_subplot(2, 2, 3)
#     # ax4 = fig.add_subplot(2, 2, 4)

#     def _config_axes(ax):
#         ax.tick_params(direction="in", which="both", top=True, right=True)
#         ax.minorticks_on()
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])

#     _config_axes(ax1)
#     _config_axes(ax2)
#     _config_axes(ax3)
#     # _config_axes(ax4)

#     e = [-0.2, 0.2, -0.1, 0.1]

#     x = np.linspace(e[0], e[1], rho.shape[1])
#     y = np.linspace(e[2], e[3], rho.shape[0])
#     xx, yy = np.meshgrid(x, y)

#     vmin, vmax = 2e-6, 2.6e-2
#     ax1.imshow(vmag, vmin=vmin, vmax=vmax, cmap=plt.cm.Spectral_r, origin="lower", extent=e)
#     # ax1.contour(xx, yy, vmag, np.linspace(vmin, vmax, 8), colors="k", linewidths=0.5)
#     ax1.set_title("Velocity", fontsize=8)

#     vmin, vmax = -4, 4
#     ax2.imshow(vort, vmin=vmin, vmax=vmax, cmap=plt.cm.RdBu, origin="lower", extent=e)
#     # ax2.contour(xx, yy, vort, np.linspace(vmin, vmax, 8), colors="k", linewidths=0.5)
#     ax2.set_title("Vorticity", fontsize=8)

#     vmin, vmax = 975, 1025
#     ax3.imshow(rho, vmin=vmin, vmax=vmax, cmap=plt.cm.plasma, origin="lower", extent=e)
#     # ax3.contour(xx, yy, rho, np.linspace(vmin, vmax, 8), colors="k", linewidths=0.5)
#     ax3.set_title("Density", fontsize=8)
#     fig.tight_layout()

#     if not show:
#         fig.savefig(path, dpi=300)
#         fig.clear()
#         plt.close()


# %% Main Loop

# write out the zeroth timestep
pref = f"example2_re{re_no:.0f}"
curl = calc_curl_2d(pidx, vel, cell, params.dx)
write_vti(f"out/{pref}_{0:06d}.vti", vel, rho, curl, cell, f1, pidx, params)
write_png(f"out/{pref}_{0:06d}.png", vel, rho, curl)

t = 0.0
out_dt = 0.1
# out_dt = sim.dt
out_i = 1
out_t = out_dt
max_i = 1000

batch_i = int((out_dt + 1e-8) // sim.dt)

log.info(f"{batch_i:,d} iters/output")

perf_total = PerfInfo()

for out_i in range(1, max_i):
    perf_batch = tick()

    if np.any(~np.isfinite(f1)):
        raise ValueError(f"Non-finite value in f.")

    loop_for_2(batch_i, vel, rho, f1, f2, is_wall, update_vel, params, pidx, d2q9)

    perf_batch = perf_batch.tock(events=np.prod(dom.counts) * batch_i)
    perf_total = perf_total + perf_batch

    mlups_batch = perf_batch.events / (1e6 * perf_batch.seconds)
    mlups_total = perf_total.events / (1e6 * perf_total.seconds)

    curl = calc_curl_2d(pidx, vel, cell, params.dx)
    write_vti(f"out/{pref}_{out_i:06d}.vti", vel, rho, curl, cell, f1, pidx, params)
    write_png(f"out/{pref}_{out_i:06d}.png", vel, rho, curl)

    log.info(f"Wrote {out_i=} {out_t=:.3f}, {mlups_batch=:.2f}, {mlups_total=:.2f}\r")

    out_t += out_dt
