# %% Imports

import logging
from itertools import product

import numpy as np
import numba
import vtk
import vtk.util.numpy_support as vtk_np

from boltzmann.utils.logger import tick, PerfInfo, basic_config
from boltzmann.core import DomainMeta, SimulationMeta, FluidMeta, D2Q9, CellType

log = logging.getLogger(__name__)

basic_config(log)


# %% Params

dom = DomainMeta.with_extent_and_counts(extent=[[-0.2, 0.2], [-0.1, 0.1]], counts=[1000, 500])
fld = FluidMeta.WATER
sim = SimulationMeta.with_cs(domain=dom, fluid=fld, cs=0.6)

log.info(f"{dom.dx=}, {dom.extent=}, {dom.counts=}, cells={np.prod(dom.counts):,d}")
log.info(f"{fld.mu=}, {fld.rho=}, {fld.nu=}")
log.info(f"{sim.tau=}, {sim.c=}, {sim.dt=}")

if sim.tau < 0.6:
    log.warning(f"Small value for tau! [tau={sim.tau}]")

# cylinder centre & radius [m]
cx, cy = -0.1, 0
r = 0.01

# flow velocity [m/s]
# v0 = 0.2
v0 = 0.005

re_no = v0 * r / fld.nu
log.info(f"Reynolds no.:  {re_no:,.0f}")


# %% Compile

log.info("Compiling using Numba...")

from boltzmann.impl2 import *

# make numba objects
pidx = PeriodicDomain(dom.counts)
params = NumbaParams(sim.dt, dom.dx, sim.c, sim.tau)
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

XX = XX.flatten()
YY = YY.flatten()


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

# ensure cell types match across boundary
pidx.copy_periodic(cell)

# flag arrays
is_wall = (cell == CellType.BC_WALL.value).astype(np.int32)
update_vel = (cell == CellType.FLUID.value).astype(np.int32)

# set wall velocity to zero
# (just for nice output, doesn't affect the simulation)
vel[cell == CellType.BC_WALL.value, 0] = 0

# initial f is equilibrium for desired values of v and rho
calc_equilibrium(vel, rho, f1, np.float32(params.cs), d2q9)
f2[:] = f1[:]


# %% Define VTK output function


def calc_curl(pidx: PeriodicDomain, v: np.ndarray, rho: np.ndarray) -> np.ndarray:
    counts = pidx.counts

    # calc curl
    pidx.copy_periodic(v)
    curl = np.zeros_like(rho)
    for yidx in range(1, counts[1] - 1):
        for xidx in range(1, counts[0] - 1):
            idx = yidx * counts[0] + xidx

            # NOTE: Assumes zero wall velocity.
            # fmt: off
            dydx1 = v[idx - counts[0], 0] * (cell[idx - counts[0]] != CellType.BC_WALL.value)
            dydx2 = v[idx + counts[0], 0] * (cell[idx + counts[0]] != CellType.BC_WALL.value)
            dxdy1 = v[idx -         1, 1] * (cell[idx -         1] != CellType.BC_WALL.value)
            dxdy2 = v[idx +         1, 1] * (cell[idx +         1] != CellType.BC_WALL.value)
            # fmt: on

            curl[idx] = ((dydx2 - dydx1) - (dxdy2 - dxdy1)) / (2 * params.dx)

    return curl


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

    counts = pidx.counts

    # need 3d vectors for vtk
    def _to3d(v: np.ndarray):
        if v.shape[-1] == 2:
            return np.pad(v, [(0, 0), (0, 1)])
        else:
            return v

    # reshape data for writing
    v_ = unflatten(pidx, _to3d(v))
    rho_ = unflatten(pidx, rho)
    curl_ = unflatten(pidx, curl)
    cell_ = unflatten(pidx, cell)
    f_ = unflatten(pidx, f)

    # v_[cell_ == CellType.BC_WALL.value, :] = np.nan
    # rho_[cell_ == CellType.BC_WALL.value] = np.nan
    # curl_[cell_ == CellType.BC_WALL.value] = np.nan

    # cut off periodic part
    # v_ = v_[1:-1, 1:-1, :]
    # rho_ = rho_[1:-1, 1:-1]
    # curl_ = curl_[1:-1, 1:-1]
    # cell_ = cell_[1:-1, 1:-1]
    # f_ = f_[1:-1, 1:-1, :]

    # counts = counts - 2

    image_data = vtk.vtkImageData()
    nx, ny = list(counts)
    image_data.SetDimensions(nx, ny, 1)

    rho_data = vtk_np.numpy_to_vtk(num_array=rho_.ravel(), deep=False, array_type=vtk.VTK_FLOAT)
    rho_data.SetName("Density")
    rho_data.SetNumberOfComponents(1)

    vel_data = vtk_np.numpy_to_vtk(num_array=v_.ravel(), deep=False, array_type=vtk.VTK_FLOAT)
    vel_data.SetName("Velocity")
    vel_data.SetNumberOfComponents(3)

    curl_data = vtk_np.numpy_to_vtk(num_array=curl_.ravel(), deep=False, array_type=vtk.VTK_FLOAT)
    curl_data.SetName("Vorticity")
    curl_data.SetNumberOfComponents(1)

    wall_data = vtk_np.numpy_to_vtk(num_array=cell_.ravel(), deep=False, array_type=vtk.VTK_FLOAT)
    wall_data.SetName("CellType")
    wall_data.SetNumberOfComponents(1)

    p_data = image_data.GetPointData()
    p_data.AddArray(rho_data)
    p_data.AddArray(vel_data)
    p_data.AddArray(curl_data)
    p_data.AddArray(wall_data)

    if save_f:
        f_data = vtk_np.numpy_to_vtk(num_array=f_.ravel(), deep=False, array_type=vtk.VTK_FLOAT)
        f_data.SetName(f"F")
        f_data.SetNumberOfComponents(9)
        p_data.AddArray(f_data)

    p_data.SetActiveAttribute("Velocity", vtk.VTK_ATTRIBUTE_MODE_DEFAULT)

    zipper = vtk.vtkZLibDataCompressor()
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(path)
    writer.SetCompressor(zipper)
    writer.SetInputData(image_data)
    writer.Write()


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
curl = calc_curl(pidx, vel, rho)
write_vti(f"out/{pref}_{0:06d}.vti", vel, rho, curl, cell, f1, pidx, params)
write_png(f"out/{pref}_{0:06d}.png", vel, rho, curl)

t = 0.0
out_dt = 0.1
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

    curl = calc_curl(pidx, vel, rho)
    write_vti(f"out/{pref}_{out_i:06d}.vti", vel, rho, curl, cell, f1, pidx, params)
    write_png(f"out/{pref}_{out_i:06d}.png", vel, rho, curl)

    log.info(f"Wrote {out_i=} {out_t=:.3f}, {mlups_batch=:.2f}, {mlups_total=:.2f}\r")

    out_t += out_dt
