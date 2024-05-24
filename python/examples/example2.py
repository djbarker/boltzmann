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

# dom = DomainMeta.with_extent_and_counts(extent=[[-1, 1], [-1, 1.1]], counts=[200, 220])
dom = DomainMeta.with_extent_and_counts(extent=[[-0.2, 0.2], [-0.1, 0.1]], counts=[1000, 500])
fld = FluidMeta(mu=0.001, rho=1000)
sim = SimulationMeta.with_cs(domain=dom, fluid=fld, cs=0.16)
# sim = SimulationMeta(domain=dom, fluid=fld, dt=0.001)

log.info(f"{dom.dx=}, {dom.extent=}, {dom.counts=}, cells={np.prod(dom.counts):,d}")
log.info(f"{fld.mu=}, {fld.rho=}, {fld.nu=}")
log.info(f"{sim.tau=}, {sim.c=}, {sim.dt=}")

if sim.tau < 0.6:
    log.warning(f"Small value for tau! [tau={sim.tau}]")

# %% Compile

log.info("Compiling using Numba...")

from boltzmann.impl2 import *

# make numba objects
pidx = PeriodicDomain(dom.counts)
params = NumbaParams(sim.dt, dom.dx, sim.c, sim.tau)
d2q9 = NumbaModel(D2Q9.ws, D2Q9.qs, D2Q9.js, pidx.counts)

# %% Initialize arrays

f = make_array(pidx, 9)
v = make_array(pidx, 2)
cell = make_array(pidx, dtype=np.int32)
rho = make_array(pidx, fill=fld.rho)
# rho *= 1 + 0.001 * np.random.uniform(size=rho.shape)
curl = make_array(pidx)
feq = np.zeros_like(f)

x = np.pad(dom.x, (1, 1), mode="edge")
y = np.pad(dom.y, (1, 1), mode="edge")
xx, yy = np.meshgrid(x, y)

xx = xx.flatten()
yy = yy.flatten()

# --- Cylinder
cell[:] = CellType.BC_WALL.value * (((xx - -0.1) ** 2 + (yy - 0.0) ** 2) < 0.01**2.0)

# --- In-flow jet
# cell[1, :150] = CellType.BC_WALL.value
# cell[1, -150:] = CellType.BC_WALL.value
# cell[1, 150:-150] = CellType.FIXED_VELOCITY.value
# v[:, 150:-150, 0] = sim.c * 0.1

cell[make_slice_y1d(pidx, 1)] = CellType.FIXED_VELOCITY.value
v[:, 0] = sim.c * 0.1 * (1 - np.exp(-((yy / 0.04) ** 2) - ((xx - -0.1) / 0.04) ** 2))
v[make_slice_y1d(pidx, 1), 0] = sim.c * 0.1
# v[make_slice_y1d(pidx, 200, 90, 110)] = sim.c * 0.099

# v[((xx - -1.5) ** 2 + (yy - 0.0) ** 2) < 0.2**2.0, 0] = sim.c * 0.1
# v[((xx - -1.2) ** 2 + (yy - 0.0) ** 2) < 0.2**2.0, 1] = sim.c * 0.02

# flag arrays
is_wall = (cell == CellType.BC_WALL.value).astype(np.int32)
update_vel = (cell == CellType.FLUID.value).astype(np.int32)

# initial f is equilibrium for desired values of v and rho
calc_equilibrium(v, rho, f, np.float32(params.cs), d2q9)
feq[:] = f[:]

# ensure cell types match across boundary
pidx.copy_periodic(cell)


# %% Define VTK out


def write_vti(
    path: str,
    pidx: PeriodicDomain,
    params: NumbaParams,
    v: np.ndarray,
    rho: np.ndarray,
    cell: np.ndarray,
    f: np.ndarray,
):

    counts = pidx.counts

    # calc curl
    pidx.copy_periodic(v)
    curl = np.zeros_like(rho)
    for yidx in range(1, counts[1] - 1):
        for xidx in range(1, counts[0] - 1):
            idx = yidx * counts[0] + xidx

            # NOTE: Assumes zero wall velocity.
            # fmt: off
            vy1 = v[idx - counts[0], 1] * (cell[idx - counts[0]] != CellType.BC_WALL.value)
            vy2 = v[idx + counts[0], 1] * (cell[idx + counts[0]] != CellType.BC_WALL.value)
            vx1 = v[idx -         1, 0] * (cell[idx -         1] != CellType.BC_WALL.value)
            vx2 = v[idx +         1, 0] * (cell[idx +         1] != CellType.BC_WALL.value)
            # fmt: on

            curl[idx] = ((vy2 - vy1) - (vx2 - vx1)) / (2 * params.dx)

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

    # v_T[cell_T == CellType.BC_WALL.value, :] = np.nan
    # rho_T[cell_T == CellType.BC_WALL.value] = np.nan
    # curl_T[cell_T == CellType.BC_WALL.value] = np.nan

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

    # f_data = vtk_np.numpy_to_vtk(num_array=f_.ravel(), deep=False, array_type=vtk.VTK_FLOAT)
    # f_data.SetName(f"F")
    # f_data.SetNumberOfComponents(9)
    # p_data.AddArray(f_data)

    p_data.SetActiveAttribute("Velocity", vtk.VTK_ATTRIBUTE_MODE_DEFAULT)

    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(path)
    writer.SetInputData(image_data)
    writer.Write()


# write out the zeroth timestep
pref = "example2"
write_vti(f"out/{pref}_{0:06d}.vti", pidx, params, v, rho, cell, f)

# %% Main Loop

t = 0.0
out_dt = 0.1
# out_dt = sim.dt
out_i = 1
out_t = out_dt
max_i = 1000

batch_i = int((out_dt + 1e-8) // sim.dt)

log.info(f"{batch_i:,d} iters/output")

prog_msg = "Wrote out_i={out_i} out_t={out_t:.3f}, mlups_batch={mlups_batch:.2f}, mlups_total={mlups_total:.2f}\r".format
perf_total = PerfInfo()

for out_i in range(1, max_i):
    perf_batch = tick()

    loop_for_2(batch_i, v, rho, f, feq, is_wall, update_vel, params, pidx, d2q9)

    perf_batch = perf_batch.tock(events=np.prod(dom.counts) * batch_i)
    perf_total = perf_total + perf_batch

    mlups_batch = perf_batch.events / (1e6 * perf_batch.seconds)
    mlups_total = perf_total.events / (1e6 * perf_total.seconds)

    log.info(prog_msg(out_i=out_i, out_t=out_t, mlups_batch=mlups_batch, mlups_total=mlups_total))

    write_vti(f"out/{pref}_{out_i:06d}.vti".format(out_i=out_i), pidx, params, v, rho, cell, f)

    out_t += out_dt
