"""
Plane Poiseuille flow.
"""

# %%

# import numpy as np
# import os


# def analytical_soln(
#     l: float, g: float, nu: float, dx: float, dt: float, t_max: float, *, order: int = 25
# ) -> np.ndarray:
#     """
#     l - Plate separation
#     g - Acceleration due to gravity
#     nu - Kinematic viscosity
#     dx - Space abscissa
#     dt - Time abscissa
#     """

#     e = 1e-8
#     d = l / 2.0
#     X = np.linspace(-d, d, int(l / dx + e))
#     T = np.linspace(0, t_max, int(t_max / dt + e))

#     XX, TT = np.meshgrid(X, T)

#     # infinite time soln
#     VV = 0.5 * (g / nu) * (XX**2 - d**2)

#     for i in range(order):

#         j = np.pi * (2 * i + 1)
#         f = (16 * (-1) ** i * d**2 * g) / (nu * j**3)
#         VV += f * np.cos(j * XX / l) * np.exp(-(j**2 * nu * TT) / (l**2))

#     return VV, XX, TT


# %%
# import matplotlib.pyplot as plt


# VV, XX, TT = analytical_soln(0.01, 9.81, 1e-6, 0.0001, 0.01, 1.0)

# plt.plot(XX[0, :], VV[0, :])
# plt.plot(XX[10, :], VV[10, :])
# plt.plot(XX[40, :], VV[40, :])
# plt.plot(XX[80, :], VV[80, :])

# %% Imports

import logging
import os

import numpy as np

from boltzmann.utils.logger import tick, PerfInfo, basic_config
from boltzmann.core import DomainMeta, SimulationMeta, FluidMeta, D2Q9, CellType

log = logging.getLogger(__name__)

basic_config(log)

log.info("Starting")


# %% Params

dom = DomainMeta.with_extent_and_counts(extent=[[0.00, 0.01], [0.00, 0.01]], counts=[101, 101])
fld = FluidMeta(mu=0.01, rho=100)
sim = SimulationMeta.with_cs(domain=dom, fluid=fld, cs=0.1)

log.info(f"{dom.dx=}, {dom.extent=}, {dom.counts=}, cells={np.prod(dom.counts):,d}")
log.info(f"{fld.mu=}, {fld.rho=}, {fld.nu=}")
log.info(f"{sim.tau=}, {sim.cs=}, {sim.dt=}")

if sim.tau < 0.6:
    log.warning(f"Small value for tau! [tau={sim.tau}]")

l_ = 0.25
nu_ = fld.nu / (sim.cs**2 * sim.dt)
# nu_ = fld.nu / sim.cs**2
wpve = 1.0 / (nu_ + 0.5)
wnve = 1.0 / (l_ / nu_ + 0.5)
# wnve = wpve

tau_pve = 1.0 / wpve
tau_nve = 1.0 / wnve

log.info(f"{tau_pve=} {tau_nve=} {nu_=}")

# pipe width [m]
l_si = dom.extent[0, 1] - dom.extent[0, 0]

# gravitational acceleration [m/s^2]
g_si = 0.1
g_lu = g_si * (sim.dt**2 / dom.dx)

# max infinite time velocity
v_si = (g_si / fld.nu) * (l_si**2) / 8
log.info(f"Max vel.: {v_si:.04f} m/s")

re_no = v_si * l_si / fld.nu
log.info(f"Reynolds no.:  {re_no:,.0f}")

g_lu = np.array([0, g_lu], dtype=np.float32)


# %% Compile

log.info("Compiling using Numba...")

from boltzmann.impl2 import *


# make numba objects
pidx = PeriodicDomain(dom.counts)
params = NumbaParams(sim.dt, dom.dx, sim.cs, sim.tau, wpve, wnve, g_lu)
d2q9 = NumbaModel(D2Q9.ws, D2Q9.qs, D2Q9.js, pidx.counts)

# %% Initialize arrays

f1 = make_array(pidx, 9)
f2 = make_array(pidx, 9)
vel = make_array(pidx, 2)
rho = make_array(pidx, fill=fld.rho)
cell = make_array(pidx, dtype=np.int32)

# rho_ = unflatten(pidx, rho)
# rho_[126, 126] = 1.05

cell_ = unflatten(pidx, cell)
cell_[:, 1] = CellType.BC_WALL.value
cell_[:, -2] = CellType.BC_WALL.value

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

from boltzmann.vtkio import VtiWriter


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


# %% Main Loop

tmpl_out = "poiseuille/out_{:06d}.vti"

path = os.path.dirname(tmpl_out)
if not os.path.exists(path):
    os.makedirs(path)

# write out the zeroth timestep
curl = calc_curl_2d(pidx, vel, cell, params.dx)
write_vti(tmpl_out.format(0), vel, rho, curl, cell, f1, pidx, params)

t = 0.0
out_dt = 0.1
# out_dt = sim.dt
out_i = 1
out_t = out_dt
max_i = 31

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
    write_vti(tmpl_out.format(out_i), vel, rho, curl, cell, f1, pidx, params)

    log.info(f"Wrote {out_i=} {out_t=:.3f}, {mlups_batch=:.2f}, {mlups_total=:.2f}\r")

    out_t += out_dt
