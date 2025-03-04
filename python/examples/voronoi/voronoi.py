# %% Imports

import logging
from boltzmann.units import Domain, Scales
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import os

from pathlib import Path
from scipy.signal import convolve2d
from scipy.ndimage import distance_transform_edt
from scipy.stats.qmc import PoissonDisk

from boltzmann.utils.logger import basic_config, dotted, timed
from boltzmann.core import (
    Simulation,
    CellFlags,
    calc_lbm_params_lu,
)
from boltzmann.simulation import TimeMeta, parse_cli, run_sim

np.random.seed(42)

basic_config()
logger = logging.getLogger(__name__)


def voronoi(
    points: np.ndarray,
    extent: np.ndarray,
    counts: np.ndarray,
    width: float,
) -> np.ndarray:
    """
    Generate a Voronoi diagram with a given width around the points.
    """
    assert points.ndim == 2
    assert points.shape[1] == 2
    assert extent.shape == (2, 2)
    assert counts.shape == (2,)

    size_x = extent[0, 1] - extent[0, 0]
    size_y = extent[1, 1] - extent[0, 0]

    idx_x = (counts[0] * (points[:, 0] - extent[0, 0]) / size_x).astype(np.int64)
    idx_y = (counts[1] * (points[:, 1] - extent[1, 0]) / size_y).astype(np.int64)

    WW = np.zeros(tuple(counts))
    WW[idx_x, idx_y] = 1

    # Tile so the middle chunk looks periodic.
    WW = np.tile(WW, (3, 3))

    KK = np.array(
        [
            [0, -1, 0],
            [-1, 4, -1],
            [0, -1, 0],
        ],
    )

    DD = distance_transform_edt(1 - WW)
    CC = convolve2d(DD, KK, mode="same") > 0

    # TODO: downsample here

    # Here we add some noise to the width check to make the channels varied.
    xx = np.linspace(extent[0, 0], extent[0, 1], counts[0])
    yy = np.linspace(extent[1, 0], extent[1, 1], counts[1])
    XX, YY = np.meshgrid(xx, yy)
    K = [[3, 5], [6, -3], [7, 23]]  # wave number
    A = [0.3, 0.2, 0.1]  # amplitude
    WW = np.ones_like(XX) * 0.8
    for k, a in zip(K, A):
        WW += a * np.sin(2 * np.pi * (k[0] * XX / size_x + k[1] * YY / size_y))
    WW = np.tile(WW, (3, 3)) * width

    VV = distance_transform_edt(1 - CC) < (WW / 2)

    # chop out middle of periodic tiling
    VV = VV[counts[0] : 2 * counts[0], counts[1] : 2 * counts[1]]

    return VV


# %% Params

x_si = 1.0  # domain size [m]
w_si = 0.04  # channel width [m]
g_si = 0.3  # Body acceleration [m/s^2]
nu_si = 1e-4  # kinematic viscosity [m^2/s]

L = 1500  # domain width [cells]

# implied quantities
u_si = (1 / 8) * (g_si / nu_si) * w_si**2  # flow velocity [m/s]
Re = int(u_si * w_si / nu_si)  # Reynolds number [1]
D = L * (w_si / x_si)  # channel width [cells]

(tau, u) = calc_lbm_params_lu(Re, D, tau=0.505, M_max=0.1)
dx = x_si / L
dt = (u / x_si) * dx

dotted(logger, "Reynolds number", Re)
dotted(logger, "tau", tau)
dotted(logger, "u", u)
dotted(logger, "L", L)
dotted(logger, "dx", dx)
dotted(logger, "dt", dt)

scales = Scales.make(dx=dx, dt=dt)

# dimensionless time does not depend on viscosity, purely on distances
out_dx_si = 0.5 * w_si  # want to output when flow has moved this far
sim_dx_si = u_si * dt  # flow moves this far in dt (i.e. one iteration)
n = out_dx_si / sim_dx_si
n = int(n + 1e-8)
out_dt_si = dt * n
out_n = int(4 * x_si / out_dx_si)

time_meta = TimeMeta.make(
    dt_step=dt,
    dt_output=out_dt_si,
    output_count=out_n,
)

dotted(logger, "outputs", out_n)

# geometry
domain = Domain.make(counts=[L, L], dx=dx)

# %% Initialize arrays

omega_ns = 1 / tau
omega_ad = 1 / 0.501

counts = domain.counts.astype(np.int32)

args = parse_cli()

# Either load the simulation or create it.
if args.resume:
    with timed(logger, "Loading simulation"):
        sim = Simulation.load_checkpoint(args.dev, str(args.base / "checkpoint.mpk"))
        tracer = sim.get_tracer(0)
else:
    with timed(logger, "Creating simulation"):
        sim = Simulation(args.dev, counts, q=9, omega_ns=omega_ns)
        tracer = sim.add_tracer(q=5, omega_ad=omega_ad)

    # Set-up wall boundaries
    p_disk = PoissonDisk(2, radius=2 * w_si, rng=np.random.default_rng(42))
    points = p_disk.fill_space()
    extent = np.array([[0, 1], [0, 1]])
    VV = voronoi(points, extent, counts, D)

    sim.cells.flags[:] |= 1 - VV

    # Set-up initial tracer
    x = np.arange(0, L, 1)
    XX, YY = np.meshgrid(x, x, indexing="ij")
    # mask = VV & (XX <= L // 11) & (5 * L // 11 <= YY) & (YY <= 6 * L // 11)
    # tracer.val[mask] = 1.0
    # sim.cells.flags[mask] |= CellType.FIXED_SCALAR_VALUE.value

    mask = VV & (YY <= L // 5)
    tracer.val[mask] = 1.0

    # IMPORTANT: convert gravity lattice units before setting
    g_lu = scales.acceleration.to_lattice_units(g_si)
    sim.set_gravity(np.array([0.0, g_lu], np.float32))


# %% Define simulation output


def write_output(base: Path, iter: int):
    cmap = plt.get_cmap("viridis")
    cmap.set_bad("gray")

    def _annotate(label: str):
        ax = plt.gca()
        ax.text(
            25,
            85,
            label,
            color="white",
            path_effects=[pe.withStroke(linewidth=2, foreground="k")],
            fontsize=15,
        )

    mask = (sim.cells.flags & CellFlags.WALL) > 0

    vmag = np.sqrt(np.sum(sim.fluid.vel**2, axis=-1))
    vmag[mask] = np.nan
    vmax = 0.9 * np.nanmax(vmag)
    plt.imshow(vmag.T, cmap=cmap, interpolation="none", vmax=vmax)
    _annotate("Velocity")
    plt.axis("off")
    plt.savefig(base / "vmag.png", bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()

    conc = tracer.val.copy()
    conc[mask] = np.nan
    plt.imshow(conc.T, cmap=cmap, interpolation="none", vmin=0.0, vmax=0.9)
    _annotate("Tracer")
    plt.axis("off")
    plt.savefig(base / f"conc_{iter:06d}.png", bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()

    cmap = plt.get_cmap("inferno")
    cmap.set_bad("gray")

    rho = sim.fluid.rho.copy()
    rho[mask] = np.nan
    plt.imshow(rho.T, cmap=cmap, interpolation="none")
    _annotate("Pressure")
    plt.axis("off")
    plt.savefig(base / "density.png", bbox_inches="tight", pad_inches=0, dpi=300)
    plt.close()

    print(scales.velocity.to_lattice_units(u_si), np.nanmax(vmag))


# %% Run the simulation

run_sim(args.base, time_meta, sim, write_output, write_checkpoints=not args.no_checkpoint)
