"""
The core lattice Boltzmann simulation classes.

The Rust extension module is re-exported here for convenience & clarity.
If you wish to import from that directly you can import ``boltzmann.boltzmann``.

See :doc:`guides/basic <guides/basic>` for more information on how to use this module.
"""

from __future__ import annotations
from typing import Literal

import numpy as np

from math import sqrt


__all__ = [
    "bgk",
    "trt",
    "Device",
    "CellFlags",
    "calc_lbm_params_lu",
    "calc_lbm_params_si",
    "check_lbm_params",
]

# ------------------------------------
# Re-export the rust module here.

import boltzmann.boltzmann  # type: ignore
from boltzmann.boltzmann import *  # type: ignore  # noqa: F403

if hasattr(boltzmann.boltzmann, "__all__"):
    __all__.extend(boltzmann.boltzmann.__all__)

# ------------------------------------


def bgk(tau: float) -> Omega:
    """
    Construct a BGK collision operator.

    Args:
        tau: The relaxation time in simulation units (i.e. iterations).
    """
    return Omega.BGK(1.0 / tau)


def trt(tau_pos: float, tau_neg: float | None = None, magic_no: float = 0.25) -> Omega:
    """
    Construct a TRT collision operator.

    Args:
        tau_pos: The relaxation time in simulation units (i.e. iterations).
        tau_neg: The relaxation time in simulation units (i.e. iterations).
            If not specified this will be inferred from `magic_no`.
        magic_no: Used to infer `tau_neg` if not specified.
            Defaults to 1/4. See Lattice Boltzmann Principles & Practice, Krueger et al, Section 10.7.2
    """
    tau_neg = tau_neg or (magic_no / (tau_pos - 0.5) + 0.5)
    return Omega.TRT(1.0 / tau_pos, 1.0 / tau_neg)


Device = Literal["cpu", "gpu"]  #: OpenCL device type on which type to run the simulations.


class CellFlags:
    """
    Flags which control the simulations behaviour at a given cell.
    Multiple flags can be combined using the bitwise OR operator.
    """

    # fmt: off
    FLUID = 0                 #: The default value, the fluid & any tracers will evolve as usual.
    WALL = 1                  #: The fluid obeys the `no-slip boundary condition <https://en.wikipedia.org/wiki/No-slip_condition>`_ & tracers will see `zero concentration gradient <https://en.wikipedia.org/wiki/Neumann_boundary_condition>`_.
    FIXED_FLUID_VELOCITY = 2  #: The fluid velocity is fixed at the initial value, the density will evolve as usual.
    FIXED_FLUID_DENSITY = 4   #: The fluid density is fixed at the initial value, the velocity will evolve as usual.
    FIXED_SCALAR_VALUE = 8    #: The scalar value is fixed at the initial value. The fluid evolves as normal.
    FIXED_FLUID = FIXED_FLUID_VELOCITY | FIXED_FLUID_DENSITY  #: Both fluid density & velocity are fixed to their initial values.
    # fmt: on


def check_lbm_params(
    Re: float, L: float, tau: float, M_max: float = 0.1, slack: float = 0.0, bgk: bool = True
):
    """
    Check if the chosen parameters are likely to be stable or not with BGK collision operator.
    For more information see https://dbarker.uk/lbm_parameterization/.

    :param Re: Desired Reynold's number of the system.
    :param L: Characteristic length of the system.
    :param tau: Relaxation time of the system.
    :param M_max: Maximum Mach number of the system.
    :param slack: Extra margin on the checks. Must be in range (-1, inf).
                  Values < 0 imply stricter checks, wheras > 0 imply more relaxed checks.
    :param bgk: Perform the bgk stability check or not.
    """

    frac = slack + 1.0

    # Maximum value of tau implied by Mach condition.
    tau_mach = np.sqrt(3) * (M_max * frac) * L / Re + 0.5

    # Minimum value of L implied by the BGK stability condition.
    L_stab = (Re / frac) / 24

    eps = 1e-8
    assert (not bgk) or (L + eps > L_stab), f"BGK stability condition violated. {L=} < {L_stab}"
    assert tau < tau_mach + eps, f"Mach condition violated. {tau=} > {tau_mach}"


def calc_lbm_params_lu(
    # params
    Re: float,
    L: float,
    tau: float | None = None,
    # bounds
    M_max: float = 0.1,
    tau_max: float = 1.0,
    tau_min: float = 0.5,
    slack: float = 0.0,
    bgk: bool = True,
) -> tuple[float, float]:
    """
    Choose the values of tau and u which maximize simulation speed & verify stability.
    If you specify tau that value will be used instead.

    :param Re: Desired Reynold's number of the system.
    :param L: Characteristic length of the system.
    :param tau: Relaxation time of the system.
    :param M_max: Maximum Mach number of the system.
    :param tau_max: Hard cap on the value of tau allowed.
    :param tau_min: Hard floor on the value of tau allowed.
    :param slack: See :py:meth:`check_lbm_params`.
    :param bgk: Perform the bgk stability check or not.
    """

    if tau is None:
        tau = sqrt(3) * M_max * L / Re + 0.5  # Maximum permissible by Mach condition
        tau = min(max(tau, tau_min), tau_max)  # Hard bounds on tau

    check_lbm_params(Re, L, tau, M_max, slack, bgk)

    nu = (tau - 0.5) / 3
    u = Re * nu / L

    return (tau, u)


def calc_lbm_params_si(
    # params
    dx: float,
    u_si: float,
    L_si: float,
    nu_si: float,
    tau: float | None = None,
    # bounds
    M_max: float = 0.1,
    tau_max: float = 10.0,
    tau_min: float = 0.5,
    slack: float = 0.0,
    bgk: bool = True,
) -> tuple[float, float]:
    """
    Choose the values of tau and u which maximize simulation speed & verify stability.
    If you specify tau that value will be used instead.

    Sometimes it is more natural to specify the physical parameters directly rather than via
    Reynold's number.
    Under the hood, however, this function calls :py:meth:`calc_lbm_params_lu`.

    .. note::

        The suffix ``_si`` means the arguments are in SI units.
        The function still returns the LBM parameters in lattice units.

    :param dx: Lattice spacing.
    :param u_si: Characteristic velocity of the system.
    :param L_si: Characteristic length-scale of the system.
    :param nu_si: Kinematic viscosity of the system.
    :param tau: Relaxation time of the BGK collision operator.
    :param M_max: Maximum Mach number of the system.
    :param tau_max: Hard cap on the value of tau allowed.
    :param tau_min: Hard floor on the value of tau allowed.
    :param slack: See :py:meth:`check_lbm_params`.
    :param bgk: Perform the bgk stability check or not.
    """

    Re = u_si * L_si / nu_si
    L = L_si / dx
    tau, u = calc_lbm_params_lu(
        Re, L, tau=tau, M_max=M_max, tau_min=tau_min, tau_max=tau_max, slack=slack, bgk=bgk
    )

    # sanity check:
    if tau is None:
        tau_ = np.sqrt(3) * nu_si * M_max / (dx * u_si) + 0.5
        tau_ = min(tau_, tau_max)
        assert abs(tau - tau_) < 1e-8, f"tau mismatch: {tau=} vs {tau_=}"

    return (tau, u)
